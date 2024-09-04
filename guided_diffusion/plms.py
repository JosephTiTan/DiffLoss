"""SAMPLING ONLY."""
import os
import torch
from torch import nn
import torchvision
import numpy as np
import numpy
from tqdm import tqdm
from functools import partial
from PIL import Image
import shutil
from torch import optim
from tqdm.auto import tqdm
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler   
import torch.utils.checkpoint as checkpoint

from ldm.modules.diffusionmodules.util import (
    make_ddim_sampling_parameters,
    make_ddim_timesteps,
)
import clip
from einops import rearrange
import random



class PLMSSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)

    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.0, verbose=True):
        if ddim_eta != 0:
            raise ValueError("ddim_eta must be 0 for PLMS")
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.ddpm_num_timesteps,
            verbose=verbose,
        )
        alphas_cumprod = self.model.alphas_cumprod
        assert (alphas_cumprod.shape[0] == self.ddpm_num_timesteps), "alphas have to be defined for each timestep"
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer("betas", to_torch(self.model.betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod.cpu())),)
        self.register_buffer("log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu())))
        self.register_buffer("sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod.cpu() - 1)),)

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=alphas_cumprod.cpu(),
            ddim_timesteps=self.ddim_timesteps,
            eta=0.0,
            verbose=verbose,
        )
        self.register_buffer("ddim_sigmas", ddim_sigmas)
        self.register_buffer("ddim_alphas", ddim_alphas)
        self.register_buffer("ddim_alphas_prev", ddim_alphas_prev)
        self.register_buffer("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev)
            / (1 - self.alphas_cumprod)
            * (1 - self.alphas_cumprod / self.alphas_cumprod_prev)
        )
        self.register_buffer("ddim_sigmas_for_original_num_steps", sigmas_for_original_sampling_steps)




    def plms_sample(
        self,
        x,
        condition,
        t,
        index,
        repeat_noise=False,
        use_original_steps=False,
        quantize_denoised=False,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        old_eps=None,
        t_next=None,
        input_image=None,
        noise_save_path=None,
    ):
        b, *_, device = *x.shape, x.device

        def optimize_model_output(x, t, condition):
            if (unconditional_conditioning is None or unconditional_guidance_scale == 1.0):
                e_t = self.model.apply_model(x, t, condition)
            else:
                x_in = torch.cat([x] * 2)
                t_in = torch.cat([t] * 2)
                c_in = torch.cat([unconditional_conditioning, condition])
                e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                e_t = e_t_uncond + unconditional_guidance_scale * (e_t - e_t_uncond)
            return e_t

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = (self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev)
        sqrt_one_minus_alphas = (self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas)
        sigmas = (self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas)

        def get_x_prev_and_pred_x0(e_t, index):
            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            if quantize_denoised:
                pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)
            # direction pointing to x_t
            dir_xt = (1.0 - a_prev - sigma_t ** 2).sqrt() * e_t
            time_curr = index * 20 + 1
            if noise_save_path and index > 16:
                noise = torch.load(noise_save_path + "_time%d.pt" % (time_curr))[:1]
            else:
                noise = torch.zeros_like(dir_xt)
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            return x_prev, pred_x0

        e_t = optimize_model_output(x, t, condition)
        if len(old_eps) == 0:
            x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t, index)
            e_t_next = optimize_model_output(x_prev, t_next, condition)
            e_t_prime = (e_t + e_t_next) / 2
        elif len(old_eps) == 1:
            e_t_prime = (3 * e_t - old_eps[-1]) / 2
        elif len(old_eps) == 2:
            e_t_prime = (23 * e_t - 16 * old_eps[-1] + 5 * old_eps[-2]) / 12
        elif len(old_eps) >= 3:
            e_t_prime = (55 * e_t - 59 * old_eps[-1] + 37 * old_eps[-2] - 9 * old_eps[-3]) / 24

        x_prev, pred_x0 = get_x_prev_and_pred_x0(e_t_prime, index)

        return x_prev, pred_x0, e_t





#########################################
    def ddim_sample_loop(
        self,
        S,
        cond=None,
        shape,
        ddim_use_original_steps=False,
        timesteps=None,
        quantize_x0=False,
        eta=0.0,
        mask=None,
        x0=None,
        temperature=1.0,
        noise_dropout=0.0,
        score_corrector=None,
        corrector_kwargs=None,
        verbose=True,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
        input_image=None,
        noise_save_path=None,
        **kwargs,
    ):
        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        device = self.model.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        img_clone = img.clone()


        timesteps = self.ddim_timesteps
        time_range = np.flip(timesteps)

        with torch.no_grad():
            img = img_clone.clone()
            total_steps = timesteps.shape[0]
            iterator = time_range
            old_eps = []
            with autocast():
                for i, step in enumerate(tqdm(iterator)):
                    index = total_steps - i - 1
                    ts = torch.full((b,), step, device=device, dtype=torch.long)
                    ts_next = torch.full((b,), time_range[min(i + 1, len(time_range) - 1)], device=device, dtype=torch.long,)
                    outs = self.plms_sample(
                            img,
                            cond,
                            ts,
                            index=index,
                            use_original_steps=ddim_use_original_steps,
                            quantize_denoised=quantize_denoised,
                            temperature=temperature,
                            noise_dropout=noise_dropout,
                            score_corrector=score_corrector,
                            corrector_kwargs=corrector_kwargs,
                            unconditional_guidance_scale=unconditional_guidance_scale,
                            unconditional_conditioning=unconditional_conditioning,
                            old_eps=old_eps,
                            t_next=ts_next,
                            input_image=input_image,
                            noise_save_path=noise_save_path,
                    )
                    img, pred_x0, e_t = outs
                    old_eps.append(e_t)
                    if len(old_eps) >= 4:
                        old_eps.pop(0)
                img_ddim = torch.clamp((img + 1.0) / 2.0, min=0.0, max=1.0)
                img_ddim = img_ddim.cpu().permute(0, 2, 3, 1).permute(0, 3, 1, 2)
                # save image
                with torch.no_grad():
                    x_sample = 255.0 * rearrange(img_ddim[0].detach().cpu().numpy(), "c h w -> h w c")
                    imgsave = Image.fromarray(x_sample.astype(np.uint8))
                    imgsave.save(image_save_path + "{}-{}.png".format(w1,w2))
        torch.cuda.empty_cache()

        return None