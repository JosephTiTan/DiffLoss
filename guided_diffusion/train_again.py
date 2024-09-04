import copy
import functools
import os

import blobfile as bf
import torch as th
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel as DDP
from torch.optim import AdamW

from . import dist_util, logger
from . import metrics_util as Metrics
from .fp16_util import MixedPrecisionTrainer
from .nn import update_ema
from .resample import LossAwareSampler, UniformSampler
from torch.cuda.amp import autocast as autocast
import torch
import torch.nn as nn
import torch.nn.init as init



# For ImageNet experiments, this was a good default value.
# We found that the lg_loss_scale quickly climbed to
# 20-21 within the first ~1K steps of training.
INITIAL_LOG_LOSS_SCALE = 20.0

result_path = "/home/tjt/code/DiffLoss_new/result"
file_path = "/home/tjt/code/DiffLoss_new/result/log.txt"
class TrainLoop:
    def __init__(
        self,
        *,
        Restoration_model,
        model,
        diffusion,
        data,
        testdata,
        batch_size,
        microbatch,
        lr,
        ema_rate,
        log_interval,
        save_interval,
        resume_checkpoint,
        use_fp16=False,
        fp16_scale_growth=1e-3,
        schedule_sampler=None,
        weight_decay=0.0,
        lr_anneal_steps=0,
    ):
        # self.Restoration_model = Restoration_model
        self.ddp_model = model
        self.diffusion = diffusion
        self.data = data
        self.val_loader = testdata
        self.batch_size = batch_size
        self.microbatch = microbatch if microbatch > 0 else batch_size
        self.lr = lr
        self.ema_rate = (
            [ema_rate]
            if isinstance(ema_rate, float)
            else [float(x) for x in ema_rate.split(",")]
        )
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.test_interval = 485
        self.resume_checkpoint = resume_checkpoint
        self.use_fp16 = use_fp16
        self.fp16_scale_growth = fp16_scale_growth
        self.schedule_sampler = schedule_sampler or UniformSampler(diffusion)
        self.weight_decay = weight_decay
        self.lr_anneal_steps = lr_anneal_steps

        self.step = 0
        self.resume_step = 0
        self.plot = False
        self.global_batch = self.batch_size * dist.get_world_size()

        self.sync_cuda = th.cuda.is_available()
        
        # for param in self.model.parameters():
        #     param.requires_grad = False
        
        # self.mp_trainer = MixedPrecisionTrainer(
        #     model=self.model,
        #     use_fp16=self.use_fp16,
        #     fp16_scale_growth=fp16_scale_growth,
        # )
        self.Restoration_model = Restoration_model
        # self.mp_trainer_Restoration = MixedPrecisionTrainer(
        #     model=Restoration_model,
        #     use_fp16=False,
        #     fp16_scale_growth=fp16_scale_growth,
        # )
        #
        self.opt = AdamW(self.Restoration_model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        # self.ema_params = [copy.deepcopy(self.mp_trainer_Restoration.master_params) for _ in range(len(self.ema_rate))]
        
        self.use_ddp = False
        # self.ddp_model = self.model
        # self.ddp_model_Restoration = self.Restoration_model

        
        # if th.cuda.is_available():
        #     self.use_ddp = True
        #     self.ddp_model = DDP(
        #         self.model,
        #         device_ids=[dist_util.dev()],
        #         output_device=dist_util.dev(),
        #         broadcast_buffers=False,
        #         bucket_cap_mb=128,
        #         find_unused_parameters=False,
        #     )
        #     self.ddp_model_Restoration = DDP(
        #         self.Restoration_model,
        #         device_ids=[dist_util.dev()],
        #         output_device=dist_util.dev(),
        #         broadcast_buffers=False,
        #         bucket_cap_mb=128,
        #         find_unused_parameters=False,
        #     )
        # else:
        #     if dist.get_world_size() > 1:
        #         logger.warn("Distributed training requires CUDA. " "Gradients will not be synchronized properly!")
        #     self.use_ddp = False
        #     self.ddp_model = self.model
        #     self.ddp_model_Restoration = self.Restoration_model


    def _load_optimizer_state(self):
        main_checkpoint = find_resume_checkpoint() or self.resume_checkpoint
        opt_checkpoint = bf.join(
            bf.dirname(main_checkpoint), f"opt{self.resume_step:06}.pt"
        )
        if bf.exists(opt_checkpoint):
            logger.log(f"loading optimizer state from checkpoint: {opt_checkpoint}")
            state_dict = dist_util.load_state_dict(
                opt_checkpoint, map_location=dist_util.dev()
            )
            self.opt.load_state_dict(state_dict)

    def run_loop(self):
        while (not self.lr_anneal_steps or self.step + self.resume_step < self.lr_anneal_steps):
            batch_degrad, batch_gt = next(self.data)
            self.run_step(batch_degrad, batch_gt)
            if self.step % self.log_interval == 0:
                logger.dumpkvs()
            if self.step % self.save_interval == 0:
                self.save()
            if self.step % self.test_interval == 0:
                self.test(self.val_loader, self.step, plot = True)
                # Run for a finite amount of time in integration tests.
                if os.environ.get("DIFFUSION_TRAINING_TEST", "") and self.step > 0:
                    return
            self.step += 1
        # Save the last checkpoint if it wasn't already saved.

        self.plot = True
        self.test(self.val_loader, self.step, self.plot)
        self.plot = False

        if (self.step - 1) % self.save_interval != 0:
            self.save()



    def run_step(self, batch_degrad, batch_gt):
        self.forward_backward(batch_degrad, batch_gt)

        # took_step = self.mp_trainer_Restoration.optimize(self.opt)
        # if took_step:
        #     self._update_ema()
        # self._anneal_lr()
        # self.log_step()

    def forward_backward(self, batch_degrad, batch_gt):
        # self.mp_trainer_Restoration.zero_grad()
        for i in range(0, batch_degrad.shape[0], self.microbatch):

            micro_degrad = batch_degrad[i : i + self.microbatch].to(dist_util.dev())
            micro_gt = batch_gt[i : i + self.microbatch].to(dist_util.dev())
            last_batch = (i + self.microbatch) >= batch_degrad.shape[0]
            t, weights = self.schedule_sampler.sample(micro_degrad.shape[0], dist_util.dev())

            losses, output = self.diffusion.training_losses(
                self.Restoration_model,
                self.ddp_model,
                micro_degrad,
                micro_gt,
                t,
            )
                # print(losses)

            if isinstance(self.schedule_sampler, LossAwareSampler):
                self.schedule_sampler.update_with_local_losses(t, losses["loss"].detach())

            loss = (losses["loss"] * weights).mean()
            log_loss_dict(self.diffusion, t, {k: v * weights for k, v in losses.items()})
            loss.backward()
            self.opt.step()

            
            # print("Gradients before backward:")
            # for name, param in self.ddp_model.named_parameters():
            #     print(f"{name}: {param.grad}")
            

            
            # 反向传播后，检查梯度
            # print("\nGradients after backward:")
            # for name, param in self.ddp_model.named_parameters():
            #     if param.requires_grad:
            #         print(f"{name}: {param.grad}")
    def test(self, val_loader, current_step, plot=False):
        print('Testing begin...')
        out_dict = {}
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_loss = 0.0
        idx = 0
        L2lloss = nn.MSELoss()
        self.Restoration_model.eval()
        with torch.no_grad():
            for i,  val_data in enumerate(val_loader):
                idx += 1
                micro_degrad, micro_gt = val_data
                micro_degrad = micro_degrad.to(dist_util.dev())
                micro_gt = micro_gt.to(dist_util.dev())

                output = self.Restoration_model.model(micro_degrad)

                loss = L2lloss(micro_gt, output)

                output = (output + 1)
                micro_gt = (micro_gt + 1)
                micro_degrad = (micro_degrad + 1)

                out_dict['Out'] = output.detach().float().cpu()
                out_dict['HR'] = micro_gt.detach().float().cpu()
                out_dict['LR'] = micro_degrad.detach().float().cpu()
                print('Detach finish')
                out_img = Metrics.tensor2img(out_dict['Out'])  # uint8
                hr_img = Metrics.tensor2img(out_dict['HR'])  # uint8
                lr_img = Metrics.tensor2img(out_dict['LR'])  # uint8
                # print('tensor2img finish')


                Metrics.save_img(hr_img, '{}/{}_hr.png'.format(result_path, idx))
                Metrics.save_img(out_img, '{}/{}_out.png'.format(result_path, idx))
                Metrics.save_img(lr_img, '{}/{}_lr.png'.format(result_path, idx))
                    # print('plot finish')

                eval_psnr = Metrics.calculate_psnr(out_img, hr_img)
                eval_ssim = Metrics.calculate_ssim(out_img, hr_img)

                avg_psnr += eval_psnr
                avg_ssim += eval_ssim
                avg_loss += loss

                if idx == 15:
                    break

            avg_psnr = avg_psnr / idx
            avg_ssim = avg_ssim / idx
            avg_loss = avg_loss / idx
            print('# Step : {:8,d}'.format(current_step))
            print('# Validation # PSNR: {:.4e}'.format(avg_psnr))
            print('# Validation # SSIM: {:.4e}'.format(avg_ssim))
            print('# Validation # LOSS: {:.4e}'.format(avg_loss))

            with open(file_path, 'a') as file:
                file.write('# Step : {:8,d}'.format(current_step) + '\n')
                file.write('# Validation # PSNR: {:.4e}'.format(avg_psnr) + '\n')
                file.write('# Validation # SSIM: {:.4e}'.format(avg_ssim) + '\n')
                file.write('# Validation # LOSS: {:.4e}'.format(avg_loss) + '\n')


        self.mp_trainer_Restoration.model.train()






    def _update_ema(self):
        for rate, params in zip(self.ema_rate, self.ema_params):
            update_ema(params, self.mp_trainer_Restoration.master_params, rate=rate)

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr

    def log_step(self):
        logger.logkv("step", self.step + self.resume_step)
        logger.logkv("samples", (self.step + self.resume_step + 1) * self.global_batch)

    def save(self):
        def save_checkpoint(rate, params):
            state_dict = self.mp_trainer_Restoration.master_params_to_state_dict(params)
            if dist.get_rank() == 0:
                logger.log(f"saving model {rate}...")
                if not rate:
                    filename = f"model{(self.step+self.resume_step):06d}.pt"
                else:
                    filename = f"ema_{rate}_{(self.step+self.resume_step):06d}.pt"
                with bf.BlobFile(bf.join(get_blob_logdir(), filename), "wb") as f:
                    th.save(state_dict, f)

        save_checkpoint(0, self.mp_trainer_Restoration.master_params)
        for rate, params in zip(self.ema_rate, self.ema_params):
            save_checkpoint(rate, params)

        if dist.get_rank() == 0:
            with bf.BlobFile(
                bf.join(get_blob_logdir(), f"opt{(self.step+self.resume_step):06d}.pt"),
                "wb",
            ) as f:
                th.save(self.opt.state_dict(), f)

        dist.barrier()


def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0


def get_blob_logdir():
    # You can change this to be a separate path to save checkpoints to
    # a blobstore or some external drive.
    return logger.get_dir()


def find_resume_checkpoint():
    # On your infrastructure, you may want to override this to automatically
    # discover the latest checkpoint on your blob storage, etc.
    return None


def find_ema_checkpoint(main_checkpoint, step, rate):
    if main_checkpoint is None:
        return None
    filename = f"ema_{rate}_{(step):06d}.pt"
    path = bf.join(bf.dirname(main_checkpoint), filename)
    if bf.exists(path):
        return path
    return None


def log_loss_dict(diffusion, ts, losses):
    for key, values in losses.items():
        logger.logkv_mean(key, values.mean().item())
        # Log the quantiles (four quartiles, in particular).
        for sub_t, sub_loss in zip(ts.cpu().numpy(), values.detach().cpu().numpy()):
            quartile = int(4 * sub_t / diffusion.num_timesteps)
            logger.logkv_mean(f"{key}_q{quartile}", sub_loss)
