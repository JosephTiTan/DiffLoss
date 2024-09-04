"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

from torchvision.utils import make_grid
from typing import Optional
from PIL import Image

import argparse
import os
import cv2
import numpy as np
import torch as th
import torch
import torch.distributed as dist
import random
from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from evaluations.evaluator import compute_fid



def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    
def save_image(images: torch.Tensor, nrow: int = 8, show: bool = True, path: Optional[str] = None, format: Optional[str] = None, to_grayscale: bool = False, **kwargs):
    images = images * 0.5 + 0.5
    grid = make_grid(images, nrow=nrow, **kwargs)  # (channels, height, width)
    #  (height, width, channels)
    grid = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
    im = Image.fromarray(grid)
    if to_grayscale:
        im = im.convert(mode="L")
    if path is not None:
        im.save(path, format=format)
    if show:
        im.show()
    return grid


def main():
    #seed_torch(seed=100)
    
    seed=1000
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    args = create_argparser().parse_args()
    #dist_util.setup_dist()
    #torch.distributed.init_process_group(backend='nccl')
    logger.configure()
    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"), strict=True)

    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()


    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=NUM_CLASSES, size=(args.batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes

        if args.sampler == "ddpm":
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        
        if args.sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
            
        if args.sampler == "dpm_solver":
            sample_fn = diffusion.dpm_solver_uncond_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                model_kwargs=model_kwargs,
                thresholding=args.thresholding,
                timestep_respacing=int(args.timestep_respacing),
            )
            
        if args.sampler == "unipc":
            sample_fn = diffusion.UniPC_uncond_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                model_kwargs=model_kwargs,
                thresholding=args.thresholding,
                timestep_respacing=int(args.timestep_respacing),
            )
        

        # save_image(sample, nrow=2, show=False, path="/mnt/workspace/workgroup/yuhu/code/guided-diffusion/samples/output.png", to_grayscale=False)
        # print(yes)
        
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.extend([sample.cpu().numpy()])
        if args.class_cond:
            all_labels.extend([classes.cpu().numpy()])
        logger.log(f"created {len(all_images) * args.batch_size} samples")
    
    arr = np.concatenate(all_images, axis=0)
    print(arr.shape)
    arr = arr[: args.num_samples]
    if args.class_cond:
        label_arr = np.concatenate(all_labels, axis=0)
        label_arr = label_arr[: args.num_samples]
        print(label_arr.shape)

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(args.sample_dir, f"DDIM5_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    if args.class_cond:
        np.savez(out_path, arr, label_arr)
    else:
        np.savez(out_path, arr)

    logger.log("sampling complete")

    torch.cuda.empty_cache()
    print("Begin to compute FID...")
    compute_fid(args.ref_batch, sample_batch=out_path)
    


#    ## save images for visualization
#     samples = sample_fn(
#             model,
#             (args.batch_size, 3, args.image_size, args.image_size),
#             clip_denoised=args.clip_denoised,
#             model_kwargs=model_kwargs,
#     )
#     save_image(samples, nrow=8, show=False, path="/mnt/workspace/workgroup/yuhu/code/guided-diffusion/samples/output.png", to_grayscale=False)

#     for i in range(len(samples)):
#         sample = ((samples[i] + 1) * 127.5).clamp(0, 255).to(th.uint8)
#         sample = sample.permute(1, 2, 0)
#         sample = sample.contiguous()
#         sample = sample.cpu().numpy()
#         #out_path = os.path.join(args.sample_dir, "{}.png".format(args.timestep_respacing))
#         out_path = os.path.join(args.sample_dir, "{}.png".format(i))
#         logger.log(f"saving to {out_path}")
#         cv2.imwrite(out_path, sample)
#     logger.log("sampling complete")



def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=32,
        sampler="ddim",
        model_path="",
        sample_dir="",
        ref_batch="",
        thresholding=False,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
