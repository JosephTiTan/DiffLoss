"""
Train a diffusion model on images.
"""

import argparse
import os
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
# from restoration_network.dehaze.dehazing_FSDGN import FSDGN
from restoration_network.enhance.IAT_main import IAT
# from restoration_network.enhance.LPF.model import DeepLPFNet
# from restoration_network.dehaze.MB_TaylorFormer import MB_TaylorFormer
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler

# my_model = '' # if use pth

def main():
    args = create_argparser().parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = "1"
    torch.distributed.init_process_group(backend='nccl')
    #dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    Restoration_model = IAT()
    Restoration_model.cuda()
    # Restoration_model.load_state_dict(torch.load(my_model)['net'])
    
    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        gt_dir = args.gt_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    testdata = load_data(
        data_dir=args.testdata_dir,
        gt_dir=args.testgt_dir,
        batch_size=1,
        image_size=args.image_size,
        class_cond=args.class_cond,
        random_flip = False
    )

    logger.log("training...")
    TrainLoop(
        Restoration_model=Restoration_model,
        model=model,
        diffusion=diffusion,
        data=data,
        testdata = testdata,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        gt_dir="",
        testdata_dir="",
        testgt_dir="",
        model_path="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        log_dir="",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
