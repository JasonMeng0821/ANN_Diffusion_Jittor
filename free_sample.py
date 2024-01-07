"""
Like image_sample.py, but use a noisy image classifier to guide the sampling
process towards more realistic images.
"""

import argparse
import os

import numpy as np
import jittor as jt
import jittor.nn as nn
import time
from PIL import Image


from free_diffusion import dist_util, logger
from free_diffusion.script_util import (
    NUM_CLASSES,
    models_and_diffusion_defaults,
    create_models_and_diffusion,
    add_dict_to_argparser,
    args_to_dict
)

if jt.has_cuda:
    jt.flags.use_cuda = 1 # jt.flags.use_cuda 表示是否使用 gpu 训练。


def main():
    args = create_argparser().parse_args()

    #dist_util.setup_dist()
    logger.configure()

    logger.log("creating model and diffusion...")
    cmodel, ucmodel, diffusion = create_models_and_diffusion(
        **args_to_dict(args, models_and_diffusion_defaults().keys())
    )
    cmodel.load_state_dict(
        dist_util.load_state_dict(args.cmodel_path)
    )
    ucmodel.load_state_dict(
        dist_util.load_state_dict(args.ucmodel_path)
    )
    #model.to(dist_util.dev())
    if args.use_fp16:
        cmodel.convert_to_fp16()
        ucmodel.convert_to_fp16()
    cmodel.eval()
    ucmodel.eval()

    def cmodel_fn(x, t, y=None):
        assert y is not None
        return cmodel(x, t, y)
    
    def ucmodel_fn(x, t):
        return ucmodel(x, t, None)

    logger.log("sampling...")
    all_images = []
    all_labels = []
    while len(all_images) * args.batch_size < args.num_samples:
        model_kwargs = {}
        classes = jt.randint(
            low=0, high=NUM_CLASSES, shape=(args.batch_size,)#, device=dist_util.dev()
        )
        model_kwargs["y"] = classes
        sample_fn = (
            diffusion.p_sample_loop
        )
        sample = sample_fn(
            w=args.w,
            cmodel=cmodel_fn,
            ucmodel=ucmodel_fn,
            shape=(args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs
        )

        sample = ((sample + 1) * 127.5).clamp(0, 255).to(jt.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        all_images.append(sample.numpy())
        all_labels.append(classes.numpy())
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]

    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)

    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        cmodel_path="",
        ucmodel_path="",
        w=0
    )
    defaults.update(models_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
