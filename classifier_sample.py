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


from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
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
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    model.load_state_dict(

        dist_util.load_state_dict(args.model_path)
    )
    #model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("loading classifier...")
    classifier = create_classifier(**args_to_dict(args, classifier_defaults().keys()))
    classifier.load_state_dict(

        dist_util.load_state_dict(args.classifier_path)
    )

    #classifier.to(dist_util.dev())
    if args.classifier_use_fp16:
        classifier.convert_to_fp16()
    classifier.eval()

    def cond_fn(x, t, y=None):
        assert y is not None
        with jt.enable_grad():
            x_in = x.detach()
            logits = classifier(x_in, t)
            log_probs = nn.log_softmax(logits, dim=-1)
            #selected = log_probs[range(len(logits)), y.view(-1)] type <range> not support for jittor array
            selected = log_probs[jt.arange(len(logits)), y.view(-1)]
            return jt.grad(selected.sum(), x_in)[0] * args.classifier_scale

    def model_fn(x, t, y=None):
        assert y is not None
        return model(x, t, y if args.class_cond else None)

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
            diffusion.p_sample_loop if not args.use_ddim else diffusion.ddim_sample_loop
        )
        # print("Start Sample_fn")
        #time0 = time.time()
        sample = sample_fn(
            model_fn,
            (args.batch_size, 3, args.image_size, args.image_size),
            clip_denoised=args.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            #device=dist_util.dev(),
        )
        #time1 = time.time()
        # print("Sample_fn Finished:", time1-time0)
        sample = ((sample + 1) * 127.5).clamp(0, 255).to(jt.uint8)
        sample = sample.permute(0, 2, 3, 1)
        sample = sample.contiguous()

        # sample is in the shape (1, 256, 256, 3), add it to all_images on the first dimension
        all_images.append(sample.numpy())
        all_labels.append(classes.numpy())
        #print("all_images length:", len(all_images))

        #gathered_samples = [jt.zeros_like(sample)]
        #dist.all_gather(gathered_samples, sample)  # gather not supported with NCCL
        #all_images.extend([s.numpy() for s in sample])
        #all_images.extend([sample for sample in gathered_samples])
        #gathered_labels = [jt.zeros_like(classes)]
        #dist.all_gather(gathered_labels, classes)
        #all_labels.extend([l.numpy() for l in classes])
        #all_labels.extend([labels for labels in gathered_labels])
        logger.log(f"created {len(all_images) * args.batch_size} samples")

    arr = np.concatenate(all_images, axis=0)
    arr = arr[: args.num_samples]
    label_arr = np.concatenate(all_labels, axis=0)
    label_arr = label_arr[: args.num_samples]
    #if dist.get_rank() == 0:
    shape_str = "x".join([str(x) for x in arr.shape])
    out_path = os.path.join(logger.get_dir(), f"samples_{shape_str}.npz")
    logger.log(f"saving to {out_path}")
    np.savez(out_path, arr, label_arr)

    #dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=10000,
        batch_size=16,
        use_ddim=False,
        model_path="",
        classifier_path="",
        classifier_scale=1.0,
    )
    defaults.update(model_and_diffusion_defaults())
    defaults.update(classifier_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
