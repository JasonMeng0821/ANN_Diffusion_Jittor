"""
Train a noised image classifier on ImageNet.
"""

import argparse
import os

import blobfile as bf
import jittor as jt
import jittor.nn as nn
from jittor.optim import AdamW

from guided_diffusion import logger
from guided_diffusion.fp16_util import MixedPrecisionTrainer
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    add_dict_to_argparser,
    args_to_dict,
    classifier_and_diffusion_defaults,
    create_classifier_and_diffusion,
)


def main():
    args = create_argparser().parse_args()

    logger.log("creating model and diffusion...")
    model, diffusion = create_classifier_and_diffusion(
        **args_to_dict(args, classifier_and_diffusion_defaults().keys())
    )

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=True,
        random_crop=True,
    )

    logger.log(f"creating optimizer...")
    opt = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    logger.log("training classifier model...")

    def forward_backward_log(data_loader, prefix="train"):
        batch, extra = next(data_loader)
        labels = extra["y"]

        batch = batch

        t = jt.zeros(batch.shape[0], dtype=jt.long)
        logits = model(batch, t)
        loss = nn.cross_entropy_loss(logits, labels)

        losses = {}
        losses[f"{prefix}/loss"] = loss.item()
        losses[f"{prefix}/top1"] = compute_top_k(logits, labels, 1)
        losses[f"{prefix}/top5"] = compute_top_k(logits, labels, 5)
        logger.log(losses)

        loss = loss.mean()
        opt.backward(loss) 

    for step in range(args.iterations):

        forward_backward_log(data)
        opt.step()

        if step % args.log_interval == 0:
            logger.log(f"step {step}:")

    logger.log("saving model...")

    jt.save(
        model.state_dict(),
        os.path.join(logger.get_dir(), f"model.pt"),
    )


def compute_top_k(logits, labels, k, reduction="mean"):
    _, top_ks = jt.topk(logits, k, dim=-1)
    if reduction == "mean":
        return (top_ks == labels[:, None]).float().sum(dim=-1).mean().item()
    elif reduction == "none":
        return (top_ks == labels[:, None]).float().sum(dim=-1)


def split_microbatches(microbatch, *args):
    bs = len(args[0])
    if microbatch == -1 or microbatch >= bs:
        yield tuple(args)
    else:
        for i in range(0, bs, microbatch):
            yield tuple(x[i : i + microbatch] if x is not None else None for x in args)


def create_argparser():
    defaults = dict(
        data_dir="",
        val_data_dir="",
        noised=True,
        iterations=150000,
        lr=3e-4,
        weight_decay=0.0,
        anneal_lr=False,
        batch_size=4,
        microbatch=-1,
        schedule_sampler="uniform",
        resume_checkpoint="",
        log_interval=10,
        eval_interval=5,
        save_interval=10000,
    )
    defaults.update(classifier_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
