"""
Sample new images from a pre-trained DiT.
"""
import torch
import jittor as jt
from torchvision.utils import save_image
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from download import find_model
from models import DiT_models
import argparse
import numpy as np


def main(args):
    jt.set_seed(args.seed)
    jt.flags.use_cuda = 1 if jt.compiler.has_cuda else 0
    
    with jt.no_grad():

      if args.ckpt is None:
          assert args.model == "DiT-XL/2", "Only DiT-XL/2 models are available for auto-download."
          assert args.image_size in [256, 512]
          assert args.num_classes == 1000

      # Load model:
      latent_size = args.image_size // 8
      model = DiT_models[args.model](
          input_size=latent_size,
          num_classes=args.num_classes
      )
      # Auto-download a pre-trained model or load a custom DiT checkpoint from train.py:
      ckpt_path = args.ckpt or f"DiT-XL-2-{args.image_size}x{args.image_size}.pt"
      state_dict = find_model(ckpt_path)
      model.load_state_dict(state_dict)
      model.eval()  # important!
      diffusion = create_diffusion(str(args.num_sampling_steps))
      # vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}")
      vae = AutoencoderKL.from_pretrained("./stabilityai") 

      # Labels to condition the model with (feel free to change):
      # class_labels = [i for i in range(1000)]
      class_labels = [jt.randint(0, 1000, [1]).data[0] for i in range(100)]

      # Create sampling noise:
      n = len(class_labels)
      z = jt.randn(n, 4, latent_size, latent_size)
      y = jt.Var(class_labels)

      # Setup classifier-free guidance:
      z = jt.cat([z, z], 0)
      y_null = jt.Var([1000] * n)
      y = jt.cat([y, y_null], 0)
      model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)

      # Sample images:
      samples = diffusion.p_sample_loop(
          model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True
      )
      samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

      # NOTE: Because we use the pretrained VAE from huggingface, we need to transfer it into Tensor here.
      #      Due to the limit of computation resources and time, we do not train the VAE from scratch.
      #      If we use the VAE from our repo after training, we can remove this line.
      samples = torch.Tensor(samples.numpy())
      samples = vae.decode(samples / 0.18215).sample

      # Change the shape of the samples to (n, image_size, image_size, 3):
      samples = samples.permute(0, 2, 3, 1)

      # Rescale images to [0, 1]:
      samples = (samples + 1) / 2

      # Clip to [0, 1]:
      # samples = np.clip(samples, 0, 1)
      # save images to npz file
      np.savez(f"./sample/sample{args.seed}.npz", samples.detach().cpu().numpy())

      # Save and display images:
      # save_image(samples, "sample.png", nrow=4, normalize=True, value_range=(-1, 1))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=list(DiT_models.keys()), default="DiT-XL/2")
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="mse")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--num-sampling-steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Optional path to a DiT checkpoint (default: auto-download a pre-trained DiT-XL/2 model).")
    args = parser.parse_args()
    main(args)
