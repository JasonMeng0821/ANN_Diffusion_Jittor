import os
import jittor as jt
import argparse
import numpy as np
from free_diffusion.train.unet import Unet
from tqdm import tqdm
from free_diffusion.train.diffusion import GaussianDiffusion
from free_diffusion.train.utils import get_named_beta_schedule
from free_diffusion.train.embedding import ConditionalEmbedding
from free_diffusion.train.dataloader_cifar import load_data

def train(params:argparse.Namespace):

    # initialize models
    net = Unet(
                in_ch = params.inch,
                mod_ch = params.modch,
                out_ch = params.outch,
                ch_mul = params.chmul,
                num_res_blocks = params.numres,
                cdim = params.cdim,
                use_conv = params.useconv,
                droprate = params.droprate,
                dtype = params.dtype
            )
    cemblayer = ConditionalEmbedding(10, params.cdim, params.cdim)
    # load last epoch
    lastpath = os.path.join(params.moddir,'last_epoch.pt')
    if os.path.exists(lastpath):
        lastepc = jt.load(lastpath)['last_epoch']
        # load checkpoints
        checkpoint = jt.load(os.path.join(params.moddir, f'ckpt_{lastepc}_checkpoint.pt'), map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        cemblayer.load_state_dict(checkpoint['cemblayer'])
    else:
        lastepc = 0
    betas = get_named_beta_schedule(num_diffusion_timesteps = params.T)
    diffusion = GaussianDiffusion(
                    dtype = params.dtype,
                    model = net,
                    betas = betas,
                    w = params.w,
                    v = params.v
                )
    

    # optimizer settings
    optimizer = jt.optim.AdamW(diffusion.parameters(), lr = params.lr, weight_decay = 0.0)
    
    # load cifar10 dataset
    dataloader = load_data(params.batchsize, params.numworkers)
    # warmup scheduler
    #warmUpScheduler = GradualWarmupScheduler(optimizer, multiplier = params.multiplier, total_epoch = 10, after_scheduler = None)

    # training
    with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
        for img, lab in tqdmDataLoader:
            print("hihihi")
            b = img.shape[0]
            optimizer.zero_grad()
            x_0 = img
            lab = lab
            cemb = cemblayer(lab)
            cemb[np.where(np.random.rand(b)<params.threshold)] = 0
            loss = diffusion.trainloss(x_0, cemb = cemb)
            print(loss)
            optimizer.backward(loss)
            optimizer.step()
            # tqdmDataLoader.set_postfix(
            #     ordered_dict={
            #         #"epoch": epc + 1,
            #         "loss: ": loss.item(),
            #         "batch per device: ":x_0.shape[0],
            #         "img shape: ": x_0.shape[1:],
            #         "LR": optimizer.state_dict()['param_groups'][0]["lr"]
            #     }
            # )

def main():
    # several hyperparameters for model
    parser = argparse.ArgumentParser(description='test for diffusion model')

    parser.add_argument('--batchsize',type=int,default=256,help='batch size per device for training Unet model')
    parser.add_argument('--numworkers',type=int,default=4,help='num workers for training Unet model')
    parser.add_argument('--inch',type=int,default=3,help='input channels for Unet model')
    parser.add_argument('--modch',type=int,default=64,help='model channels for Unet model')
    parser.add_argument('--T',type=int,default=1000,help='timesteps for Unet model')
    parser.add_argument('--outch',type=int,default=3,help='output channels for Unet model')
    parser.add_argument('--chmul',type=list,default=[1,2,2,2],help='architecture parameters training Unet model')
    parser.add_argument('--numres',type=int,default=2,help='number of resblocks for each block in Unet model')
    parser.add_argument('--cdim',type=int,default=10,help='dimension of conditional embedding')
    parser.add_argument('--useconv',type=bool,default=True,help='whether use convlution in downsample')
    parser.add_argument('--droprate',type=float,default=0.1,help='dropout rate for model')
    parser.add_argument('--dtype',default=jt.float32)
    parser.add_argument('--lr',type=float,default=2e-4,help='learning rate')
    parser.add_argument('--w',type=float,default=1.8,help='hyperparameters for classifier-free guidance strength')
    parser.add_argument('--v',type=float,default=0.3,help='hyperparameters for the variance of posterior distribution')
    parser.add_argument('--epoch',type=int,default=1500,help='epochs for training')
    parser.add_argument('--multiplier',type=float,default=2.5,help='multiplier for warmup')
    parser.add_argument('--threshold',type=float,default=0.1,help='threshold for classifier-free guidance')
    parser.add_argument('--interval',type=int,default=20,help='epoch interval between two evaluations')
    parser.add_argument('--moddir',type=str,default='model',help='model addresses')
    parser.add_argument('--samdir',type=str,default='sample',help='sample addresses')
    parser.add_argument('--genbatch',type=int,default=80,help='batch size for sampling process')
    parser.add_argument('--clsnum',type=int,default=10,help='num of label classes')
    parser.add_argument('--num_steps',type=int,default=50,help='sampling steps for DDIM')
    parser.add_argument('--eta',type=float,default=0,help='eta for variance during DDIM sampling process')
    parser.add_argument('--select',type=str,default='linear',help='selection stragies for DDIM')
    parser.add_argument('--ddim',type=lambda x:(str(x).lower() in ['true','1', 'yes']),default=False,help='whether to use ddim')
    parser.add_argument('--local_rank',default=-1,type=int,help='node rank for distributed training')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()