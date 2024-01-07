# 扩散模型的复现及隐式分类器实现和骨架网络改进
此仓库是2023秋清华大学人工神经网络大作业开源代码，我们在Jittor框架下复现了[Diffusion Models Beat GANS on Image Synthesis](https://arxiv.org/abs/2105.05233)论文中的模型，同时实现了[Classifier-free diffusion guidance](https://arxiv.org/abs/2207.12598)中的隐式分类器引导采样和[Scalable diffusion models with transformers](https://openaccess.thecvf.com/content/ICCV2023/html/Peebles_Scalable_Diffusion_Models_with_Transformers_ICCV_2023_paper.html)中的DiT模型。

## 预训练模型链接

### 扩散模型
Conditional Model:https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion.pt
Unconditional Model:https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_diffusion_uncond.pt
Classifier:https://openaipublic.blob.core.windows.net/diffusion/jul-2021/256x256_classifier.pt

### DiT模型
https://dl.fbaipublicfiles.com/DiT/models/DiT-XL-2-256x256.pt

## 训练模型

### 基础扩散模型训练

```
MODEL_FLAGS="--image_size 256 --num_channels 128 --num_res_blocks 3"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule linear"
TRAIN_FLAGS="--lr 1e-4 --batch_size 128"
python3 image_train.py --data_dir path/to/images $MODEL_FLAGS $DIFFUSION_FLAGS $TRAIN_FLAGS
```

### 分类器训练

```
TRAIN_FLAGS="--iterations 300000 --anneal_lr True --batch_size 256 --lr 3e-4 --save_interval 10000 --weight_decay 0.05"
CLASSIFIER_FLAGS="--image_size 128 --classifier_attention_resolutions 32,16,8 --classifier_depth 2 --classifier_width 128 --classifier_pool attention --classifier_resblock_updown True --classifier_use_scale_shift_norm True"
python3 classifier_train.py --data_dir path/to/imagenet $TRAIN_FLAGS $CLASSIFIER_FLAGS
```

### 隐式分类器扩散模型训练

默认使用Cifar10数据集进行训练。

```
python3 free_train.py
```

### DiT模型训练

进入DiT目录，运行下面的指令进行train，--data-path设置数据集目录，--results-dir设置结果保存的目录。

```
python3 train.py --model DiT-XL/2 --data-path ./imagenet --image-size 256 --batch-size 128 --epochs 300 --num-workers 4 --global-seed 0 --results-dir ./res_dir
```

## 生成采样图片

对于DDPM和DDIM的分类器引导采样，我们提供了一个脚本来生成采样图片。对于隐式分类器引导采样，我们提供了另一个脚本来生成采样图片。

### 分类器引导DDPM采样
对于DDPM采样，以下六个参数需要设置或调整，即：
要生成的采样图片的数量（num_samples）、批次大小（batch size）、时间步长（timestep_respacing）、分类器尺度（classifier_scale）、分类器路径（classifier_path）
模型路径（model_path）。

该模型在时间步长为250时，生成的采样图片质量最高。增大分类器尺度，可以在生成的采样图片中获得更多的多样性，但是会降低采样图片的质量。

例如，要生成10000张ImageNet 256x256、时间步长为250，分类器尺度为10.0的采样图片，可以运行以下命令：

```
python3 classifier_sample.py --batch_size 8 --num_samples 10000 --timestep_respacing 250 --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --classifier_scale 10.0 --classifier_path path/to/classifier.pt --model_path model_path /path/to/model.pt
```

该脚本将生成一个.npz文件，其中包含生成的采样图片。该文件的大小约为1.8GB。


### 分类器引导DDIM采样
对于DDIM采样，需要额外添加和调整以下两个参数：
--use_ddim True
--timestep_respacing ddim25


例如，要生成10000张ImageNet 256x256、使用DDIM、时间步长为25、分类器尺度为1.0的采样图片，可以运行以下命令：

```
python3 classifier_sample.py --batch_size 8 --num_samples 64 --timestep_respacing ddim25 --use_ddim True --attention_resolutions 32,16,8 --class_cond True --diffusion_steps 1000 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --classifier_scale 10.0 --classifier_path path/to/classifier.pt --model_path model_path /path/to/model.pt
```

### 隐式分类器引导DDIM采样

与DDIM采样的参数基本相同，需要删除以下两个参数：
 --class_cond True 
 --diffusion_steps 1000

需要额外添加一个超参数：分类器权重（w）。

增大分类器权重，可以在生成的采样图片中获得更多的多样性，但是会降低采样图片的质量。


例如，要生成10000张ImageNet 256x256、使用DDIM、时间步长为25、分类器权重为3、隐式分类器引导采样的采样图片，可以运行以下命令：


```
python3 free_sample.py --attention_resolutions 32,16,8 --image_size 256 --learn_sigma True --noise_schedule linear --num_channels 256 --num_head_channels 64 --num_res_blocks 2 --resblock_updown True --use_scale_shift_norm True --cmodel_path /path/to/conditional_model.pt  --ucmodel_path /path/to/unconditional_model.pt --batch_size 60 --num_samples 10000 --timestep_respacing ddim25 --use_ddim True --w 3
```

### DiT模型采样

Load AutoEncoder模型的方法：将vae模型放在`./stabilityai`下。

将预训练模型放在`./pretrained_models`下。

进入DiT目录，运行下面的指令进行sample，--seed可以设置随机数种子。

```
python3 sample.py --model DiT-XL/2 --image-size 256 --ckpt ./pretrained_models/DiT-XL-2-256x256.pt --seed 0
```

## 评价采样质量

我们使用FID，sFID，Precision，Recall和Inception Score来比较不同模型的采样质量。这些指标都可以使用evaluator、采样图片、标准参考图片来计算，所有采样的图片都存储在.npz（numpy）文件中。标准参考图片由复现论文的原作者给出，可以从下面给出的链接下载。

ImageNet 256x256 标准参考图片（VIRTUAL_imagenet256_labeled.npz）: https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz


### 运行评价脚本

进入evaluations文件夹，运行evaluator.py脚本，它需要两个参数：标准参考图片、采样图片。该脚本将下载用于评估的InceptionV3模型到当前工作目录（如果尚未存在）。该文件大约为100MB。

```
python3 evaluator.py VIRTUAL_imagenet256_labeled.npz path/to/yoursamples.npz
```