from jittor import transform
import jittor as jt
from jittor.dataset import Dataset
from jittor.dataset import DataLoader
from jittor.dataset.cifar import CIFAR10


def load_data(batchsize:int, numworkers:int):
    trans = transform.Compose([
        transform.Resize(32),
        transform.CenterCrop(32),
        transform.ToTensor(),
        transform.ImageNormalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    data_train = CIFAR10(
                        root = './',
                        train = True,
                        download = True,
                        transform = trans
                    )
    sampler = jt.dataset.RandomSampler(data_train)
    trainloader = DataLoader(
                        data_train,
                        batch_size = batchsize,
                        num_workers = numworkers,
                        sampler = sampler,
                        drop_last = True
                    )
    return trainloader

def transback(data):
    return data / 2 + 0.5
