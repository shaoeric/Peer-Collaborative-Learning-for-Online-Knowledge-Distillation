try:
    from .cifar import *
except:
    from sys import path
    path.append('../datasets')
    from cifar import *

import os

import yaml
from easydict import EasyDict
import re

__all__ = [
    "DatasetBuilder"
]


DATASET_CONFIG = os.path.join(os.path.dirname(__file__), "dataset_config.yml")


def parse_dataset_config():
    loader = yaml.SafeLoader
    loader.add_implicit_resolver(
        u'tag:yaml.org,2002:float',
        re.compile(u'''^(?:
        [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
        |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
        |\\.[0-9_]+(?:[eE][-+][0-9]+)?
        |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
        |[-+]?\\.(?:inf|Inf|INF)
        |\\.(?:nan|NaN|NAN))$''', re.X),
        list(u'-+0123456789.'))
    with open(DATASET_CONFIG, 'r') as f:
        data = yaml.load(f, Loader=loader)
    return EasyDict(data)


class DatasetBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(dataset_name: str = 'CIFAR10',
             *args, **kwargs):
        config = parse_dataset_config()[dataset_name]
        config.update(kwargs)
        dataset = globals()[dataset_name](*args, **config)
        return dataset, {dataset_name: config}


if __name__ == '__main__':
    from torchvision.transforms import transforms
    from torch.utils.data import DataLoader
    from matplotlib import pyplot as plt

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    # val_transform = transforms.Compose([
    #     transforms.ToTensor(),
    #     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # ])
    trainset, trainset_config = DatasetBuilder.load(dataset_name="CIFAR100", transform=train_transform, train=True)
    # valset, valset_config = DatasetBuilder.load(dataset_name="CIFAR100", transform=val_transform, train=False)
    print(trainset_config)
    # print(valset_config)
    trainloader = DataLoader(trainset, batch_size=16, shuffle=True, num_workers=4)
    # val_loader = DataLoader(valset, batch_size=16, shuffle=False, num_workers=4)
    for img, label in trainloader:
        print(img.shape, label.shape)  # torch.Size([16, 3, 32, 32]) torch.Size([16])
        img1 = img[0, 0].cpu().numpy().transpose(1, 2, 0) * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.4914, 0.4822, 0.4465))
        img2 = img[0, 1].cpu().numpy().transpose(1, 2, 0) * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.4914, 0.4822, 0.4465))
        img3 = img[0, 2].cpu().numpy().transpose(1, 2, 0) * np.array((0.2023, 0.1994, 0.2010)) + np.array((0.4914, 0.4822, 0.4465))
        plt.figure(figsize=(8, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(img1)
        plt.subplot(1, 3, 2)
        plt.imshow(img2)
        plt.subplot(1, 3, 3)
        plt.imshow(img3)
        plt.show()
        break