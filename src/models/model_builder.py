import torch.nn as nn

import torch

try:
    from .resnet import *
    from .vgg import vgg16
    from .densenet import densenetd40k12
except:
    from sys import path
    path.append('../models')
    from resnet import *
    from vgg import vgg16
    from densenet import densenetd40k12



class ModelBuilder:
    def __init__(self) -> None:
        pass
    
    @staticmethod
    def load(model_name, *args, **kwargs) -> nn.Module:
        model = globals()[model_name]
        return model(*args, **kwargs)


if __name__ == '__main__':
    model = ModelBuilder.load("densenetd40k12", num_classes=100, ema=False)
    x1 = torch.randn(size=(4, 3, 32, 32))
    x2 = torch.randn(size=(4, 3, 32, 32))
    x3 = torch.randn(size=(4, 3, 32, 32))
    out1, out2, out3, enout = model(x1, x2, x3)  # [4, 100]
    print(out1.shape, out2.shape, out3.shape)
    print(out1.requires_grad)
    # print(sum(out1 == out2), sum(out1 == out3), sum(out2 == out3))