# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import init, Sequential, Conv2d, BatchNorm2d, ReLU, Identity
from torchvision.ops.misc import Conv2dNormActivation

from collections import OrderedDict


class TripleConvHead(Sequential):
    def __init__(self, in_channels: int = 1024, out_channels: int = 2048, hidden_channels: int = 256, **kwargs):
        self.in_channels = in_channels
        self.out_channels = out_channels
        defaults = dict(
            kernel_size=3, stride=1, padding=1,
            groups=1, norm_layer=BatchNorm2d, activation_layer=ReLU, dilation=1, inplace=True, bias=False
        )
        defaults.update(kwargs)
        super().__init__(OrderedDict([
            (f'layer1', Conv2dNormActivation(in_channels, hidden_channels, **defaults)),
            (f'layer2', Conv2dNormActivation(hidden_channels, hidden_channels, **defaults)),
            (f'layer3', Conv2dNormActivation(hidden_channels, out_channels, **defaults)),
        ]))

    def reset_parameters(self) -> None:
        for layer in self.children():
            for module in layer.children():
                if isinstance(module, Conv2d):
                    init.kaiming_uniform_(module.weight, mode='fan_out', nonlinearity='relu')
                    if module.bias is not None:
                        init.zeros_(module.bias)
                elif isinstance(module, BatchNorm2d):
                    module.reset_parameters()


if __name__ == '__main__':
    model = TripleConvHead(1024)
    model.reset_parameters()
    print(model(torch.randn(5, 1024, 14, 14)).shape)
