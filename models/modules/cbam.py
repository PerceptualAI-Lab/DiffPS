# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Linear, Conv2d, ReLU, init, functional as F
from collections import OrderedDict


class ChannelAttention(Module):
    def __init__(self, channels: int, reduction: int = 16, bias: bool = False):
        super().__init__()
        neck = channels // reduction
        self.body = Sequential(OrderedDict([
            ('fc1', Linear(channels, neck, bias=bias)),
            ('relu', ReLU(inplace=True)),
            ('fc2', Linear(neck, channels, bias=bias)),
        ]))

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.body.fc1.weight, mode='fan_out', nonlinearity='relu')
        init.kaiming_uniform_(self.body.fc2.weight, mode='fan_in', nonlinearity='relu')
        if self.body.fc1.bias is not None:
            init.zeros_(self.body.fc1.bias)
        if self.body.fc2.bias is not None:
            init.zeros_(self.body.fc2.bias)

    def forward(self, x: Tensor) -> Tensor:
        attention = F.sigmoid(
            self.body(torch.mean(x, dim=(2, 3))) +
            self.body(torch.amax(x, dim=(2, 3)))
        )[..., None, None]
        return x * attention


class SpatialAttention(Module):
    def __init__(self, kernel_size: int = 7, bias: bool = False):
        super().__init__()
        self.body = Conv2d(2, 1, kernel_size, 1, 'same', bias=bias)

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.body.weight)
        if self.body.bias is not None:
            init.zeros_(self.body.bias)

    def forward(self, x: Tensor) -> Tensor:
        attention = F.sigmoid(self.body(torch.cat([
            torch.mean(x, dim=1, keepdim=True),
            torch.amax(x, dim=1, keepdim=True)
        ], dim=1)))
        return x * attention


class RobustCBAM(Sequential):
    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7, bias: bool = False):
        super().__init__(OrderedDict([
            ('channel_attention', ChannelAttention(channels, reduction, bias)),
            ('spatial_attention', SpatialAttention(kernel_size, bias)),
        ]))

    def reset_parameters(self) -> None:
        self.channel_attention.reset_parameters()
        self.spatial_attention.reset_parameters()

