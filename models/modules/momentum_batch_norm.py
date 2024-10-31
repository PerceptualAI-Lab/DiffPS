# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module


class MomentumBatchNorm(Module):
    def __init__(self, channels: int, momentum: float = 0.9, eps: float = 1e-5):
        super().__init__()
        self.channels = channels
        self.momentum = momentum
        self.eps = eps
        self.register_buffer('mean', torch.empty((channels,)))
        self.register_buffer('var', torch.empty((channels,)))
        self.update = self._reset

    @torch.no_grad()
    def _reset(self, norms: Tensor) -> None:
        self.mean = torch.mean(norms, dim=(0, 2))
        self.var = torch.var(norms, dim=(0, 2), unbiased=True)
        self.update = self._shift

    @torch.no_grad()
    def _shift(self, norms: Tensor) -> None:
        self.mean = self.momentum * self.mean + (1.0 - self.momentum) * torch.mean(norms, dim=(0, 2))
        self.var = self.momentum * self.var + (1.0 - self.momentum) * torch.var(norms, dim=(0, 2), unbiased=True)

    def forward(self, inputs: Tensor) -> Tensor:
        shape = inputs.shape
        inputs = inputs.view(*shape[:2], -1)
        if self.training:
            self.update(inputs.detach())
        return ((inputs - self.mean.view(1, -1, 1)) / torch.sqrt(self.var + self.eps).view(1, -1, 1)).view(shape)


if __name__ == '__main__':
    model = MomentumBatchNorm(1)
    print(model(torch.randn(64, 1)).shape)
    print(model.state_dict())
    print(model(torch.randn(64, 1)).shape)
    print(model.mean, model.var)
