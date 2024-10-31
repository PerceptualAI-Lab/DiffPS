# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module


class DropPath(Module):
    def __init__(self, drop_probability: float, entire_batch: bool = False):
        assert 0. <= drop_probability <= 1.
        super().__init__()
        self.register_buffer('keep_prob', torch.tensor(1. - drop_probability))
        self.entire_batch = entire_batch

    def forward(self, x: Tensor) -> Tensor:
        if self.keep_prob == 1. or not self.training:
            return x
        return x / self.keep_prob * torch.bernoulli(
            self.keep_prob if self.entire_batch else self.keep_prob.expand((len(x),) + (1,) * (x.ndim - 1))
        )


if __name__ == '__main__':
    model = DropPath(0.1)
    a = torch.randn(12, 8)
    b = model(a)
    print(a)
    print(b)
