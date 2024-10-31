# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import init, Module, Sequential, Linear, BatchNorm1d, Identity

from typing import Tuple, Optional
from collections import OrderedDict


class BoxPredictor(Module):
    def __init__(self, in_channels: int, num_classes: int, quality: bool = False, batch_norm: bool = True):
        super().__init__()
        self.classifier = Linear(in_channels, num_classes)
        self.regressor = Sequential(OrderedDict([
            ('fc', Linear(in_channels, 4, bias=not batch_norm)),
            ('bn', BatchNorm1d(4) if batch_norm else Identity()),
        ]))
        self.evaluator = Linear(in_channels, 1) if quality else None

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.classifier.weight)
        init.zeros_(self.classifier.bias)
        init.xavier_uniform_(self.regressor.fc.weight)
        if self.regressor.fc.bias is not None:
            init.zeros_(self.regressor.fc.bias)
        getattr(self.regressor.bn, 'reset_parameters', lambda: None)()
        if self.evaluator is not None:
            init.xavier_uniform_(self.evaluator.weight)
            init.zeros_(self.evaluator.bias)

    def forward(self, feat_map: Tensor) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
        feat_vector = torch.mean(feat_map, dim=(2, 3))
        return (
            self.classifier(feat_vector),
            self.regressor(feat_vector),
            self.evaluator(feat_vector) if self.evaluator is not None else None,
        )


if __name__ == '__main__':
    model = BoxPredictor(2048, 2, batch_norm=False)
    model.reset_parameters()
    print(model(torch.randn(5, 2048, 14, 14)))
