# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module, BatchNorm1d, functional as F

from torchvision.ops.focal_loss import sigmoid_focal_loss

from typing import Optional

from models.modules.momentum_batch_norm import MomentumBatchNorm


class BackgroundNoiseReductionLoss(Module):
    def __init__(self, mapping: Optional[Module] = None):
        super().__init__()
        self.mapping = mapping or MomentumBatchNorm(1)

    def forward(self, vectors: Tensor, labels: Tensor) -> Tensor:
        norms = torch.norm(vectors, p=2, dim=1, keepdim=True)
        distributions = self.mapping(norms).view(-1)  # [0,+inf) -> (-inf,+inf)
        binaries = labels.clamp(min=0.0, max=1.0)
        return sigmoid_focal_loss(distributions, binaries, alpha=-1., gamma=2., reduction='mean')

