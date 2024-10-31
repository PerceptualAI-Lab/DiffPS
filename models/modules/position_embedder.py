# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module, Parameter, Identity, init, functional as F, LayerNorm, Dropout

from typing import Tuple, Union
from abc import ABC, abstractmethod


class PositionEmbedder(Module, ABC):
    def __init__(
            self,
            size: Union[int, Tuple[int, int]],
            dim: int,
            batch_first: bool = True,
    ):
        super().__init__()
        self.height, self.width = (size, size) if isinstance(size, int) else size
        self.dim = dim
        self.batch_dim = 1 - batch_first
        self._register_position_embeddings()

    @abstractmethod
    def _register_position_embeddings(self) -> None:
        pass

    @abstractmethod
    def _get_position_embeddings(self) -> Tensor:  # (H, W, E)
        pass

    def reset_parameters(self) -> None:
        for param in self.parameters(False):
            init.normal_(param)

    def forward(self, inputs: Tensor) -> Tensor:
        embeddings = self._get_position_embeddings()  # (H, W, E)
        embeddings = embeddings.view(-1, self.dim).unsqueeze(self.batch_dim)  # (H, W, E) -> (1, H*W, E) or (H*W, 1, E)
        return inputs + embeddings


class LearnablePositionEmbedder1D(PositionEmbedder):
    def _register_position_embeddings(self) -> None:
        self.register_parameter('pos', Parameter(torch.empty(self.height, self.width, self.dim)))

    def _get_position_embeddings(self) -> Tensor:
        return self.pos


class LearnablePositionEmbedder2D(PositionEmbedder):
    def _register_position_embeddings(self) -> None:
        self.register_parameter('pos_row', Parameter(torch.empty(self.height, 1, self.dim // 2)))
        self.register_parameter('pos_col', Parameter(torch.empty(1, self.width, self.dim // 2)))

    def _get_position_embeddings(self) -> Tensor:
        return torch.cat([self.pos_row.repeat(1, self.width, 1), self.pos_col.repeat(self.height, 1, 1)], dim=-1)


if __name__ == '__main__':
    model = LearnablePositionEmbedder2D(7, 1024, batch_first=False)
    model.reset_parameters()
    print(model(torch.randn(49, 5, 1024)).shape)
