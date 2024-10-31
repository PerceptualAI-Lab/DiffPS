# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import init, Module, ModuleList, Sequential, Linear, BatchNorm1d, AdaptiveMaxPool2d, Flatten
from typing import List, Dict, Callable, Optional, Any
from collections import OrderedDict

from models.modules.cbam import RobustCBAM
from models.modules.drop_path import DropPath


def num_split(num: int, divider: int) -> List[int]:
    nums = [num // divider] * divider
    for index in range(num % divider):
        nums[index] += 1
    return nums


class StripProjector(Sequential):
    def __init__(self, in_channels: int, dim_out: int, drop_path: float = 0.1):
        super().__init__(OrderedDict([
            ('cbam', RobustCBAM(in_channels, bias=True)),
            ('gmp', AdaptiveMaxPool2d(1)),
            ('flatten', Flatten(start_dim=1, end_dim=-1)),
            ('linear', Linear(in_channels, dim_out, bias=False)),
            ('norm', BatchNorm1d(dim_out)),
            ('drop_path', DropPath(drop_path)),
        ]))

    def reset_parameters(self) -> None:
        self.cbam.reset_parameters()
        init.xavier_uniform_(self.linear.weight)
        self.norm.reset_parameters()


class Branch(Module):
    def __init__(self, in_channels: int, dim_out: int, num_strips: int, drop_path: float = 0.1):
        super().__init__()
        self.strip_projectors = ModuleList(
            StripProjector(in_channels, dim_part, drop_path)
            for dim_part in num_split(dim_out, num_strips)
        )

    def reset_parameters(self) -> None:
        for strip_projector in self.strip_projectors:
            strip_projector.reset_parameters()

    def forward(self, feat_map: Tensor) -> Tensor:
        return torch.cat([
            strip_projector(strip_of_feat_map) for strip_projector, strip_of_feat_map in zip(
                self.strip_projectors, feat_map.split(num_split(feat_map.shape[2], len(self.strip_projectors)), dim=2)
            )
        ], dim=1)


class MultiGranularityEmbedding(Module):
    def __init__(self, in_channels: int, dim_out: int, num_branches: int = 3, drop_path: float = 0.1):
        super().__init__()
        self.branches = ModuleList(
            Branch(in_channels, dim_branch, num_strips, drop_path)
            for num_strips, dim_branch in enumerate(num_split(dim_out, num_branches), 1)
        )

    def reset_parameters(self) -> None:
        for branch in self.branches:
            branch.reset_parameters()

    def forward(self, feat_map: Tensor) -> Tensor:
        return torch.cat([branch(feat_map) for branch in self.branches], dim=1)


class GlobalFeatureEmbedding(Sequential):
    def __init__(self, in_channels: int, dim_out: int):
        super().__init__(OrderedDict([
            ('gmp', AdaptiveMaxPool2d(1)),
            ('flatten', Flatten(start_dim=1, end_dim=-1)),
            ('linear', Linear(in_channels, dim_out)),
            ('norm', BatchNorm1d(dim_out)),
        ]))

    def reset_parameters(self) -> None:
        init.xavier_uniform_(self.linear.weight)
        init.zeros_(self.linear.bias)


class Embedder(Module):
    def __init__(
            self,
            in_feat_names: List[str],
            in_channels_list: List[int],
            dim_out: int = 1024,
            emb_type: Callable[..., Module] = MultiGranularityEmbedding,
            extra_cfg: Optional[Dict[str, Any]] = None,
    ):
        assert len(in_feat_names) == len(in_channels_list)
        super().__init__()
        self.in_feat_names = in_feat_names
        self.dim_out = dim_out
        self.encoders = ModuleList(
            emb_type(channels, dim, **extra_cfg or {})
            for channels, dim in zip(in_channels_list, num_split(dim_out, len(in_channels_list)))
        )

    def reset_parameters(self) -> None:
        for encoder in self.encoders:
            encoder.reset_parameters()

    def forward(self, feat_maps: Dict[str, Tensor]) -> Tensor:
        embeddings = torch.cat([
            encoder(feat_maps[feat_nm]) for encoder, feat_nm in zip(self.encoders, self.in_feat_names)
        ], dim=1)
        return embeddings


if __name__ == '__main__':
    _type = GlobalFeatureEmbedding
    model = Embedder(
        ['res4', 'res5'], [1024, 2048], 1024, _type,
        dict(num_branches=3, drop_path=0.1) if _type == MultiGranularityEmbedding else None
    )
    model.reset_parameters()
    print(model({'res4': torch.randn(5, 1024, 14, 14), 'res5': torch.randn(5, 2048, 14, 14)}).shape)
