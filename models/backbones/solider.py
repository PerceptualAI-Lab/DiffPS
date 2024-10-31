# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module, Sequential, functional as F

from torchvision.models.swin_transformer import swin_b

from collections import OrderedDict
from typing import Dict, List
import yaml
import os.path as osp

from models.backbones.swin_transformer import (
    swin_tiny_patch4_window7_224, swin_small_patch4_window7_224, swin_base_patch4_window7_224
)
from models.backbones.base import BaseBackbone, BaseBackboneHead

_arch = 'small'  # tiny, small or base


def _make_model():
    config = dict(drop_path_rate=0.1, semantic_weight=0.6)
    return {
        'tiny': swin_tiny_patch4_window7_224,
        'small': swin_small_patch4_window7_224,
        'base': swin_base_patch4_window7_224
    }[_arch](**config)


def _get_weights():
    with open('configs/_path_solider_weights.yaml', 'r', encoding='UTF-8') as file:
        paths = yaml.load(file.read(), yaml.FullLoader)
    return {
        nm: value for nm, value in torch.load(
            osp.join(paths['base_dir'], paths[_arch]) if 'base_dir' in paths else paths[_arch], map_location='cpu'
        )['teacher'].items() if 'backbone' in nm
    }


class SOLIDERBackbone(BaseBackbone):
    def __init__(self, return_layers_idx: List[int]):
        return_layers_idx.sort()
        assert 1 <= return_layers_idx[0] and return_layers_idx[-1] <= 4
        super().__init__()
        self.backbone = _make_model()
        self.backbone.patch_embed.requires_grad_(False)  # frozen stem
        self._out_indices = [idx - 1 for idx in return_layers_idx]
        self.out_channels_list = [self.backbone.num_features[indic] for indic in self._out_indices]
        self.out_feat_names = [self.FEATURE_NAMES[indic] for indic in self._out_indices]

    def load_pretrained_weights(self) -> None:
        self.load_state_dict(_get_weights())

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        if self.backbone.semantic_weight >= 0:
            w = torch.ones((len(x), 1), device=x.device) * self.backbone.semantic_weight
            semantic_weight = torch.cat([w, 1 - w], dim=-1)

        x, hw_shape = self.backbone.patch_embed(x)

        if self.backbone.use_abs_pos_embed:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.drop_after_pos(x)

        outs = []
        for i, stage in enumerate(self.backbone.stages[:self._out_indices[-1] + 1]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if self.backbone.semantic_weight >= 0:
                sw = self.backbone.semantic_embed_w[i](semantic_weight).unsqueeze(1)
                sb = self.backbone.semantic_embed_b[i](semantic_weight).unsqueeze(1)
                x = x * self.backbone.softplus(sw) + sb
            if i in self._out_indices:
                out = getattr(self.backbone, f'norm{i}')(out)
                out = out.view(-1, *out_hw_shape, self.backbone.num_features[i]).permute(0, 3, 1, 2).contiguous()
                outs.append(out)
        return OrderedDict([(nm, out) for nm, out in zip(self.out_feat_names, outs)])


_head_index = 3


class SOLIDERHead(BaseBackboneHead):
    def __init__(self, down_sampling: bool = True):
        super().__init__()
        self.backbone = _make_model()
        self.in_channels = self.backbone.num_features[-2]
        self.out_channels = self.backbone.num_features[-1]

    def load_pretrained_weights(self) -> None:
        self.load_state_dict(_get_weights())

    def forward(self, x: Tensor) -> Tensor:
        if self.backbone.semantic_weight >= 0:
            w = torch.ones((len(x), 1), device=x.device) * self.backbone.semantic_weight
            semantic_weight = torch.cat([w, 1 - w], dim=-1)

        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2)
        x = x.permute(0, 2, 1)
        x, hw_shape = self.backbone.stages[_head_index - 1].downsample(x, hw_shape)
        if self.backbone.semantic_weight >= 0:
            sw = self.backbone.semantic_embed_w[_head_index - 1](semantic_weight).unsqueeze(1)
            sb = self.backbone.semantic_embed_b[_head_index - 1](semantic_weight).unsqueeze(1)
            x = x * self.backbone.softplus(sw) + sb
        x, hw_shape, out, out_hw_shape = self.backbone.stages[_head_index](x, hw_shape)
        out = getattr(self.backbone, f'norm{_head_index}')(out)
        out = out.view(-1, *out_hw_shape, self.backbone.num_features[_head_index]).permute(0, 3, 1, 2).contiguous()
        return out


if __name__ == '__main__':
    backbone = SOLIDERBackbone([3])
    backbone.load_pretrained_weights()
    print(backbone.out_channels_list)
    print(backbone.out_feat_names)
    print(backbone(torch.randn(5, 3, 224, 224)))
    print('---------------------------')
    head = SOLIDERHead(True)
    head.load_pretrained_weights()
    print(head.in_channels)
    print(head.out_channels)
    print(head(torch.randn(5, head.in_channels, 14, 14)).shape)
