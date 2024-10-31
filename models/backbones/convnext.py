# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from typing import List, Dict
from collections import OrderedDict

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Conv2d

from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights
from torchvision.models._utils import IntermediateLayerGetter

from models.backbones.base import BaseBackbone, BaseBackboneHead


_make_model = convnext_base


def _get_weights():
    return ConvNeXt_Base_Weights.IMAGENET1K_V1.get_state_dict(progress=True)


class ConvNeXtBackbone(BaseBackbone):
    """
    convnext.features:
    index   | 0     | 1         | 2             | 3         | 4             | 5         | 6             | 7         |
    content | stem  | block seq | downsample    | block seq | downsample    | block seq | downsample    | block seq |
    layer   | stem  | layer1    | layer2                    | layer3                    | layer4                    |
    feature |       P1          P2                          P3                          P4                          P5
    """
    def __init__(self, return_layers_idx: List[int]):
        return_layers_idx.sort()
        assert 1 <= return_layers_idx[0] and return_layers_idx[-1] <= 4
        super().__init__()
        return_layers = {str(2 * idx - 1): self.FEATURE_NAMES[idx - 1] for idx in return_layers_idx}
        self.features = IntermediateLayerGetter(_make_model().features, return_layers=return_layers)
        self.features['0'].requires_grad_(False)  # frozen stem
        self.out_channels_list = [self.features[nm][-1].block[5].out_features for nm in return_layers]
        self.out_feat_names = list(return_layers.values())

    def load_pretrained_weights(self) -> None:
        self.load_state_dict(_get_weights(), strict=False)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.features(x)


class ConvNeXtHead(BaseBackboneHead):
    def __init__(self, down_sampling: bool = True):
        super().__init__()
        features = _make_model().features
        self.features = Sequential(OrderedDict([
            ('6', Sequential(OrderedDict([
                ('0', features[6][0]),  # norm
                (
                    '1', features[6][1]
                ) if down_sampling else (
                    'conv', Conv2d(features[6][1].in_channels, features[6][1].out_channels, 1, 1, 0)
                ),
            ]))),
            ('7', features[7]),
        ]))
        self.in_channels = self.features[0][1].in_channels
        self.out_channels = self.features[1][-1].block[5].out_features

    def load_pretrained_weights(self) -> None:
        self.load_state_dict(_get_weights(), strict=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x)


if __name__ == '__main__':
    backbone = ConvNeXtBackbone([3])
    print(backbone)
    print(backbone.out_channels_list)
    print('---------------------------')
    head = ConvNeXtHead(True)
    print(head)
    print(head.in_channels)
    print(head.out_channels)
    print(head(torch.randn(5, 512, 14, 14)).shape)
