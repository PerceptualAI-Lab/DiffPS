# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from torch import Tensor
from torch.nn import Module, Sequential, Conv2d, BatchNorm2d

from torchvision.models.resnet import resnet50, ResNet50_Weights, Bottleneck
from torchvision.models._utils import IntermediateLayerGetter

from typing import List, Tuple, Dict

from models.backbones.base import BaseBackbone, BaseBackboneHead


_make_model = resnet50


def _get_weights():
    return ResNet50_Weights.IMAGENET1K_V1.get_state_dict(progress=True)


class ResNetBackbone(BaseBackbone):
    def __init__(self, return_layers_idx: List[int]):
        return_layers_idx.sort()
        assert 1 <= return_layers_idx[0] and return_layers_idx[-1] <= 4
        super().__init__()
        return_layers = {f'layer{idx}': self.FEATURE_NAMES[idx - 1] for idx in return_layers_idx}
        self.body = IntermediateLayerGetter(model=_make_model(), return_layers=return_layers)
        self.body['conv1'].requires_grad_(False)
        self.body['bn1'].requires_grad_(False)
        self.out_channels_list = [self.body[nm][-1].conv3.out_channels for nm in return_layers]
        self.out_feat_names = list(return_layers.values())

    def load_pretrained_weights(self) -> None:
        self.body.load_state_dict(_get_weights(), strict=False)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        return self.body(x)


class ResNetHead(BaseBackboneHead):
    def __init__(self, down_sampling: bool = True):
        super().__init__()
        self.layer4 = Sequential(  # If the stride of the first block is 1, this layer do not have downsampling.
            Bottleneck(
                1024, 512,
                stride=1 + down_sampling,
                downsample=Sequential(
                    Conv2d(1024, 2048, 1, 1 + down_sampling, 0, bias=False),
                    BatchNorm2d(2048)
                ),
            ),
            Bottleneck(2048, 512),
            Bottleneck(2048, 512),
        )
        self.in_channels = self.layer4[0].conv1.in_channels
        self.out_channels = self.layer4[-1].conv3.out_channels

    def load_pretrained_weights(self) -> None:
        self.load_state_dict({name: value for name, value in _get_weights().items() if 'layer4' in name})

    def forward(self, x: Tensor) -> Tensor:
        return self.layer4(x)


if __name__ == '__main__':
    backbone = ResNetBackbone([3])
    backbone.load_pretrained_weights()
    print(backbone.out_feat_names)
    print(backbone.out_channels_list)
    print(backbone.body['conv1'].weight.requires_grad)
    head = ResNetHead()
    head.load_pretrained_weights()
    print(head)
    print(head.out_channels)
