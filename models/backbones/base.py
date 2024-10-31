# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from torch.nn import Module
from abc import ABC, abstractmethod


class BaseBackbone(Module, ABC):
    FEATURE_NAMES = [f'feat_map_{idx}' for idx in (2, 3, 4, 5)]

    @abstractmethod
    def load_pretrained_weights(self) -> None:
        pass


class BaseBackboneHead(Module, ABC):
    @abstractmethod
    def load_pretrained_weights(self) -> None:
        pass
