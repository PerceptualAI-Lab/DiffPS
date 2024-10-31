# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

from pathlib import Path
import re
import torch
import numpy as np
import random
import os
from typing import Iterable, Tuple, Optional, List, Union, Dict, Any, Callable
from collections import OrderedDict
import functools


def make_log_dir(root: Path, subfolder_name: str = '', output_dir: str = '') -> Path:
    """ Creating './logs/subfolder_name/runX' in the same directory as this file, and return it. The X is equal to
    maximum+1. """
    base = root.joinpath(r'logs', subfolder_name)
    maximum = 1
    if base.exists():
        pattern = re.compile(r'^run([1-9]+[0-9]*)$')
        for folder in base.iterdir():
            number = pattern.findall(folder.name)
            if number:
                maximum = max(maximum, int(number[0]) + 1)
    #path = base.joinpath(f'run{maximum}')
    path = base.joinpath(output_dir)
    path.mkdir(mode=0o755, parents=True, exist_ok=True)
    return path


def set_random_seed(seed) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    # torch.use_deterministic_algorithms(True)


def normalize_weight_zero_bias(module: torch.nn.Module) -> None:
    if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor) and module.weight.requires_grad:
        torch.nn.init.normal_(module.weight, std=0.01)
    if hasattr(module, 'bias') and isinstance(module.bias, torch.Tensor) and module.bias.requires_grad:
        torch.nn.init.zeros_(module.bias)


def make_optimizer(
        named_parameters: Iterable[Tuple[str, torch.nn.Parameter]],
        type_: str,
        lr: float,
        weight_decay: float = 0.,
        bias_lr_factor: float = 1.,
        bias_decay: Optional[float] = None,
        sgd_momentum: float = 0.,
) -> torch.optim.Optimizer:
    param_biases, param_others = [], []
    for nm, param in named_parameters:
        if param.requires_grad:
            (param_biases if 'bias' in nm else param_others).append(param)
    cfgs = [{'params': param_others, 'lr': lr, 'weight_decay': weight_decay}]
    if len(param_biases) > 0:
        cfgs.append({'params': param_biases, 'lr': lr * bias_lr_factor, 'weight_decay': bias_decay or weight_decay})
    if type_ in {'SGD', 'sgd'}:
        return torch.optim.SGD(cfgs, lr=lr, momentum=sgd_momentum)
    elif type_ in {'Adam', 'adam', 'ADAM'}:
        return torch.optim.Adam(cfgs, lr=lr)
    elif type_ in {'AdamW', 'adamw', 'ADAMW'}:
        return torch.optim.AdamW(cfgs, lr=lr)
    else:
        raise ValueError


def penetrate_list_or_dict(function: Callable):
    @functools.wraps(function)
    def convert(data: Any, *args, **kwargs):
        if isinstance(data, list):
            return [convert(item, *args, **kwargs) for item in data]
        elif isinstance(data, dict):
            return {key: convert(value, *args, **kwargs) for key, value in data.items()}
        else:
            return function(data, *args, **kwargs)

    return convert


@penetrate_list_or_dict
def ndarray_to_tensor(data: np.ndarray, device: torch.DeviceObjType) -> torch.Tensor:
    return torch.from_numpy(data).to(device)


@penetrate_list_or_dict
def tensor_to_ndarray(data: torch.Tensor) -> np.ndarray:
    return data.detach().cpu().numpy()


def make_scheduler(
        optimizer: torch.optim.Optimizer,
        iters_within_epoch: int,
        milestones: List[int],
        gamma: float = 0.1,
        warmup_factor: float = 0.001,
        warmup_epochs: int = 0,
) -> torch.optim.lr_scheduler.LRScheduler:
    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer=optimizer,
        start_factor=warmup_factor,
        total_iters=iters_within_epoch * warmup_epochs,
    ) if warmup_epochs > 0 else None
    multistep = torch.optim.lr_scheduler.MultiStepLR(
        optimizer=optimizer,
        milestones=list(map(lambda epoch: iters_within_epoch * (epoch - warmup_epochs), milestones)),
        gamma=gamma,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer=optimizer,
        schedulers=[warmup, multistep],
        milestones=[warmup.total_iters],
    ) if warmup_epochs > 0 else multistep


class Pack(torch.nn.Module):
    def __init__(
            self,
            body: torch.nn.Module,
            in_feat_name: Optional[str] = None,
            output_both_ends: bool = False,
    ):
        super().__init__()
        self.body = body
        self.in_feat_name = in_feat_name
        if output_both_ends:
            self.out_feat_names = ['head_input', 'head_output']
            self.out_channels_list = [body.in_channels, body.out_channels]
        else:
            self.out_channels = body.out_channels

    def forward(self, x: Union[torch.Tensor, Dict[str, torch.Tensor]]) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        x = x[self.in_feat_name] if self.in_feat_name is not None else x
        y = self.body(x)
        return OrderedDict(
            [(self.out_feat_names[0], x), (self.out_feat_names[1], y)]
        ) if hasattr(self, 'out_feat_names') else y

