# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn.modules import Module
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import wandb
from typing import Dict, Optional, Any, Iterable
from tqdm import tqdm

from utils.general import ndarray_to_tensor


class Trainer:
    def __init__(
            self,
            model: Module,
            optimizer: Optimizer,
            scheduler: Optional[LRScheduler] = None,
            clip_grad: float = -1.,
            amp: bool = False,
    ):
        self.epoch = 0
        self.iteration = 0
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.clip_grad = clip_grad
        self.scaler = GradScaler(init_scale=2. ** 13, enabled=amp)
        self.device = next(model.parameters()).device

    def train(self, dataloader: DataLoader) -> Iterable[Dict[str, Tensor]]:
        self.epoch += 1
        self.model.train()
        self.optimizer.zero_grad()
        for images, targets, _ in tqdm(dataloader, desc=f'Epoch-{self.epoch}', unit='batches'):
            self.iteration += 1
            images = ndarray_to_tensor(images, self.device)
            targets = ndarray_to_tensor(targets, self.device)

            # Forward
            with autocast(enabled=self.scaler.is_enabled()):
                losses = self.model(images, targets)
                loss = sum(losses.values())

            # Backward
            if torch.isnan(loss) or torch.isinf(loss):  # check loss value
                raise ValueError(f'Loss value error! {losses}')
            self.scaler.scale(loss).backward()

            # Optimize
            if self.clip_grad > 0:
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), max_norm=self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

            # Scheduler
            self.scheduler.step()
            
            # Log to wandb
            wandb.log({
                "train/loss": loss.item(),
                "train/lr": self.optimizer.param_groups[0]["lr"],
                "epoch": self.epoch,
                "iter": self.iteration
            })
            for loss_name, loss_value in losses.items():
                wandb.log({f"train/{loss_name}": loss_value.item()})
                
            yield losses

    def save_ckpt(self, file: str) -> None:
        torch.save({
            'epoch': self.epoch,
            'iteration': self.iteration,
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler else None,
            'scaler': self.scaler.state_dict() if self.scaler.is_enabled() else None,
        }, file)

    def load_ckpt(self, file: str) -> None:
        ckpt = torch.load(file)
        self.epoch = ckpt['epoch']
        self.iteration = ckpt['iteration']
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        if self.scheduler:
            self.scheduler.load_state_dict(ckpt['scheduler'])
        if self.scaler.is_enabled():
            self.scaler.load_state_dict(ckpt['scaler'])

    @staticmethod
    def get_model_state_dict_from_ckpt(file: str) -> Dict[str, Any]:
        return torch.load(file)['model']
