# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module
from torchvision.ops import boxes as box_ops

from typing import Tuple, List, Dict, Callable, Optional, Union
from numpy import ndarray


class Sampler:
    def __init__(
            self,
            # Match
            fg_thresh: float,
            bg_thresh: float,
            # Balanced positive negative
            batch_size_per_image: Optional[int] = None,  # If None, will not balance pos and neg
            positive_fraction: float = 0.5,
            keep_positive_fraction: bool = True,  # If True, even num_pos + num_neg < batch_size, the pos_frac still be keep
            # Placeholder
            bg_label: int = 0,
            neg_idx_code: int = -1,
            between_idx_code: int = -2,
            # others
            append_gt_boxes: bool = False,
            box_similarity_function: Callable[[Tensor, Tensor], Tensor] = box_ops.box_iou
    ):
        self.fg_thresh = fg_thresh
        self.bg_thresh = bg_thresh
        self.num_pos_for_img = None if batch_size_per_image is None else int(batch_size_per_image * positive_fraction)
        self.num_neg_for_img = None if batch_size_per_image is None else (batch_size_per_image - self.num_pos_for_img)
        self.keep_positive_fraction = keep_positive_fraction
        self.bg_label = bg_label
        self.neg_idx_code = neg_idx_code
        self.between_idx_code = between_idx_code
        self.append_gt_boxes = append_gt_boxes
        self.box_similarity_function = box_similarity_function

    def __call__(
            self, boxes: List[Tensor], targets: List[Dict[str, Tensor]]
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor], List[Tensor]]:
        truths, labels = [target['boxes'] for target in targets], [target['labels'] for target in targets]
        if self.append_gt_boxes:
            boxes = [torch.cat([boxes_in_img, truths_in_img]) for boxes_in_img, truths_in_img in zip(boxes, truths)]
        sample_boxes, sample_truths, sample_labels, all_matched_idxes = [], [], [], []
        for boxes_in_img, truths_in_img, labels_in_img in zip(boxes, truths, labels):
            sim_matrix = self.box_similarity_function(truths_in_img, boxes_in_img)
            sim_val_per_match, gt_idx_per_match = torch.max(sim_matrix, dim=0)
            pos_idx = torch.nonzero(sim_val_per_match >= self.fg_thresh).view(-1)
            neg_idx = torch.nonzero(sim_val_per_match < self.bg_thresh).view(-1)
            if self.num_pos_for_img is not None:
                shrinkage_factor = min(len(pos_idx) / self.num_pos_for_img, len(neg_idx) / self.num_neg_for_img, 1.0) \
                    if self.keep_positive_fraction else 1.0
                pos_idx = pos_idx[
                    torch.randperm(len(pos_idx), device=pos_idx.device)[:int(self.num_pos_for_img * shrinkage_factor)]
                ]
                neg_idx = neg_idx[
                    torch.randperm(len(neg_idx), device=neg_idx.device)[:int(self.num_neg_for_img * shrinkage_factor)]
                ]  # Force positive and negative samples to maintain balance proportionally by shrinkage_factor
            matched_truths, matched_labels = truths_in_img[gt_idx_per_match], labels_in_img[gt_idx_per_match]
            matched_labels[neg_idx] = self.bg_label
            sample_idxes = torch.cat([pos_idx, neg_idx])
            sample_boxes.append(boxes_in_img[sample_idxes])
            sample_truths.append(matched_truths[sample_idxes])
            sample_labels.append(matched_labels[sample_idxes])
            matched_idx = torch.full((len(boxes_in_img),), self.between_idx_code, device=boxes_in_img.device)
            matched_idx[pos_idx] = gt_idx_per_match[pos_idx]
            matched_idx[neg_idx] = self.neg_idx_code
            all_matched_idxes.append(matched_idx)
        return sample_boxes, sample_truths, sample_labels, all_matched_idxes


def compute_inters(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    inters_wh = (
        torch.min(boxes1[:, 2:], boxes2[:, 2:]) - torch.max(boxes1[:, :2], boxes2[:, :2])
    ).clamp(min=0)
    return inters_wh[:, 0] * inters_wh[:, 1]


def compute_valid_area_fraction(truths: Tensor, boxes: Tensor) -> Tensor:
    return compute_inters(truths, boxes) / box_ops.box_area(boxes)


def compute_iou(truths: Tensor, boxes: Tensor) -> Tensor:
    inters = compute_inters(truths, boxes)
    unions = box_ops.box_area(truths) + box_ops.box_area(boxes) - inters
    return inters / unions


def compute_centerness(truths: Tensor, boxes: Tensor) -> Tensor:
    box_ctrs = (boxes[:, :2] + boxes[:, 2:]) / 2
    margins_left_top = box_ctrs - truths[:, :2]
    margins_right_bottom = truths[:, 2:] - box_ctrs
    ratio_horizontal_vertical = (
        torch.min(margins_left_top, margins_right_bottom).clamp(min=0) /
        torch.max(margins_left_top, margins_right_bottom)
    )
    return torch.sqrt(ratio_horizontal_vertical[:, 0] * ratio_horizontal_vertical[:, 1])


class BoxConverter:
    @staticmethod
    def xywh_to_xyxy_(boxes: Union[Tensor, ndarray]) -> None:
        boxes[..., 2:] += boxes[..., :2]

    @staticmethod
    def xywh_to_cxcywh_(boxes: Union[Tensor, ndarray]) -> None:
        boxes[..., :2] += 0.5 * boxes[..., 2:]

    @staticmethod
    def xyxy_to_xywh_(boxes: Union[Tensor, ndarray]) -> None:
        boxes[..., 2:] -= boxes[..., :2]

    @staticmethod
    def xyxy_to_cxcywh_(boxes: Union[Tensor, ndarray]) -> None:
        BoxConverter.xyxy_to_xywh_(boxes)
        BoxConverter.xywh_to_cxcywh_(boxes)

    @staticmethod
    def cxcywh_to_xywh_(boxes: Union[Tensor, ndarray]) -> None:
        boxes[..., :2] -= 0.5 * boxes[..., 2:]

    @staticmethod
    def cxcywh_to_xyxy_(boxes: Union[Tensor, ndarray]) -> None:
        BoxConverter.cxcywh_to_xywh_(boxes)
        BoxConverter.xywh_to_xyxy_(boxes)

    @staticmethod
    def convert_(boxes: Union[Tensor, ndarray], in_fmt: str, out_fmt: str):
        getattr(BoxConverter, f'{in_fmt}_to_{out_fmt}_')(boxes)

    def __init__(self, in_fmt: str, out_fmt: str, copy: bool = True):
        self._convert_ = getattr(BoxConverter, f'{in_fmt}_to_{out_fmt}_')
        self._copy = (lambda data: data.copy()) if copy else (lambda data: data)

    def __call__(self, boxes: Union[Tensor, ndarray]) -> Union[Tensor, ndarray]:
        boxes = self._copy(boxes)
        self._convert_(boxes)
        return boxes
