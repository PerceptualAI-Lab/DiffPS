# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module, functional as F


class TripletCosineLoss(Module):
    def __init__(self, margin: float):
        super().__init__()
        self.margin = margin

    def forward(
            self,
            anchor_vectors: Tensor,
            anchor_labels: Tensor,
            sample_vectors: Tensor,
            sample_labels: Tensor,
            reduction: str = "mean",
            normalized: bool = False,
    ) -> Tensor:
        if not normalized:
            anchor_vectors = F.normalize(anchor_vectors, dim=-1)
            sample_vectors = F.normalize(sample_vectors, dim=-1)
        cosine_sim_mat = anchor_vectors @ sample_vectors.t()
        labels_sam_mat = anchor_labels[:, None] == sample_labels
        positive, negative = [], []
        for cosine_sim, labels_sam in zip(cosine_sim_mat, labels_sam_mat):
            positive.append(torch.amin(cosine_sim[labels_sam]))  # Min cosine similarity in the same labels
            negative.append(torch.amax(cosine_sim[~labels_sam]))  # Max cosine similarity in the different labels
        return F.margin_ranking_loss(
            input1=torch.stack(positive),
            input2=torch.stack(negative),
            target=torch.ones_like(anchor_labels),
            margin=self.margin,
            reduction=reduction,
        )
