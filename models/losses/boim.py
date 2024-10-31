# -*- coding: UTF-8 -*-
# @Author: Yimin Jiang

import torch
from torch import Tensor
from torch.nn import Module, functional as F

from typing import Optional

from models.losses.triplet import TripletCosineLoss


class BidirectionalOnlineInstanceMatchingLoss(Module):
    def __init__(
            self,
            dim: int,
            lut_size: int,
            cq_size: int,
            momentum: Optional[float] = None,
            scalar: float = 30.0,
            margin: float = 0.25,
            weight_softmax: float = 1.0,
            weight_triplet: float = 1.0,
    ):
        """
        Args:
            dim: dimension of person feature vectors
            lut_size: the number of the persons who have label in the dataset
            cq_size: the size of the circular queue. If you do not use cq, set this value to 0.
            momentum: lut momentum
            scalar: similarity matrix scalar
        """
        super().__init__()
        self.dim = dim
        self.lut_size = lut_size
        self.cq_size = cq_size
        self.ratio = ((1.0 - momentum) / momentum) if momentum is not None else None
        self.scalar = scalar
        self.triplet = TripletCosineLoss(margin)
        self.weight_softmax = weight_softmax
        self.weight_triplet = weight_triplet
        self.register_buffer('lut', torch.zeros((lut_size, dim)))
        self.register_buffer('cq', torch.zeros((cq_size, dim)))
        self._update_lut = self._adaptively_update_lut if momentum is None else self._momentum_update_lut

    ''' Update LUT and CQ method ----------------------------------------------------------------------------------- '''
    @torch.no_grad()
    def _momentum_update_lut(self, vectors: Tensor, labels: Tensor) -> None:
        for label, vector in zip(labels, vectors):
            self.lut[label] = F.normalize(self.lut[label] + self.ratio * vector, dim=-1)

    @torch.no_grad()
    def _adaptively_update_lut(self, vectors: Tensor, labels: Tensor) -> None:
        for label, vector in zip(labels, vectors):
            center = self.lut[label]
            hard = self.lut[torch.topk(self.lut @ center, k=2).indices[1]]  # get the hard negative of the center
            self.lut[label] = F.normalize(center + torch.exp(vector @ (center - hard)) * vector, dim=-1)

    @torch.no_grad()
    def _update_cq(self, vectors: Tensor) -> None:
        self.cq = torch.cat([vectors, self.cq])[:self.cq_size]

    ''' Loss calculation methods ----------------------------------------------------------------------------------- '''
    def _softmax_loss(self, vectors: Tensor, labels: Tensor) -> Tensor:
        cos_sim_mat = vectors @ torch.cat([self.lut, self.cq]).t()
        sim_mat = self.scalar * cos_sim_mat  # [-1,+1] -> [-scalar,+scaler]
        return F.cross_entropy(sim_mat, labels, reduction='mean')

    def _triplet_loss(self, l_vectors: Tensor, l_labels: Tensor, u_vectors: Tensor) -> Tensor:
        m_labels = l_labels.unique()
        m_vectors = self.lut[m_labels]
        return self.triplet(
            anchor_vectors=torch.cat([l_vectors, m_vectors]),
            anchor_labels=torch.cat([l_labels, m_labels]),
            sample_vectors=torch.cat([l_vectors, u_vectors.detach(), m_vectors]),
            sample_labels=torch.cat([l_labels, l_labels.new_full((len(u_vectors),), -100), m_labels]),
            reduction='mean',
            normalized=True,
        )

    def forward(self, vectors: Tensor, labels: Tensor, *, normalized: bool = False) -> Optional[Tensor]:
        mask = labels > 0  # filter out background
        vectors, labels = vectors[mask] if normalized else F.normalize(vectors[mask], dim=-1), labels[mask] - 1
        mask = labels < self.lut_size  # filter out unlabeled person
        l_vectors, l_labels, u_vectors = vectors[mask], labels[mask], vectors[~mask]
        losses = []
        if len(l_vectors) > 0:
            if self.weight_softmax > 0.0:
                losses.append(self.weight_softmax * self._softmax_loss(l_vectors, l_labels))
            if self.weight_triplet > 0.0:
                losses.append(self.weight_triplet * self._triplet_loss(l_vectors, l_labels, u_vectors))
            self._update_lut(l_vectors.detach(), l_labels)
        if len(u_vectors) > 0:
            self._update_cq(u_vectors.detach())
        return sum(losses) if losses else None


if __name__ == '__main__':
    model = BidirectionalOnlineInstanceMatchingLoss(1024, 5532, 5000, None, 30, 0.25, 1.0, 1.0)
    a = torch.randn((64, 1024), requires_grad=True)
    b = torch.arange(5000, 5064)
    c = model(a, b)
    c.backward()
    print(a.grad)
