"""
DGRPN (Diffusion-Guided Region Proposal Network) Modulator.
DiffPS 논문 Sec 3.4, Eq.1~4 구현.
"""
import torch
import torch.nn as nn
from torch.nn import functional as F


class DGRPNModulator(nn.Module):
    def __init__(self, tau=0.5, delta=3.0, peak_window=7, neigh_window=9, topk=50,
                 learnable_beta=True, init_beta=1.0, init_gamma=0.5):
        super().__init__()
        self.tau = tau
        self.delta = delta
        self.peak_window = peak_window
        self.neigh_window = neigh_window 
        self.topk = topk

        if learnable_beta:
            self.beta = nn.Parameter(torch.tensor(float(init_beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(init_beta)))
        self.gamma = nn.Parameter(torch.tensor(float(init_gamma)))

    @torch.no_grad()
    def _find_topk_peaks(self, M_th):
        B, _, H, W = M_th.shape
        # local maxima mask
        k = self.peak_window
        pooled = F.max_pool2d(M_th, kernel_size=k, stride=1, padding=k//2)
        is_peak = (M_th == pooled) & (M_th > 0)               # [B,1,H,W]
        scores = M_th.masked_fill(~is_peak, 0.0)              # keep only peaks

        # take Top-K per batch
        flat = scores.view(B, -1)                             # [B, H*W]
        topk_scores, topk_idx = torch.topk(flat, k=min(self.topk, H*W), dim=1)
        cy = (topk_idx // W).long()                           # [B,K]
        cx = (topk_idx %  W).long()
        return cx, cy, topk_scores

    def forward(self, agg_detection_feats, detection_attn_map):
        B, C, H, W = agg_detection_feats.shape
        device = agg_detection_feats.device

        # 1) Thresholding (Eq. 1)
        M = detection_attn_map.unsqueeze(1).clamp(min=0)      # [B,1,H,W]
        if M.max() > 0:
            M = M / (M.max() + 1e-8)                          # normalize to [0,1] (robust)
        M_th = torch.where(M > self.tau, M, torch.zeros_like(M))

        # 2) Peak detection (local maxima) & select Top-K
        cx, cy, peak_scores = self._find_topk_peaks(M_th)     # each [B,K]
        K = cx.shape[1]
        if K == 0:
            # no peaks -> return identity (no modulation)
            return agg_detection_feats

        # 3) Local weighted std around each peak (Eq. 2)
        # build coordinate grids
        xs = torch.arange(W, device=device).view(1, 1, 1, W).expand(B, 1, H, W)
        ys = torch.arange(H, device=device).view(1, 1, H, 1).expand(B, 1, H, W)

        # unfold to get neighborhood patches around every location
        k = self.neigh_window
        pad = k // 2
        unfold = nn.Unfold(kernel_size=k, padding=pad)

        # [B, k*k, H*W]
        patch_w = unfold(M_th)               # weights
        patch_x = unfold(xs.to(M_th))        # x-coords
        patch_y = unfold(ys.to(M_th))        # y-coords

        # gather patches at peak indices
        lin_idx = (cy * W + cx)              # [B,K]
        # expand batch for gather
        b_idx = torch.arange(B, device=device)[:, None].expand(B, K)  # [B,K]

        # gather columns for each (b, peak)
        # result shapes: [B, K, k*k]
        w_vec = patch_w.permute(0,2,1)[b_idx, lin_idx, :]      # weights
        x_vec = patch_x.permute(0,2,1)[b_idx, lin_idx, :]
        y_vec = patch_y.permute(0,2,1)[b_idx, lin_idx, :]

        # centers as float
        cx_f = cx.float().unsqueeze(-1)                        # [B,K,1]
        cy_f = cy.float().unsqueeze(-1)

        # Eq.2: s_w^k = max(δ, sqrt(Σ (i - c_x^k)^2 · M_th)); paper uses unnormalized weighted sum
        sum_sq_x = ((x_vec - cx_f) ** 2 * w_vec).sum(dim=-1)
        sum_sq_y = ((y_vec - cy_f) ** 2 * w_vec).sum(dim=-1)
        s_w = torch.sqrt(sum_sq_x.clamp_min(1e-8)).clamp_min(self.delta)
        s_h = torch.sqrt(sum_sq_y.clamp_min(1e-8)).clamp_min(self.delta)

        # 4) Build per-peak Gaussian maps G_k and aggregate by max (Eq. 3)
        # broadcast grids to [B,K,H,W]
        X = xs[:, :, :, :].float()                             # [B,1,H,W]
        Y = ys[:, :, :, :].float()
        X = X.expand(B, K, H, W)
        Y = Y.expand(B, K, H, W)

        cx_b = cx_f.squeeze(-1).unsqueeze(-1).unsqueeze(-1).float()  # [B,K,1,1]
        cy_b = cy_f.squeeze(-1).unsqueeze(-1).unsqueeze(-1).float()
        sw_b = s_w.unsqueeze(-1).unsqueeze(-1)
        sh_b = s_h.unsqueeze(-1).unsqueeze(-1)

        beta = self.beta.abs() + 1e-6                           # ensure positive
        Gk = torch.exp(- ((X - cx_b)**2) / (beta * (sw_b**2))
                       - ((Y - cy_b)**2) / (beta * (sh_b**2)))  # [B,K,H,W]

        G_final, _ = torch.max(Gk, dim=1)                       # [B,H,W]
        G_final = G_final.unsqueeze(1)                          # [B,1,H,W] for channel broadcast

        # 5) Modulate features: F_det = F + gamma * (G_final ⊙ F)  (Eq. 4)
        F_det = agg_detection_feats + self.gamma * (G_final * agg_detection_feats)
        return F_det
