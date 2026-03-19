import torch
import torch.nn as nn
import torch.nn.functional as F

class SFAN(nn.Module):
    def __init__(self, embed_dim, num_parts=4):
        super(SFAN, self).__init__()
        self.part_weights = nn.Parameter(torch.ones(num_parts))
        self.alpha = nn.Parameter(torch.tensor(0.0))
        self.num_parts = num_parts

    def set_text_embeddings(self, text_embeds):
        self.text_embeds = text_embeds

    def forward(self, F_refined):

        B, C, H, W = F_refined.shape

        # 1. feature normalization
        feat = F.normalize(F_refined, dim=1)  # (B, C, H, W)
        feat_flat = feat.view(B, C, -1).permute(0, 2, 1)  # (B, HW, C)

        # 2. text embedding normalization
        text_embeds = F.normalize(self.text_embeds, dim=1)  # (num_parts, C)

        # 3. cosine similarity (Eq. 4): (B, HW, num_parts)
        sim = torch.matmul(feat_flat, text_embeds.t())
        sim = sim.permute(0, 2, 1).view(B, self.num_parts, H, W)  # (B, num_parts, H, W)

        # 4. softmax over parts → probability maps Ŝ_c
        sim_soft = F.softmax(sim, dim=1)  # (B, num_parts, H, W)

        # 5. semantic weighting: F_c = Ŝ_c ⊙ F_refined
        F_refined_exp = F_refined.unsqueeze(1)  # (B,1,C,H,W)
        sim_soft_exp = sim_soft.unsqueeze(2)    # (B,num_parts,1,H,W)
        F_parts = sim_soft_exp * F_refined_exp  # (B,num_parts,C,H,W)

        # 6. weighted sum over parts (Eq. 6): F_sem_branch = Σ_c W_c F_c
        w = F.softmax(self.part_weights, dim=0)  # (num_parts,)
        F_sem_branch = torch.sum(w.view(1, self.num_parts, 1, 1, 1) * F_parts, dim=1)

        # 7. skip + balance: F_sem = F_refined + alpha * (F_sem_branch - F_refined)
        alpha = self.alpha.to(F_refined.dtype)
        F_sem = F_refined + alpha * (F_sem_branch - F_refined)
        return F_sem