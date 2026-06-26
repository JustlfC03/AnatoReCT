# utils/simsiam_loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimSiamLoss(nn.Module):
    def __init__(self, proj_dim=2048, pred_dim=512):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(proj_dim, pred_dim),
            nn.BatchNorm1d(pred_dim),
            nn.ReLU(),
            nn.Linear(pred_dim, proj_dim)
        )
    
    def forward(self, z1, z2):
        p1 = self.predictor(z1)
        p2 = self.predictor(z2)
        
        z1 = z1.detach()
        z2 = z2.detach()
        
        loss = -F.cosine_similarity(p1, z2).mean() - F.cosine_similarity(p2, z1).mean()
        
        return loss / 2