# utils/hasc_losses.py
import torch
import torch.nn as nn


class HASCLoss(nn.Module):
    def __init__(self, slice_num=4, patch_num=8, margin=0.5):
        super().__init__()
        self.slice_num = slice_num
        self.patch_num = patch_num
        self.margin = margin
    
    def forward(self, features_ld, features_nd):
        """
        Args:
            features_ld: [B, slice_num, patch_num, D]
            features_nd: [B, slice_num, patch_num, D]
        """
        d_sl_ld = self._intra_slice_distance(features_ld)
        d_sl_nd = self._intra_slice_distance(features_nd)
        d_slvl_ld = self._inter_slice_distance(features_ld)
        d_slvl_nd = self._inter_slice_distance(features_nd)
        d_diff = self._cross_level_distance(features_ld, features_nd)
        
        avg_same_slice = (d_sl_ld + d_sl_nd) / 2.0
        avg_same_level = (d_slvl_ld + d_slvl_nd) / 2.0
        
        loss_same_slice = avg_same_slice
        loss_same_level = torch.relu(avg_same_slice - avg_same_level + self.margin)
        loss_diff_level = torch.relu(avg_same_level - d_diff + self.margin)
        loss_ranking = self._ranking_loss(d_sl_ld, d_sl_nd, d_slvl_ld, d_slvl_nd, d_diff)
        
        total_loss = (0.3 * loss_same_slice + 
                      0.2 * loss_same_level + 
                      0.3 * loss_diff_level + 
                      0.2 * loss_ranking)
        
        return total_loss
    
    def _intra_slice_distance(self, features):
        B, S, P, D = features.shape
        total = 0.0
        cnt = 0
        
        for s in range(S):
            slice_feat = features[:, s, :, :]
            for i in range(P):
                for j in range(i + 1, P):
                    dist = torch.sqrt(((slice_feat[:, i] - slice_feat[:, j]) ** 2).sum(dim=-1) + 1e-8)
                    total += dist.mean()
                    cnt += 1
        
        return total / max(cnt, 1)
    
    def _inter_slice_distance(self, features):
        B, S, P, D = features.shape
        slice_mean = features.mean(dim=2)
        
        total = 0.0
        cnt = 0
        
        for i in range(S):
            for j in range(i + 1, S):
                dist = torch.sqrt(((slice_mean[:, i] - slice_mean[:, j]) ** 2).sum(dim=-1) + 1e-8)
                total += dist.mean()
                cnt += 1
        
        return total / max(cnt, 1)
    
    def _cross_level_distance(self, features_ld, features_nd):
        B, S, P, D = features_ld.shape
        ld_flat = features_ld.reshape(B, -1)
        nd_flat = features_nd.reshape(B, -1)
        dist = torch.sqrt(((ld_flat - nd_flat) ** 2).sum(dim=-1) + 1e-8)
        return dist.mean()
    
    def _ranking_loss(self, d_sl_ld, d_sl_nd, d_slvl_ld, d_slvl_nd, d_diff, margin=0.5):
        loss_a_ld = torch.relu(d_sl_ld - d_slvl_ld + margin)
        loss_a_nd = torch.relu(d_sl_nd - d_slvl_nd + margin)
        avg_same_level = (d_slvl_ld + d_slvl_nd) / 2.0
        loss_b = torch.relu(avg_same_level - d_diff + margin)
        return (loss_a_ld + loss_a_nd + loss_b) / 3.0