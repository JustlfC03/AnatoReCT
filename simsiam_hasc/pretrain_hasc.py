import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transct_model import FeatureExtractor
from datasets.hasc_dataset import HASCDataset
from utils.hasc_losses import HASCLoss


class HASCPretrainer:
    def __init__(self, model, config, device='cuda'):
        self.model = model
        self.device = device
        self.config = config
        
        self.criterion = HASCLoss(
            slice_num=config['slice_num'],
            patch_num=config['patch_num'],
            margin=config.get('margin', 0.5)
        )
        
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.get('lr', 1e-4),
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        def lambda_rule(epoch):
            return 0.96 ** (epoch // 1000)
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda_rule
        )
    
    def train(self, dataloader, epochs=2500):
        self.model.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"HASC Epoch {epoch+1}/{epochs}")
            
            for ld_batch, nd_batch in pbar:
                ld_batch = ld_batch.to(self.device)
                nd_batch = nd_batch.to(self.device)
                
                B, S, P, C, H, W = ld_batch.shape
                
                ld_flat = ld_batch.view(B * S * P, C, H, W)
                nd_flat = nd_batch.view(B * S * P, C, H, W)
                
                ld_feat = self.model(ld_flat, return_features=True)
                nd_feat = self.model(nd_flat, return_features=True)
                
                D = ld_feat.shape[-1]
                ld_feat = ld_feat.view(B, S, P, D)
                nd_feat = nd_feat.view(B, S, P, D)
                
                loss = self.criterion(ld_feat, nd_feat)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            self.scheduler.step()
            avg_loss = total_loss / len(dataloader)
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, LR = {current_lr:.2e}")
            
            if (epoch + 1) % 100 == 0:
                self._save_checkpoint(epoch)
        
        return self.model
    
    def _save_checkpoint(self, epoch):
        save_path = f"./checkpoints/hasc_epoch_{epoch}.pt"
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
        }, save_path)
        print(f"Saved: {save_path}")


def main():
    config = {
        'slice_num': 4,
        'patch_num': 8,
        'patch_size': 64,
        'feature_dim': 64,
        'margin': 0.5,
        'lr': 1e-4,
        'epochs': 2500,
        'batch_size': 2,
        'weight_decay': 1e-5,
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    data_root = "/data/Mayo2016/1mm_B30"
    train_patients = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333']
    
    dataset = HASCDataset(
        data_root=data_root,
        train_patients=train_patients,
        slice_num=config['slice_num'],
        patch_num=config['patch_num'],
        patch_size=config['patch_size']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )
    
    
    model = FeatureExtractor(
        in_channels=1,
        feature_dim=config['feature_dim']
    ).to(device)
    
    pretrainer = HASCPretrainer(model, config, device)
    trained_model = pretrainer.train(dataloader, epochs=config['epochs'])
    
    torch.save(trained_model.state_dict(), "./checkpoints/hasc_pretrained.pth")


if __name__ == '__main__':
    main()