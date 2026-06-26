import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transct_model import FeatureExtractor
from datasets.simsiam_dataset import SimSiamDataset
from utils.ct_augmentation import CTDataAugmentation
from utils.simsiam_loss import SimSiamLoss


class SimSiamProjector(nn.Module):
    def __init__(self, in_dim=256, proj_dim=2048):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
            nn.ReLU(),
            nn.Linear(proj_dim, proj_dim),
            nn.BatchNorm1d(proj_dim),
        )
    
    def forward(self, x):
        return self.net(x)


class SimSiamPretrainer:
    def __init__(self, backbone, config, device='cuda'):
        self.backbone = backbone
        self.device = device
        self.config = config
        
        proj_dim = config.get('proj_dim', 2048)
        
        self.projector = SimSiamProjector(
            in_dim=config['feature_dim'],
            proj_dim=proj_dim
        ).to(device)
        
        self.criterion = SimSiamLoss(proj_dim=proj_dim, pred_dim=config.get('pred_dim', 512))
        self.criterion = self.criterion.to(device)
        
        self.augmenter = CTDataAugmentation()
        
        self.optimizer = torch.optim.Adam(
            list(backbone.parameters()) + list(self.projector.parameters()),
            lr=config.get('lr', 2e-4),  # 改：1e-4 -> 2e-4
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        total_iterations = config.get('total_iterations', 100000)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_iterations, eta_min=1e-6
        )
        self.global_step = 0
        self.total_iterations = total_iterations
    
    def train(self, dataloader, epochs=300):
        self.backbone.train()
        self.projector.train()
        
        for epoch in range(epochs):
            total_loss = 0.0
            pbar = tqdm(dataloader, desc=f"SimSiam Epoch {epoch+1}/{epochs}")
            
            for batch in pbar:
                if self.global_step >= self.total_iterations:
                    print(f"Reached {self.total_iterations} iterations, stopping")
                    return self.backbone
                
                batch = batch.to(self.device)
                
                
                batch_np = batch.cpu().numpy()
                view1, view2 = self.augmenter.create_two_views(batch_np)
                view1 = torch.from_numpy(view1).float().to(self.device)
                view2 = torch.from_numpy(view2).float().to(self.device)
                
                feat1 = self.backbone(view1, return_features=True)
                feat2 = self.backbone(view2, return_features=True)
                
                z1 = self.projector(feat1)
                z2 = self.projector(feat2)
                
                loss = self.criterion(z1, z2)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                self.scheduler.step()
                self.global_step += 1
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'step': self.global_step})
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f}, Step = {self.global_step}")
            
            if (epoch + 1) % 10 == 0:
                self._save_checkpoint(epoch)
        
        return self.backbone
    
    def _save_checkpoint(self, epoch):
        save_path = f"./checkpoints/simsiam_epoch_{epoch}.pt"
        os.makedirs("./checkpoints", exist_ok=True)
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'backbone_state_dict': self.backbone.state_dict(),
            'projector_state_dict': self.projector.state_dict(),
        }, save_path)
        print(f"Saved: {save_path}")
    
    def load_checkpoint(self, checkpoint_path):
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
            self.projector.load_state_dict(checkpoint['projector_state_dict'])
            self.global_step = checkpoint['global_step']
            print(f"Loaded checkpoint, resuming from step {self.global_step}")
            return True
        return False


def main():
    config = {
        'feature_dim': 256,
        'proj_dim': 2048,
        'pred_dim': 512,
        'lr': 2e-4,
        'epochs': 300,
        'batch_size': 8,
        'weight_decay': 1e-5,
        'total_iterations': 100000
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_root = "/data/Mayo2016/1mm_B30"
    train_patients = ['L067', 'L096', 'L109', 'L143', 'L192', 'L286', 'L291', 'L310', 'L333']
    
    dataset = SimSiamDataset(
        data_root=data_root,
        patient_list=train_patients,
        dose_types=['quarter_1mm', 'full_1mm']
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0
    )
    
    backbone = FeatureExtractor(
        in_channels=1,
        feature_dim=config['feature_dim']
    ).to(device)
    
    pretrainer = SimSiamPretrainer(backbone, config, device)
    
    checkpoint_path = "./checkpoints/simsiam_epoch_xx.pt"
    pretrainer.load_checkpoint(checkpoint_path)
    
    
    trained_backbone = pretrainer.train(dataloader, epochs=config['epochs'])
    
    
    torch.save(trained_backbone.state_dict(), "./checkpoints/simsiam_pretrained_final.pth")

if __name__ == '__main__':
    main()