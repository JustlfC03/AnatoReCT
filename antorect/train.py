import os
import sys
import torch
import torch.nn as nn
from pathlib import Path

from src.diff import (
    Trainer, set_seed, FeatureFusionConditionalUnet, ImprovedConditionalDDPM
)
from transct_model import FeatureExtractor

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
set_seed(42)

debug = False
if debug:
    save_and_sample_every = 2
    timesteps = 100
    sampling_timesteps = 50
    train_num_steps = 200
else:
    save_and_sample_every = 1000
    timesteps = 1000
    sampling_timesteps = 20
    train_num_steps = 120000

condition = True
flist_dir = "/flist_dir"

folder = [
    os.path.join(flist_dir, "train_gt.flist"),
    os.path.join(flist_dir, "train_input.flist"),
    os.path.join(flist_dir, "test_gt.flist"),
    os.path.join(flist_dir, "test_input.flist")
]

train_batch_size = 1
num_samples = 4
image_size = 512
channels = 1

HASC_GLOBAL_DIM = 64
SIMSIAM_LOCAL_DIM = 128

print("=" * 80)
print(f"Image size: {image_size}")
print(f"Timesteps: {timesteps}")
print(f"Sampling timesteps: {sampling_timesteps}")
print("-" * 40)
print(f"Feature Fusion: ENABLED")
print(f"  HASC Global Feature: {HASC_GLOBAL_DIM} dim")
print(f"  SimSiam Local Feature: {SIMSIAM_LOCAL_DIM} dim")
print("=" * 80)

class DualFeatureExtractor(nn.Module):
    def __init__(self, hasc_backbone, simsiam_backbone, 
                 hasc_global_dim=64, simsiam_local_dim=128):
        super().__init__()
        self.hasc_backbone = hasc_backbone
        self.simsiam_backbone = simsiam_backbone
        
        if hasc_global_dim != 64:
            self.global_expand = nn.Sequential(
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.Linear(128, hasc_global_dim)
            )
            print(f"  HASC global expand: 64 -> {hasc_global_dim}")
        else:
            self.global_expand = nn.Identity()
        
        if simsiam_local_dim < 256:
            self.local_proj = nn.Sequential(
                nn.Conv2d(256, simsiam_local_dim, kernel_size=1),
                nn.GroupNorm(8, simsiam_local_dim),
                nn.SiLU()
            )
            print(f"  SimSiam local project: 256 -> {simsiam_local_dim}")
        else:
            self.local_proj = nn.Identity()
    
    def forward(self, x):
        global_feat = self.hasc_backbone(x, return_features=True)
        global_feat = self.global_expand(global_feat)
        
        local_feat = self.simsiam_backbone(x, return_features=False)
        local_feat = self.local_proj(local_feat)
        
        return global_feat, local_feat

hasc_backbone = FeatureExtractor(in_channels=1, feature_dim=64)
hasc_ckpt = "/checkpoints/hasc_epoch_xxx.pt"

print(f"  Loading HASC checkpoint: {hasc_ckpt}")
state_dict = torch.load(hasc_ckpt, map_location='cpu')
model_state = state_dict.get('model_state_dict', state_dict)
hasc_backbone.load_state_dict(model_state, strict=False)
print("  HASC loaded")

simsiam_backbone = FeatureExtractor(in_channels=1, feature_dim=256)
simsiam_ckpt = "/checkpoints/simsiam_epoch_xxx.pt"
print(f"  Loading SimSiam checkpoint: {simsiam_ckpt}")
state_dict = torch.load(simsiam_ckpt, map_location='cpu')
model_state = state_dict.get('backbone_state_dict', state_dict)
simsiam_backbone.load_state_dict(model_state, strict=False)
print("  SimSiam loaded")

feature_extractor = DualFeatureExtractor(
    hasc_backbone=hasc_backbone,
    simsiam_backbone=simsiam_backbone,
    hasc_global_dim=HASC_GLOBAL_DIM,
    simsiam_local_dim=SIMSIAM_LOCAL_DIM
)

for param in feature_extractor.parameters():
    param.requires_grad = False
feature_extractor.eval()

print(f"\n  Feature extractor ready:")
print(f"    Global (HASC): [{HASC_GLOBAL_DIM}]")
print(f"    Local (SimSiam): [{SIMSIAM_LOCAL_DIM}, 32, 32]")

model = FeatureFusionConditionalUnet(
    dim=64,
    channels=channels,
    cond_channels=channels,
    dim_mults=(1, 2, 4, 8),
    resnet_block_groups=8,
    enable_feature_fusion=True,
    global_feat_dim=HASC_GLOBAL_DIM,
    local_feat_dim=SIMSIAM_LOCAL_DIM,
)

diffusion = ImprovedConditionalDDPM(
    model,
    image_size=image_size,
    timesteps=timesteps,
    sampling_timesteps=sampling_timesteps,
    loss_type='l1',
    gamma_schedule='quadratic',
    sigma_max=0.3,
    enable_feature_fusion=True,
    global_feat_dim=HASC_GLOBAL_DIM,
    local_feat_dim=SIMSIAM_LOCAL_DIM,
)

print(f"Model created. Total parameters: {sum(p.numel() for p in model.parameters()):,}")

trainer = Trainer(
    diffusion,
    folder,
    train_batch_size=train_batch_size,
    num_samples=num_samples,
    train_lr=2e-4,
    train_num_steps=train_num_steps,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=False,
    convert_image_to="L",
    condition=condition,
    save_and_sample_every=save_and_sample_every,
    equalizeHist=False,
    crop_patch=False,
    generation=False,
    feature_extractor=feature_extractor
)

device = next(feature_extractor.parameters()).device
print(f"  Feature extractor device: {device}")

test_input = torch.randn(1, 1, 512, 512).to(device)
with torch.no_grad():
    g, l = feature_extractor(test_input)

trainer.train()
