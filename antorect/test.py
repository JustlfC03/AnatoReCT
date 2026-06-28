import os
import torch
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np

from src.diff import (
    set_seed, FeatureFusionConditionalUnet, ImprovedConditionalDDPM
)
from transct_model import FeatureExtractor
from datasets.get_dataset import dataset
from ema_pytorch import EMA

os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(e) for e in [0])
set_seed(42)


class DualFeatureExtractor(nn.Module):
    def __init__(self, hasc_backbone, simsiam_backbone, 
                 hasc_global_dim=64, simsiam_local_dim=128):
        super().__init__()
        self.hasc_backbone = hasc_backbone
        self.simsiam_backbone = simsiam_backbone
        self.global_expand = nn.Identity() if hasc_global_dim == 64 else nn.Sequential(
            nn.Linear(64, 128), nn.ReLU(), nn.Linear(128, hasc_global_dim)
        )
        self.local_proj = nn.Identity() if simsiam_local_dim >= 256 else nn.Sequential(
            nn.Conv2d(256, simsiam_local_dim, kernel_size=1),
            nn.GroupNorm(8, simsiam_local_dim), nn.SiLU()
        )
    
    def forward(self, x):
        global_feat = self.global_expand(self.hasc_backbone(x, return_features=True))
        local_feat = self.local_proj(self.simsiam_backbone(x, return_features=False))
        return global_feat, local_feat


timesteps = 1000
sampling_timesteps = 20
image_size = 512
channels = 1
HASC_GLOBAL_DIM = 64
SIMSIAM_LOCAL_DIM = 128

flist_dir = "/flist_dir"
folder = [
    os.path.join(flist_dir, "train_gt.flist"),
    os.path.join(flist_dir, "train_input.flist"),
    os.path.join(flist_dir, "test_gt.flist"),
    os.path.join(flist_dir, "test_input.flist")
]

checkpoint_path = '/antorect/results/sample/model-120.pt'

print("=" * 80)
print("Test - Save Grayscale Output Images")
print(f"Checkpoint: {checkpoint_path}")
print("=" * 80)


device = torch.device('cuda:0')

hasc_backbone = FeatureExtractor(in_channels=1, feature_dim=64)
hasc_ckpt = "/checkpoints/hasc_epoch_xxx.pt"
if os.path.exists(hasc_ckpt):
    state_dict = torch.load(hasc_ckpt, map_location='cpu')
    hasc_backbone.load_state_dict(state_dict.get('model_state_dict', state_dict), strict=False)
    print("  HASC loaded")

simsiam_backbone = FeatureExtractor(in_channels=1, feature_dim=256)
simsiam_ckpt = "/checkpoints/simsiam_epoch_xxx.pt"
if os.path.exists(simsiam_ckpt):
    state_dict = torch.load(simsiam_ckpt, map_location='cpu')
    simsiam_backbone.load_state_dict(state_dict.get('backbone_state_dict', state_dict), strict=False)
    print("  SimSiam loaded")

feature_extractor = DualFeatureExtractor(hasc_backbone, simsiam_backbone, HASC_GLOBAL_DIM, SIMSIAM_LOCAL_DIM)
feature_extractor.eval().to(device)
for p in feature_extractor.parameters():
    p.requires_grad = False
print("  Feature extractor loaded")

model = FeatureFusionConditionalUnet(
    dim=64, channels=channels, cond_channels=channels,
    dim_mults=(1, 2, 4, 8), resnet_block_groups=8,
    enable_feature_fusion=True,
    global_feat_dim=HASC_GLOBAL_DIM,
    local_feat_dim=SIMSIAM_LOCAL_DIM,
)

diffusion = ImprovedConditionalDDPM(
    model, image_size=image_size,
    timesteps=timesteps, sampling_timesteps=sampling_timesteps,
    loss_type='l1', gamma_schedule='quadratic', sigma_max=0.3,
    enable_feature_fusion=True,
)

diffusion = diffusion.to(device)



if not os.path.exists(checkpoint_path):
    print(f"  Error: Checkpoint not found at {checkpoint_path}")
    model_dir = Path(checkpoint_path).parent
    if model_dir.exists():
        model_files = list(model_dir.glob("model-*.pt"))
        if model_files:
            model_files.sort(key=lambda x: int(x.stem.split('-')[-1]))
            checkpoint_path = str(model_files[-1])
            print(f"  Using latest: {checkpoint_path}")
        else:
            print(f"  No model files found")
            exit(1)
    else:
        print(f"  Directory not found")
        exit(1)

checkpoint = torch.load(checkpoint_path, map_location=device)
print(f"  Checkpoint keys: {list(checkpoint.keys())}")

ema_model = EMA(diffusion, beta=0.995, update_every=10, update_after_step=0)

if 'ema' in checkpoint:
    ema_model.load_state_dict(checkpoint['ema'])
    print("  Loaded EMA state")
else:
    diffusion.load_state_dict(checkpoint['model'], strict=False)
    print("  Loaded model directly")

ema_model.to(device)
ema_model.eval()
inference_model = ema_model.ema_model
print("  EMA model ready")

test_dataset = dataset(
    folder[2:4], image_size,
    augment_flip=False, convert_image_to="L",
    condition=1, equalizeHist=False,
    crop_patch=False, sample=True, generation=False
)

test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
print(f"  Test samples: {len(test_loader)}")


output_dir = Path('./results/output_2020_120')
output_dir.mkdir(parents=True, exist_ok=True)

def save_grayscale(tensor, path):
    if tensor.dim() == 4:
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] == 1:
        tensor = tensor[0]
    if tensor.dim() == 3 and tensor.shape[0] == 3:
        tensor = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
    
    img_np = tensor.detach().cpu().numpy()
    img_np = np.clip(img_np * 255, 0, 255).astype(np.uint8)
    
    img_pil = Image.fromarray(img_np, mode='L')
    img_pil.save(path)
    
    saved = Image.open(path)
    print(f"    Saved: {path.name}, mode={saved.mode}, size={saved.size}")

print("\nProcessing...")

with torch.no_grad():
    for idx, batch in enumerate(test_loader):
        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1}/{len(test_loader)}")
        
        ldct = batch[1].to(device)
        
        ldct_norm = ldct * 2 - 1
        global_feat, local_feat = feature_extractor(ldct_norm)
        
        result = inference_model.sample(
            cond_img=ldct,
            batch_size=1,
            last=True,
            features=(global_feat, local_feat)
        )
        
        output = result[-1] if isinstance(result, list) else result
        output = torch.clamp(output, 0, 1)
        
        try:
            name = test_dataset.load_name(idx, sub_dir=False)
            name = os.path.splitext(name)[0]
        except:
            name = f'{idx:04d}'
        
        save_grayscale(output, output_dir / f'{name}.png')

print(f"\n" + "=" * 80)
print(f"Done! Output images saved to: {output_dir}")
print(f"Total: {len(list(output_dir.glob('*.png')))} images")
print("=" * 80)
