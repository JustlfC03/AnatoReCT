import math
import os
import random
from collections import namedtuple
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from datasets.get_dataset import dataset
from einops import rearrange, reduce
from ema_pytorch import EMA
from torch import einsum, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import utils
from tqdm.auto import tqdm

ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])

def set_seed(SEED):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

def cycle(dl):
    while True:
        for data in dl:
            yield data

def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr

def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1

def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )

def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)

class WeightStandardizedConv2d(nn.Conv2d):
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()
        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)
        q = q * self.scale
        v = v / (h * w)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        q = q * self.scale
        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class FeatureFusionConditionalUnet(nn.Module):
    def __init__(
        self,
        dim=64,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        cond_channels=1,
        resnet_block_groups=8,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        enable_feature_fusion=True,
        global_feat_dim=64,
        local_feat_dim=128,
    ):
        super().__init__()
        self.channels = channels
        self.enable_feature_fusion = enable_feature_fusion
        
        input_channels = channels + cond_channels
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        block_klass = partial(ResnetBlock, groups=resnet_block_groups)
        
        time_dim = dim * 4
        if learned_sinusoidal_cond or random_fourier_features:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim
        
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        
        if enable_feature_fusion:
            self.global_feat_proj = nn.Sequential(
                nn.Linear(global_feat_dim, time_dim * 2),
                nn.GELU(),
                nn.Linear(time_dim * 2, time_dim)
            )
            
            self.local_feat_projs = nn.ModuleList()
            
            level_channels = []
            current_dim = init_dim
            for dim_mult in dim_mults:
                level_channels.append(current_dim)
                current_dim = dim * dim_mult
            
            for ch in level_channels:
                proj = nn.Sequential(
                    nn.Conv2d(local_feat_dim, ch, kernel_size=3, padding=1),
                    nn.GroupNorm(8, ch),
                    nn.SiLU()
                )
                self.local_feat_projs.append(proj)
            
            self.time_gate = nn.Sequential(
                nn.Linear(time_dim, 1),
                nn.Sigmoid()
            )

        self.downs = nn.ModuleList([])
        num_resolutions = len(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))
        
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        self.ups = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))
        
        default_out_dim = channels
        self.out_dim = default(out_dim, default_out_dim)
        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)
    
    def forward(self, noise_img, time, cond_img, global_feat=None, local_feat=None):
        if cond_img is None:
            cond_img = torch.zeros_like(noise_img)
        elif isinstance(cond_img, list):
            cond_img = cond_img[0] if len(cond_img) > 0 else torch.zeros_like(noise_img)
        else:
            cond_img = cond_img.float()

        noise_channels = noise_img.shape[1]
        cond_channels = cond_img.shape[1]
        if noise_channels != cond_channels:
            if noise_channels == 1 and cond_channels == 3:
                cond_img = cond_img.mean(dim=1, keepdim=True)
            elif noise_channels == 3 and cond_channels == 1:
                cond_img = cond_img.repeat(1, 3, 1, 1)
            else:
                cond_img = cond_img[:, :1] if cond_channels > 1 else cond_img
        
        if noise_img.shape[2:] != cond_img.shape[2:]:
            cond_img = F.interpolate(cond_img, size=noise_img.shape[2:], mode='bilinear', align_corners=False)
        
        x = torch.cat([noise_img, cond_img], dim=1)
        x = self.init_conv(x)
        r = x.clone()
        
        t = self.time_mlp(time)
        
        local_feat_buffers = None
        local_weight = None
        
        if self.enable_feature_fusion and (global_feat is not None or local_feat is not None):
            time_norm = time.float() / 1000.0
            gate_weight = self.time_gate(t)
            time_weight = time_norm.view(-1, 1)
            
            if global_feat is not None:
                global_t = self.global_feat_proj(global_feat)
                global_weight = time_weight * gate_weight
                t = t + global_weight * global_t
            
            if local_feat is not None:
                local_feat_buffers = []
                for proj in self.local_feat_projs:
                    projected = proj(local_feat)
                    local_feat_buffers.append(projected)
                local_weight = (1.0 - time_weight) * (1.0 - gate_weight)
        
        h = []
        level_idx = 0
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            
            if local_feat_buffers is not None and local_weight is not None:
                if level_idx < len(local_feat_buffers):
                    local_resized = F.interpolate(
                        local_feat_buffers[level_idx],
                        size=x.shape[2:],
                        mode='bilinear'
                    )
                    if local_resized.shape[1] == x.shape[1]:
                        x = x + local_weight.view(-1, 1, 1, 1) * local_resized
            
            h.append(x)
            x = block2(x, t)
            x = attn(x)
            h.append(x)
            x = downsample(x)
            level_idx += 1
        
        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)
        
        for block1, block2, attn, upsample in self.ups:
            x = torch.cat([x, h.pop()], dim=1)
            x = block1(x, t)
            x = torch.cat([x, h.pop()], dim=1)
            x = block2(x, t)
            x = attn(x)
            x = upsample(x)
        
        x = torch.cat([x, r], dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class ImprovedConditionalDDPM(nn.Module):
    def __init__(
        self,
        model,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        loss_type='l1',
        gamma_schedule='quadratic',
        sigma_max=0.3,
        enable_feature_fusion=False,
        global_feat_dim=64,
        local_feat_dim=128,
    ):
        super().__init__()
        self.model = model
        self.image_size = image_size
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.sigma_max = sigma_max
        self.enable_feature_fusion = enable_feature_fusion
        
        # 生成调度
        gamma = self._create_gamma_schedule(timesteps, gamma_schedule)
        sigma = self._create_sigma_schedule(timesteps)
        
        self.register_buffer('gamma', gamma)
        self.register_buffer('sigma', sigma)
        self.register_buffer('one_minus_gamma', 1.0 - gamma)
        
        print(f"ImprovedConditionalDDPM initialized: timesteps={timesteps}, sampling={sampling_timesteps}")
        print(f"  enable_feature_fusion={enable_feature_fusion}")
    
    def _create_gamma_schedule(self, timesteps, schedule_type):
        t = torch.linspace(0, 1, timesteps, dtype=torch.float64)
        if schedule_type == 'linear':
            gamma = t
        elif schedule_type == 'quadratic':
            gamma = t ** 2
        elif schedule_type == 'cosine':
            gamma = 1 - torch.cos(t * math.pi / 2)
        else:
            gamma = t
        return gamma.float()
    
    def _create_sigma_schedule(self, timesteps):
        t = torch.linspace(0, 1, timesteps, dtype=torch.float64)
        sigma = 4 * t * (1 - t) * self.sigma_max
        return sigma.float()
    
    def estimate_noise_level(self, x_cond):
        with torch.no_grad():
            laplacian = torch.tensor([[[[0, 1, 0], [1, -4, 1], [0, 1, 0]]]], device=x_cond.device).float()
            high_freq = F.conv2d(x_cond, laplacian, padding=1)
            noise_estimate = torch.std(high_freq.view(high_freq.shape[0], -1), dim=1)
            noise_level = torch.sigmoid((noise_estimate - 0.05) * 10)
        return noise_level.view(-1, 1, 1, 1)
    
    def q_sample(self, x_target, x_cond, t, noise=None):
        gamma_t = extract(self.gamma, t, x_target.shape)
        one_minus_gamma_t = extract(self.one_minus_gamma, t, x_target.shape)
        sigma_t = extract(self.sigma, t, x_target.shape)
        
        if noise is None:
            noise = torch.randn_like(x_target)
        
        noise_level = self.estimate_noise_level(x_cond)
        x_t = one_minus_gamma_t * x_target + gamma_t * x_cond + sigma_t * noise_level * noise
        return x_t
    
    def p_losses(self, x_target, x_cond, t, features=None):
        x_target = x_target.float()
        x_cond = x_cond.float()
        
        noise = torch.randn_like(x_target)
        x_t = self.q_sample(x_target, x_cond, t, noise)
        
        global_feat, local_feat = None, None
        if features is not None:
            global_feat, local_feat = features
        
        if self.enable_feature_fusion:
            pred_noise = self.model(x_t, t, x_cond, global_feat=global_feat, local_feat=local_feat)
        else:
            pred_noise = self.model(x_t, t, x_cond)
        
        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_noise, noise)
        else:
            loss = F.mse_loss(pred_noise, noise)
        
        return loss
    
    @torch.no_grad()
    def p_sample(self, x_t, t, x_cond, pred_noise):
        gamma_t = extract(self.gamma, t, x_t.shape)
        one_minus_gamma_t = extract(self.one_minus_gamma, t, x_t.shape)
        sigma_t = extract(self.sigma, t, x_t.shape)
        
        noise_level = self.estimate_noise_level(x_cond)
        x_target_est = (x_t - gamma_t * x_cond - sigma_t * noise_level * pred_noise) / (one_minus_gamma_t + 1e-8)
        x_target_est = torch.clamp(x_target_est, -1.0, 1.0)
        
        t_val = t[0].item() if isinstance(t, torch.Tensor) else t
        t_prev = t_val - 1
        
        if t_prev >= 0:
            t_prev_tensor = torch.full((x_t.shape[0],), t_prev, device=x_t.device, dtype=torch.long)
            gamma_prev = extract(self.gamma, t_prev_tensor, x_t.shape)
            one_minus_gamma_prev = extract(self.one_minus_gamma, t_prev_tensor, x_t.shape)
            x_prev = one_minus_gamma_prev * x_target_est + gamma_prev * x_cond
        else:
            x_prev = x_target_est
        
        return x_prev, x_target_est
    
    @torch.no_grad()
    def sample(self, cond_img, batch_size=None, last=True, features=None):
        if isinstance(cond_img, list):
            cond_img = cond_img[0] if len(cond_img) > 0 else None
        
        if cond_img is None:
            raise ValueError("cond_img is None")
        
        cond_img = cond_img.float()
        cond_img_norm = cond_img * 2 - 1
        
        if batch_size is None:
            batch_size = cond_img_norm.shape[0]
        
        device = cond_img_norm.device
        shape = (batch_size, self.model.channels, self.image_size, self.image_size)
        
        global_feat, local_feat = None, None
        if features is not None:
            global_feat, local_feat = features
        
        noise_level = self.estimate_noise_level(cond_img_norm)
        
        start_t = self.timesteps - 1
        sigma_start = extract(self.sigma, torch.full((batch_size,), start_t, device=device), shape)
        noise = torch.randn(shape, device=device)
        x = cond_img_norm + sigma_start * noise_level * noise
        x = torch.clamp(x, -1.0, 1.0)
        
        for t in reversed(range(self.sampling_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            if self.enable_feature_fusion:
                pred_noise = self.model(x, t_tensor, cond_img_norm, 
                                       global_feat=global_feat, local_feat=local_feat)
            else:
                pred_noise = self.model(x, t_tensor, cond_img_norm)
            
            x, _ = self.p_sample(x, t_tensor, cond_img_norm, pred_noise)
            x = torch.clamp(x, -1.0, 1.0)
        
        result = torch.clamp((x + 1) * 0.5, 0.0, 1.0)
        return [result]
    
    def forward(self, img, *args, features=None, **kwargs):
        if isinstance(img, list):
            x_target = img[0].float()
            x_cond = img[1].float()
        else:
            x_target = img.float()
            x_cond = None
        
        b = x_target.shape[0]
        device = x_target.device
        
        x_target = x_target * 2 - 1
        x_cond = x_cond * 2 - 1
        
        t = torch.randint(0, self.timesteps, (b,), device=device).long()
        return self.p_losses(x_target, x_cond, t, features=features)


# ========== Trainer 类 ==========
class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        folder,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_samples=25,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
        equalizeHist=False,
        crop_patch=False,
        generation=False,
        feature_extractor=None
    ):
        super().__init__()
        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.crop_patch = crop_patch
        self.accelerator.native_amp = amp
        self.feature_extractor = feature_extractor
        
        if feature_extractor is not None:
            for param in feature_extractor.parameters():
                param.requires_grad = False
            feature_extractor.eval()
        
        self.model = diffusion_model
        assert has_int_squareroot(num_samples)
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition
        
        # 设置数据集
        if self.condition:
            if len(folder) == 4:
                self.condition_type = 2
                ds = dataset(folder[2:4], self.image_size,
                             augment_flip=False, convert_image_to=convert_image_to, 
                             condition=1, equalizeHist=equalizeHist, crop_patch=crop_patch, 
                             sample=True, generation=generation)
                trian_folder = folder[0:2]
                self.sample_dataset = ds
                self.sample_loader = cycle(self.accelerator.prepare(
                    DataLoader(self.sample_dataset, batch_size=num_samples, shuffle=True,
                               pin_memory=True, num_workers=4)))
                ds = dataset(trian_folder, self.image_size, augment_flip=augment_flip,
                             convert_image_to=convert_image_to, condition=1, 
                             equalizeHist=equalizeHist, crop_patch=crop_patch, 
                             generation=generation)
                self.dl = cycle(self.accelerator.prepare(
                    DataLoader(ds, batch_size=train_batch_size, shuffle=True, 
                               pin_memory=True, num_workers=4)))
            else:
                raise ValueError(f"folder length {len(folder)} not supported")
        else:
            self.condition_type = 0
            ds = dataset(folder, self.image_size, augment_flip=augment_flip,
                         convert_image_to=convert_image_to, condition=0, 
                         equalizeHist=equalizeHist, crop_patch=crop_patch, 
                         generation=generation)
            self.dl = cycle(self.accelerator.prepare(
                DataLoader(ds, batch_size=train_batch_size, shuffle=True, 
                           pin_memory=True, num_workers=4)))
        
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)
        
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay, update_every=ema_update_every)
            self.set_results_folder(results_folder)
        
        self.step = 0
        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        device = self.accelerator.device
        self.device = device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            model = self.accelerator.unwrap_model(self.model)
            model.load_state_dict(data['model'])
            self.step = data['step']
            self.opt.load_state_dict(data['opt'])
            self.ema.load_state_dict(data['ema'])
            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])
            print("load model - " + str(path))
        self.ema.to(self.device)

    def train(self):
        accelerator = self.accelerator
        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:
            while self.step < self.train_num_steps:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        data = next(self.dl)
                        data = [item.to(self.device) for item in data]
                        ldct_img = data[1]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)
                        ldct_img = data
                    
                    features = None
                    if self.feature_extractor is not None:
                        with torch.no_grad():
                            if next(self.feature_extractor.parameters()).device != self.device:
                                self.feature_extractor = self.feature_extractor.to(self.device)
                            if ldct_img.max() <= 1.0 and ldct_img.min() >= 0.0:
                                ldct_img_for_feat = normalize_to_neg_one_to_one(ldct_img)
                            else:
                                ldct_img_for_feat = ldct_img
                            global_feat, local_feat = self.feature_extractor(ldct_img_for_feat)
                            features = (global_feat, local_feat)
                    
                    with self.accelerator.autocast():
                        loss = self.model(data, features=features)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                    
                    self.accelerator.backward(loss)
                
                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                accelerator.wait_for_everyone()
                self.opt.step()
                self.opt.zero_grad()
                accelerator.wait_for_everyone()
                self.step += 1
                
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()
                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.sample(milestone)
                        if self.step % (self.save_and_sample_every * 10) == 0:
                            self.save(milestone)
                
                pbar.set_description(f'loss: {total_loss:.4f}')
                pbar.update(1)
        
        accelerator.print('training complete')

    def sample(self, milestone, last=True, FID=False):
        self.ema.ema_model.eval()
        with torch.no_grad():
            batches = self.num_samples
            
            if self.condition_type == 0:
                x_input_sample = [0]
                show_x_input_sample = []
                features = None
            elif self.condition_type == 2:
                x_input_sample = next(self.sample_loader)
                x_input_sample = [item.to(self.device) for item in x_input_sample]
                show_x_input_sample = x_input_sample
                x_input_sample = x_input_sample[1:]  # 去掉GT
                
                # 修复：正确检查是否为list且非空
                features = None
                if self.feature_extractor is not None:
                    if x_input_sample and len(x_input_sample) > 0:
                        ldct_img = x_input_sample[0]
                        if ldct_img.max() <= 1.0 and ldct_img.min() >= 0.0:
                            ldct_img_for_feat = normalize_to_neg_one_to_one(ldct_img)
                        else:
                            ldct_img_for_feat = ldct_img
                        global_feat, local_feat = self.feature_extractor(ldct_img_for_feat)
                        features = (global_feat, local_feat)
            else:
                x_input_sample = [0]
                show_x_input_sample = []
                features = None
            
            all_images_list = show_x_input_sample + \
                list(self.ema.ema_model.sample(x_input_sample, batch_size=batches, last=last, features=features))
            
            all_images = torch.cat(all_images_list, dim=0)
            nrow = int(math.sqrt(self.num_samples)) if last else all_images.shape[0]
            
            file_name = f'sample-{milestone}.png'
            utils.save_image(all_images, str(self.results_folder / file_name), nrow=nrow)
            print("sample-save " + file_name)
        return milestone

    def test(self, sample=False, last=True, FID=False):
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(dataset=self.sample_dataset, batch_size=1)
            i = 0
            for items in loader:
                file_name = self.sample_dataset.load_name(i, sub_dir=self.sub_dir) if self.condition else f'{i}.png'
                i += 1
                
                with torch.no_grad():
                    if self.condition_type == 2:
                        x_input_sample = [item.to(self.device) for item in items]
                        show_x_input_sample = x_input_sample
                        x_input_sample = x_input_sample[1:]
                        
                        features = None
                        if self.feature_extractor is not None and x_input_sample:
                            ldct_img = x_input_sample[0]
                            if ldct_img.max() <= 1.0 and ldct_img.min() >= 0.0:
                                ldct_img_for_feat = normalize_to_neg_one_to_one(ldct_img)
                            else:
                                ldct_img_for_feat = ldct_img
                            global_feat, local_feat = self.feature_extractor(ldct_img_for_feat)
                            features = (global_feat, local_feat)
                    else:
                        x_input_sample = [items.to(self.device)]
                        show_x_input_sample = x_input_sample
                        features = None
                    
                    if sample:
                        all_images_list = show_x_input_sample + \
                            list(self.ema.ema_model.sample(x_input_sample, batch_size=1, features=features))
                    else:
                        all_images_list = list(self.ema.ema_model.sample(x_input_sample, batch_size=1, last=last, features=features))
                        all_images_list = [all_images_list[-1]]
                        if self.crop_patch:
                            pad_size = self.sample_dataset.get_pad_size(i-1)
                            for k, img in enumerate(all_images_list):
                                _, _, h, w = img.shape
                                img = img[:, :, 0:h-pad_size[0], 0:w-pad_size[1]]
                                all_images_list[k] = img
                
                all_images = torch.cat(all_images_list, dim=0)
                nrow = int(math.sqrt(self.num_samples)) if last else all_images.shape[0]
                utils.save_image(all_images, str(self.results_folder / file_name), nrow=nrow)
                print(f"test-save {i}: {file_name}")
        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)