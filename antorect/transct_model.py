# transct_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _tf_fspecial_gauss(size, sigma):
    coords = torch.arange(size) - (size - 1) / 2
    x, y = torch.meshgrid(coords, coords, indexing='ij')
    g = torch.exp(-(x**2 + y**2) / (2 * sigma**2))
    g = g / g.sum()
    return g.view(1, 1, size, size)


class GaussianLowPass(nn.Module):
    def __init__(self, kernel_size=11, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer('kernel', _tf_fspecial_gauss(kernel_size, sigma))
    
    def forward(self, x):
        pad = self.kernel_size // 2
        return F.conv2d(x, self.kernel, padding=pad, groups=x.shape[1])


class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, Q, K, V):
        d_k = Q.shape[-1]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        output = torch.matmul(attn, V)
        return output


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model, bias=True)
        self.W_K = nn.Linear(d_model, d_model, bias=True)
        self.W_V = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.attention = ScaledDotProductAttention()
    
    def forward(self, queries, keys, values):
        Q = self.W_Q(queries)
        K = self.W_K(keys)
        V = self.W_V(values)
        
        batch_size = Q.size(0)
        
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        output = self.attention(Q, K, V)
        
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_proj(output)
        
        return output


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=None):
        super().__init__()
        d_ff = d_ff or 8 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.activation = nn.LeakyReLU(0.2)
    
    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ff(x)
        x = x + residual
        
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads=8):
        super().__init__()
        self.self_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.cross_attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
    
    def forward(self, x, memory):
        residual = x
        x = self.norm1(x)
        x = self.self_attn(x, x, x)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.cross_attn(x, memory, memory)
        x = x + residual
        
        residual = x
        x = self.norm3(x)
        x = self.ff(x)
        x = x + residual
        
        return x


class FeatureExtractor(nn.Module):
    def __init__(self, in_channels=1, feature_dim=256):
        super().__init__()
        self.feature_dim = feature_dim
        
        self.gaussian_filter = GaussianLowPass(kernel_size=11, sigma=1.5)
        
        self.conv1_lr = nn.Conv2d(in_channels, 16, kernel_size=5, stride=2, padding=2)
        self.act1_lr = nn.LeakyReLU(0.2)
        
        self.conv2_lr = nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2)
        self.act2_lr = nn.LeakyReLU(0.2)
        
        self.conv3_lr = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.act3_lr = nn.LeakyReLU(0.2)
        
        self.conv4_lr = nn.Conv2d(64, 256, kernel_size=5, stride=2, padding=2)
        self.act4_lr = nn.LeakyReLU(0.2)
        
        self.conv3_hr = nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2)
        self.act3_hr = nn.LeakyReLU(0.2)
        
        self.conv4_hr = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.act4_hr = nn.LeakyReLU(0.2)
        
        self.conv5_lr = nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2)
        self.act5_lr = nn.LeakyReLU(0.2)
        
        self.hr_convs = nn.ModuleList([
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        ])
        
        d_model = 256
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads=8) for _ in range(3)
        ])
        
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads=8) for _ in range(3)
        ])
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, padding=0),
            nn.LeakyReLU(0.2)
        )
        
        self.feature_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, feature_dim),
        )
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, return_features=True):
        img_lr = self.gaussian_filter(x)
        img_hr = x - img_lr
        
        x_lr = self.conv1_lr(img_lr)
        x_lr = self.act1_lr(x_lr)
        
        x_lr = self.conv2_lr(x_lr)
        x_lr = self.act2_lr(x_lr)
        x_128 = x_lr
        
        x_64_lr = self.conv3_lr(x_128)
        x_64_lr = self.act3_lr(x_64_lr)
        
        x_32_lr = self.conv4_lr(x_64_lr)
        x_32_lr = self.act4_lr(x_32_lr)
        
        x_64_hr = self.conv3_hr(x_128)
        x_64_hr = self.act3_hr(x_64_hr)
        
        x_32_hr = self.conv4_hr(x_64_hr)
        x_32_hr = self.act4_hr(x_32_hr)
        
        x_lr_final = self.conv5_lr(x_32_hr)
        x_lr_final = self.act5_lr(x_lr_final)
        
        B, C, H, W = x_lr_final.shape
        memory = x_lr_final.flatten(2).transpose(1, 2)
        
        for encoder in self.encoder_layers:
            memory = encoder(memory)
        
        img_hr_patch = F.pixel_unshuffle(img_hr, 16)
        x_hr = img_hr_patch
        
        for conv in self.hr_convs:
            x_hr = conv(x_hr)
            x_hr = F.leaky_relu(x_hr, 0.2)
        
        B, C, H, W = x_hr.shape
        x_hr_seq = x_hr.flatten(2).transpose(1, 2)
        
        for decoder in self.decoder_layers:
            x_hr_seq = decoder(x_hr_seq, memory)
        
        x_hr_decoded = x_hr_seq.transpose(1, 2).reshape(B, C, H, W)
        
        fused_features = x_hr_decoded + x_32_lr
        fused_features = self.fusion_conv(fused_features)

        if return_features:
            features = self.feature_head(fused_features)
            features = F.normalize(features, dim=-1)
            return features
        
        return fused_features