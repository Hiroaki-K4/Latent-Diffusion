import math

import torch
import torch.nn as nn


# Sinusoidal Time Embedding
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        half_dim = self.embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb


# Residual Block
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, time_emb_dim=None):
        super().__init__()
        self.time_mlp = (
            nn.Sequential(nn.Linear(time_emb_dim, out_channels))
            if time_emb_dim
            else None
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
        )
        self.residual_conv = (
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x, t):
        h = self.block1(x)
        if self.time_mlp:
            t_emb = self.time_mlp(t)[:, :, None, None]
            h = h + t_emb
        h = self.block2(h)
        return h + self.residual_conv(x)


# Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_norm = self.norm(x)
        q, k, v = self.qkv(x_norm).chunk(3, dim=1)

        q = q.view(B, C, H * W).permute(0, 2, 1)
        k = k.view(B, C, H * W)
        v = v.view(B, C, H * W).permute(0, 2, 1)

        attn = torch.softmax(q @ k / math.sqrt(C), dim=-1)
        h = attn @ v
        h = h.permute(0, 2, 1).view(B, C, H, W)
        return self.proj(h) + x


# U-Net for Diffusion Model
class DiffusionUNet(nn.Module):
    def __init__(
        self, in_channels, out_channels, time_emb_dim=128, features=[64, 128, 256, 512]
    ):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionalEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.encoder = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(ResBlock(in_channels, feature, time_emb_dim))
            self.downs.append(
                nn.Conv2d(feature, feature, kernel_size=4, stride=2, padding=1)
            )
            in_channels = feature

        # Bottleneck
        self.bottleneck = nn.Sequential(
            ResBlock(features[-1], features[-1] * 2, time_emb_dim),
            AttentionBlock(features[-1] * 2),
            ResBlock(features[-1] * 2, features[-1] * 2, time_emb_dim),
        )

        # Decoder
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2, feature, kernel_size=4, stride=2, padding=1
                )
            )
            self.decoder.append(ResBlock(feature * 2, feature, time_emb_dim))

        self.final_block = nn.Sequential(
            ResBlock(features[0], features[0]),
            nn.Conv2d(features[0], out_channels, kernel_size=1),
        )

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        skip_connections = []

        # Encoder
        for block, down in zip(self.encoder, self.downs):
            x = block(x, t_emb)
            skip_connections.append(x)
            x = down(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for up, block, skip in zip(self.ups, self.decoder, skip_connections):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = block(x, t_emb)

        return self.final_block(x)


# Test the Diffusion U-Net
if __name__ == "__main__":
    model = DiffusionUNet(in_channels=3, out_channels=3)
    x = torch.randn(1, 3, 64, 64)  # Example batch of images
    t = torch.tensor([10])  # Example timestep
    out = model(x, t)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
