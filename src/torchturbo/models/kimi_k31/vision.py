"""MoonViT-V2 image encoder and spatial patch projector."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig


class KimiK31PatchPositions(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(
            config.init_pos_emb_height, config.init_pos_emb_width, config.vt_hidden_size,
        ))
        self.interpolation_mode = config.pos_emb_interpolation_mode
        nn.init.normal_(self.weight)

    def forward(self, grids: tuple[tuple[int, int], ...]) -> torch.Tensor:
        weight = self.weight
        positions = []
        for height, width in grids:
            if (height, width) == weight.shape[:2]:
                grid = weight
            else:
                grid = F.interpolate(
                    weight.permute(2, 0, 1).unsqueeze(0).float(),
                    size=(height, width), mode=self.interpolation_mode,
                )
                grid = grid.squeeze(0).permute(1, 2, 0).to(weight.dtype)
            positions.append(grid.flatten(0, 1))
        return torch.cat(positions)


class KimiK31PatchEmbed(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(
            3, config.vt_hidden_size, kernel_size=config.patch_size, stride=config.patch_size,
            bias=config.patch_embed_proj_bias,
        )
        self.pos_emb = KimiK31PatchPositions(config)

    def forward(
        self, pixels: torch.Tensor, grids: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        patches = self.proj(pixels.to(self.proj.weight.dtype)).flatten(1)
        return patches + self.pos_emb(grids).to(patches.dtype)


class KimiK31VisionMLP(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.fc0 = nn.Linear(config.vt_hidden_size, config.vt_intermediate_size, bias=config.linear_bias)
        self.fc1 = nn.Linear(config.vt_intermediate_size, config.vt_hidden_size, bias=config.linear_bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc1(F.gelu(self.fc0(hidden_states), approximate="tanh"))


class KimiK31VisionBlock(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.num_heads = config.vt_num_attention_heads
        hidden_size = config.vt_hidden_size
        qkv_size = config.qkv_hidden_size
        self.head_dim = qkv_size // self.num_heads
        self.norm0 = nn.RMSNorm(hidden_size)
        self.norm1 = nn.RMSNorm(hidden_size)
        self.wqkv = nn.Linear(hidden_size, 3 * qkv_size, bias=config.attn_bias)
        self.wo = nn.Linear(qkv_size, hidden_size, bias=config.attn_bias)
        self.mlp = KimiK31VisionMLP(config)

    @staticmethod
    def _rotate(tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        even, odd = tensor.float().unflatten(-1, (-1, 2)).unbind(-1)
        return torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2).to(tensor.dtype)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: tuple[int, ...],
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        qkv = self.wqkv(self.norm0(hidden_states)).unflatten(-1, (3, self.num_heads, self.head_dim))
        query, key, value = qkv.unbind(dim=1)
        query = self._rotate(query, cos, sin).transpose(0, 1)
        key = self._rotate(key, cos, sin).transpose(0, 1)
        value = value.transpose(0, 1)
        attention = [
            F.scaled_dot_product_attention(q, k, v)
            for q, k, v in zip(query.split(lengths, dim=1), key.split(lengths, dim=1), value.split(lengths, dim=1))
        ]
        attention = torch.cat(attention, dim=1).transpose(0, 1).flatten(1)
        hidden_states = hidden_states + self.wo(attention)
        return hidden_states + self.mlp(self.norm1(hidden_states))


class KimiK31VisionEncoder(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.head_dim = config.qkv_hidden_size // config.vt_num_attention_heads
        self.blocks = nn.ModuleList(KimiK31VisionBlock(config) for _ in range(config.vt_num_hidden_layers))
        self.final_layernorm = nn.RMSNorm(config.vt_hidden_size)

    def forward(
        self, hidden_states: torch.Tensor, grids: tuple[tuple[int, int], ...],
    ) -> torch.Tensor:
        device = hidden_states.device
        frequency = 10000 ** (-torch.arange(0, self.head_dim, 4, device=device, dtype=torch.float32) / self.head_dim)
        phases = []
        for height, width in grids:
            x = torch.arange(width, device=device, dtype=torch.float32).repeat(height)
            y = torch.arange(height, device=device, dtype=torch.float32).repeat_interleave(width)
            phases.append(torch.stack((x[:, None] * frequency, y[:, None] * frequency), dim=-1).flatten(1))
        phase = torch.cat(phases).unsqueeze(1)
        cos, sin = phase.cos(), phase.sin()
        lengths = tuple(height * width for height, width in grids)
        for block in self.blocks:
            hidden_states = block(hidden_states, lengths, cos, sin)
        return self.final_layernorm(hidden_states)


class KimiK31VisionTower(nn.Module):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__()
        self.patch_embed = KimiK31PatchEmbed(config)
        self.encoder = KimiK31VisionEncoder(config)
        self.merge_kernel = tuple(config.merge_kernel_size)

    def forward(self, pixel_values: torch.Tensor, grid_thws: torch.Tensor) -> torch.Tensor:
        grids = []
        kernel_h, kernel_w = self.merge_kernel
        for frames, height, width in grid_thws.tolist():
            if frames != 1 or height % kernel_h or width % kernel_w:
                raise ValueError("Kimi K3.1 expects image grids divisible by the merge kernel")
            grids.append((height, width))
        embedding = self.patch_embed(pixel_values, tuple(grids))
        features = self.encoder(embedding, tuple(grids))
        merged = []
        lengths = [height * width for height, width in grids]
        for states, (height, width) in zip(features.split(lengths), grids):
            rows, cols = height // kernel_h, width // kernel_w
            grid = states.reshape(rows, kernel_h, cols, kernel_w, -1)
            merged.append(grid.permute(0, 2, 1, 3, 4).reshape(rows * cols, -1))
        return torch.cat(merged)


class KimiK31VisionProjector(nn.Module):
    def __init__(self, config: PretrainedConfig, text_hidden_size: int) -> None:
        super().__init__()
        kh, kw = config.merge_kernel_size
        merged_size = config.mm_hidden_size * kh * kw
        self.proj = nn.Sequential(
            nn.Linear(merged_size, merged_size, bias=False),
            nn.GELU(),
            nn.Linear(merged_size, text_hidden_size, bias=False),
        )
        self.post_norm = nn.RMSNorm(text_hidden_size, eps=config.projector_ln_eps)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.post_norm(self.proj(features))
