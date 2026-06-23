from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, sequence_length: int, embed_dim: int):
        super().__init__()
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        encoding = torch.zeros(sequence_length, embed_dim, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1], :]


class TrafficTransformerEncoder(nn.Module):
    def __init__(
        self,
        state_dim: int,
        sequence_length: int,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.state_dim = state_dim
        self.sequence_length = sequence_length
        self.embed = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.position = SinusoidalPositionEncoding(sequence_length, embed_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        if states.ndim != 3:
            raise ValueError("states must have shape (batch, sequence, state_dim)")
        if states.shape[-1] != self.state_dim:
            raise ValueError(f"expected state_dim={self.state_dim}, got {states.shape[-1]}")
        x = self.embed(states)
        x = self.position(x)
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.output_norm(x)
