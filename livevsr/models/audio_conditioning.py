"""Temporal HuBERT feature projection for audio-conditioned video inference.

Parameter paths such as ``attn.to_*`` and ``ff.net.0.proj`` match the released
checkpoint. Component attribution is documented in THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


AUDIO_FRAMES = 20
AUDIO_WINDOW = 8
AUDIO_FEATURE_DIM = 4096
AUDIO_INTERMEDIATE_DIM = 1536
AUDIO_OUTPUT_DIM = 3072


class SinusoidalPositionalEmbedding(nn.Module):
    """Add the fixed positional table used by the audio projector."""

    def __init__(self, embed_dim: int, max_seq_length: int = 100):
        super().__init__()
        # Build the fixed table in float32 before casting the complete model,
        # including when audio modules are constructed directly as bfloat16.
        position = torch.arange(
            max_seq_length, dtype=torch.float32
        ).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2, dtype=torch.float32)
            * (-math.log(10000.0) / embed_dim)
        )
        pe = torch.zeros(
            1, max_seq_length, embed_dim, dtype=torch.float32
        )
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        pe = pe.to(dtype=torch.get_default_dtype())
        self.register_buffer("pe", pe)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states + self.pe[:, : hidden_states.shape[1]]


class AudioAttention(nn.Module):
    """Diffusers ``Attention`` equivalent with checkpoint-compatible names."""

    def __init__(self, query_dim: int, heads: int, dim_head: int):
        super().__init__()
        inner_dim = heads * dim_head
        self.heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_out = nn.ModuleList(
            [nn.Linear(inner_dim, query_dim, bias=True), nn.Dropout(0.0)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        query = rearrange(
            self.to_q(hidden_states), "b s (h d) -> b h s d", h=self.heads
        )
        key = rearrange(
            self.to_k(hidden_states), "b s (h d) -> b h s d", h=self.heads
        )
        value = rearrange(
            self.to_v(hidden_states), "b s (h d) -> b h s d", h=self.heads
        )
        hidden_states = F.scaled_dot_product_attention(query, key, value)
        hidden_states = rearrange(hidden_states, "b h s d -> b s (h d)")
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class GEGLU(nn.Module):
    """Checkpoint-compatible GEGLU used by diffusers ``FeedForward``."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states, gate = self.proj(hidden_states).chunk(2, dim=-1)
        return hidden_states * F.gelu(gate)


class AudioFeedForward(nn.Module):
    """Diffusers ``FeedForward(dim=...)`` equivalent for the audio checkpoint."""

    def __init__(self, dim: int, mult: int = 4):
        super().__init__()
        inner_dim = dim * mult
        self.net = nn.ModuleList(
            [GEGLU(dim, inner_dim), nn.Dropout(0.0), nn.Linear(inner_dim, dim)]
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class AudioProjLayer(nn.Module):
    """One temporal-window audio projection block."""

    def __init__(
        self,
        dim: int = AUDIO_INTERMEDIATE_DIM,
        use_attn: bool = True,
        attn_head_dim: int = 64,
        attn_num_heads: int = 24,
    ):
        super().__init__()
        if dim != attn_head_dim * attn_num_heads:
            raise ValueError("dim must equal attn_head_dim * attn_num_heads")
        self.use_attn = use_attn
        self.norm1 = nn.LayerNorm(dim) if use_attn else None
        self.attn = (
            AudioAttention(dim, heads=attn_num_heads, dim_head=attn_head_dim)
            if use_attn
            else None
        )
        self.norm2 = nn.LayerNorm(dim)
        self.ff = AudioFeedForward(dim)
        self.pos_embed = SinusoidalPositionalEmbedding(
            embed_dim=dim, max_seq_length=100
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, frames, window, channels = hidden_states.shape
        hidden_states = rearrange(
            hidden_states, "b f w c -> (b f) w c"
        )
        if self.use_attn:
            hidden_states = hidden_states + self.attn(
                self.pos_embed(self.norm1(hidden_states))
            )
        hidden_states = hidden_states + self.ff(self.norm2(hidden_states))
        return rearrange(
            hidden_states,
            "(b f) w c -> b f w c",
            b=batch_size,
            f=frames,
            w=window,
            c=channels,
        )


class AudioProjModule(nn.Module):
    """Project raw HuBERT windows ``[B,20,8,4096]`` to DiT width 3072."""

    def __init__(
        self,
        audio_feat_layers: tuple[int, ...] = (4,),
        audio_feat_channels: tuple[int, ...] = (1024,),
        intermediate_dim: int = AUDIO_INTERMEDIATE_DIM,
        output_dim: int = AUDIO_OUTPUT_DIM,
        num_layers: int = 4,
        use_attn: bool = True,
        attn_head_dim: int = 64,
        attn_num_heads: int = 24,
        expected_frames: int = AUDIO_FRAMES,
        expected_window: int = AUDIO_WINDOW,
    ):
        super().__init__()
        if len(audio_feat_layers) != len(audio_feat_channels):
            raise ValueError(
                "audio_feat_layers and audio_feat_channels must have equal length"
            )
        if intermediate_dim != attn_head_dim * attn_num_heads:
            raise ValueError(
                "intermediate_dim must equal attn_head_dim * attn_num_heads"
            )
        self.audio_feat_layers = tuple(audio_feat_layers)
        self.audio_feat_channels = tuple(audio_feat_channels)
        self.input_dim = sum(
            layers * channels
            for layers, channels in zip(
                self.audio_feat_layers, self.audio_feat_channels
            )
        )
        self.intermediate_dim = intermediate_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.expected_frames = expected_frames
        self.expected_window = expected_window

        self.proj_in = nn.Linear(self.input_dim, intermediate_dim)
        self.layers = nn.ModuleList(
            [
                AudioProjLayer(
                    dim=intermediate_dim,
                    use_attn=use_attn,
                    attn_head_dim=attn_head_dim,
                    attn_num_heads=attn_num_heads,
                )
                for _ in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(intermediate_dim, output_dim)

    def forward(self, audio_features: torch.Tensor) -> torch.Tensor:
        expected = (
            self.expected_frames,
            self.expected_window,
            self.input_dim,
        )
        if audio_features.ndim != 4 or tuple(audio_features.shape[1:]) != expected:
            raise ValueError(
                "Audio features must have shape "
                f"[B,{expected[0]},{expected[1]},{expected[2]}], got "
                f"{tuple(audio_features.shape)}"
            )
        hidden_states = self.proj_in(audio_features)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.proj_out(hidden_states)


def prepare_audio_features(
    audio_features: torch.Tensor | None,
    *,
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return the HuBERT tensor or the raw all-zero audio condition."""

    expected_shape = (
        batch_size,
        AUDIO_FRAMES,
        AUDIO_WINDOW,
        AUDIO_FEATURE_DIM,
    )
    if audio_features is None:
        return torch.zeros(expected_shape, device=device, dtype=dtype)
    if tuple(audio_features.shape) != expected_shape:
        raise ValueError(
            f"Audio features must have shape {expected_shape}, got "
            f"{tuple(audio_features.shape)}"
        )
    return audio_features.to(device=device, dtype=dtype)


__all__ = [
    "AudioProjLayer",
    "AudioProjModule",
    "AUDIO_FEATURE_DIM",
    "AUDIO_FRAMES",
    "AUDIO_INTERMEDIATE_DIM",
    "AUDIO_OUTPUT_DIM",
    "AUDIO_WINDOW",
    "prepare_audio_features",
]
