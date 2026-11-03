"""Audio-conditioned Wan video inference with checkpoint-compatible tensor names.

Derived from DiffSynth-Studio (Apache-2.0); see THIRD_PARTY_NOTICES.md.
Only dense bidirectional attention is supported in this release.
"""
import math
from typing import Tuple, Optional
import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange
from .audio_conditioning import (AudioProjModule, AUDIO_FEATURE_DIM, AUDIO_FRAMES,
    AUDIO_INTERMEDIATE_DIM, AUDIO_OUTPUT_DIM, AUDIO_WINDOW,
    prepare_audio_features)

def flash_attention(q, k, v, num_heads, compatibility_mode=False, attention_mask=None):
    q, k, v = [rearrange(x, 'b s (n d) -> b n s d', n=num_heads) for x in (q, k, v)]
    x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
    return rearrange(x, 'b n s d -> b s (n d)')


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)

def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)

def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis

def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis

def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device.type == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight

class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads

    def forward(self, q, k, v, attention_mask=None):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads,
                            attention_mask=attention_mask)
        return x

class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)
        # Causal/Sparse 模式的缓存状态
        self.local_attn_mask = None
        self.local_attn_mask_h = None
        self.local_attn_mask_w = None
        self._local_range = None
        # VSA gate 参数: 不在 __init__ 中创建 (保持现有 checkpoint 的严格加载接口)
        # 通过 inject_vsa_gates() 在模型加载后动态注入
        # self.vsa_gate_coarse = None
        # self.vsa_gate_fine = None

    def forward(self, x, freqs, f=None, h=None, w=None,
                local_num=None, topk=None, kv_len=None,
                is_stream=False, pre_cache_k=None, pre_cache_v=None,
                local_range=9, enable_causal=False, enable_sparse=False,
                enable_vsa=False, num_frame_per_block=1,
                kv_cache=None, is_native_5b=False,
                contains_reference=False, condition_tokens=None):
        B, L, D = x.shape
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        condition_k = condition_v = None
        if condition_tokens is not None:
            if condition_tokens.shape != x.shape:
                raise ValueError(
                    "DTI LR token shape mismatch: "
                    f"x={tuple(x.shape)}, condition={tuple(condition_tokens.shape)}"
                )
            condition_k = self.norm_k(self.k(condition_tokens))
            condition_v = self.v(condition_tokens)
            condition_k = rope_apply(condition_k, freqs, self.num_heads)

        # --- 回退: 无 f/h/w 或所有开关都关闭 → 原始全注意力 ---
        if f is None or h is None or w is None or (
                not enable_causal and not enable_sparse and not enable_vsa):
            if condition_k is not None:
                k = torch.cat([k, condition_k], dim=1)
                v = torch.cat([v, condition_v], dim=1)
            x = self.attn(q, k, v)
            return self.o(x)

        raise ValueError("This inference release uses dense bidirectional attention only")


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)

        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)

class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

        # ★ 新增：参考图交叉注意力（默认为 None，按需注入）
        self.ref_cross_attn = None
        self.norm_ref = None
        self.reference_cachekv_attn = None
        self.evidence_adapter = None
        # Audio modules are attached only for an audio-enabled checkpoint.
        # Keeping them absent by default preserves strict loading of legacy Wan
        # and VSR checkpoints.
        self.norm4 = None
        self.audio_attn = None

    def forward(self, x, context, t_mod, freqs, ref_context=None,
                f=None, h=None, w=None, local_num=None, topk=None,
                kv_len=None, is_stream=False, pre_cache_k=None, pre_cache_v=None,
                local_range=9, enable_causal=False, enable_sparse=False,
                enable_vsa=False, num_frame_per_block=1,
                kv_cache=None, is_native_5b=False,
                contains_reference=False, evidence_cond=None,
                reference_kv=None, condition_tokens=None,
                reference_spatial_shape=None, audio_embeds=None):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        condition_input = None
        if condition_tokens is not None:
            condition_input = modulate(
                self.norm1(condition_tokens), shift_msa, scale_msa
            )

        # Self-Attention: 支持 causal/sparse/vsa 模式
        self_attn_result = self.self_attn(
            input_x, freqs, f=f, h=h, w=w,
            local_num=local_num, topk=topk, kv_len=kv_len,
            is_stream=is_stream, pre_cache_k=pre_cache_k, pre_cache_v=pre_cache_v,
            local_range=local_range, enable_causal=enable_causal, enable_sparse=enable_sparse,
            enable_vsa=enable_vsa, num_frame_per_block=num_frame_per_block,
            kv_cache=kv_cache, is_native_5b=is_native_5b,
            contains_reference=contains_reference,
            condition_tokens=condition_input)
        if is_stream and f is not None:
            if is_native_5b:
                self_attn_out, updated_cache = self_attn_result
                cache_k = cache_v = None
            else:
                self_attn_out, cache_k, cache_v = self_attn_result
                updated_cache = None
        else:
            self_attn_out = self_attn_result
            cache_k = cache_v = None
            updated_cache = None

        x = self.gate(x, gate_msa, self_attn_out)

        # V7 Zero-Conv 注入 (self-attn 之后, cross-attn 之前)
        v7_zc = getattr(self, 'v7_zero_conv', None)
        if v7_zc is not None and ref_context is not None:
            x = x + v7_zc(ref_context)

        # Evidence Adapter 注入 (regional policy, zero-init γ)
        ev_adapter = getattr(self, 'evidence_adapter', None)
        if ev_adapter is not None and evidence_cond is not None:
            x = x + ev_adapter(x, evidence_cond)

        # Cross-Attention (V7 模式下 cross_attn 已删除, 跳过)
        if self.cross_attn is not None:
            x = x + self.cross_attn(self.norm3(x), context)

        # Audio cross-attention.  The HuBERT projector produces one
        # eight-token window for each of the 20 VAE38 frames.  Attention is
        # therefore evaluated independently for every video frame, exactly
        # after text cross-attention and before the FFN.
        if audio_embeds is not None:
            if self.audio_attn is None or self.norm4 is None:
                raise RuntimeError(
                    "audio_embeds were provided before audio modules "
                    "were attached"
                )
            if audio_embeds.ndim != 4:
                raise ValueError(
                    "projected audio must have shape [B,F,8,D], got "
                    f"{tuple(audio_embeds.shape)}"
                )
            batch_size, audio_frames, audio_window, audio_dim = (
                audio_embeds.shape
            )
            if (
                audio_frames != AUDIO_FRAMES
                or audio_window != AUDIO_WINDOW
            ):
                raise ValueError(
                    "projected audio must have shape "
                    f"[B,{AUDIO_FRAMES},{AUDIO_WINDOW},D], got "
                    f"{tuple(audio_embeds.shape)}"
                )
            if batch_size != x.shape[0] or audio_dim != self.dim:
                raise ValueError(
                    "projected audio does not match the video tokens: "
                    f"video={tuple(x.shape)}, audio={tuple(audio_embeds.shape)}"
                )
            if f is not None and f != audio_frames:
                raise ValueError(
                    "video/audio frame mismatch: "
                    f"video_frames={f}, audio_frames={audio_frames}"
                )
            if x.shape[1] % audio_frames != 0:
                raise ValueError(
                    "video token count must be divisible by audio "
                    f"frames: tokens={x.shape[1]}, frames={audio_frames}"
                )
            spatial_tokens = x.shape[1] // audio_frames
            frame_tokens = rearrange(
                x,
                "b (f s) d -> (b f) s d",
                f=audio_frames,
                s=spatial_tokens,
            )
            frame_audio = rearrange(
                audio_embeds, "b f p d -> (b f) p d"
            )
            frame_tokens = frame_tokens + self.audio_attn(
                self.norm4(frame_tokens), frame_audio
            )
            x = rearrange(
                frame_tokens,
                "(b f) s d -> b (f s) d",
                b=batch_size,
                f=audio_frames,
            )

        # Full-face static reference K/V. This cache is separate from the
        # causal video cache and is therefore never truncated by kv_len.
        # NOTE: do not reuse cache_k/cache_v here — those names carry the
        # streaming self-attention KV cache returned to the caller.
        if self.reference_cachekv_attn is not None and reference_kv is not None:
            if len(reference_kv) == 2:
                ref_cache_k, ref_cache_v = reference_kv
                reference_scale = None
            else:
                ref_cache_k, ref_cache_v, reference_scale = reference_kv
            reference_residual = self.reference_cachekv_attn(
                x,
                (ref_cache_k, ref_cache_v),
                spatial_shape=reference_spatial_shape,
            )
            if reference_scale is None:
                x = x + reference_residual
            else:
                x = x + reference_residual * reference_scale.to(
                    device=reference_residual.device,
                    dtype=reference_residual.dtype,
                )

        # 参考图交叉注意力 (V1/V2 模式)
        if self.ref_cross_attn is not None and ref_context is not None:
            input_ref_x = modulate(self.norm_ref(x), shift_msa, scale_msa)
            x = x + self.ref_cross_attn(input_ref_x, ref_context)

        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        if is_stream and f is not None:
            if is_native_5b:
                return x, updated_cache
            return x, cache_k, cache_v
        return x

def inject_audio_conditioning(dit, *, device=None, dtype=None):
    """Attach the checkpoint-compatible audio branch to a Wan2.2 5B DiT."""

    if getattr(dit, "audio_embedding", None) is not None:
        return dit
    if (
        dit.dim != AUDIO_OUTPUT_DIM
        or dit.num_heads != 24
        or len(dit.blocks) != 30
    ):
        raise ValueError(
            "Audio conditioning requires Wan2.2 5B "
            "(dim=3072, heads=24, blocks=30), got "
            f"dim={dit.dim}, heads={dit.num_heads}, blocks={len(dit.blocks)}"
        )

    reference_weight = dit.patch_embedding.weight
    if device is None:
        device = reference_weight.device
    if dtype is None:
        dtype = reference_weight.dtype
    previous_default_dtype = torch.get_default_dtype()
    change_default_dtype = dtype != previous_default_dtype
    try:
        if change_default_dtype:
            torch.set_default_dtype(dtype)
        # ``assign=True`` checkpoint loading instantiates directly on meta and
        # then replaces every parameter/buffer from the merged state dict.
        # Explicit injection constructs directly on the backbone device/dtype,
        # avoiding a second float32 multi-GB allocation.
        with torch.device(device):
            dit.audio_embedding = AudioProjModule()
            for block in dit.blocks:
                block.norm4 = nn.LayerNorm(dit.dim, eps=dit.eps)
                block.audio_attn = CrossAttention(
                    dit.dim, dit.num_heads, dit.eps
                )
    finally:
        if change_default_dtype:
            torch.set_default_dtype(previous_default_dtype)
    dit.audio_conditioning_enabled = True
    return dit

class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)

class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x

def drop_text_cross_attn(dit):
    """删除每个 DiTBlock 的 text cross_attn + norm3, 释放显存。
    独立于 V7/区域策略, 由 --drop_text_cross_attn 控制。返回释放的参数量。"""
    n_deleted = 0
    for block in dit.blocks:
        if getattr(block, "cross_attn", None) is not None:
            n_deleted += sum(p.numel() for p in block.cross_attn.parameters())
            del block.cross_attn
            block.cross_attn = None
        if getattr(block, "norm3", None) is not None:
            n_deleted += sum(p.numel() for p in block.norm3.parameters())
            del block.norm3
            block.norm3 = None
    print(f"[drop_text_cross_attn] freed {n_deleted / 1e6:.2f}M params")
    return n_deleted

def inject_spark_sparse_ch_v2_multiref(
    dit,
    device="cpu",
    dtype=torch.bfloat16,
):
    """Attach v2's zero-init parallel projection without changing v1.

    The checkpoint parameter deliberately remains ``spark_ref_proj.weight`` so
    the architecture is the same zero-init channel-concat half used by v1.  A
    distinct marker selects the multi-reference canvas contract.
    """
    if getattr(dit, "spark_sparse_ch_v2_multiref", False):
        raise ValueError("spark_sparse_ch_v2_multiref is already attached")
    if getattr(dit, "spark_sparse_ch_v1", False):
        raise ValueError("spark_sparse_ch_v2_multiref and v1 are exclusive")
    projection = nn.Conv3d(
        dit.in_dim,
        dit.dim,
        kernel_size=dit.patch_size,
        stride=dit.patch_size,
        bias=False,
    ).to(device=device, dtype=dtype)
    nn.init.zeros_(projection.weight)
    dit.spark_ref_proj = projection
    dit.spark_sparse_ch_v2_multiref = True
    num_params = sum(parameter.numel() for parameter in projection.parameters())
    print(
        "[SparkSparseChV2MultiRef] injected zero-init reference projection: "
        f"Conv3d({dit.in_dim}, {dit.dim}, {dit.patch_size}), "
        f"params={num_params / 1e6:.2f}M; temporal_concat=False",
        flush=True,
    )
    return dit

class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        enable_audio_conditioning: bool = False,
    ):
        super().__init__()
        # ... 全部原有 __init__ 代码不变 ...
        self.dim = dim
        self.in_dim = in_dim
        self.ffn_dim = ffn_dim
        self.eps = eps
        self.out_dim = out_dim
        self.text_dim = text_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.num_heads = num_heads  # ★ 新增：保存以便注入时使用
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.audio_embedding = None
        self.audio_conditioning_enabled = False
        if enable_audio_conditioning:
            inject_audio_conditioning(self)
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        self.control_adapter = None

        # ★ 新增：参考图编码器（默认 None，由外部注入）
        # self.hr_ref_encoder = None
        # self.lq4x_proj = None
        self.hr_ref_proj = None
        self.lr_proj = None

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Attach audio modules before recursively loading an audio checkpoint.

        Legacy Wan/VSR checkpoints do not contain ``audio_embedding`` and keep
        their original strict-loading contract.  The complete audio-conditioned
        checkpoint contains the 4096-wide projector and activates this branch.
        """

        projector_key = prefix + "audio_embedding.proj_in.weight"
        if projector_key in state_dict and self.audio_embedding is None:
            projector_weight = state_dict[projector_key]
            if tuple(projector_weight.shape) != (
                AUDIO_INTERMEDIATE_DIM,
                AUDIO_FEATURE_DIM,
            ):
                raise ValueError(
                    "Audio initialization requires the complete audio-conditioned "
                    "checkpoint with audio_embedding.proj_in.weight shape "
                    f"({AUDIO_INTERMEDIATE_DIM}, {AUDIO_FEATURE_DIM}); "
                    f"got {tuple(projector_weight.shape)}"
                )
            assign_parameters = local_metadata.get(
                "assign_to_params_buffers", False
            )
            inject_audio_conditioning(
                self,
                device=("meta" if assign_parameters else projector_weight.device),
                dtype=projector_weight.dtype,
            )
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def project_audio(
        self,
        audio_feats,
        *,
        batch_size,
        device,
        dtype,
    ):
        """Project real or all-zero raw HuBERT features for every sample."""

        if self.audio_embedding is None:
            raise RuntimeError("Audio modules are not attached")
        audio_feats = prepare_audio_features(
            audio_feats,
            batch_size=batch_size,
            device=device,
            dtype=dtype,
        )
        return self.audio_embedding(audio_feats)

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2],
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )
