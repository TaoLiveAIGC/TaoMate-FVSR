from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn


def _linear_fp32(layer: nn.Linear, inputs: torch.Tensor) -> torch.Tensor:
    """Run one Linear in FP32 without replacing or detaching its parameters."""

    bias = None if layer.bias is None else layer.bias.float()
    return F.linear(inputs.float(), layer.weight.float(), bias)


def _sequential_fp32(module: nn.Sequential, inputs: torch.Tensor) -> torch.Tensor:
    """Run a Wan conditioning Sequential with FP32 Linear kernels."""

    output = inputs.float()
    for layer in module:
        if isinstance(layer, nn.Linear):
            output = _linear_fp32(layer, output)
            continue
        if any(True for _ in layer.parameters(recurse=True)):
            raise TypeError(
                "FP32 Wan conditioning MLP only supports parameter-free non-linear "
                f"layers, got {type(layer).__name__}"
            )
        output = layer(output)
    return output


def fp32_wan_text_embedding(
    dit,
    raw_context: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Evaluate Wan's text-conditioning MLP in FP32, preserving gradients."""

    if not isinstance(dit.text_embedding, nn.Sequential):
        raise TypeError("dit.text_embedding must be torch.nn.Sequential")
    if not raw_context.is_floating_point():
        raise TypeError("raw_context must use a floating-point dtype")
    if output_dtype is None:
        output_dtype = raw_context.dtype

    with torch.autocast(
        device_type=raw_context.device.type,
        enabled=False,
    ):
        embedded_context = _sequential_fp32(
            dit.text_embedding,
            raw_context,
        )
    return embedded_context.to(dtype=output_dtype)


def fp32_wan_timestep_mlp(
    dit,
    embedded_timestep: torch.Tensor,
    *,
    time_embedding_shape: Optional[tuple[int, ...]] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate Wan's timestep embedding and projection MLPs in FP32.

    PyTorch 2.5.1 BF16 autocast can select a failing batch-1 GEMM kernel for
    Wan2.2-5B's 3072-by-3072 timestep layer on H20.  Disabling autocast alone
    is insufficient when a checkpoint was loaded in BF16, because FP32 inputs
    and BF16 weights cannot be multiplied directly.  Functional Linear calls
    over differentiable FP32 parameter views avoid that kernel while preserving
    gradients to the original parameters.

    The two results are cast back only at the helper boundary.  This keeps the
    existing pipeline dtype contract and prevents FP32 timestep tensors from
    promoting every downstream DiT modulation.
    """

    if not isinstance(dit.time_embedding, nn.Sequential):
        raise TypeError("dit.time_embedding must be torch.nn.Sequential")
    if not isinstance(dit.time_projection, nn.Sequential):
        raise TypeError("dit.time_projection must be torch.nn.Sequential")
    if not embedded_timestep.is_floating_point():
        raise TypeError("embedded_timestep must use a floating-point dtype")

    if output_dtype is None:
        output_dtype = embedded_timestep.dtype

    with torch.autocast(
        device_type=embedded_timestep.device.type,
        enabled=False,
    ):
        timestep_embedding = _sequential_fp32(
            dit.time_embedding,
            embedded_timestep,
        )
        if time_embedding_shape is not None:
            timestep_embedding = timestep_embedding.reshape(time_embedding_shape)
        timestep_modulation = _sequential_fp32(
            dit.time_projection,
            timestep_embedding,
        )

    return (
        timestep_embedding.to(dtype=output_dtype),
        timestep_modulation.to(dtype=output_dtype),
    )


def fp32_wan_timestep_projection(
    dit,
    timestep_embedding: torch.Tensor,
    *,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Re-project an existing Wan timestep embedding with an FP32 kernel.

    Unified sequence parallelism partitions ``t`` and projects it again after
    the common timestep builder, so that call site must not fall back to BF16
    autocast.
    """

    if not isinstance(dit.time_projection, nn.Sequential):
        raise TypeError("dit.time_projection must be torch.nn.Sequential")
    if not timestep_embedding.is_floating_point():
        raise TypeError("timestep_embedding must use a floating-point dtype")
    if output_dtype is None:
        output_dtype = timestep_embedding.dtype

    with torch.autocast(
        device_type=timestep_embedding.device.type,
        enabled=False,
    ):
        timestep_modulation = _sequential_fp32(
            dit.time_projection,
            timestep_embedding,
        )
    return timestep_modulation.to(dtype=output_dtype)
