"""Bound high-resolution decoder convolution workspace without splitting time."""
from types import MethodType
import torch
from torch.nn import functional as F
from .models.wan_video_vae import CausalConv3d


def striped_conv(conv, inputs, weight, bias):
    # CausalConv3d has already applied padding and temporal-cache concatenation.
    stride, dilation = conv.stride, conv.dilation
    kernels = weight.shape[-3:]
    depth, height, width = [(s - d * (k - 1) - 1) // t + 1
        for s, d, k, t in zip(inputs.shape[-3:], dilation, kernels, stride)]
    if height < 256 or width < 256:
        return F.conv3d(inputs, weight, bias, stride, conv.padding, dilation, conv.groups)
    output = None
    for start in range(0, height, 64):
        stop = min(start + 64, height)
        left = start * stride[1]
        right = (stop - 1) * stride[1] + dilation[1] * (kernels[1] - 1) + 1
        part = F.conv3d(inputs[..., left:right, :].contiguous(), weight, bias,
                        stride, conv.padding, dilation, conv.groups)
        if output is None:
            output = part.new_empty(inputs.shape[0], weight.shape[0], depth, height, width)
        output[..., start:stop, :] = part
    return output


def limit_decoder_workspace(vae):
    for module in vae.model.decoder.modules():
        if isinstance(module, CausalConv3d):
            module._conv_forward = MethodType(striped_conv, module)
