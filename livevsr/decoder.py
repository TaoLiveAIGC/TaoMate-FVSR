import torch
from .models.wan_video_vae import WanVideoVAE, count_conv3d, unpatchify

class StreamingVAEDecoder:
    """
    流式时域 VAE 解码器。

    直接操控 VideoVAE_ 的 Decoder3d, 在多次调用间维持 feat_cache (CausalConv3d 的
    时域缓存), 从而实现真正的流式解码: 每次仅送入 1 个 latent 帧即可获得对应的
    pixel 帧, 无需一次性将全部 latent 载入 GPU。

    用法:
        dec = StreamingVAEDecoder(vae, device, dtype)
        dec.reset()
        pix0 = dec.decode_frame(z[:, :, 0:1])    # 首帧 → 1 pixel frame  (B,3,1,H,W)
        pix1 = dec.decode_frame(z[:, :, 1:2])    # → 4 pixel frames (B,3,4,H,W)
        pix2 = dec.decode_frame(z[:, :, 2:3])    # → 4 pixel frames (B,3,4,H,W)
        ...
    """

    def __init__(self, vae: WanVideoVAE, device='cuda', dtype=torch.bfloat16):
        self.vae = vae
        self.model = vae.model          # VideoVAE_ instance
        self.scale = vae.scale           # [mean, 1/std]
        self.device = device
        self.dtype = dtype
        self._feat_map = None
        self._is_first_chunk = True

    def reset(self):
        """在解码新视频前调用, 清空所有时域缓存。"""
        conv_num = count_conv3d(self.model.decoder)
        self._feat_map = [None] * conv_num
        self._is_first_chunk = True

    def decode_frame(self, latent_frame: torch.Tensor) -> torch.Tensor:
        """
        解码 1 个 latent 帧。

        Args:
            latent_frame: (B, z_dim, 1, H_lat, W_lat), 已经是去噪后的 latent

        Returns:
            pixel_frames: (B, 3, T_out, H_pix, W_pix), clamp 到 [-1, 1]
                首帧 T_out=1, 后续 T_out=4
        """
        latent_frame = latent_frame.to(device=self.device, dtype=self.dtype)

        # Un-scale (与 VideoVAE_.decode 一致)
        scale = self.scale
        if isinstance(scale[0], torch.Tensor):
            s = [s_.to(dtype=latent_frame.dtype, device=latent_frame.device) for s_ in scale]
            z = latent_frame / s[1].view(1, self.model.z_dim, 1, 1, 1) + s[0].view(
                1, self.model.z_dim, 1, 1, 1)
        else:
            s = scale.to(dtype=latent_frame.dtype, device=latent_frame.device)
            z = latent_frame / s[1] + s[0]

        # conv2: CausalConv3d(kernel=1), 逐位置独立
        x = self.model.conv2(z)

        # decoder: 逐帧流式, feat_cache 跨调用维持
        feat_idx = [0]
        out, self._feat_map, _ = self.model.decoder(
            x,
            feat_cache=self._feat_map,
            feat_idx=feat_idx,
            first_chunk=self._is_first_chunk,
        )
        self._is_first_chunk = False
        out = unpatchify(out, self.vae.pixel_patch_size)
        expected_h = latent_frame.shape[3] * self.vae.upsampling_factor
        expected_w = latent_frame.shape[4] * self.vae.upsampling_factor
        if out.shape[1] != 3 or out.shape[3:] != (expected_h, expected_w):
            raise ValueError(
                f"Streaming VAE output mismatch: got {tuple(out.shape)}, "
                f"expected channels=3, size=({expected_h}, {expected_w})"
            )
        return out.clamp_(-1, 1)
