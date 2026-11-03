import torch
from .timestep import fp32_wan_text_embedding, fp32_wan_timestep_mlp
from .models.wan_video_dit import sinusoidal_embedding_1d

def build_native_context(
    dit,
    batch_size: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    raw_context = torch.zeros(
        batch_size,
        1,
        dit.text_dim,
        dtype=dtype,
        device=device,
    )
    return fp32_wan_text_embedding(
        dit,
        raw_context,
        output_dtype=dtype,
    )

def build_wan_timestep_modulation(
    dit,
    timestep: torch.Tensor,
    *,
    frames: int,
    tokens_per_frame: int,
    contains_reference: bool,
    clean_prefix_frames: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    timestep = timestep.reshape(-1)
    clean_prefix_frames = int(clean_prefix_frames)
    if contains_reference:
        clean_prefix_frames = max(clean_prefix_frames, 1)
    if clean_prefix_frames:
        if frames < 1:
            raise ValueError("frames must be positive")
        if tokens_per_frame < 1:
            raise ValueError("tokens_per_frame must be positive")
        if not 0 <= clean_prefix_frames <= frames:
            raise ValueError(
                "clean_prefix_frames cannot exceed the frame count"
            )

        batch_size = timestep.shape[0]
        clean_timesteps = torch.zeros(
            batch_size,
            clean_prefix_frames,
            tokens_per_frame,
            dtype=timestep.dtype,
            device=timestep.device,
        )
        normal_timesteps = timestep[:, None, None].expand(
            batch_size,
            frames - clean_prefix_frames,
            tokens_per_frame,
        )
        token_timesteps = torch.cat(
            [clean_timesteps, normal_timesteps], dim=1
        ).reshape(batch_size, frames * tokens_per_frame)
        embedded = sinusoidal_embedding_1d(
            dit.freq_dim, token_timesteps.flatten()
        )
        t, t_mod = fp32_wan_timestep_mlp(
            dit,
            embedded,
            time_embedding_shape=(
                batch_size,
                frames * tokens_per_frame,
                dit.dim,
            ),
            output_dtype=timestep.dtype,
        )
        return t, t_mod.unflatten(2, (6, dit.dim))

    t, t_mod = fp32_wan_timestep_mlp(
        dit,
        sinusoidal_embedding_1d(dit.freq_dim, timestep),
        output_dtype=timestep.dtype,
    )
    return t, t_mod.unflatten(1, (6, dit.dim))
