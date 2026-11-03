"""Dense DiT forward; joint reference/K2 CFG outside raw-HuBERT audio CFG."""
import torch
from einops import rearrange
from .native import build_native_context, build_wan_timestep_modulation


def forward(dit, state, reference, audio, timestep):
    batch, _, frames, height, width = state.shape
    tokens_per_frame = height * width // (dit.patch_size[1] * dit.patch_size[2])
    t, modulation = build_wan_timestep_modulation(dit, timestep.reshape(-1).to(state.dtype),
        frames=frames, tokens_per_frame=tokens_per_frame, contains_reference=False)
    context = build_native_context(dit, batch, state.dtype, state.device)
    x = dit.patchify(state) + dit.spark_ref_proj(reference)
    f, h, w = x.shape[2:]
    x = rearrange(x, 'b c f h w -> b (f h w) c').contiguous()
    freqs = torch.cat([
        dit.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
        dit.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
        dit.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
    ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
    audio_embeds = dit.project_audio(audio, batch_size=batch, device=x.device, dtype=x.dtype)
    for block in dit.blocks:
        x = block(x, context, modulation, freqs, h=h, w=w,
                  local_range=0, audio_embeds=audio_embeds)
    return dit.unpatchify(dit.head(x, t), (f, h, w))


def guided_forward(dit, state, reference, audio, timestep, audio_cfg, ref_cfg):
    # All reference slots, including K2 in slot 5, are zero in the negative
    # branch. K2 is not shared across branches or scaled before projection.
    def audio_branch(ref):
        positive = forward(dit, state, ref, audio, timestep)
        if audio_cfg == 1:
            return positive
        negative = forward(dit, state, ref, torch.zeros_like(audio), timestep)
        return (negative.float() + audio_cfg * (positive.float() - negative.float())).to(positive.dtype)

    positive = audio_branch(reference)
    if ref_cfg == 1:
        return positive
    negative = audio_branch(torch.zeros_like(reference))
    return (negative.float() + ref_cfg * (positive.float() - negative.float())).to(positive.dtype)


def schedule(steps, shift):
    # Same 1000-entry shifted Wan schedule and rounded boundaries as evaluation.
    base = torch.linspace(1.0, 0.0, 1001)[:-1]
    base = shift * base / (1 + (shift - 1) * base)
    indices = torch.linspace(0, 1000, steps + 1).round().long()[:-1]
    sigmas = base[indices]
    following = torch.cat([sigmas[1:], sigmas.new_zeros(1)])
    return sigmas * 1000, sigmas, following
