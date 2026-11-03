"""Checkpoint-only loading and overlap-consistent multi-step inference."""
import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from safetensors.torch import load_file

from .audio import HubertAudioProcessor
from .decoder import StreamingVAEDecoder
from .denoise import guided_forward, schedule
from .geometry import windows, input_indices
from .media import write_audio
from .models.wan_video_dit import (WanModel, drop_text_cross_attn,
    inject_spark_sparse_ch_v2_multiref, precompute_freqs_cis_3d)
from .models.wan_video_vae import WanVideoVAE38

logger = logging.getLogger(__name__)
MODEL_CONFIG = dict(has_image_input=False, patch_size=(1, 2, 2), in_dim=48,
    dim=3072, ffn_dim=14336, freq_dim=256, text_dim=4096, out_dim=48,
    num_heads=24, num_layers=30, eps=1e-6, seperated_timestep=True,
    require_clip_embedding=False, require_vae_embedding=False,
    fuse_vae_embedding_in_latents=True, enable_audio_conditioning=True)


def window_lr(global_lr, start):
    """Keep global motion, but anchor slots 0/1 to the first window's LR."""
    lr = global_lr[:, :, start:start + 20].clone()
    if start:
        lr[:, :, :2] = global_lr[:, :, :2]
    return lr


class LiveVSR:
    def __init__(self, checkpoint, vae, hubert, *, device='cuda', offload=False):
        self.device = torch.device(device)
        self.dtype = torch.bfloat16
        self.offload = offload
        with torch.device('meta'):
            self.dit = WanModel(**MODEL_CONFIG)
            drop_text_cross_attn(self.dit)
            inject_spark_sparse_ch_v2_multiref(self.dit, device='meta', dtype=self.dtype)
            self.vae = WanVideoVAE38()
        self.dit.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True, assign=True)
        self.dit.freqs = precompute_freqs_cis_3d(self.dit.dim // self.dit.num_heads)
        state = torch.load(vae, map_location='cpu', weights_only=True)
        if 'model_state' in state:
            state = state['model_state']
        if not next(iter(state)).startswith('model.'):
            state = {'model.' + k: v for k, v in state.items()}
        self.vae.load_state_dict(state, strict=True, assign=True)
        del state
        self.dit.eval().requires_grad_(False).to(dtype=self.dtype)
        self.vae.eval().requires_grad_(False).to(dtype=self.dtype)
        self.audio = HubertAudioProcessor(str(hubert), device='cpu' if offload else self.device)
        if not offload:
            self.dit.to(self.device)
            self.vae.to(self.device)
        logger.info('Loaded complete DiT (%d tensors), VAE and HuBERT', len(self.dit.state_dict()))

    def activate(self, name):
        if self.offload:
            for key, model in [('dit', self.dit), ('vae', self.vae), ('audio', self.audio)]:
                model.to(self.device if key == name else 'cpu')
            torch.cuda.empty_cache()

    def encode(self, images, size):
        frames = np.stack([np.asarray(image.resize((size, size), Image.Resampling.BICUBIC)) for image in images])
        tensor = torch.from_numpy(frames).permute(3, 0, 1, 2)[None].to(dtype=self.dtype)
        tensor = tensor * (2.0 / 255) - 1
        return self.vae.encode(tensor, device=self.device, tiled=size > 512,
                               tile_size=(32, 32), tile_stride=(16, 16)).cpu()

    @torch.inference_mode()
    def restore(self, paths, reference, pcm, workspace, *, size, seed=42, steps=10,
                noise=.8, shift=5., audio_cfg=4., ref_cfg=1.7):
        geometry = windows(len(paths))
        if size > 512:
            from .vae_memory import limit_decoder_workspace
            limit_decoder_workspace(self.vae)
        self.activate('vae')
        lr_indices = input_indices(len(paths))
        images = []
        for index in lr_indices:
            with Image.open(paths[index]) as image:
                images.append(image.convert('RGB'))
        global_lr = self.encode(images, size)
        del images
        latent_hw = size // 16
        rng = torch.Generator(device='cpu').manual_seed(seed)
        first_noise = torch.randn((1, 48, 20, latent_hw, latent_hw), generator=rng)
        if global_lr.shape[2] > 20:
            tail = torch.randn((1, 48, global_lr.shape[2] - 20, latent_hw, latent_hw), generator=rng)
            global_noise = torch.cat([first_noise, tail], dim=2)
        else:
            global_noise = first_noise
        ref_image = Image.fromarray(reference)
        identity = self.encode([ref_image], size)
        bootstrap = self.encode([ref_image] * 5, size)
        decoder = StreamingVAEDecoder(self.vae, device=self.device, dtype=self.dtype)
        decoder.reset()
        ts, sigmas, following = schedule(steps, shift)
        output = np.lib.format.open_memmap(Path(workspace) / 'output.npy', mode='w+', dtype=np.uint8,
                                         shape=(len(paths), size, size, 3))
        previous_trace, last_rgb = None, None
        for wi, plan in enumerate(geometry):
            self.activate('vae')
            reference_latents = identity.new_zeros((1, 48, 20, latent_hw, latent_hw))
            reference_latents[:, :, :1] = identity
            reference_latents[:, :, 1:2] = bootstrap[:, :, 1:2]
            if wi:
                reference_latents[:, :, 5:6] = self.encode([Image.fromarray(last_rgb)], size)
            window_wav = Path(workspace) / 'window.wav'
            write_audio(window_wav, pcm[plan['input_indices']].reshape(-1))
            self.activate('audio')
            audio = self.audio.build_condition(audio_path=str(window_wav), audio_feature_path=None,
                audio_valid=True, frame_start=0, output_device=self.device, output_dtype=self.dtype)
            start = plan['latent_start']
            lr = window_lr(global_lr, start).to(self.device, self.dtype)
            epsilon = global_noise[:, :, start:start+20].to(self.device, self.dtype)
            strength = lr.new_tensor(noise)
            state = (1 - strength) * lr + strength * epsilon
            trace = [state[:, :, 16:20].detach().cpu().clone()]
            reference_latents = reference_latents.to(self.device, self.dtype)
            self.activate('dit')
            for si, timestep in enumerate(ts):
                if wi:
                    state[:, :, 2:6] = previous_trace[si].to(self.device, self.dtype)
                velocity = guided_forward(self.dit, state, reference_latents, audio,
                    timestep.reshape(1).to(self.device, self.dtype), audio_cfg, ref_cfg)
                state = state + (following[si] - sigmas[si]).to(self.device, self.dtype) * velocity
                if wi:
                    state[:, :, 2:6] = previous_trace[si+1].to(self.device, self.dtype)
                trace.append(state[:, :, 16:20].detach().cpu().clone())
            if not torch.isfinite(state).all():
                raise RuntimeError('Non-finite decoder input')
            previous_trace = trace
            generated = state.cpu()
            del state, velocity, audio, reference_latents, lr, epsilon
            self.activate('vae')
            pixels = []
            for slot in range(plan['decode_start'], plan['decode_stop']):
                rgb = decoder.decode_frame(generated[:, :, slot:slot+1])
                if not torch.isfinite(rgb).all():
                    raise RuntimeError('Non-finite decoded RGB')
                pixels.append(((rgb + 1) * 127.5).clamp(0, 255).byte().cpu().numpy()[0].transpose(1, 2, 3, 0))
            emitted = np.concatenate(pixels)[plan['discard']:][:plan['emit_frames']]
            output[plan['retained_start']:plan['retained_stop']] = emitted
            last_rgb = emitted[-1].copy()
            logger.info('Window %d/%d: %d/%d frames', wi+1, len(geometry), plan['retained_stop'], len(paths))
        output.flush()
        return output
