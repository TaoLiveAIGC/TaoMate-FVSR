"""Original-precision low-memory adaptation of TaoMate-FVSR bc924d8 (Apache-2.0)."""
from pathlib import Path
import gc
import logging
import threading
import traceback
import sys
import numpy as np
from PIL import Image
import torch

log = logging.getLogger(__name__)


def _move_cache(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, list):
        return [_move_cache(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_cache(v, device) for v in value)
    return value


class ModelHandle:
    """Lazy CPU model cache. Each restore owns and finally releases its GPU tensors.

    Upstream livevsr must be installed. budget_gib caps the PyTorch allocator,
    not total process VRAM; leave additional CUDA/context room on physical cards.
    progress(done, total) and cancel() callbacks run between diffusion steps.
    """
    def __init__(self, model_dir, device='cuda', budget_gib=15.0):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.budget_gib = float(budget_gib)
        self.pipeline = None
        self.decoder = None
        self._lock = threading.Lock()
        self.interrupt_callback = None
        self.cancel_callback = None

    def _load(self):
        if self.pipeline is None:
            from livevsr.pipeline import LiveVSR
            self.pipeline = LiveVSR(self.model_dir / 'TaoMate-FVSR.safetensors',
                self.model_dir / 'Wan2.2_VAE.pth', self.model_dir / 'hubert',
                device=self.device, offload=True)

    def _activate(self, target):
        p = self.pipeline
        if self.decoder is not None and target != 'vae':
            self.decoder._feat_map = _move_cache(self.decoder._feat_map, 'cpu')
        # Evict FIRST: uploading DiT while VAE is resident exceeds small budgets.
        for name, module in [('dit', p.dit), ('vae', p.vae)]:
            if name != target:
                module.to('cpu')
        p.audio.to('cpu')
        torch.cuda.empty_cache()
        if target in ('dit', 'vae'):
            getattr(p, target).to(self.device)
        if self.decoder is not None and target == 'vae':
            self.decoder._feat_map = _move_cache(self.decoder._feat_map, self.device)

    def unload(self):
        self.decoder = None
        if self.pipeline is not None:
            self.pipeline.vae.model.clear_cache()
            self._activate(None)
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()

    def _encode(self, images):
        """Same causal 1+4 frame encoding, uploading only the current chunk."""
        from livevsr.models.wan_video_vae import patchify
        p, outputs = self.pipeline, []
        model = p.vae.model
        model.clear_cache()
        try:
            for begin in range(0, len(images), 4):
                if self.interrupt_callback:
                    self.interrupt_callback()
                if self.cancel_callback and self.cancel_callback():
                    raise InterruptedError('TaoMate restoration cancelled')
                # Iteration 0 uses 1 frame; all following chunks use 4 frames.
                start, stop = (0, 1) if begin == 0 else (begin - 3, begin + 1)
                batch = []
                for image in images[start:stop]:
                    if isinstance(image, (str, Path)):
                        with Image.open(image) as opened:
                            image = opened.convert('RGB')
                    batch.append(np.asarray(image.resize((512, 512), Image.Resampling.BICUBIC)))
                x = torch.from_numpy(np.stack(batch)).permute(3, 0, 1, 2)[None]
                # Preserve upstream BF16 normalization order.
                x = x.to(dtype=p.dtype) * (2.0 / 255) - 1
                x = patchify(x.to(self.device), patch_size=2)
                out, model._enc_feat_map, model._enc_conv_idx = model.encoder(
                    x, feat_cache=model._enc_feat_map, feat_idx=[0])
                mu = model.conv1(out).chunk(2, dim=1)[0]
                mean, scale = [s.to(mu.device, mu.dtype) for s in p.vae.scale]
                mu = (mu - mean.view(1, 48, 1, 1, 1)) * scale.view(1, 48, 1, 1, 1)
                outputs.append(mu.cpu())
            return torch.cat(outputs, dim=2)
        finally:
            model.clear_cache()

    @torch.inference_mode()
    def restore(self, paths, reference, pcm, workspace, *, seed=42, steps=10,
                noise=.8, shift=5., audio_cfg=4., ref_cfg=1.7, size=512,
                progress=None, cancel=None, interrupt_callback=None, progress_callback=None):
        if size != 512:
            raise ValueError('The validated low-memory profile requires size=512')
        if steps < 1 or steps > 1000:
            raise ValueError('steps must be between 1 and 1000')
        if np.asarray(pcm).shape != (len(paths), 640):
            raise ValueError('pcm must have shape [frame_count,640] at 16kHz / 25fps')
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('This model handle is already running')
        old_fraction = None
        try:
            self.interrupt_callback = interrupt_callback
            self.cancel_callback = cancel
            if self.device.type != 'cuda':
                raise ValueError('The restoration runtime requires CUDA')
            if self.device.index is None:
                self.device = torch.device('cuda', torch.cuda.current_device())
            # PyTorch 2.9 cudaMallocAsync can return an invalid previous fraction
            # (observed 180698976.7 in real Comfy). Do not mutate a policy that
            # cannot be faithfully restored. Native allocator supports the cap.
            if torch.cuda.get_allocator_backend() == 'native':
                old_fraction = torch.cuda.get_per_process_memory_fraction(self.device)
                total = torch.cuda.get_device_properties(self.device).total_memory
                torch.cuda.set_per_process_memory_fraction(min(1., self.budget_gib * 2**30 / total), self.device)
            else:
                log.warning('Allocator %s: component offload is active, but allocator budget is not enforced. '
                    'Start ComfyUI with --disable-cuda-malloc to enforce the budget.',
                    torch.cuda.get_allocator_backend())
            self._load()
            return self._restore(paths, reference, pcm, Path(workspace), seed, steps,
                noise, shift, audio_cfg, ref_cfg, progress_callback or progress, cancel)
        except BaseException as exc:
            # Comfy retains exception tracebacks for its UI; release GPU locals
            # in unwound frames before that traceback can outlive this call.
            traceback.clear_frames(exc.__traceback__)
            raise
        finally:
            active_error = sys.exc_info()[0] is not None
            cleanup_error = None
            try:
                self.unload()
            except Exception as exc:
                cleanup_error = exc
                log.exception('TaoMate model cleanup failed')
            try:
                if old_fraction is not None:
                    torch.cuda.set_per_process_memory_fraction(old_fraction, self.device)
            except Exception as exc:
                cleanup_error = cleanup_error or exc
                log.exception('TaoMate allocator policy restoration failed')
            finally:
                self.interrupt_callback = None
                self.cancel_callback = None
                self._lock.release()
            if cleanup_error is not None and not active_error:
                raise cleanup_error

    def _restore(self, paths, reference, pcm, workspace, seed, steps, noise, shift,
                 audio_cfg, ref_cfg, progress, cancel):
        # Window/CFG equations are adapted from livevsr/pipeline.py bc924d8.
        from livevsr.geometry import windows, input_indices
        from livevsr.pipeline import window_lr
        from livevsr.denoise import schedule, guided_forward
        from livevsr.decoder import StreamingVAEDecoder
        from livevsr.media import write_audio
        def check():
            if self.interrupt_callback:
                self.interrupt_callback()
            if cancel and cancel():
                raise InterruptedError('TaoMate restoration cancelled')
        workspace.mkdir(parents=True, exist_ok=True)
        check()
        p = self.pipeline
        plans = windows(len(paths))
        self._activate('vae')
        global_lr = self._encode([paths[i] for i in input_indices(len(paths))])
        rng = torch.Generator(device='cpu').manual_seed(seed)
        first_noise = torch.randn((1, 48, 20, 32, 32), generator=rng)
        global_noise = first_noise if global_lr.shape[2] == 20 else torch.cat([
            first_noise, torch.randn((1, 48, global_lr.shape[2] - 20, 32, 32), generator=rng)], dim=2)
        ref = Image.fromarray(np.asarray(reference, dtype=np.uint8))
        identity, bootstrap = self._encode([ref]), self._encode([ref] * 5)
        self.decoder = StreamingVAEDecoder(p.vae, device=self.device, dtype=p.dtype)
        self.decoder.reset()
        ts, sigmas, following = schedule(steps, shift)
        output = np.lib.format.open_memmap(workspace / 'output.npy', mode='w+',
            dtype=np.uint8, shape=(len(paths), 512, 512, 3))
        previous_trace, last_rgb = None, None
        for wi, plan in enumerate(plans):
            check()
            self._activate('vae')
            references = identity.new_zeros((1, 48, 20, 32, 32))
            references[:, :, :1] = identity
            references[:, :, 1:2] = bootstrap[:, :, 1:2]
            if wi:
                references[:, :, 5:6] = self._encode([Image.fromarray(last_rgb)])
            wav = workspace / 'window.wav'
            write_audio(wav, pcm[plan['input_indices']].reshape(-1))
            self._activate('audio')
            audio = p.audio.build_condition(audio_path=str(wav), audio_feature_path=None,
                audio_valid=True, frame_start=0, output_device=self.device, output_dtype=p.dtype)
            start = plan['latent_start']
            lr = window_lr(global_lr, start).to(self.device, p.dtype)
            epsilon = global_noise[:, :, start:start+20].to(self.device, p.dtype)
            strength = lr.new_tensor(noise)
            state = (1-strength) * lr + strength * epsilon
            trace = [state[:, :, 16:20].cpu().clone()]
            references = references.to(self.device, p.dtype)
            self._activate('dit')
            for si, timestep in enumerate(ts):
                check()
                if wi:
                    state[:, :, 2:6] = previous_trace[si].to(self.device, p.dtype)
                velocity = guided_forward(p.dit, state, references, audio,
                    timestep.reshape(1).to(self.device, p.dtype), audio_cfg, ref_cfg)
                state = state + (following[si]-sigmas[si]).to(self.device, p.dtype) * velocity
                if wi:
                    state[:, :, 2:6] = previous_trace[si+1].to(self.device, p.dtype)
                trace.append(state[:, :, 16:20].cpu().clone())
                log.info('Window %s/%s step %s/%s', wi+1, len(plans), si+1, steps)
                if progress:
                    progress(wi * steps + si+1, len(plans) * steps)
            if not torch.isfinite(state).all():
                raise RuntimeError('Non-finite decoder input')
            previous_trace, generated = trace, state.cpu()
            del state, velocity, audio, references, lr, epsilon
            self._activate('vae')
            pixels = []
            for slot in range(plan['decode_start'], plan['decode_stop']):
                check()
                rgb = self.decoder.decode_frame(generated[:, :, slot:slot+1])
                if not torch.isfinite(rgb).all():
                    raise RuntimeError('Non-finite decoded RGB')
                pixels.append(((rgb+1)*127.5).clamp(0,255).byte().cpu().numpy()[0].transpose(1,2,3,0))
            emitted = np.concatenate(pixels)[plan['discard']:][:plan['emit_frames']]
            output[plan['retained_start']:plan['retained_stop']] = emitted
            last_rgb = emitted[-1].copy()
            del rgb
            output.flush()
            log.info('Window %s/%s completed: %s frames', wi+1, len(plans), plan['retained_stop'])
        return output
