"""Crop, restore and paste a talking face back into its original video."""
import argparse
import logging
import math
import os
from pathlib import Path
import tempfile
import sys


def parser():
    from .weights import DEFAULT_CONFIG
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', required=True, type=Path, help='Full video or ordered 25 fps PNG directory')
    p.add_argument('--output', required=True, type=Path, help='Output MP4')
    p.add_argument('--audio', type=Path, help='Target speech; defaults to input video audio')
    p.add_argument('--scale', type=int, choices=[1, 2], default=1,
                   help='Inference resolution: 512 or 1024; return to 512 before paste-back')
    p.add_argument('--cropped', action='store_true', help='Input is already a 512 face crop; skip scene crop/paste')
    p.add_argument('--pose-python', default=sys.executable, help='Python with MMPose/MMDetection installed')
    p.add_argument('--pose-model-dir', type=Path, help='YOLOX/RTMW configs and weights (default: models/dwpose)')
    p.add_argument('--pose-device', default='cpu', help='DWPose device, e.g. cpu or cuda:0')
    p.add_argument('--blend-mask', type=Path,
                   default=Path(__file__).with_name('assets') / 'paste_face_mask.png',
                   help='Custom paste mask; defaults to the bundled TaoMate-FVSR mask')
    references = p.add_mutually_exclusive_group()
    references.add_argument('--reference', type=Path, help='Final prepared reference; bypass all GPEN processing')
    references.add_argument('--edited-reference', type=Path,
                            help='Edited first crop (e.g. GPT-Image); apply GPEN alignment and enhancement')
    p.add_argument('--gpen-repo', type=Path,
                   help='Local yangxy/GPEN checkout; 1x uses GPEN-BFR-512, 2x uses GPEN-BFR-1024')
    p.add_argument('--gpen-passes', type=int, choices=[1, 2], default=1)
    p.add_argument('--model-dir', type=Path, help='Directory with all model assets; no download in this mode')
    p.add_argument('--model-config', type=Path, default=DEFAULT_CONFIG)
    for name in ('checkpoint', 'vae', 'hubert', 'gpen-weights', 'gpen-detector', 'gpen-parser'):
        p.add_argument('--' + name, type=Path, help='Override local ' + name + ' path')
    p.add_argument('--offline', action='store_true', help='Only local files and existing model caches')
    p.add_argument('--cache-dir', type=Path, help='Model download cache; never stores input media')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--cpu-offload', action='store_true', help='Keep only the active model on GPU')
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--noise', type=float, default=.8)
    p.add_argument('--audio-cfg', type=float, default=4.)
    p.add_argument('--ref-cfg', type=float, default=1.7)
    p.add_argument('--shift', type=float, default=5.)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--overwrite', action='store_true')
    return p


def validate(a):
    a.input = a.input.expanduser().resolve()
    a.output = a.output.expanduser().resolve()
    for name in ('input', 'audio', 'reference', 'edited_reference'):
        value = getattr(a, name)
        valid = value is None or value.expanduser().is_file() or (name == 'input' and value.expanduser().is_dir())
        if not valid:
            raise ValueError(f'--{name} is not a local file: {value}')
        if value is not None:
            setattr(a, name, value.expanduser().resolve())
    if a.gpen_repo is not None:
        a.gpen_repo = a.gpen_repo.expanduser().resolve()
    if a.output == a.input or any(a.output == p.expanduser().resolve()
                                for p in (a.audio, a.reference, a.edited_reference) if p is not None):
        raise ValueError('Output must not overwrite an input')
    if a.output.suffix.lower() != '.mp4':
        raise ValueError('--output must end in .mp4')
    if a.output.exists() and not a.overwrite:
        raise ValueError('Output already exists; use --overwrite to replace it')
    if a.reference is None and (a.gpen_repo is None or not (a.gpen_repo / 'face_model/gpen_model.py').is_file()):
        raise ValueError('Supply --gpen-repo for first-frame enhancement, or --reference for a prepared image')
    if a.input.is_dir() and a.audio is None:
        raise ValueError('A PNG directory requires --audio')
    if not a.cropped:
        from .pose import FILES
        root = (a.model_dir or Path('models')).expanduser().resolve()
        a.pose_model_dir = (a.pose_model_dir or root / 'dwpose').expanduser().resolve()
        a.blend_mask = a.blend_mask.expanduser().resolve()
        for path in [a.blend_mask, *(a.pose_model_dir / name for name in FILES.values())]:
            if not path.is_file():
                raise ValueError(f'Missing full-scene component: {path}; see README, or use --cropped for prepared face inputs')
    if not 2 <= a.steps <= 1000 or not 0 <= a.noise <= 1:
        raise ValueError('--steps must be 2..1000 and --noise must be 0..1')
    if not all(math.isfinite(v) for v in (a.shift, a.audio_cfg, a.ref_cfg)) or a.shift <= 0:
        raise ValueError('Guidance values must be finite and --shift must be positive')


def main(argv=None):
    a = parser().parse_args(argv)
    validate(a)
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    import torch
    import numpy as np
    from PIL import Image
    from . import media
    from .pipeline import LiveVSR
    from .reference import prepare_reference
    from .weights import load_sources, resolve_asset
    from . import scene
    media.check_tools()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('LiveVSR inference requires an NVIDIA CUDA GPU with BF16 support')
    torch.cuda.set_device(torch.device(a.device))
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError('This checkpoint requires BF16 support')
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    sources = load_sources(a.model_config, scale=1 if a.reference else a.scale)
    names = ['checkpoint', 'vae', 'hubert'] + ([] if a.reference else
            ['gpen_weights', 'gpen_detector', 'gpen_parser'])
    assets = {name: resolve_asset(name, getattr(a, name), a.model_dir, sources,
              offline=a.offline, cache_dir=a.cache_dir) for name in names}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.livevsr-', dir=a.output.parent) as temporary:
        workspace = Path(temporary)
        frame_dir = workspace / 'frames'
        frame_dir.mkdir()
        if a.cropped:
            paths, model_size, _ = media.decode_video(a.input, frame_dir, a.scale)
            if model_size != 512 * a.scale:
                raise ValueError('--cropped expects 512x512 face inputs')
            width = height = 512
        else:
            native_paths, width, height, fps = media.decode_scene(a.input, frame_dir)
            paths, scene_paths, boxes = scene.prepare(native_paths, fps, workspace,
                pose_python=a.pose_python, pose_models=a.pose_model_dir, device=a.pose_device)
            model_size = 512 * a.scale
        audio_path = workspace / 'audio.wav'
        pcm = media.prepare_audio(a.audio or a.input, audio_path, len(paths))
        if a.reference:
            with Image.open(a.reference) as image:
                reference = np.array(image.convert('RGB'))
        else:
            with Image.open(a.edited_reference or paths[0]) as image:
                first = np.array(image.convert('RGB'))
            reference = prepare_reference(first, repo=a.gpen_repo, weights=assets['gpen_weights'],
                detector_weights=assets['gpen_detector'], parser_weights=assets['gpen_parser'],
                size=model_size, passes=a.gpen_passes, seed=a.seed)
        if reference.shape[0] != reference.shape[1]:
            raise ValueError('Reference must be square and match the first input crop framing')
        model = LiveVSR(assets['checkpoint'], assets['vae'], assets['hubert'],
                        device=a.device, offload=a.cpu_offload)
        frames = model.restore(paths, reference, pcm, workspace, size=model_size, seed=a.seed,
            steps=a.steps, noise=a.noise, shift=a.shift, audio_cfg=a.audio_cfg, ref_cfg=a.ref_cfg)
        del model
        torch.cuda.empty_cache()
        encoded = workspace / 'output.mp4'
        if a.cropped:
            result = (scene.resize_head(frame) for frame in frames)
        else:
            result = scene.composite(scene_paths, paths, frames, boxes, a.blend_mask)
        logging.info('Encoding %d frames at %dx%d; model resolution %d -> RGB512',
                     len(paths), width, height, model_size)
        media.encode_video(result, audio_path, encoded, size=(width, height), count=len(paths))
        del frames
        # Only publish a complete, decodable video. Existing output survives failures.
        if a.output.exists() and not a.overwrite:
            raise FileExistsError(a.output)
        encoded.replace(a.output)
    logging.info('Saved %dx%d, 25 fps video: %s', width, height, a.output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
