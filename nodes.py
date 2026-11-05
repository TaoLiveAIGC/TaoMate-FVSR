"""ComfyUI front end. Models are loaded only during Restore execution."""
from pathlib import Path
import gc
import uuid

import numpy as np
import torch

from .node_io import prepare_video, read_rgb, RestoredVideo, save_video


def model_roots():
    import folder_paths
    roots = [Path(folder_paths.models_dir) / 'TaoMate-FVSR']
    for key in ('taomate', 'TaoMate-FVSR'):
        try:
            roots.extend(Path(p) for p in folder_paths.get_folder_paths(key))
        except KeyError:
            pass
    return list(dict.fromkeys(roots))


def resolve_model_dir(selection, local_directory):
    if local_directory.strip():
        directory = Path(local_directory).expanduser().resolve()
        if directory.is_dir():
            return directory
        raise ValueError(f'Model directory does not exist: {directory}')
    for root in model_roots():
        candidate = root if selection == '(root)' else root / selection
        if (candidate / 'config.json').is_file():
            return candidate
    raise ValueError('Install TaoMate assets in models/TaoMate-FVSR, configure extra_model_paths.yaml, or enter local_directory.')


def interrupt():
    from comfy import model_management
    model_management.throw_exception_if_processing_interrupted()


class TaoMateModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        choices = {'(root)'}
        for root in model_roots():
            if root.is_dir():
                choices.update(p.name for p in root.iterdir() if p.is_dir() and (p / 'config.json').is_file())
        return {'required': {'model': (sorted(choices),),
                            'local_directory': ('STRING', {'default': ''}),
                            'budget_gib': ('FLOAT', {'default': 15.0, 'min': 10.0, 'max': 15.0, 'step': .1})}}
    RETURN_TYPES = ('TAOMATE_MODEL',)
    FUNCTION = 'load'
    CATEGORY = 'TaoMate FVSR'

    def load(self, model, local_directory, budget_gib):
        from .taomate_runtime import ModelHandle
        return (ModelHandle(str(resolve_model_dir(model, local_directory)), device='cuda', budget_gib=budget_gib),)


class TaoMatePrepareVideo:
    @classmethod
    def IS_CHANGED(cls, video_path, audio_path='', **kwargs):
        # Comfy otherwise reuses PNGs when a source file is replaced in place.
        signatures = []
        for filename in (video_path, audio_path):
            if filename.strip():
                try:
                    stat = Path(filename).expanduser().stat()
                    signatures.append((stat.st_size, stat.st_mtime_ns))
                except OSError:
                    signatures.append(None)
        return repr(signatures)

    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {
            'video_path': ('STRING', {'default': ''}),
            'audio_path': ('STRING', {'default': '', 'tooltip': 'Blank keeps original audio; otherwise provide replacement speech.'}),
            'crop_mode': (['cropped_face', 'manual_roi', 'yolo_face'],),
            'roi_x': ('INT', {'default': 0, 'min': 0}),
            'roi_y': ('INT', {'default': 0, 'min': 0}),
            'roi_side': ('INT', {'default': 512, 'min': 16}),
            'max_frames': ('INT', {'default': 0, 'min': 0, 'tooltip': '0 = whole video; 77 = one short test.'})},
            'optional': {'detector_path': ('STRING', {'default': '', 'tooltip': 'Local YOLO face .pt; blank searches models/ultralytics/face_yolov8n.pt.'}),
                         'crop_scale': ('FLOAT', {'default': .8, 'min': .3, 'max': 3., 'step': .05})}}
    RETURN_TYPES = ('TAOMATE_VIDEO', 'IMAGE')
    RETURN_NAMES = ('prepared_video', 'first_crop')
    FUNCTION = 'prepare'
    CATEGORY = 'TaoMate FVSR'

    def prepare(self, video_path, audio_path, crop_mode, roi_x, roi_y, roi_side, max_frames,
                detector_path='', crop_scale=.8):
        interrupt()
        if crop_mode == 'yolo_face' and not detector_path.strip():
            import folder_paths
            roots = [Path(folder_paths.models_dir)/'ultralytics', *model_roots()]
            candidates = [root/'face_yolov8n.pt' for root in roots]
            detector_path = str(next((path for path in candidates if path.is_file()), candidates[0]))
        prepared = prepare_video(video_path, audio_path, crop_mode, roi_x, roi_y, roi_side, max_frames,
                                 detector_path=detector_path, crop_scale=crop_scale)
        image = torch.from_numpy(read_rgb(prepared.crops[0]).astype(np.float32) / 255)[None]
        return prepared, image


class TaoMateGPENReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'first_crop': ('IMAGE',), 'model': ('TAOMATE_MODEL',),
                             'gpen_repo': ('STRING', {'default': ''}),
                             'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffff})}}
    RETURN_TYPES = ('IMAGE',)
    FUNCTION = 'enhance'
    CATEGORY = 'TaoMate FVSR'

    def enhance(self, first_crop, model, gpen_repo, seed):
        import os
        import subprocess
        import sys
        import tempfile
        from PIL import Image
        if not Path(gpen_repo).is_dir() or not gpen_repo.strip():
            raise ValueError('Set local GPEN source checkout, or use Load Image with an already prepared reference.')
        image = reference_array(first_crop)
        interrupt()
        with tempfile.TemporaryDirectory(prefix='taomate-gpen-cpu-') as tmp:
            source, output = Path(tmp)/'input.png', Path(tmp)/'output.png'
            Image.fromarray(image).save(source)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
            subprocess.run([sys.executable, str(Path(__file__).with_name('node_io.py')),
                '--input', str(source), '--output', str(output), '--models', str(model.model_dir),
                '--gpen-repo', gpen_repo, '--seed', str(seed)], env=env, check=True)
            result = read_rgb(output)
        interrupt()
        return (torch.from_numpy(result.astype(np.float32) / 255)[None],)


def reference_array(image):
    if image.ndim != 4 or image.shape[0] != 1 or tuple(image.shape[1:]) != (512, 512, 3):
        raise ValueError('Reference must be one 512 × 512 RGB image matching the first crop composition.')
    return np.clip(image.detach().cpu().numpy()[0] * 255, 0, 255).round().astype(np.uint8)


class TaoMateRestore:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'model': ('TAOMATE_MODEL',), 'video': ('TAOMATE_VIDEO',), 'reference': ('IMAGE',),
            'seed': ('INT', {'default': 42, 'min': 0, 'max': 0xffffffff}),
            'steps': ('INT', {'default': 10, 'min': 1, 'max': 100}),
            'noise': ('FLOAT', {'default': .8, 'min': .01, 'max': 1}),
            'shift': ('FLOAT', {'default': 5., 'min': .1, 'max': 20}),
            'audio_cfg': ('FLOAT', {'default': 4., 'min': 0., 'max': 20}),
            'ref_cfg': ('FLOAT', {'default': 1.7, 'min': 0., 'max': 20})}}
    RETURN_TYPES = ('TAOMATE_RESULT',)
    FUNCTION = 'restore'
    CATEGORY = 'TaoMate FVSR'

    def restore(self, model, video, reference, seed, steps, noise, shift, audio_cfg, ref_cfg):
        from comfy import model_management
        from comfy.utils import ProgressBar
        from livevsr.media import prepare_audio
        ref = reference_array(reference)
        interrupt()
        workspace = video.workspace / ('restore-' + uuid.uuid4().hex)
        workspace.mkdir()
        pcm = prepare_audio(video.audio, workspace / 'conditioning.wav', len(video.crops))
        model_management.unload_all_models()
        gc.collect()
        model_management.soft_empty_cache()
        progress = ProgressBar(max(1, len(video.crops)))
        def cancel():
            interrupt()
            return False
        try:
            frames = model.restore(video.crops, ref, pcm, workspace, seed=seed, steps=steps,
                                   noise=noise, shift=shift, audio_cfg=audio_cfg, ref_cfg=ref_cfg,
                                   progress=lambda done, total: progress.update_absolute(done, total), cancel=cancel)
            interrupt()
            return (RestoredVideo(video, frames),)
        finally:
            model.unload()


class TaoMateSaveVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {'required': {'result': ('TAOMATE_RESULT',), 'filename_prefix': ('STRING', {'default': 'TaoMate-FVSR'})}}
    RETURN_TYPES = ('STRING',)
    RETURN_NAMES = ('video_path',)
    OUTPUT_NODE = True
    FUNCTION = 'save'
    CATEGORY = 'TaoMate FVSR'

    def save(self, result, filename_prefix):
        import folder_paths
        prefix = Path(filename_prefix)
        if prefix.is_absolute() or '..' in prefix.parts:
            raise ValueError('Use a relative filename prefix inside ComfyUI output.')
        destination = Path(folder_paths.get_output_directory()) / f'{prefix}_{uuid.uuid4().hex[:12]}.mp4'
        interrupt()
        save_video(result, destination)
        return {'ui': {'text': [str(destination)]}, 'result': (str(destination),)}


NODE_CLASS_MAPPINGS = {c.__name__: c for c in (TaoMateModelLoader, TaoMatePrepareVideo,
    TaoMateGPENReference, TaoMateRestore, TaoMateSaveVideo)}
NODE_DISPLAY_NAME_MAPPINGS = {
    'TaoMateModelLoader': 'TaoMate · Load Model (16 GB budget)',
    'TaoMatePrepareVideo': 'TaoMate · Prepare Video / Preview Crop',
    'TaoMateGPENReference': 'TaoMate · GPEN Reference (CPU)',
    'TaoMateRestore': 'TaoMate · Restore Face Video',
    'TaoMateSaveVideo': 'TaoMate · Save / Paste Back Video',
}
