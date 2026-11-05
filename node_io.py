"""Disk-backed video interchange; no CUDA or automatic downloads."""
from dataclasses import dataclass
from pathlib import Path
import subprocess
import uuid

import numpy as np
from PIL import Image


@dataclass
class PreparedVideo:
    crops: list
    source: Path
    audio: Path
    workspace: Path
    size: tuple
    roi: tuple | None
    scene_paths: list
    boxes: object = None
    valid: object = None


@dataclass
class RestoredVideo:
    prepared: PreparedVideo
    frames: object


def new_workspace():
    import folder_paths
    path = Path(folder_paths.get_temp_directory()) / 'taomate' / uuid.uuid4().hex
    path.mkdir(parents=True)
    return path


def read_rgb(path):
    with Image.open(path) as im:
        return np.array(im.convert('RGB'))


def prepare_video(video_path, audio_path='', mode='cropped_face', x=0, y=0,
                  side=512, max_frames=0, workspace=None, detector_path='', crop_scale=.8):
    from livevsr.media import probe, check_tools
    check_tools()
    source = Path(video_path).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f'Video does not exist: {source}')
    audio = Path(audio_path).expanduser().resolve() if audio_path.strip() else source
    if not audio.is_file():
        raise ValueError(f'Audio does not exist: {audio}')
    streams = probe(source)
    stream = next(s for s in streams if s['codec_type'] == 'video')
    width, height = int(stream['width']), int(stream['height'])
    if mode == 'cropped_face':
        if width != height:
            raise ValueError('cropped_face requires a square face video. Use manual_roi for a full scene.')
        roi = None
    elif mode == 'manual_roi':
        if side < 16 or x < 0 or y < 0 or x + side > width or y + side > height:
            raise ValueError(f'ROI ({x}, {y}, {side}) is outside {width} × {height}')
        if width % 2 or height % 2:
            raise ValueError('Full-scene H.264 output requires even dimensions')
        roi = (x, y, side)
    elif mode == 'yolo_face':
        if width % 2 or height % 2:
            raise ValueError('Full-scene H.264 output requires even dimensions')
        return prepare_yolo_video(source, audio, width, height, max_frames,
                                  workspace, detector_path, crop_scale)
    else:
        raise ValueError(f'Unknown crop mode: {mode}')
    workspace = Path(workspace) if workspace else new_workspace()
    workspace.mkdir(parents=True, exist_ok=True)
    crops_dir = workspace / 'crops'
    crops_dir.mkdir()
    limit = ['-frames:v', str(max_frames)] if max_frames else []
    # Crop RGB, so YUV420 chroma subsampling cannot round odd ROI coordinates.
    filters = 'fps=25,format=rgb24'
    if roi:
        filters += f',crop={side}:{side}:{x}:{y}'
    filters += ',scale=512:512:flags=lanczos'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                    '-map', '0:v:0', '-vf', filters, *limit, '-threads', '2',
                    str(crops_dir / '%08d.png')], check=True)
    crops = sorted(crops_dir.glob('*.png'))
    if not crops:
        raise ValueError('No decoded frames')
    scene_paths = []
    if roi:
        scene_dir = workspace / 'scene'
        scene_dir.mkdir()
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source),
                        '-map', '0:v:0', '-vf', 'fps=25', *limit, '-threads', '2',
                        str(scene_dir / '%08d.png')], check=True)
        scene_paths = sorted(scene_dir.glob('*.png'))
        if len(scene_paths) != len(crops):
            raise ValueError('Scene/crop frame count differs')
    return PreparedVideo(crops, source, audio, workspace, (width, height), roi, scene_paths)


def prepare_yolo_video(source, audio, width, height, max_frames, workspace, detector_path, crop_scale):
    from .crop_yolo import prepare_yolo_crops
    from livevsr.crop import crop_frame
    if not Path(detector_path).is_file():
        raise ValueError('YOLO mode requires a local face_yolov8n.pt detector; no automatic download is performed.')
    workspace = Path(workspace) if workspace else new_workspace()
    scene_dir, crop_dir = workspace/'scene', workspace/'crops'
    scene_dir.mkdir(parents=True)
    crop_dir.mkdir()
    limit = ['-frames:v', str(max_frames)] if max_frames else []
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(source), '-map', '0:v:0',
        '-vf', 'fps=25', *limit, '-threads', '2', str(scene_dir/'%08d.png')], check=True)
    scenes = sorted(scene_dir.glob('*.png'))
    if not scenes:
        raise ValueError('No decoded frames')
    from comfy import model_management
    def cancel():
        model_management.throw_exception_if_processing_interrupted()
        return False
    boxes, valid, cuts = prepare_yolo_crops(scenes, detector_path, crop_scale=crop_scale, cancel=cancel)
    if cuts:
        raise ValueError(f'Possible shot cuts at 25fps frame indices {cuts}; split the clip into single shots before restoration.')
    boxes, valid = np.array(boxes, copy=True), np.asarray(valid, dtype=bool)
    good = np.flatnonzero(valid)
    if not len(good):
        raise ValueError('No tracked face found; try manual_roi or another clip.')
    # Missing detections retain their original time slot and are never pasted.
    for index in np.flatnonzero(~valid):
        boxes[index] = boxes[good[np.argmin(abs(good-index))]]
    crops = []
    for index, (path, box) in enumerate(zip(scenes, boxes)):
        cancel()
        destination = crop_dir/f'{index:08d}.png'
        Image.fromarray(crop_frame(read_rgb(path), box)).save(destination)
        crops.append(destination)
    np.savez(workspace/'crop-track.npz', boxes=boxes, valid=valid, crop_scale=crop_scale)
    return PreparedVideo(crops, source, audio, workspace, (width,height), None, scenes, boxes, valid)


def composite_frames(result):
    """Use upstream histogram matching and soft mask for full-scene paste-back."""
    import cv2
    import livevsr
    from livevsr.composite import paste_head
    p = result.prepared
    if p.roi is None and p.boxes is None:
        yield from result.frames
        return
    mask_path = Path(livevsr.__file__).parent / 'assets' / 'paste_face_mask.png'
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Cannot read upstream paste mask: {mask_path}')
    alpha = cv2.resize(mask, (512, 512)).astype(np.float32) / 255.
    for index, (path, generated) in enumerate(zip(p.scene_paths, result.frames)):
        scene = read_rgb(path)
        if p.valid is not None and not p.valid[index]:
            yield scene
            continue
        if p.boxes is not None:
            box = p.boxes[index]
        else:
            x, y, side = p.roi
            box = (x, y, x+side, y+side)
        yield paste_head(scene, read_rgb(p.crops[index]), np.asarray(generated), box, alpha)


def save_video(result, destination):
    """Mux original/replacement audio, retaining its sample rate and stereo channels."""
    p = result.prepared
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if len(result.frames) != len(p.crops):
        raise ValueError('Restored frame count differs from prepared video')
    width, height = p.size if p.scene_paths else (512, 512)
    process = subprocess.Popen(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo',
        '-pix_fmt', 'rgb24', '-s', f'{width}x{height}', '-r', '25', '-i', 'pipe:0',
        '-i', str(p.audio), '-map', '0:v:0', '-map', '1:a:0', '-af', 'apad',
        '-c:v', 'libx264', '-threads', '2', '-crf', '18', '-pix_fmt', 'yuv420p',
        '-c:a', 'aac', '-b:a', '192k', '-t', str(len(p.crops)/25),
        '-movflags', '+faststart', str(destination)], stdin=subprocess.PIPE)
    try:
        for frame in composite_frames(result):
            process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
    finally:
        process.stdin.close()
        code = process.wait()
    if code:
        raise RuntimeError(f'FFmpeg save failed ({code})')
    return destination


if __name__ == '__main__':
    # Standalone CPU worker: GPEN imports otherwise inspect CUDA visibility and
    # JIT-build CUDA ops even when prepare_reference(device='cpu') is requested.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--gpen-repo', required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    from livevsr.reference import prepare_reference
    root = Path(args.models)
    result = prepare_reference(read_rgb(args.input), repo=args.gpen_repo,
        weights=root/'GPEN-BFR-512.pth', detector_weights=root/'RetinaFace-R50.pth',
        parser_weights=root/'ParseNet-latest.pth', size=512, passes=1, seed=args.seed, device='cpu')
    Image.fromarray(result).save(args.output)
