"""Native-frame DWPose crops and RGB512 histogram/mask paste-back."""
import logging
import math
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image

from .crop import crop_frame, detect_cuts, shot_local_boxes
from .composite import paste_head


def rgb(path):
    with Image.open(path) as image:
        return np.array(image.convert('RGB'))


def frame_indices(count, fps):
    """Floor mapping shared by native geometry and scene frames, as in the demo."""
    numerator, denominator = fps.numerator, fps.denominator
    total = math.ceil(count * 25 * denominator / numerator)
    return np.minimum(np.arange(total) * numerator // (25 * denominator), count-1)


def prepare(paths, fps, workspace, *, pose_python, pose_models, device):
    track_path = workspace / 'dwpose.npy'
    subprocess.run([str(pose_python), str(Path(__file__).with_name('pose.py')),
        '--frames', str(paths[0].parent), '--model-dir', str(pose_models),
        '--device', device, '--output', str(track_path)], check=True)
    track = np.load(track_path, allow_pickle=False)
    first = rgb(paths[0])
    height, width = first.shape[:2]
    cuts = detect_cuts(rgb(path) for path in paths)
    native_boxes = shot_local_boxes(track, width, height, cuts, adaptive=True)
    indices = frame_indices(len(paths), fps)
    boxes = native_boxes[indices]
    scene_paths = [paths[i] for i in indices]
    folder = workspace / 'crops'
    folder.mkdir()
    crops = []
    for i, (path, box) in enumerate(zip(scene_paths, boxes)):
        output = folder / f'{i:08d}.png'
        Image.fromarray(crop_frame(rgb(path), box)).save(output)
        crops.append(output)
    logging.info('DWPose crops: %d native -> %d frames at 25 fps, %d shots',
                 len(paths), len(crops), len(cuts)+1)
    return crops, scene_paths, boxes


def resize_head(frame):
    """Both 1x and 2x inference feed the same RGB512 paste geometry."""
    if frame.shape[:2] == (512, 512):
        return frame
    import cv2
    return cv2.resize(frame, (512, 512), interpolation=cv2.INTER_AREA)


def composite(scene_paths, crops, generated, boxes, mask_path):
    import cv2
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f'Cannot read blend mask: {mask_path}')
    alpha = cv2.resize(mask, (512, 512)).astype(np.float32) / 255.
    for path, crop, frame, box in zip(scene_paths, crops, generated, boxes):
        yield paste_head(rgb(path), rgb(crop), resize_head(frame), box, alpha)
