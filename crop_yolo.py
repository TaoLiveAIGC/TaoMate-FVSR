"""CPU YOLO face crops with explicit frame validity; no automatic downloads.

Independent bbox mode, not landmark-equivalent native DWPose geometry. Input
paths must already be on the node's 25 fps timeline. crop_scale scales the
crop's side, not the model's scale2 or the output video dimensions.
"""
from pathlib import Path
import os
import tempfile

import numpy as np


def _overlap(a, b):
    a, b = np.asarray(a), np.asarray(b)
    inter = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
    return float(inter / max(1e-8, (a[2:]-a[:2]).prod() + (b[2:]-b[:2]).prod() - inter))


def _context_square(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    y1 -= .2 * (y2-y1)
    side = min(max(x2-x1, y2-y1) * 1.45, width, height)
    x = float(np.clip((x1+x2-side)/2, 0, width-side))
    y = float(np.clip((y1+y2-side)/2, 0, height-side))
    return np.array([x, y, x+side, y+side])


def scale_square(box, scale, width, height):
    """Keep center, translate at edges before limiting to the frame short side."""
    if not np.isfinite(scale) or not 0 < scale <= 2:
        raise ValueError('crop_scale must be finite and in (0, 2]')
    box = np.asarray(box, dtype=float)
    center = (box[:2]+box[2:])/2
    side = min(max(box[2:]-box[:2])*scale, width, height)
    left = np.clip(center-side/2, [0, 0], [width-side, height-side])
    actual = np.round(np.r_[left, left+side]).astype(np.int64)
    if np.any(actual[2:] <= actual[:2]):
        raise ValueError('crop_scale produces an empty integer crop')
    return actual


def boxes_from_detections(detections, cuts, width, height, crop_scale=.8):
    """Pure geometry adapter, also usable to replay saved CPU detections.

    Largest initial face; IoU >= .2 association. No detection is invalid but
    may reacquire at the previous location. An incompatible candidate locks
    association until the next cut. This is not identity recognition.
    """
    from livevsr.crop import _adaptive_boxes
    if not np.isfinite(crop_scale) or not 0 < crop_scale <= 2:
        raise ValueError('crop_scale must be finite and in (0, 2]')
    count = len(detections)
    cutset = set(cuts)
    if any(not isinstance(i, (int, np.integer)) or i <= 0 or i >= count for i in cutset):
        raise ValueError('Cut indices must be interior frame indices')
    raw, previous, lost = [], None, False
    for i, candidates in enumerate(detections):
        if i in cutset:
            previous, lost = None, False
        candidates = [np.asarray(b[:4], dtype=float) for b in candidates
                      if len(b) >= 4 and np.isfinite(b[:4]).all() and b[2] > b[0] and b[3] > b[1]]
        if not candidates or lost:
            raw.append(None)
            continue
        if previous is None:
            chosen = max(candidates, key=lambda b: (b[2:]-b[:2]).prod())
        else:
            matches = [b for b in candidates if _overlap(b, previous) >= .2]
            if not matches:
                lost = True
                raw.append(None)
                continue
            chosen = max(matches, key=lambda b: _overlap(b, previous))
        previous = chosen
        raw.append(_context_square(chosen, width, height))
    boxes = np.zeros((count, 4), dtype=np.int64)
    valid = np.zeros(count, dtype=bool)
    start = None
    for i in range(count+1):
        if start is not None and (i == count or i in cutset or raw[i] is None):
            # Integer native adaptive output is intentionally preserved before
            # scale, matching the full-sequence C + scale ablation exactly.
            smoothed = _adaptive_boxes(raw[start:i], width, height)
            for j, box in enumerate(smoothed, start):
                boxes[j] = scale_square(np.asarray(box, dtype=int), crop_scale, width, height)
                valid[j] = True
            start = None
        if i < count and raw[i] is not None and start is None:
            start = i
    return boxes, valid


def prepare_yolo_crops(paths, detector_path, crop_scale=.8, progress=None, cancel=None):
    """Return (xyxy int64[N,4], valid bool[N], cuts list[int]) on CPU.

    Reads one PNG at a time, preserving input order and length. progress takes
    (completed, total); cancel takes no arguments and may return true or raise.
    Invalid boxes are zero and must NEVER be pasted. Consumers may duplicate
    a nearby valid crop as model input only, while retaining valid=False.
    """
    paths = list(paths)
    if not paths:
        raise ValueError('No scene frames supplied')
    if not Path(detector_path).is_file():
        raise FileNotFoundError(f'Local YOLO face weights not found: {detector_path}')
    if not np.isfinite(crop_scale) or not 0 < crop_scale <= 2:
        raise ValueError('crop_scale must be finite and in (0, 2]')

    def check_cancel():
        if cancel is not None and cancel():
            raise RuntimeError('TaoMate face crop cancelled')

    check_cancel()
    # Ultralytics may create its settings at import time. Respect an explicit
    # caller setting; otherwise keep those files out of shared HOME config.
    config = Path(tempfile.gettempdir()) / f'taomate-ultralytics-{os.getpid()}'
    os.environ.setdefault('YOLO_CONFIG_DIR', str(config))
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError('YOLO crop mode requires optional dependency ultralytics; use manual ROI otherwise') from exc
    import cv2
    model = YOLO(str(detector_path))
    detections, cuts, previous_gray, size = [], [], None, None
    for i, path in enumerate(paths):
        check_cancel()
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Cannot read scene frame: {path}')
        height, width = image.shape[:2]
        if size is None:
            size = (width, height)
        elif size != (width, height):
            raise ValueError('Scene frame dimensions must remain constant')
        # Same native cut heuristic, evaluated in a streaming pass.
        gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (240, 135)).astype(np.float32)
        if previous_gray is not None and np.abs(gray-previous_gray).mean() > 20.:
            cuts.append(i)
        previous_gray = gray
        result = model.predict(image, device='cpu', imgsz=640, conf=.25, verbose=False)[0]
        detections.append(np.c_[result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()].tolist())
        if progress is not None:
            progress(i+1, len(paths))
    check_cancel()
    boxes, valid = boxes_from_detections(detections, cuts, *size, crop_scale=crop_scale)
    if not valid.any():
        raise ValueError('No valid face track; choose a manual ROI or another clip')
    return boxes, valid, cuts
