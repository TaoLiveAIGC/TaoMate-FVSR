"""Shot-local square DWPose geometry, independent of detector/deployment code.

Input: normalized DWPose/RTMW tracks [frames, 1, 134, 3] (x, y, score).
Use the same decoded source-frame order for tracks, cuts, crops and paste-back.
"""
import numpy as np
from scipy.signal import savgol_filter

FACE_INDEX = [63, 66, 27, 37, 25, 26, 24, 40, 39, 38] + list(range(24, 92)) + [32]


def _window(requested, size, minimum=3):
    length = min(requested, size)
    if length % 2 == 0:
        length -= 1
    length = max(minimum, length)
    if length > size:
        length = size if size % 2 else size - 1
    return max(1, length)


def _mean(values, width):
    if len(values) <= 1:
        return values
    return [np.mean(values[max(0, i-width//2):min(len(values), i+width//2+1)],
                    axis=0).astype(np.float32) for i in range(len(values))]


def _sg(values, width=25):
    if len(values) <= 2:
        return np.asarray(values)
    width = _window(width, len(values))
    pad = width // 2
    array = np.asarray(values).reshape(len(values), -1)
    padded = np.pad(array, ((pad, pad), (0, 0)), mode='edge')
    return savgol_filter(padded, width, min(2, width-1), axis=0)[pad:-pad].astype(np.float32)


def _pose_filter(raw):
    result = raw.copy()
    previous = result[0].copy()
    for i, pose in enumerate(result):
        delta = pose[..., :2] - previous[..., :2]
        rate = .1 + .9 / (np.exp(3 - np.abs(delta) * 600) + 1)
        rate[pose[..., 2] < .3] = .1
        rate[:, 14:16] = 1
        result[i, ..., :2] = previous[..., :2] + delta * rate
        previous = result[i].copy()
    return result


def _landmark_boxes(points, width, height):
    points = points.copy()
    base = _window(15, len(points), 5)
    for i in range(5):
        length = _window(base + i * 4, len(points), 5)
        points = savgol_filter(points, length, min(2, length-1), axis=0, mode='mirror')
    for length in (17, 7):
        length = min(length, len(points))
        if length > 1:
            points = np.asarray(_mean(points, length), dtype=points.dtype)
    top = (points[:, 0] + points[:, 1]) / 2
    bottom = (points[:, 2] + points[:, 3]) / 2
    forehead = top + 1.1 * (top - bottom)
    points = np.concatenate([points, forehead[:, None]], axis=1)
    boxes = []
    for face in points:
        # Multiply each float32 channel independently, as in the native cropper.
        x, y = np.clip(face[:, 0], 0, 1) * width, np.clip(face[:, 1], 0, 1) * height
        x1, y1, x2, y2 = float(x.min()), float(y.min()), float(x.max()), float(y.max())
        side = min(max(x2-x1, y2-y1) * 1.45, width, height)
        left = np.clip((x1+x2-side)/2, 0, width-side)
        upper = np.clip((y1+y2-side)/2, 0, height-side)
        boxes.append([float(left), float(upper), float(left+side), float(upper+side)])
    if len(boxes) > 1:
        boxes = _mean(boxes, min(7, len(boxes)))
        for box in boxes:
            w, h = box[2]-box[0], box[3]-box[1]
            if w != h:
                side = max(w, h)
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                box[:] = [cx-side/2, cy-side/2, cx+side/2, cy+side/2]
    return boxes


def _smooth_boxes(boxes, width, height):
    boxes = _mean(_mean(boxes, 5), 11)
    boxes = _mean(_mean(_sg(boxes), 7), 3)
    sides = [min(box[2]-box[0], box[3]-box[1]) for box in boxes]
    sides = _mean(_sg(_mean(sides, 11)), 7)
    squared = []
    for box, side in zip(boxes, sides):
        x, y = max(0., float(np.float32(box[0]))), max(0., float(np.float32(box[1])))
        side = min(float(np.asarray(side).reshape(-1)[0]), height-y, width-x)
        squared.append(np.array([x, y, x+side, y+side], dtype=np.float32))
    for length in (11, 7, 3):
        squared = _mean(squared, length)
    result = []
    for x1, y1, x2, y2 in squared:
        x1 = max(0, min(width-1, int(np.floor(x1))))
        y1 = max(0, min(height-1, int(np.floor(y1))))
        x2 = max(x1+1, min(width, int(np.ceil(x2))))
        y2 = max(y1+1, min(height, int(np.ceil(y2))))
        result.append([x1, y1, x2, y2])
    return result


def _adaptive_boxes(boxes, width, height):
    """Choose fixed, fixed-size moving, or dynamic crop geometry."""
    array = np.asarray(boxes)
    union = np.r_[array[:, :2].min(axis=0), array[:, 2:].max(axis=0)]
    tolerance = (union[2:] - union[:2]) * .16
    fixed = np.all(array[:, :2] - union[:2] <= tolerance) and np.all(union[2:] - array[:, 2:] <= tolerance)
    sides = array[:, 2] - array[:, 0]
    fixed_size = np.all(np.abs(sides - sides.mean()) <= sides.mean() * .1)
    if not fixed and not fixed_size:
        return _smooth_boxes(boxes, width, height)
    array = np.asarray(_mean(boxes, 5))
    if fixed:
        union = np.r_[array[:, :2].min(axis=0), array[:, 2:].max(axis=0)]
        side = min(max(union[2:] - union[:2]), width, height)
        centers = np.tile((union[:2] + union[2:]) / 2, (len(array), 1))
    else:
        side = min(float((array[:, 2] - array[:, 0]).max()), width, height)
        centers = _mean(_mean(_mean(_sg(_mean((array[:, :2] + array[:, 2:]) / 2, 15)), 11), 7), 3)
    result = []
    for center in centers:
        x, y = np.clip(center - side/2, (0, 0), (width-side, height-side))
        result.append([int(np.floor(x)), int(np.floor(y)), int(np.ceil(x+side)), int(np.ceil(y+side))])
    return result


def shot_local_boxes(keypoints, width, height, cuts=(), *, adaptive=False):
    """1.45x square framing; every temporal filter restarts at cuts.

    `cuts` are zero-based source-frame indices of each new shot. These reset
    only geometry smoothing, never the model's reference/history/decoder.
    `adaptive=True` selects the current fixed/fixed-size/dynamic modes per shot;
    False retains the earlier always-dynamic Python helper behavior.
    Integer rounding may make the two bbox sides differ by one pixel.
    """
    raw = np.asarray(keypoints, dtype=np.float32)
    if raw.ndim != 4 or raw.shape[1:] != (1, 134, 3) or len(raw) == 0:
        raise ValueError('Expected nonempty DWPose [frames, 1, 134, 3]')
    if not np.isfinite(raw).all():
        raise ValueError('DWPose track contains non-finite values')
    cuts = list(cuts)
    if cuts != sorted(set(cuts)) or any(c <= 0 or c >= len(raw) for c in cuts):
        raise ValueError('Cuts must be increasing frame indices inside the source clip')
    boundaries = [0, *cuts, len(raw)]
    boxes = []
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        poses = _pose_filter(raw[start:stop])
        initial = _landmark_boxes(poses[:, 0, FACE_INDEX, :2], width, height)
        boxes.extend((_adaptive_boxes if adaptive else _smooth_boxes)(initial, width, height))
    return np.asarray(boxes, dtype=np.int64)


def detect_cuts(frames, threshold=20.):
    """Demo shot heuristic: adjacent grayscale 240x135 mean difference > 20.

    Review the returned indices for flashes/fast motion; explicit cuts can be
    passed to shot_local_boxes instead. `frames` are uint8 RGB arrays.
    """
    import cv2
    previous, cuts = None, []
    for i, rgb in enumerate(frames):
        gray = cv2.resize(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY), (240, 135)).astype(np.float32)
        if previous is not None and np.abs(gray-previous).mean() > threshold:
            cuts.append(i)
        previous = gray
    return cuts


def crop_frame(rgb, box):
    from PIL import Image
    return np.asarray(Image.fromarray(rgb).crop(tuple(box)).resize((512, 512), Image.Resampling.BILINEAR))
