"""Inference-only color matching and soft-mask paste, without a private SDK."""
import numpy as np


def blend_head(source_head, generated_head, alpha):
    """Per-channel histogram matching against the source crop, then soft blend.

    Inputs share the same geometry. Alpha is a caller-supplied HxW soft mask
    in [0,1]. No segmentation weights or private mask assets are bundled.
    """
    from skimage.exposure import match_histograms
    if source_head.shape != generated_head.shape or alpha.shape != source_head.shape[:2]:
        raise ValueError('Source, generated crop and mask must share spatial dimensions')
    if not np.isfinite(alpha).all() or alpha.min() < 0 or alpha.max() > 1:
        raise ValueError('Soft-mask alpha must be finite and in [0,1]')
    matched = np.stack([match_histograms(generated_head[..., c], source_head[..., c])
                        for c in range(3)], axis=-1).astype(np.float32)
    alpha = alpha[..., None]
    return np.clip(matched * alpha + source_head.astype(np.float32) * (1-alpha), 0, 255).astype(np.uint8)


def paste_head(scene_rgb, source_head, generated_head, box, alpha):
    """Paste to the corresponding source bbox, preserving pixels outside it.

    The CLI reduces 2x model output back to RGB512 before calling this function.
    Source, prediction and mask share that geometry; the scene is not enlarged.
    """
    import cv2
    x1, y1, x2, y2 = map(int, box)
    h, w = scene_rgb.shape[:2]
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError('Paste box must lie inside the source frame')
    result = scene_rgb.copy()
    blended = blend_head(source_head, generated_head, alpha)
    result[y1:y2, x1:x2] = cv2.resize(blended, (x2-x1, y2-y1), interpolation=cv2.INTER_LINEAR)
    return result
