"""Inference-only YOLOX/RTMW extraction in the DWPose 134-point order.

Can run independently in a separate OpenMMLab inference environment.
"""
import argparse
from pathlib import Path

import numpy as np

FILES = {
    'det_config': 'yolox_l_8xb8-300e_coco.py',
    'det_weights': 'yolox_l_8x8_300e_coco_20211126_140236-d3bd2b23.pth',
    'pose_config': 'rtmw-x_8xb320-270e_cocktail14-384x288.py',
    'pose_weights': 'rtmw-x_simcc-cocktail14_pt-ucoco_270e-384x288-f840f204_20231122.pth',
}


def dwpose_order(points, scores, width, height):
    """Match the current RTMW adapter, including its score normalization."""
    info = np.concatenate((points, scores[..., None]), axis=-1)
    neck = info[:, [5, 6]].mean(axis=1)
    neck[:, 2] = ((scores[:, 5] > .5) & (scores[:, 6] > .5)) * 10
    info = np.insert(info, 17, neck, axis=1)
    src = [17, 6, 8, 10, 7, 9, 12, 14, 16, 13, 15, 2, 1, 4, 3]
    dst = [1, 2, 3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 17]
    info[:, dst] = info[:, src]
    info = info[:1].astype(np.float32)
    info[..., 0] /= width
    info[..., 1] /= height
    info[..., 2] /= 10
    return info


def extract(paths, model_dir, device):
    import cv2
    import torch
    from mmdet.apis import init_detector, inference_detector
    from mmpose.apis import init_model, inference_topdown
    from mmpose.evaluation.functional import nms
    from mmpose.structures import merge_data_samples
    from mmpose.utils import adapt_mmdet_pipeline

    torch.set_num_threads(2)
    cv2.setNumThreads(1)
    assets = {key: str(Path(model_dir) / name) for key, name in FILES.items()}
    detector = init_detector(assets['det_config'], assets['det_weights'], device=device)
    detector.cfg = adapt_mmdet_pipeline(detector.cfg)
    pose = init_model(assets['pose_config'], assets['pose_weights'], device=device)
    track = np.empty((len(paths), 1, 134, 3), dtype=np.float32)
    with torch.inference_mode():
        for i, path in enumerate(paths):
            bgr = cv2.imread(str(path))
            height, width = bgr.shape[:2]
            pred = inference_detector(detector, bgr).pred_instances.cpu().numpy()
            boxes = np.concatenate((pred.bboxes, pred.scores[:, None]), axis=1)
            boxes = boxes[(pred.labels == 0) & (pred.scores > .5)]
            boxes = boxes[nms(boxes, .7)]
            if len(boxes):
                best = int(np.argmax(boxes[:, 4]))
                boxes = boxes[best:best+1, :4]
                results = inference_topdown(pose, bgr, boxes)
            else:
                results = inference_topdown(pose, bgr)
            result = merge_data_samples(results).pred_instances
            points = result.get('transformed_keypoints', result.keypoints)
            track[i] = dwpose_order(points, result.keypoint_scores, width, height)
            if (i+1) % 100 == 0 or i+1 == len(paths):
                print(f'DWPose: {i+1}/{len(paths)} native frames', flush=True)
    return track


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--frames', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cpu')
    a = p.parse_args()
    np.save(a.output, extract(sorted(a.frames.glob('*.png')), a.model_dir, a.device))
