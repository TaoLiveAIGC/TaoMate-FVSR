"""Align once, enhance in the same coordinates, inverse once onto the input."""
import importlib
import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import torch


def prepare_reference(first_frame, *, repo, weights, detector_weights, parser_weights,
                      size=512, passes=1, seed=42, device='cpu'):
    """GPEN official geometry; no background SR and no second alignment.

    Supply the actual first model crop, or a composition-preserving edit of it.
    Detector/parser/generator weights are local assets, never downloaded here.
    """
    import cv2
    if first_frame.shape[0] != first_frame.shape[1]:
        raise ValueError('Edit the square model crop; do not pass a full-scene reference')
    if passes not in (1, 2):
        raise ValueError('GPEN passes must be 1 or 2')
    if size not in (512, 1024):
        raise ValueError('GPEN reference size must be 512 or 1024')
    repo = Path(repo).resolve()
    sys.path.insert(0, str(repo))
    try:
        importlib.import_module('__init_paths')
        from face_detect.retinaface_detection import RetinaFaceDetection
        from face_parse.face_parsing import FaceParse
        from face_model.face_gan import FaceGAN
        from align_faces import get_reference_facial_points, warp_and_crop_face
    finally:
        sys.path.remove(str(repo))

    class LocalFaceGAN(FaceGAN):
        def load_model(self, channel_multiplier=2, narrow=1):
            self.mfile = str(Path(weights).resolve())
            return super().load_model(channel_multiplier, narrow)

    # Upstream detector/parser use fixed relative filenames. Stage symlinks in
    # a private temporary directory rather than changing the GPEN checkout.
    with tempfile.TemporaryDirectory(prefix='livevsr-gpen-') as tmp:
        directory = Path(tmp) / 'weights'
        directory.mkdir()
        for name, path in [('RetinaFace-R50.pth', detector_weights), ('ParseNet-latest.pth', parser_weights)]:
            (directory / name).symlink_to(Path(path).resolve())
        detector = RetinaFaceDetection(tmp, device=device)
        parser = FaceParse(tmp, device=device)
        generator = LocalFaceGAN(base_dir=str(repo), in_size=size, out_size=size,
            model=f'GPEN-BFR-{size}', channel_multiplier=2, narrow=1, key=None, is_norm=True, device=device)
    rgb = np.asarray(Image.fromarray(first_frame).resize((size, size), Image.Resampling.LANCZOS))
    bgr = np.ascontiguousarray(rgb[..., ::-1])
    with torch.inference_mode():
        boxes, landmarks = detector.detect(bgr)
        faces = [(box, points) for box, points in zip(boxes, landmarks) if box[4] >= .9]
        if len(faces) != 1:
            raise ValueError(f'Reference must contain one detected face; found {len(faces)}. Supply --reference to use a final prepared image.')
        box, points = faces[0]
        template = get_reference_facial_points((size, size), .25, (0, 0), True)
        enhanced, inverse = warp_and_crop_face(bgr, points.reshape(2, 5),
                                              reference_pts=template, crop_size=(size, size))
        for _ in range(passes):
            torch.manual_seed(seed)
            tensor = generator.img2tensor(enhanced)
            output, _ = generator.model(tensor)
            enhanced = generator.tensor2img(output)
        mask = parser.process(enhanced)[0] / 255.
        # GPEN's native mask_postprocess, without importing its background-SR stack.
        mask[:26] = mask[-26:] = 0
        mask[:, :26] = mask[:, -26:] = 0
        mask = cv2.GaussianBlur(cv2.GaussianBlur(mask, (101, 101), 4), (101, 101), 4).astype(np.float32)
        alpha = cv2.warpAffine(cv2.resize(mask, (size, size)), inverse, (size, size), flags=3)[..., None]
        if min(box[2] - box[0], box[3] - box[1]) < 100:
            kernel = np.array([[.0625, .125, .0625], [.125, .25, .125], [.0625, .125, .0625]], np.float32)
            enhanced = cv2.filter2D(enhanced, -1, kernel)
        warped = cv2.warpAffine(enhanced, inverse, (size, size), flags=3)
        result = cv2.convertScaleAbs(bgr * (1 - alpha) + warped * alpha)
    logging.info('GPEN-BFR-%d reference: %dx%d, %d pass(es)', size,
                 result.shape[1], result.shape[0], passes)
    return np.ascontiguousarray(result[..., ::-1])
