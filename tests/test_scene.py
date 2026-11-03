import tempfile
import unittest
from pathlib import Path


class SceneTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import numpy
            import scipy
            import cv2
            import skimage
        except ImportError:
            raise unittest.SkipTest('Scene tests require the scene extra')

    def test_all_crop_filters_are_shot_local(self):
        import numpy as np
        from livevsr.crop import shot_local_boxes
        rng = np.random.default_rng(42)
        track = rng.uniform(.25, .5, (30, 1, 134, 3)).astype(np.float32)
        track[..., 2] = .9
        track[17:, ..., :2] += .2
        whole = shot_local_boxes(track, 960, 540, [17])
        separate = np.concatenate([shot_local_boxes(track[:17], 960, 540),
                                   shot_local_boxes(track[17:], 960, 540)])
        np.testing.assert_array_equal(whole, separate)
        self.assertTrue(np.all(np.abs((whole[:, 2]-whole[:, 0])-(whole[:, 3]-whole[:, 1])) <= 1))

    def test_short_shots(self):
        import numpy as np
        from livevsr.crop import shot_local_boxes
        rng = np.random.default_rng(42)
        track = rng.uniform(.2, .7, (3, 1, 134, 3)).astype(np.float32)
        boxes = shot_local_boxes(track, 640, 480, [1])
        self.assertEqual(boxes.shape, (3, 4))
        self.assertTrue(np.all(boxes[:, 2:] > boxes[:, :2]))

    def test_detect_cuts(self):
        import numpy as np
        from livevsr.crop import detect_cuts
        dark = np.zeros((80, 120, 3), dtype=np.uint8)
        self.assertEqual(detect_cuts([dark, dark, dark+255, dark+255]), [2])

    def test_native_crop_and_png_input(self):
        import numpy as np
        from PIL import Image
        from livevsr.crop import crop_frame
        from livevsr.media import decode_video
        rgb = np.random.default_rng(3).integers(0, 256, (100, 120, 3), dtype=np.uint8)
        crop = crop_frame(rgb, [10, 20, 80, 90])
        with tempfile.TemporaryDirectory() as tmp:
            Image.fromarray(crop).save(Path(tmp) / '00000001.png')
            paths, w, h = decode_video(Path(tmp), None, 2)
            self.assertEqual((w, h), (1024, 1024))
            np.testing.assert_array_equal(np.asarray(Image.open(paths[0])), crop)

    def test_color_match_and_bbox_paste(self):
        import numpy as np
        from livevsr.composite import blend_head, paste_head
        source = np.full((32, 32, 3), 80, dtype=np.uint8)
        generated = np.full_like(source, 190)
        alpha = np.ones((32, 32), np.float32)
        np.testing.assert_array_equal(blend_head(source, generated, alpha), source)
        scene = np.zeros((64, 80, 3), np.uint8)
        result = paste_head(scene, source, generated, [10, 12, 42, 44], alpha)
        np.testing.assert_array_equal(result[12:44, 10:42], source)
        result[12:44, 10:42] = 0
        np.testing.assert_array_equal(result, scene)

    def test_native_fps_mapping(self):
        from fractions import Fraction
        import numpy as np
        from livevsr.scene import frame_indices
        indices = frame_indices(240, Fraction(24))
        self.assertEqual(len(indices), 250)
        np.testing.assert_array_equal(indices, np.arange(250) * 24 // 25)

    def test_adaptive_crop_does_not_cross_cuts(self):
        import numpy as np
        from livevsr.crop import shot_local_boxes
        rng = np.random.default_rng(42)
        track = rng.uniform(.25, .4, (20, 1, 134, 3)).astype(np.float32)
        track[10:, ..., :2] += .3
        whole = shot_local_boxes(track, 960, 540, [10], adaptive=True)
        separate = np.concatenate([shot_local_boxes(track[:10], 960, 540, adaptive=True),
                                   shot_local_boxes(track[10:], 960, 540, adaptive=True)])
        np.testing.assert_array_equal(whole, separate)

    def test_dwpose_order_and_score_contract(self):
        import numpy as np
        from livevsr.pose import dwpose_order
        points = np.arange(266, dtype=np.float32).reshape(1, 133, 2)
        scores = np.full((1, 133), .8, dtype=np.float32)
        result = dwpose_order(points, scores, 640, 480)
        self.assertEqual(result.shape, (1, 134, 3))
        np.testing.assert_allclose(result[0, 1, :2], points[0, [5, 6]].mean(0) / [640, 480])
        self.assertEqual(result[0, 1, 2], 1.)
        np.testing.assert_allclose(result[0, 24:92, :2], points[0, 23:91] / [640, 480])

    def test_two_x_returns_to_512_before_paste(self):
        import cv2
        import numpy as np
        from PIL import Image
        from livevsr.scene import composite, resize_head
        from livevsr.composite import paste_head
        rng = np.random.default_rng(7)
        original = rng.integers(0, 256, (600, 800, 3), dtype=np.uint8)
        source = rng.integers(0, 256, (512, 512, 3), dtype=np.uint8)
        prediction = rng.integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
        reduced = cv2.resize(prediction, (512, 512), interpolation=cv2.INTER_AREA)
        np.testing.assert_array_equal(resize_head(prediction), reduced)
        box = [120, 80, 520, 480]
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp) / name for name in ('scene.png', 'crop.png', 'mask.png')]
            for path, array in zip(paths, [original, source, np.full((512,512), 128, np.uint8)]):
                Image.fromarray(array).save(path)
            actual = next(composite([paths[0]], [paths[1]], [prediction], [box], paths[2]))
        expected = paste_head(original, source, reduced, box, np.full((512,512),128/255,np.float32))
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual[:80], original[:80])
        self.assertEqual(actual.shape, original.shape)


if __name__ == '__main__':
    unittest.main()
