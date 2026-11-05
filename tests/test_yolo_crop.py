import unittest
import numpy as np
from crop_yolo import boxes_from_detections, scale_square, prepare_yolo_crops


class YoloCropTests(unittest.TestCase):
    def test_edge_translation_and_limit(self):
        np.testing.assert_equal(scale_square([-20, 20, 60, 100], .75, 160, 160), [0, 30, 60, 90])
        box = scale_square([-100, -100, 200, 200], 1, 100, 80)
        self.assertEqual(box[2]-box[0], 80)
        with self.assertRaises(ValueError):
            scale_square([0, 0, 10, 10], float('nan'), 100, 100)

    def test_invalid_preserves_length_and_reacquires(self):
        face = [30, 30, 60, 60, .9]
        boxes, valid = boxes_from_detections([[face], [], [face]], [], 200, 200)
        self.assertEqual(boxes.shape, (3, 4))
        self.assertEqual(valid.tolist(), [True, False, True])
        np.testing.assert_equal(boxes[1], 0)
        np.testing.assert_equal(boxes[0], boxes[2])

    def test_nonoverlap_locks_until_cut(self):
        a, b = [20, 20, 40, 40], [140, 140, 160, 160]
        boxes, valid = boxes_from_detections([[a], [b], [a], [b]], [3], 200, 200)
        self.assertEqual(valid.tolist(), [True, False, False, True])
        self.assertTrue(np.all(boxes[3][:2] > 100))

    def test_cut_resets_adaptive_geometry(self):
        a, b = [20, 20, 40, 40], [140, 140, 160, 160]
        boxes, _ = boxes_from_detections([[a]]*4+[[b]]*4, [4], 200, 200)
        first, _ = boxes_from_detections([[a]]*4, [], 200, 200)
        second, _ = boxes_from_detections([[b]]*4, [], 200, 200)
        np.testing.assert_equal(boxes, np.concatenate([first, second]))

    def test_empty_sequence(self):
        with self.assertRaises(ValueError):
            prepare_yolo_crops([], 'missing.pt')


if __name__ == '__main__':
    unittest.main()
