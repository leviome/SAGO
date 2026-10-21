"""Adaptive backgrounds use unpadded foreground edges, independent of padding."""
import unittest
import cv2
import numpy as np
from backend.drone_background import choose_drone_background


class DroneBackgroundTest(unittest.TestCase):
    def test_inner_boundary_ignores_interior_and_outside_background(self):
        mask = np.zeros((25, 25), np.uint8)
        mask[2:23, 2:23] = 255
        image = np.full((25, 25, 3), 255, np.uint8)
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8)) > 0
        edge = (mask > 0) & ~inner
        image[edge] = 15
        before_image, before_mask = image.copy(), mask.copy()
        result = choose_drone_background(image, mask)
        self.assertEqual(result['color'], [1., 1., 1.])
        self.assertEqual(result['edge_pixels'], int(edge.sum()))
        self.assertAlmostEqual(result['edge_luminance'], 15 / 255)
        # Bright interior/outside must not outweigh a dark object perimeter.
        np.testing.assert_array_equal(image, before_image)
        np.testing.assert_array_equal(mask, before_mask)
        image[:] = 0
        image[edge] = 240
        self.assertEqual(choose_drone_background(image, mask)['color'], [0., 0., 0.])

    def test_rgb_luminance_threshold_and_original_mask(self):
        mask = np.zeros((9, 9), np.uint8);mask[3:6, 3:6] = 1
        image = np.full((9, 9, 3), 255, np.uint8);image[mask > 0] = 0
        self.assertEqual(choose_drone_background(image, mask)['color'], [1., 1., 1.])
        # A padded mask includes bright scene pixels, proving why raw masks matter.
        padded = cv2.dilate(mask, np.ones((3, 3), np.uint8))
        self.assertEqual(choose_drone_background(image, padded)['color'], [0., 0., 0.])
        for rgb, expected in (([127]*3, [1.]*3), ([128]*3, [0.]*3),
                              ([255, 0, 0], [1.]*3), ([0, 255, 0], [0.]*3)):
            image[:] = rgb
            self.assertEqual(choose_drone_background(image, mask)['color'], expected)

    def test_full_frame_thin_masks_holes_and_manual_override(self):
        image = np.zeros((7, 7, 3), np.uint8)
        full = np.ones((7, 7), bool)
        self.assertEqual(choose_drone_background(image, full)['edge_pixels'], 24)
        full[2:5, 2:5] = False
        self.assertGreater(choose_drone_background(image, full)['edge_pixels'], 24)
        tiny = np.zeros((7, 7), np.uint8);tiny[0, 0] = 255
        self.assertEqual(choose_drone_background(image, tiny)['edge_pixels'], 1)
        report = choose_drone_background(None, None, False, [.2, .4, .6])
        self.assertEqual(report['color'], [.2, .4, .6])
        self.assertFalse(report['enabled'])
        self.assertIsNone(report['edge_luminance'])
        for rgb, mask, enabled in ((image, np.zeros((7, 7)), True),
                                    (image, np.ones((2, 2)), True),
                                    (image / 255, tiny, True), (image, tiny, 'false')):
            with self.assertRaises(ValueError):choose_drone_background(rgb, mask, enabled)
