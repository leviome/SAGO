"""Text boxes and independent SAM2 candidate states, without GPU weights."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from backend.engine import Engine
from backend.grounding import candidate_boxes


class GroundingTest(unittest.TestCase):
    def test_xyxy_nms_scores_and_invalid_boxes(self):
        boxes = [[.5, .5, .8, .6], [.5, .5, .79, .59], [.9, .9, .1, .1],
                 [.1, .1, -.1, .1], [float('nan'), .5, .2, .2]]
        result = candidate_boxes(boxes, [.9, .8, .7, .99, .99])
        self.assertEqual(len(result), 2)
        np.testing.assert_allclose(result[0]['box'], [[.1, .2], [.9, .8]])
        self.assertEqual(result[0]['score'], .9)

    def test_foreground_region_and_background_refinement(self):
        boxes = [[.25, .5, .4, .6], [.75, .5, .4, .6]]
        self.assertEqual(len(candidate_boxes(boxes, [.9, .8], foreground=[(.25, .5)])), 1)
        self.assertEqual(candidate_boxes(boxes, [.9, .8], foreground=[(.5, .1)]), [])
        limited = candidate_boxes(boxes, [.9, .8], region=[[.1, .3], [.5, .6]])
        np.testing.assert_allclose(limited[0]['box'], [[.1, .3], [.45, .6]])
        self.assertEqual(candidate_boxes(boxes, [.1, .2]), [])

    def fixture(self, folder):
        engine = Engine('unused.pt', folder)
        engine.torch = torch
        engine.capture = dict(id='test', version=0, rgb=np.zeros((4, 6, 3), np.uint8), encoded=None)
        engine.load_model = Mock(return_value={'model_load_seconds': 0.})
        engine.image_url = lambda rgb: 'image'
        engine.grounding = Mock()
        engine.grounding.detect.return_value = [dict(box=[[0., 0.], [.4, 1.]], score=.9),
                                               dict(box=[[.6, 0.], [1., 1.]], score=.8)]
        calls = []
        def decode(*args):
            value = len(calls)+1
            calls.append(value)
            return (torch.full((1, 4, 2, 2), float(value)),
                    torch.full((1, 4, 3), float(value)), torch.tensor([[float(value)]]))
        engine.tracker = SimpleNamespace(image_encoder=Mock(return_value=[torch.zeros(1)]),
            prompt_encoder=Mock(return_value='prompt'), decode_mask=decode,
            make_hires_mask_uint8=lambda *args, **kw: torch.ones((4, 6), dtype=torch.uint8))
        return engine

    def test_multiple_detections_keep_matching_prediction_pointer_and_score(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(torch.cuda, 'synchronize'):
            engine = self.fixture(folder)
            result = engine.preview(dict(capture='test', model='sam2', text='toy'))
            self.assertEqual(len(result['masks']), 8)
            self.assertEqual(result['candidate_labels'][4], '目标 2 · Level 0')
            self.assertEqual(result['scores'], [.9]*4+[.8]*4)
            for i, candidate in enumerate(engine.capture['sam2_candidates']):
                value = 1. if i < 4 else 2.
                self.assertEqual(candidate['score'].item(), value)
                self.assertTrue((candidate['preds'] == value).all())
                self.assertTrue((candidate['pointers'] == value).all())
            engine.preview(dict(capture='test', model='sam2', foreground=[[1, 1]]))
            self.assertEqual(len(engine.capture['sam2_candidates']), 4)
            engine.tracker.image_encoder.assert_called_once()

    def test_empty_detection_invalidates_previous_masks(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(torch.cuda, 'synchronize'):
            engine = self.fixture(folder)
            engine.capture['masks'] = ['stale']
            engine.grounding.detect.return_value = []
            with self.assertRaisesRegex(ValueError, '未找到匹配物体'):
                engine.preview(dict(capture='test', model='sam2', text='unknown'))
            self.assertIsNone(engine.capture['masks'])
