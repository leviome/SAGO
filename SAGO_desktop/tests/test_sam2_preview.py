"""Spatial SAM2 prompts retain candidate state and reuse image encoding."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from backend.engine import Engine


class Sam2PreviewTest(unittest.TestCase):
    def test_point_box_prompts_and_repeated_preview_keep_candidate_state(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(torch.cuda, 'synchronize'):
            engine=Engine('unused.pt',folder);engine.torch=torch
            engine.capture=dict(id='test',version=0,rgb=np.zeros((4,6,3),np.uint8),encoded=None)
            engine.load_model=Mock(return_value={'model_load_seconds':0.})
            engine.image_url=lambda rgb:'image'
            predictions=torch.arange(16).reshape(1,4,2,2).float()
            pointers=torch.arange(12).reshape(1,4,3).float();score=torch.tensor([[.8]])
            engine.tracker=SimpleNamespace(image_encoder=Mock(return_value=[torch.zeros(1)]),
                prompt_encoder=Mock(return_value='prompt'),decode_mask=Mock(return_value=(predictions,pointers,score)),
                make_hires_mask_uint8=lambda *args,**kw:torch.ones((4,6),dtype=torch.uint8))
            result=engine.preview(dict(capture='test',foreground=[[3,2]],background=[[1,1]]))
            self.assertEqual(result['candidate_labels'],['Level 0','Level 1','Level 2','Level 3'])
            self.assertEqual(len(result['masks']),4)
            engine.tracker.prompt_encoder.assert_called_with(box=[],fg_pts=[(.5,.5)],bg_pts=[(1/6,1/4)])
            for index,candidate in enumerate(engine.capture['sam2_candidates']):
                self.assertTrue(torch.equal(candidate['preds'],predictions[:,[index]]))
                self.assertTrue(torch.equal(candidate['pointers'],pointers[:,[index]]))
                self.assertIs(candidate['score'],score)
            engine.preview(dict(capture='test',box=[[0,0],[5,3]]))
            engine.tracker.image_encoder.assert_called_once()
            engine.tracker.prompt_encoder.assert_called_with(box=[[(0.,0.),(5/6,3/4)]],fg_pts=[],bg_pts=[])
            with self.assertRaises(ValueError):engine.preview(dict(capture='test'))

    def test_removed_prompts_are_rejected_before_capture_or_gpu(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=Engine('unused.pt',folder);engine.initialize=Mock()
            for data in ({'model':'sam3'},{'model':'sam3.1'},{'text':'chair'}):
                with self.assertRaises(ValueError):engine.preview(data)
            engine.initialize.assert_not_called()
