"""Camera motion follows rendered axes and preserves segmentation captures."""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from backend.engine import Engine
from backend.navigation import translated_target

VIEW=dict(target=[1.,2.,3.],radius=4.,fov=90.,aspect=2.,yaw=0.,pitch=0.,up=[0.,-1.,0.],size=768)

class NavigationTest(unittest.TestCase):
    def test_walk_translates_target_without_zoom_or_rotation(self):
        view=deepcopy(VIEW);before=deepcopy(view)
        target=translated_target(view,np.eye(4),{'move':[.1,.2]})
        np.testing.assert_allclose(target,[1.4,2.,3.8])
        self.assertEqual(view,before)
    def test_right_drag_moves_against_screen_motion_with_correct_aspect(self):
        target=translated_target(VIEW,np.eye(4),{'pan':[.1,.2]})
        np.testing.assert_allclose(target,[-.6,.4,3.])
    def test_translation_follows_tilted_camera_not_world_up(self):
        axes=np.eye(4);axes[:3,:3]=[[0,0,-1],[0,1,0],[1,0,0]]
        np.testing.assert_allclose(translated_target(VIEW,axes,{'move':[.1,.2]}),[1.8,2.,2.6])
        np.testing.assert_allclose(translated_target(VIEW,axes,{'pan':[.1,0]}),[1.,2.,4.6])
    def test_invalid_motion_rejected(self):
        for value in ([1,2],[],[1,2,3],[float('nan'),0],[float('inf'),0],'bad'):
            for key in ('move','pan'):
                with self.assertRaises((ValueError,TypeError)):
                    translated_target(VIEW,np.eye(4),{key:value})
    def engine(self,folder):
        engine=Engine('unused.pt',folder);engine.view=deepcopy(VIEW)
        engine.require_scene=Mock();engine.frame=Mock(return_value={'ok':True})
        engine.camera=Mock(return_value=Mock(world_view_transform=torch.eye(4)))
        return engine
    def test_engine_updates_target_and_preserves_orientation(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=self.engine(folder)
            self.assertEqual(engine.update_view({'move':[0,.1]}),{'ok':True})
            np.testing.assert_allclose(engine.view['target'],[1,2,3.4])
            for key in ('radius','yaw','pitch','up'):self.assertEqual(engine.view[key],VIEW[key])
    def test_frozen_navigation_rejected_without_changing_view(self):
        with tempfile.TemporaryDirectory() as folder:
            for attr in ('capture','ground_capture','correction_pending'):
                engine=self.engine(folder);setattr(engine,attr,{'id':'frozen'})
                with self.assertRaisesRegex(ValueError,'Exit prompts'):
                    engine.update_view({'pan':[.1,0]})
                self.assertEqual(engine.view,VIEW);engine.frame.assert_not_called()
