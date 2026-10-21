"""Physical display resolution, supported sizes and render invalidation."""
import tempfile
import unittest
from unittest.mock import Mock
from backend.engine import Engine
from desktop.render_quality import target_size

class RenderQualityTest(unittest.TestCase):
    def test_auto_uses_physical_pixels_rounds_up_and_caps(self):
        self.assertEqual(target_size(1022,752),1024)
        self.assertEqual(target_size(1022,752,2),2048)
        self.assertEqual(target_size(700,1100,1.5),1664)
        self.assertEqual(target_size(3840,2160,2),2560)
        self.assertEqual(target_size(400,300),512)
        self.assertEqual(target_size(1022,752,2,1920),1920)
    def test_extended_sizes_and_invalid_values(self):
        for size in (512,768,1024,1088,1536,1920,2560):self.assertEqual(Engine.valid_render_size(size),size)
        for size in (0,511,2561,True,768.5,'1920',None):
            with self.assertRaises(ValueError):Engine.valid_render_size(size)
    def test_frame_changes_size_once_then_keeps_idle_capture_valid(self):
        with tempfile.TemporaryDirectory() as folder:
            engine=Engine('unused.pt',folder);engine.view={'size':768,'aspect':1.5}
            engine.frame=Mock(return_value={})
            engine.execute('frame',{'render_size':1920,'aspect':2},None,None)
            self.assertEqual(Engine.dimensions(engine.view),(1920,960))
            self.assertEqual(engine.scene_version,1)
            engine.capture={'frozen':True}
            engine.execute('frame',{'render_size':1920,'aspect':2},None,None)
            self.assertEqual(engine.scene_version,1);self.assertIsNotNone(engine.capture)
