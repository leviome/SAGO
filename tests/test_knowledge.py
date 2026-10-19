"""Ground persistence, scene identity, atomic writes and reload behavior."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from backend.engine import Engine
from backend.ground import ground_plane, flip_plane
from backend.knowledge import SceneKnowledge


def plane():
    return ground_plane([[0, 0, 2], [1, 0, 2], [0, 1, 2]], [0, 0, 3], [0, 0, 1])


class KnowledgeTest(unittest.TestCase):
    def engine(self, root):
        root = Path(root)
        ply = root/'scene.ply'
        ply.touch(exist_ok=True)
        engine = Engine('unused.pt', root/'outputs')
        engine.initialize = Mock()
        engine.torch = torch
        model = SimpleNamespace(get_xyz=torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]), load_ply=Mock())
        engine.GaussianModel = lambda *a: model
        engine.camera = lambda: SimpleNamespace(world_view_transform=torch.eye(4))
        engine.utils = SimpleNamespace(params2sphere=lambda *a: (2., .1, .2))
        engine.frame = lambda *a: engine.snapshot()
        engine.load(dict(path=str(ply)))
        return engine

    def test_standard_scene_and_flat_ply_locations(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            standard = SceneKnowledge(root/'point_cloud/iteration_30000/point_cloud.ply')
            self.assertEqual(standard.path, root/'SAGO_knowledge.json')
            self.assertEqual(standard.key, 'point_cloud/iteration_30000/point_cloud.ply')
            self.assertEqual(SceneKnowledge(root/'point_cloud/point_cloud.ply').path, standard.path)
            self.assertEqual(SceneKnowledge(root/'thing.ply').path, standard.path)

    def test_roundtrip_merge_and_two_ply_isolation(self):
        with TemporaryDirectory() as folder:
            first, second = SceneKnowledge(Path(folder)/'one.ply'), SceneKnowledge(Path(folder)/'two.ply')
            first.path.write_text(json.dumps(dict(objects=['keep'], scenes={'one.ply': {'label': 'keep too'}})))
            first.save_ground(plane())
            second.save_ground(flip_plane(plane()))
            self.assertEqual(json.loads(first.path.read_text())['objects'], ['keep'])
            self.assertEqual(json.loads(first.path.read_text())['scenes']['one.ply']['label'], 'keep too')
            np.testing.assert_allclose(first.load_ground()['normal'], [0, 0, 1])
            np.testing.assert_allclose(second.load_ground()['normal'], [0, 0, -1])
            self.assertIsNone(SceneKnowledge(Path(folder)/'unknown.ply').load_ground())

    def test_accept_flip_and_fresh_engine_automatically_restores(self):
        with TemporaryDirectory() as folder:
            engine = self.engine(folder)
            engine.ground_capture = dict(id='test', version=engine.scene_version, plane=plane())
            engine.ground_action(dict(operation='apply', capture='test'))
            self.assertTrue(engine.knowledge.path.is_file())
            restored = self.engine(folder)
            self.assertTrue(restored.ground_restored)
            np.testing.assert_allclose(restored.view['up'], [0, 0, 1])
            restored.ground_action(dict(operation='flip'))
            reloaded = self.engine(folder)
            np.testing.assert_allclose(reloaded.view['up'], [0, 0, -1])
            self.assertEqual(reloaded.ground_plane['offset'], 2.)

    def test_save_failure_leaves_existing_file_view_and_draft_unchanged(self):
        with TemporaryDirectory() as folder:
            engine = self.engine(folder)
            engine.knowledge.save_ground(plane())
            original = engine.knowledge.path.read_bytes()
            previous_view = dict(engine.view)
            draft = dict(id='draft', version=engine.scene_version, plane=flip_plane(plane()))
            engine.ground_capture = draft
            with patch('backend.knowledge.os.replace', side_effect=PermissionError('read only')):
                with self.assertRaisesRegex(ValueError, '地面未应用.*无法保存'):
                    engine.ground_action(dict(operation='apply', capture='draft'))
            self.assertEqual(engine.view, previous_view)
            self.assertIs(engine.ground_capture, draft)
            self.assertEqual(engine.knowledge.path.read_bytes(), original)
            self.assertFalse(list(Path(folder).glob('.SAGO_knowledge-*.tmp')))

    def test_corrupt_file_is_reported_and_never_overwritten(self):
        with TemporaryDirectory() as folder:
            path = Path(folder)/'SAGO_knowledge.json'
            path.write_text('{broken')
            engine = self.engine(folder)
            self.assertIsNone(engine.ground_plane)
            self.assertIn('无法读取地面知识', engine.knowledge_warning)
            with self.assertRaisesRegex(ValueError, '地面未应用'):
                engine.orient_to_ground(plane())
            self.assertEqual(path.read_text(), '{broken')

    def test_invalid_or_inconsistent_plane_is_not_restored(self):
        with TemporaryDirectory() as folder:
            knowledge = SceneKnowledge(Path(folder)/'scene.ply')
            knowledge.save_ground(plane())
            valid = json.loads(knowledge.path.read_text())
            for value in ([0, 1, 0], [0, 0, 0], [float('nan'), 0, 1]):
                record = json.loads(json.dumps(valid))
                record['scenes']['scene.ply']['ground_plane']['normal'] = value
                knowledge.path.write_text(json.dumps(record))
                with self.assertRaises(ValueError):
                    knowledge.load_ground()
