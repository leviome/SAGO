"""Semantic persistence, orbit consistency, cache correctness and ranked retrieval."""
import copy
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from backend.engine import Cancelled
from backend.gif_export import orbit_geometry
from backend.knowledge import SceneKnowledge
from backend.semantic import MODEL_INFO, normalized, semantic_action
import test_gif_export as gif_fixtures


class SemanticTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine, self.renders = gif_fixtures.GifExportTest().fixture(self.tmp.name)
        e = self.engine
        e.capture = None
        e.instance_ids = np.array([1, 1, 2], dtype=np.int32)
        e.objects.append(dict(id='second', number=2, name='Second', hidden=False))
        e.object_mask = lambda number: torch.from_numpy(e.labels() == number)
        e.next_object_id = 3
        e.gaussian_fingerprint = 'original-scene-order'
        e.knowledge = SceneKnowledge(Path(self.tmp.name)/'scene.ply')
        e.ids_dirty = True
        self.cancel = threading.Event()
        self.progress = Mock()
        self.clip = Mock()
        count = [0]
        def features(images):
            value = np.zeros((4, 512), dtype=np.float32)
            value[:, count[0] % 2] = 1
            count[0] += 1
            return value
        self.clip.images.side_effect = features
        text = np.zeros(512, dtype=np.float32); text[1] = 1
        self.clip.text.return_value = text
        e._clip_encoder = self.clip

    def call(self, operation, **data):
        return semantic_action(self.engine, dict(operation=operation, **data), self.progress, self.cancel)

    def read(self):
        return self.engine.knowledge.read()['scenes'][self.engine.knowledge.key]

    def test_four_views_match_gif_geometry_and_preserve_camera_and_selection(self):
        e = self.engine
        view = copy.deepcopy(e.view); selection = e.selection.clone()
        self.call('build')
        self.assertEqual(len(self.renders), 8)  # Includes hidden ID 1.
        center, up, radius = orbit_geometry(e.model.get_xyz[:2].numpy(), e.model.get_scaling[:2].numpy(), e.view['up'], e.view['fov'])
        cameras = [item[0] for item in self.renders[:4]]
        for i, camera in enumerate(cameras):
            self.assertAlmostEqual(camera.arguments[0], radius)
            self.assertAlmostEqual(camera.arguments[1], np.radians(view['yaw']) + i*np.pi/2)
            self.assertAlmostEqual(camera.arguments[2], np.radians(60))
            np.testing.assert_allclose(camera.arguments[3], center)
            self.assertEqual(camera.width, 512)
        self.assertTrue(all(item[2]['bg'] == 'white' for item in self.renders))
        self.assertEqual(e.view, view);self.assertTrue(torch.equal(e.selection, selection))
        self.assertEqual(e.render_mode, 'depth');self.assertTrue(e.objects[0]['hidden'])

    def test_atomic_scene_persistence_and_reopen_without_reencoding(self):
        e = self.engine
        e.knowledge.save_record(unrelated={'preserve': True})
        result = self.call('build')
        record = self.read()
        self.assertEqual(record['semantic_index']['encoder'], MODEL_INFO)
        self.assertEqual(record['unrelated'], {'preserve': True})
        self.assertEqual(record['segmentation']['gaussian_ids'], [1,1,2])
        self.assertFalse(e.ids_dirty);self.assertEqual(result['indexed'], 2)
        e._clip_encoder = None
        with patch('backend.semantic.ClipEncoder', return_value=self.clip) as loader:
            result = self.call('search', text='a second object')
            self.assertEqual(result['results'][0]['number'], 2)
            loader.assert_called_once()
        self.assertEqual(self.clip.images.call_count, 2)
        self.assertEqual(self.call('build')['updated'], 0)

    def test_changed_masks_are_excluded_then_rebuilt(self):
        self.call('build')
        self.engine.instance_ids[:] = [1,2,2]
        with self.assertRaisesRegex(ValueError, 'outdated'):
            self.call('search', text='object')
        self.assertEqual(self.call('build')['updated'], 2)
        self.assertEqual(self.call('search', text='object')['stale_ids'], [])

    def test_partial_stale_and_deleted_ids_never_retrieved(self):
        self.call('build')
        self.engine.instance_ids[:] = [1,0,2]
        result = self.call('search', text='object')
        self.assertEqual(result['stale_ids'], [1])
        self.assertEqual([r['number'] for r in result['results']], [2])
        self.engine.objects.pop(0)
        self.assertEqual([r['number'] for r in self.call('search', text='object')['results']], [2])

    def test_renumber_and_rename_keep_features_and_return_current_metadata(self):
        self.call('build')
        self.engine.instance_ids[:] = [4,4,2]
        self.engine.objects[0].update(number=4, name='New name')
        self.engine.next_object_id = 5
        result = self.call('search', text='object')
        self.assertEqual(result['results'][1]['number'], 4)
        self.assertEqual(result['results'][1]['name'], 'New name')
        self.assertEqual(self.call('build')['updated'], 0)

    def test_maximum_view_score_not_average_and_normalization(self):
        self.call('build')
        record = self.read()['semantic_index']
        features = np.zeros((4,512));features[:3,0] = 4;features[3,1] = 3
        record['objects']['test']['embeddings'] = features.tolist()
        self.engine.knowledge.save_record(semantic_index=record)
        scores = {r['number']:r['score'] for r in self.call('search', text='object')['results']}
        self.assertAlmostEqual(scores[1], 1)
        self.assertAlmostEqual(scores[2], 1)

    def test_cancelled_build_never_overwrites_previous_knowledge(self):
        self.call('build')
        before = self.engine.knowledge.path.read_bytes()
        self.engine.ids_dirty = True
        def encode(images):
            self.cancel.set()
            values = np.zeros((4,512));values[:,0] = 1
            return values
        self.clip.images.side_effect = encode
        with self.assertRaises(Cancelled):
            self.call('build', elevation_degrees=30)
        self.assertEqual(self.engine.knowledge.path.read_bytes(), before)
        self.assertTrue(self.engine.ids_dirty)

    def test_write_failure_keeps_old_record_and_dirty_flag(self):
        self.call('build');before=self.engine.knowledge.path.read_bytes();self.engine.ids_dirty=True
        with patch('backend.knowledge.os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):self.call('build')
        self.assertTrue(self.engine.ids_dirty)
        self.assertEqual(self.engine.knowledge.path.read_bytes(), before)
        self.assertFalse(list(Path(self.tmp.name).glob('.SAGO_knowledge-*.tmp')))

    def test_settings_changes_rebuild_and_up_change_invalidates_search(self):
        self.call('build')
        self.assertEqual(self.call('build', elevation_degrees=30)['updated'], 2)
        self.engine.view['up'] = [0,1,0]
        with self.assertRaisesRegex(ValueError, 'outdated'):self.call('search', text='object')

    def test_incompatible_model_and_invalid_features_are_not_searched(self):
        self.call('build');record=self.read()['semantic_index']
        record['objects']['test']['embeddings'][0][0] = float('inf')
        # JSON persistence forbids NaN/Infinity; use an invalid dimension instead.
        record['objects']['test']['embeddings'] = [[1.0]]
        self.engine.knowledge.save_record(semantic_index=record)
        self.assertEqual(self.call('search', text='object')['stale_ids'], [1])
        self.assertEqual(self.call('build')['updated'], 1)
        record=self.read()['semantic_index'];record['encoder']['revision']='other'
        self.engine.knowledge.save_record(semantic_index=record)
        with self.assertRaisesRegex(ValueError, 'Build semantic'):self.call('search', text='object')

    def test_invalid_requests_and_empty_objects_do_not_load_encoder(self):
        with self.assertRaises(ValueError):self.call('build', elevation_degrees=float('nan'))
        with self.assertRaises(ValueError):self.call('build', size=True)
        self.call('build')
        for text in ('', ' '*4, 'x'*513, None):
            with self.assertRaises(ValueError):self.call('search', text=text)
        self.engine.objects=[]
        with self.assertRaises(ValueError):self.call('build')
        for value in (np.zeros((4,512)), np.full(512,float('nan'))):
            with self.assertRaises(ValueError):normalized(value)

    def test_engine_dispatch(self):
        result=self.engine.execute('semantic',dict(operation='build'),self.progress,self.cancel)
        self.assertEqual(result['indexed'],2)


if __name__ == '__main__':unittest.main()
