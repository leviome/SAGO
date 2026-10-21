"""Dense instance IDs, scene alignment and atomic knowledge persistence."""
import copy
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch

from backend.knowledge import SceneKnowledge
import test_knowledge


class InstanceIdsTest(unittest.TestCase):
    def setUp(self):
        cuda = patch.object(torch.Tensor, 'cuda', lambda tensor: tensor)
        cuda.start()
        self.addCleanup(cuda.stop)

    def engine(self, folder):
        return test_knowledge.KnowledgeTest.engine(self, folder)

    def test_dense_vector_roundtrip_ground_merge_and_restore_visible(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            e.knowledge.save_ground(test_knowledge.plane())
            e.selection = torch.tensor([True, False, False])
            e.objects_action({'action': 'add_id', 'name': 'first'})
            e.selection = torch.tensor([True, True, False])  # Already assigned point keeps ID 1.
            e.objects_action({'action': 'add_id', 'name': 'second'})
            np.testing.assert_array_equal(e.instance_ids, [1, 2, 0])
            self.assertEqual(e.instance_ids.dtype, np.int32)
            self.assertTrue(all('mask' not in obj for obj in e.objects))
            self.assertTrue(all(obj['hidden'] for obj in e.objects))
            self.assertFalse(e.knowledge.load_segmentation(3, e.gaussian_fingerprint))
            self.assertTrue(e.ids_dirty)
            e.objects_action({'action': 'save_ids'})
            self.assertFalse(e.ids_dirty)
            record = json.loads(e.knowledge.path.read_text())['scenes']['scene.ply']
            self.assertEqual(record['segmentation']['gaussian_ids'], [1, 2, 0])
            self.assertIn('ground_plane', record)
            # A later ground edit must preserve the entire ID record.
            saved = copy.deepcopy(record['segmentation'])
            e.knowledge.save_ground(test_knowledge.plane())
            self.assertEqual(json.loads(e.knowledge.path.read_text())['scenes']['scene.ply']['segmentation'], saved)
            restored = self.engine(folder)
            self.assertTrue(restored.ids_restored)
            self.assertEqual(restored.next_object_id, 3)
            self.assertEqual(restored.objects, [dict(obj, hidden=False) for obj in e.objects])
            self.assertEqual(restored.active().tolist(), [True, True, True])
            self.assertFalse(restored.ids_dirty)
            restored.objects_action({'action': 'select', 'id': restored.objects[1]['id']})
            self.assertEqual(restored.selection.tolist(), [False, True, False])
            # Editing history marks the saved state dirty, but never changes disk until Save.
            restored.objects_action({'action': 'delete', 'id': restored.objects[0]['id']})
            np.testing.assert_array_equal(restored.instance_ids, [0, 2, 0])
            restored.objects_action({'action': 'undo'})
            np.testing.assert_array_equal(restored.instance_ids, [1, 2, 0])
            restored.objects_action({'action': 'redo'})
            np.testing.assert_array_equal(restored.instance_ids, [0, 2, 0])
            restored.selection = torch.tensor([True, False, False])
            restored.objects_action({'action': 'add_id'})
            np.testing.assert_array_equal(restored.instance_ids, [3, 2, 0])
            self.assertTrue(restored.ids_dirty)
            self.assertEqual(json.loads(e.knowledge.path.read_text())['scenes']['scene.ply']['segmentation'], saved)

    def test_rename_preserves_segmentation_and_supports_history_and_persistence(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            e.selection = torch.tensor([True, False, False])
            e.objects_action({'action': 'add_id', 'name': 'old'})
            obj = e.objects[0]
            key = obj['id']
            e.objects_action({'action': 'save_ids'})
            saved = e.knowledge.path.read_bytes()
            e.capture = {'id': 'current'}
            e.background_frame = {'id': 'multi'}
            result = e.objects_action({'action': 'rename', 'id': key, 'name': '  红色椅子  '})
            self.assertEqual(result['objects'][0]['name'], '红色椅子')
            self.assertTrue(e.ids_dirty)
            self.assertEqual(e.capture, {'id': 'current'})
            self.assertEqual(e.background_frame, {'id': 'multi'})
            np.testing.assert_array_equal(e.labels(), [1, 0, 0])
            self.assertEqual((obj['id'], obj['number'], obj['hidden']), (key, 1, True))
            self.assertIsNone(e.selection)
            self.assertEqual(e.knowledge.path.read_bytes(), saved)
            history = len(e.undo_stack)
            e.objects_action({'action': 'rename', 'id': key, 'name': '红色椅子'})
            self.assertEqual(len(e.undo_stack), history)
            for name in (' ', None, 1, 'a'*81):
                with self.assertRaises(ValueError):
                    e.objects_action({'action': 'rename', 'id': key, 'name': name})
            with self.assertRaisesRegex(ValueError, '物体不存在'):
                e.objects_action({'action': 'rename', 'id': 'missing', 'name': 'chair'})
            self.assertEqual(len(e.undo_stack), history)
            e.objects_action({'action': 'undo'})
            self.assertEqual(e.objects[0]['name'], 'old')
            e.objects_action({'action': 'redo'})
            self.assertEqual(e.objects[0]['name'], '红色椅子')
            e.objects_action({'action': 'save_ids'})
            self.assertEqual(self.engine(folder).objects[0]['name'], '红色椅子')

    def test_invalid_vectors_metadata_and_order_are_rejected(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            e.selection = torch.tensor([False, True, False])
            e.objects_action({'action': 'add_id'})
            e.objects_action({'action': 'save_ids'})
            original = json.loads(e.knowledge.path.read_text())
            def invalid(change):
                data = copy.deepcopy(original)
                change(data['scenes']['scene.ply']['segmentation'])
                e.knowledge.path.write_text(json.dumps(data))
                with self.assertRaises(ValueError):
                    e.knowledge.load_segmentation(3, e.gaussian_fingerprint)
                restored = self.engine(folder)
                self.assertFalse(restored.ids_restored)
                self.assertIn('无法读取三维分割知识', restored.knowledge_warning)
                np.testing.assert_array_equal(restored.instance_ids, [0, 0, 0])
                self.assertEqual(restored.objects, [])
            for values in ([0, 1], [0, -1, 0], [0, 1.5, 0], [0, True, 0], [0, 2147483648, 0], [0, 2, 0]):
                invalid(lambda record, values=values: record.update(gaussian_ids=values))
            invalid(lambda r: r.update(gaussian_order_sha256='wrong order'))
            invalid(lambda r: r.update(gaussian_count=4))
            invalid(lambda r: r.update(next_id=1))
            invalid(lambda r: r['objects'].append(copy.deepcopy(r['objects'][0])))
            invalid(lambda r: r['objects'][0].update(hidden='yes'))

    def test_atomic_failure_empty_save_and_scene_isolation(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            e.objects_action({'action': 'save_ids'})
            self.assertEqual(e.knowledge.load_segmentation(3, e.gaussian_fingerprint)[0].tolist(), [0, 0, 0])
            before = e.knowledge.path.read_bytes()
            e.selection = torch.tensor([True, False, False])
            e.objects_action({'action': 'add_id'})
            labels = e.instance_ids.copy()
            with patch('backend.knowledge.os.replace', side_effect=PermissionError('read only')):
                with self.assertRaisesRegex(ValueError, '分割结果未保存'):
                    e.objects_action({'action': 'save_ids'})
            self.assertTrue(e.ids_dirty)
            np.testing.assert_array_equal(e.instance_ids, labels)
            self.assertEqual(e.knowledge.path.read_bytes(), before)
            self.assertFalse(list(Path(folder).glob('.SAGO_knowledge-*.tmp')))
            other = SceneKnowledge(Path(folder)/'other.ply')
            self.assertIsNone(other.load_segmentation(3, e.gaussian_fingerprint))
            e.knowledge.path.write_text('{corrupt')
            with self.assertRaisesRegex(ValueError, '分割结果未保存'):
                e.objects_action({'action': 'save_ids'})
            self.assertEqual(e.knowledge.path.read_text(), '{corrupt')

    def test_merge_union_preserves_other_ids_history_and_persistence(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            for i in range(3):
                e.selection = torch.arange(3) == i
                e.objects_action({'action': 'add_id', 'name': f'target {i+1}'})
            ids = [o['id'] for o in e.objects]
            e.objects_action({'action': 'save_ids'})
            e.capture = {'id': 'stale'}
            result = e.objects_action({'action': 'merge', 'ids': [ids[2], ids[0]]})
            np.testing.assert_array_equal(e.instance_ids, [1, 2, 1])
            self.assertEqual([o['number'] for o in e.objects], [1, 2])
            self.assertEqual(e.objects[0]['name'], 'target 1')
            self.assertEqual(e.objects[0]['id'], ids[0])
            self.assertFalse(e.objects[0]['hidden'])
            self.assertTrue(e.objects[1]['hidden'])
            self.assertEqual(result['objects'][0]['count'], 2)
            self.assertEqual(e.selection.tolist(), [True, False, True])
            self.assertEqual(e.active().tolist(), [True, False, True])
            self.assertEqual(e.next_object_id, 4)
            self.assertTrue(e.ids_dirty)
            self.assertIsNone(e.capture)
            e.objects_action({'action': 'undo'})
            np.testing.assert_array_equal(e.instance_ids, [1, 2, 3])
            self.assertIsNone(e.selection)
            self.assertTrue(all(o['hidden'] for o in e.objects))
            e.objects_action({'action': 'redo'})
            np.testing.assert_array_equal(e.instance_ids, [1, 2, 1])
            self.assertEqual(e.selection.tolist(), [True, False, True])
            e.objects_action({'action': 'save_ids'})
            restored = self.engine(folder)
            np.testing.assert_array_equal(restored.instance_ids, [1, 2, 1])
            self.assertEqual(restored.objects, [dict(obj, hidden=False) for obj in e.objects])
            self.assertEqual(restored.next_object_id, 4)

    def test_invalid_merge_is_rejected_without_mutation(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            e.selection = torch.tensor([True, False, False])
            e.objects_action({'action': 'add_id'})
            e.selection = torch.tensor([False, True, False])
            e.objects_action({'action': 'add_id'})
            e.objects_action({'action': 'save_ids'})
            ids = [o['id'] for o in e.objects]
            original = e.instance_ids.copy(), copy.deepcopy(e.objects), len(e.undo_stack), e.knowledge.path.read_bytes()
            e.capture = {'id': 'valid'}
            for value in (None, 'bad', [], ids[:1], [ids[0], ids[0]], [ids[0], 'missing'], [False, ids[0]], ids+[ids[0]]):
                with self.assertRaises(ValueError):
                    e.objects_action({'action': 'merge', 'ids': value})
                np.testing.assert_array_equal(e.instance_ids, original[0])
                self.assertEqual(e.objects, original[1])
                self.assertEqual(len(e.undo_stack), original[2])
                self.assertEqual(e.knowledge.path.read_bytes(), original[3])
                self.assertFalse(e.ids_dirty)
                self.assertEqual(e.capture, {'id': 'valid'})
            # An actual merge changes only the chosen IDs; background stays zero.
            e.objects_action({'action': 'merge', 'ids': ids})
            np.testing.assert_array_equal(e.instance_ids, [1, 1, 0])
