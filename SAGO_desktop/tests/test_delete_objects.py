"""Deleting an object changes ownership, never Gaussian geometry."""
import copy
import json
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import numpy as np
import torch
from backend.engine import Engine
import test_knowledge


class DeleteObjectTest(unittest.TestCase):
    def setUp(self):
        cuda = patch.object(torch.Tensor, 'cuda', lambda tensor: tensor)
        cuda.start(); self.addCleanup(cuda.stop)

    def engine(self, folder):
        return test_knowledge.KnowledgeTest.engine(self, folder)

    def test_delete_background_selection_history_and_saved_roundtrip(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            for i in range(2):
                e.selection = torch.arange(3) == i
                e.objects_action({'action': 'add_id', 'name': f'object {i + 1}'})
            first, second = copy.deepcopy(e.objects)
            e.foreground_ids = {first['id'], second['id']}
            e.selection = torch.tensor([True, True, False])
            e.drone_path = {'origin': [1, 2, 3], 'views': []}
            e.objects_action({'action': 'save_ids'})
            saved = e.knowledge.path.read_bytes()
            original_geometry = e.model.get_xyz.clone()
            model = e.model
            e.capture = {'id': 'old'}; e.background_frame = {'id': 'old'}
            e.correction_pending = {'id': 'old'}
            result = e.objects_action({'action': 'delete', 'id': first['id']})
            np.testing.assert_array_equal(e.labels(), [0, 2, 0])
            self.assertEqual(e.objects, [second])
            self.assertEqual(e.active().tolist(), [True, False, True])
            self.assertEqual(e.selection.tolist(), [False, True, False])
            self.assertIsNone(e.drone_path)
            self.assertEqual(e.foreground_ids, {second['id']})
            self.assertEqual(result['display']['foreground_ids'], [second['id']])
            self.assertEqual((result['count'], result['selected']), (3, 1))
            self.assertIs(e.model, model)
            torch.testing.assert_close(e.model.get_xyz, original_geometry)
            self.assertTrue(e.ids_dirty)
            self.assertIsNone(e.capture); self.assertIsNone(e.background_frame); self.assertIsNone(e.correction_pending)
            self.assertEqual(e.next_object_id, 3)
            self.assertEqual(e.knowledge.path.read_bytes(), saved)
            e.objects_action({'action': 'undo'})
            np.testing.assert_array_equal(e.labels(), [1, 2, 0])
            self.assertEqual(e.objects, [first, second])
            self.assertEqual(e.selection.tolist(), [True, True, False])
            self.assertEqual(e.foreground_ids, {first['id'], second['id']})
            self.assertEqual(e.drone_path, {'origin': [1, 2, 3], 'views': []})
            e.objects_action({'action': 'redo'})
            np.testing.assert_array_equal(e.labels(), [0, 2, 0])
            self.assertEqual(e.selection.tolist(), [False, True, False])
            self.assertEqual(e.foreground_ids, {second['id']})
            e.objects_action({'action': 'save_ids'})
            record = json.loads(e.knowledge.path.read_text())['scenes']['scene.ply']['segmentation']
            self.assertEqual(record['gaussian_ids'], [0, 2, 0])
            restored = self.engine(folder)
            np.testing.assert_array_equal(restored.labels(), [0, 2, 0])
            self.assertEqual([o['number'] for o in restored.objects], [2])
            self.assertEqual(restored.next_object_id, 3)

    def test_delete_entire_selection_unselected_object_and_invalid_id(self):
        with TemporaryDirectory() as folder:
            e = self.engine(folder)
            e.selection = torch.tensor([True, False, False])
            e.objects_action({'action': 'save'})
            key = e.objects[0]['id']
            e.foreground_ids = set()  # Preserve unchecked visibility through undo.
            e.objects_action({'action': 'delete', 'id': key})
            self.assertIsNone(e.selection)
            self.assertEqual(e.objects, [])
            np.testing.assert_array_equal(e.labels(), [0, 0, 0])
            e.objects_action({'action': 'undo'})
            self.assertEqual(e.foreground_ids, set())
            self.assertEqual(e.selection.tolist(), [True, False, False])
            e.selection = torch.tensor([False, False, True])
            e.objects_action({'action': 'delete', 'id': key})
            self.assertEqual(e.selection.tolist(), [False, False, True])
            before, history = e.labels().copy(), len(e.undo_stack)
            with self.assertRaisesRegex(ValueError, '物体不存在'):
                e.objects_action({'action': 'delete', 'id': key})
            np.testing.assert_array_equal(e.labels(), before)
            self.assertEqual(len(e.undo_stack), history)


class DesktopDeleteObjectTest(DeleteObjectTest):
    def engine(self, folder):
        from desktop.engine import DesktopEngine
        e = super().engine(folder)
        e.__class__ = DesktopEngine
        return e
