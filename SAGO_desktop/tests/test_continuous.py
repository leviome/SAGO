"""Continuous object commits are atomic, undoable, and excluded from later views."""
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from backend.engine import Engine


class ContinuousSegmentationTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.engine = Engine('unused.pt', self.tmp.name)
        self.engine.torch = SimpleNamespace(
            bool=torch.bool, from_numpy=torch.from_numpy,
            ones=lambda n, **kw: torch.ones(n, dtype=kw['dtype']))
        self.engine.model = SimpleNamespace(get_xyz=torch.zeros(6, 3))
        self.engine.frame = self.engine.snapshot
        self.engine.selection = torch.tensor([True, True, False, False, False, False])
        cuda = patch.object(torch.Tensor, 'cuda', lambda tensor: tensor)
        cuda.start()
        self.addCleanup(cuda.stop)

    def test_two_rounds_hide_and_restore_atomically(self):
        e = self.engine
        e.capture = {'id': 'old capture'}
        first_mask = e.selection.clone()
        first = e.objects_action({'action': 'add_id', 'name': 'First'})
        self.assertEqual(first['selected'], 0)
        self.assertEqual(first['objects'][0]['number'], 1)
        self.assertTrue(first['objects'][0]['hidden'])
        self.assertEqual(first['objects'][0]['name'], 'First')
        self.assertIsNone(e.capture)
        self.assertTrue(torch.equal(e.active(), ~first_mask))
        e.objects_action({'action': 'undo'})
        self.assertEqual(e.objects, [])
        self.assertTrue(torch.equal(e.selection, first_mask))
        e.objects_action({'action': 'redo'})
        self.assertIsNone(e.selection)
        self.assertTrue(torch.equal(e.active(), ~first_mask))
        e.selection = torch.tensor([False, False, True, False, False, False])
        second = e.objects_action({'action': 'add_id'})
        self.assertEqual([o['number'] for o in second['objects']], [1, 2])
        self.assertNotEqual(second['objects'][0]['id'], second['objects'][1]['id'])
        self.assertEqual(e.active().tolist(), [False, False, False, True, True, True])
        first_id = first['objects'][0]['id']
        e.objects_action({'action': 'select', 'id': first_id})
        self.assertTrue(torch.equal(e.selection, first_mask))
        e.objects_action({'action': 'hide', 'id': first_id})
        self.assertEqual(e.active().tolist(), [True, True, False, True, True, True])

    def test_empty_cannot_create_duplicate_and_save_still_visible(self):
        e = self.engine
        e.objects_action({'action': 'save'})
        self.assertFalse(e.objects[0]['hidden'])
        self.assertIsNotNone(e.selection)
        with self.assertRaisesRegex(ValueError, '已属于现有 ID'):
            e.objects_action({'action': 'add_id'})
        e.selection = torch.tensor([False, False, True, False, False, False])
        e.objects_action({'action': 'add_id'})
        before = len(e.undo_stack)
        with self.assertRaisesRegex(ValueError, '选区为空'):
            e.objects_action({'action': 'add_id'})
        self.assertEqual(len(e.objects), 2)
        self.assertEqual(len(e.undo_stack), before)
