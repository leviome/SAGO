"""Consecutive ID remapping and read-only multi-object rendering."""
import copy
from types import SimpleNamespace
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from backend.engine import Engine
import test_knowledge


class MultiObjectsTest(unittest.TestCase):
    def setUp(self):
        cuda=patch.object(torch.Tensor, 'cuda', lambda tensor: tensor)
        cuda.start(); self.addCleanup(cuda.stop)

    def engine(self, folder):
        return test_knowledge.KnowledgeTest.engine(self, folder)

    def test_renumber_preserves_geometry_metadata_color_and_history(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            for i in range(3):
                e.selection=torch.arange(3)==i
                e.objects_action({'action':'add_id','name':f'name {i+1}'})
            e.objects_action({'action':'delete','id':e.objects[0]['id']})
            e.selection=torch.tensor([False,True,False])
            objects=copy.deepcopy(e.objects)
            colors={o['id']:e.object_color(o) for o in objects}
            e.capture={'id':'old'}
            e.objects_action({'action':'renumber'})
            np.testing.assert_array_equal(e.instance_ids,[0,1,2])
            self.assertEqual([o['number'] for o in e.objects],[1,2])
            self.assertEqual([o['name'] for o in e.objects],['name 2','name 3'])
            self.assertEqual([o['id'] for o in e.objects],[o['id'] for o in objects])
            self.assertEqual([o['hidden'] for o in e.objects],[True,True])
            self.assertEqual(e.selection.tolist(),[False,True,False])
            self.assertEqual(e.next_object_id,3)
            self.assertIsNone(e.capture)
            self.assertTrue(e.ids_dirty)
            self.assertEqual({o['id']:e.object_color(o) for o in e.objects},colors)
            e.objects_action({'action':'undo'})
            np.testing.assert_array_equal(e.instance_ids,[0,2,3])
            self.assertEqual(e.objects,objects)
            self.assertEqual(e.next_object_id,4)
            e.objects_action({'action':'redo'})
            np.testing.assert_array_equal(e.instance_ids,[0,1,2])
            e.objects_action({'action':'save_ids'})
            loaded=self.engine(folder)
            np.testing.assert_array_equal(loaded.instance_ids,[0,1,2])
            self.assertEqual(loaded.objects,[dict(obj,hidden=False) for obj in e.objects])
            self.assertEqual(loaded.next_object_id,3)
            history=len(e.undo_stack)
            e.objects_action({'action':'renumber'})
            self.assertEqual(len(e.undo_stack),history)  # Already consecutive is a no-op.

    def test_renumber_empty_and_reset_next_number(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            with self.assertRaises(ValueError):e.objects_action({'action':'renumber'})
            e.selection=torch.tensor([True,False,False])
            e.objects_action({'action':'add_id'})
            e.next_object_id=20
            e.objects_action({'action':'renumber'})
            self.assertEqual(e.next_object_id,2)
            np.testing.assert_array_equal(e.instance_ids,[1,0,0])

    def test_multi_masks_colors_hidden_objects_raw_and_empty(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            for i in range(2):
                e.selection=torch.arange(3)==i
                e.objects_action({'action':'add_id'})
            original=e.instance_ids.copy(),copy.deepcopy(e.objects)
            e.selection=torch.tensor([False,False,True])
            e.frame=Engine.frame.__get__(e)
            e.camera=Mock(return_value=object())
            e.visualization=lambda cam:{'bbox':{'unused':True}}
            e.image_url=lambda image:'image'
            e.render_simple=Mock(return_value={'render':torch.zeros(3,1,1)})
            e.id_colors=True
            result=e.frame('multi')
            options=e.render_simple.call_args.kwargs
            self.assertEqual(options['extra_gau_mask'].tolist(),[True,True,False])
            colors=options['override_color']
            self.assertEqual(tuple(colors.shape),(3,3))
            self.assertFalse(torch.equal(colors[0],colors[1]))
            np.testing.assert_allclose(colors[0].numpy(),e.object_color(e.objects[0]),rtol=1e-6)
            self.assertIsNone(result['visualization']['bbox'])
            self.assertEqual(e.active().tolist(),[False,False,True])
            e.foreground_options({'foreground_ids':[e.objects[1]['id']],'id_colors':False})
            e.frame('multi')
            options=e.render_simple.call_args.kwargs
            self.assertEqual(options['extra_gau_mask'].tolist(),[False,True,False])
            self.assertNotIn('override_color',options)
            # Standard rendering still hides committed objects.
            e.frame('all')
            self.assertEqual(e.render_simple.call_args.kwargs['extra_gau_mask'].tolist(),[False,False,True])
            e.foreground_options({'foreground_ids':[]})
            e.render_simple.reset_mock()
            e.frame('multi')
            e.render_simple.assert_not_called()
            np.testing.assert_array_equal(e.instance_ids,original[0])
            self.assertEqual(e.objects,original[1])
            self.assertEqual(e.selection.tolist(),[False,False,True])

    def test_invalid_multi_options_are_rejected(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            for data in ({'foreground_ids':'bad'},{'foreground_ids':[1]},{'foreground_ids':['missing']},{'id_colors':'yes'},
                         {'show_background':0},{'show_foreground':'false'}):
                with self.assertRaises(ValueError):e.foreground_options(data)

    def test_all_composes_background_and_ids_without_changing_current_round(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            for i in range(2):
                e.selection=torch.arange(3)==i
                e.objects_action({'action':'save'})
            e.selection=torch.tensor([False,True,False])
            original=e.instance_ids.copy(),copy.deepcopy(e.objects)
            e.model.get_features=torch.tensor([[[.1,.2,.3]],[[.4,.5,.6]],[[.7,.8,.9]]])
            e.model.active_sh_degree=0
            e.frame=Engine.frame.__get__(e)
            e.camera=lambda:SimpleNamespace(camera_center=torch.tensor([0.,0.,-1.]))
            e.visualization=lambda cam:{}
            e.image_url=lambda image:'image'
            e.render_simple=Mock(return_value={'render':torch.zeros(3,1,1)})
            e.foreground_options({'foreground_ids':[e.objects[0]['id']], 'id_colors':True})
            result=e.frame('all')
            options=e.render_simple.call_args.kwargs
            self.assertEqual(options['extra_gau_mask'].tolist(),[True,False,True])
            colors=options['override_color']
            np.testing.assert_allclose(colors[0],e.object_color(e.objects[0]),rtol=1e-6)
            # Background retains its original SH color, rather than black or an ID tint.
            np.testing.assert_allclose(colors[2],.5+.28209479177387814*e.model.get_features[2,0],rtol=1e-6)
            self.assertEqual(e.background_frame['numbers'],[1])
            self.assertIn('frame_id',result)
            e.foreground_options({'show_background':False,'id_colors':False})
            e.frame('all')
            self.assertEqual(e.render_simple.call_args.kwargs['extra_gau_mask'].tolist(),[True,False,False])
            self.assertNotIn('override_color',e.render_simple.call_args.kwargs)
            e.foreground_options({'show_background':True,'show_foreground':False})
            e.frame('all')
            self.assertEqual(e.render_simple.call_args.kwargs['extra_gau_mask'].tolist(),[False,False,True])
            self.assertEqual(e.background_frame['numbers'],[])
            e.foreground_options({'show_background':False})
            e.render_simple.reset_mock()
            e.frame('all')
            e.render_simple.assert_not_called()
            e.frame('foreground')
            self.assertEqual(e.render_simple.call_args.kwargs['extra_gau_mask'].tolist(),[False,True,False])
            e.frame('background')
            self.assertEqual(e.render_simple.call_args.kwargs['extra_gau_mask'].tolist(),[True,False,True])
            e.foreground_options({'show_foreground':True})
            e.objects[0]['hidden']=True
            e.render_simple.reset_mock()
            e.frame('all')
            e.render_simple.assert_not_called()
            e.objects[0]['hidden']=False
            np.testing.assert_array_equal(e.instance_ids,original[0])
            self.assertEqual(e.objects,original[1])
            self.assertEqual(e.selection.tolist(),[False,True,False])
