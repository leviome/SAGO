"""Upright volume containment, degeneracies, ownership and undo/redo."""
import itertools
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
import numpy as np
import torch
from backend.completion import complete_foreground
from backend.engine import Engine


def box_scene():
    corners=list(itertools.product((-2.,2.),(-1.,1.),(-3.,3.)))
    xyz=torch.tensor(corners+[[0,0,0],[1,0,2],[2.1,0,0],[0,0,3.1]],dtype=torch.float32)
    selection=torch.arange(len(xyz))<8
    return xyz,selection

class CompletionGeometryTest(unittest.TestCase):
    def test_cube_fills_occluded_centers_and_excludes_outside(self):
        xyz,selection=box_scene();original=selection.clone()
        result,volume=complete_foreground(xyz,selection,[0,0,1],'cube')
        self.assertEqual(result.tolist(),[True]*10+[False,False])
        self.assertEqual(selection.tolist(),original.tolist())
        np.testing.assert_allclose(sorted(volume['size']),[2,4,6])
    def test_cube_matches_tilted_up_and_rotated_horizontal_rectangle(self):
        xyz,selection=box_scene()
        a,b=np.deg2rad([37,61])
        horizontal=np.array([[np.cos(a),-np.sin(a),0],[np.sin(a),np.cos(a),0],[0,0,1]])
        tilt=np.array([[np.cos(b),0,np.sin(b)],[0,1,0],[-np.sin(b),0,np.cos(b)]])
        rotation=torch.tensor(tilt@horizontal,dtype=torch.float32)
        rotated=xyz@rotation.T+torch.tensor([100.,-60.,7.])
        result,_=complete_foreground(rotated,selection,rotation[:,2],'cube')
        self.assertEqual(result.tolist(),[True]*10+[False,False])
    def test_cylinder_uses_enclosing_circle_not_foreground_mean(self):
        xyz=torch.tensor([[0.,0,0],[4,0,2],[4,0,0],[2,1,1],[0,1,1],[2,0,2.1],[2,0,-.1]])
        selection=torch.arange(len(xyz))<3
        result,volume=complete_foreground(xyz,selection,[0,0,1],'cylinder')
        self.assertEqual(result.tolist(),[True]*4+[False]*3)
        np.testing.assert_allclose(volume['center'],[2,0,1],atol=1e-6)
        self.assertAlmostEqual(volume['radius'],2,places=6)
        self.assertAlmostEqual(volume['height'],2,places=6)
    def test_cylinder_follows_tilted_axis_and_flipped_up(self):
        xyz=torch.tensor([[0.,0,0],[4,0,2],[4,0,0],[2,1,1],[0,1,1]])
        rotation=torch.tensor([[0.,0,1],[0,1,0],[-1,0,0]])
        selection=torch.arange(len(xyz))<3
        for sign in (1,-1):
            result,_=complete_foreground(xyz@rotation.T,selection,rotation[:,2]*sign,'cylinder')
            self.assertEqual(result.tolist(),[True]*4+[False])
    def test_single_point_line_and_flat_shapes(self):
        for shape in ('cube','cylinder'):
            xyz=torch.tensor([[1.,2,3],[1,2,3],[1,2,3.1]])
            result,_=complete_foreground(xyz,torch.tensor([True,False,False]),[0,0,1],shape)
            self.assertEqual(result.tolist(),[True,True,False])
            xyz=torch.tensor([[0.,0,0],[2,0,0],[1,0,0],[1,0,.1]])
            result,_=complete_foreground(xyz,torch.tensor([True,True,False,False]),[0,0,1],shape)
            self.assertEqual(result.tolist(),[True,True,True,False])
    def test_chunk_boundary_and_nonfinite_background(self):
        xyz=torch.ones((262146,3));xyz[0]=0;xyz[-2]=0;xyz[-1]=float('nan')
        selection=torch.zeros(len(xyz),dtype=torch.bool);selection[0]=True
        result,_=complete_foreground(xyz,selection,[0,0,1],'cube')
        self.assertEqual(result.nonzero().flatten().tolist(),[0,262144])
    def test_empty_invalid_up_and_invalid_foreground_rejected(self):
        xyz=torch.tensor([[0.,0,0]])
        for selection,up,shape in ((None,[0,0,1],'cube'),(torch.tensor([False]),[0,0,1],'cylinder'),(torch.tensor([True]),[0,0,0],'cube'),(torch.tensor([True]),[0,0,float('nan')],'cylinder'),(torch.tensor([True]),[0,0,1],'sphere')):
            with self.assertRaises(ValueError):complete_foreground(xyz,selection,up,shape)
        with self.assertRaises(ValueError):complete_foreground(xyz*float('nan'),torch.tensor([True]),[0,0,1],'cube')

class CompletionEngineTest(unittest.TestCase):
    def setUp(self):
        cuda=patch.object(torch.Tensor,'cuda',lambda value:value);cuda.start();self.addCleanup(cuda.stop)
        self.temp=TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.engine=Engine('unused.pt',self.temp.name);e=self.engine;e.torch=torch
        xyz,e.selection=box_scene();e.model=SimpleNamespace(get_xyz=xyz);e.view=dict(up=[0,0,1])
        e.frame=Mock(side_effect=lambda mode='all':dict(count=len(xyz),selected=int(e.selection.sum())))
    def test_completion_undo_redo_and_id_ownership_unchanged(self):
        e=self.engine;e.instance_ids=np.array([1]+[0]*7+[2,0,0,0],dtype=np.int32)
        e.objects=[dict(id='old',number=2,name='old object',hidden=True)]
        labels=e.labels().copy();objects=[dict(o) for o in e.objects];xyz=e.model.get_xyz.clone()
        result=e.objects_action(dict(action='complete',shape='cube'))
        self.assertEqual(result['completion']['added'],2)
        self.assertEqual(result['selected'],10);self.assertFalse(e.ids_dirty)
        np.testing.assert_array_equal(e.labels(),labels);self.assertEqual(e.objects,objects)
        torch.testing.assert_close(e.model.get_xyz,xyz)
        e.objects_action(dict(action='undo'));self.assertEqual(int(e.selection.sum()),8)
        e.objects_action(dict(action='redo'));self.assertEqual(int(e.selection.sum()),10)
        np.testing.assert_array_equal(e.labels(),labels)
    def test_no_op_does_not_add_history(self):
        e=self.engine;e.objects_action(dict(action='complete',shape='cube'));count=len(e.undo_stack)
        result=e.objects_action(dict(action='complete',shape='cube'))
        self.assertEqual(result['completion']['added'],0);self.assertEqual(len(e.undo_stack),count)
    def test_frozen_views_and_empty_selection_leave_history_unchanged(self):
        e=self.engine
        for attr in ('capture','ground_capture','correction_pending'):
            setattr(e,attr,{'id':'frozen'})
            with self.assertRaises(ValueError):e.objects_action(dict(action='complete',shape='cube'))
            setattr(e,attr,None)
        e.selection=None
        with self.assertRaises(ValueError):e.objects_action(dict(action='complete',shape='cylinder'))
        self.assertFalse(e.undo_stack);e.frame.assert_not_called()
