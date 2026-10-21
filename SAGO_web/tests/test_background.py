"""Screen-box correction edits dense labels without touching unshown objects."""
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from backend.background import projected_box_mask, projected_ellipse_mask, ellipse_box_mask
from backend.engine import Engine
import test_knowledge


class BackgroundBoxTest(unittest.TestCase):
    def setUp(self):
        cuda=patch.object(torch.Tensor,'cuda',lambda tensor:tensor)
        cuda.start();self.addCleanup(cuda.stop)

    def engine(self,folder):
        e=test_knowledge.KnowledgeTest.engine(self,folder)
        e.model.get_xyz=torch.tensor([[0.,0.,1.],[0.,0.,2.],[.8,0.,1.]])
        e.selection=torch.tensor([True,False,True])
        e.objects_action({'action':'add_id','name':'editable'})
        e.selection=torch.tensor([False,True,False])
        e.objects_action({'action':'add_id','name':'unshown'})
        e.foreground_ids={e.objects[0]['id']}
        e.camera=lambda:SimpleNamespace(full_proj_transform=torch.eye(4))
        e.frame=Engine.frame.__get__(e)
        e.view.update(size=100,aspect=1.25)
        e.visualization=lambda cam:{}
        def render(*args, **kw):
            owners=torch.full((1,80,100),-1.)
            # Stub native front-contributor output, respecting displayed layers.
            if kw['extra_gau_mask'][0]:
                owners[0,40,50]=0
                if kw.get('scaling_modifier',1.) == 1.: owners[0,40,66]=0
            if kw['extra_gau_mask'][1] and not kw['extra_gau_mask'][0]: owners[0,40,50]=1
            if kw['extra_gau_mask'][2]: owners[0,40,90]=2
            return {'render':torch.zeros(3,80,100),'gs_per_pixel':owners}
        e.render_simple=render
        e.natural_colors=lambda cam, mask:torch.zeros(int(mask.sum()),3)
        e.image_url=lambda image:'image'
        return e

    def test_projection_direction_depth_invalid_and_reversed_rectangle(self):
        xyz=np.array([[0,0,1],[0,0,10],[.8,0,1],[0,0,-1],[np.nan,0,1]])
        matrix=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,0]])
        result=projected_box_mask(xyz,matrix,100,80,[[60,50],[40,30]])
        self.assertEqual(result.tolist(),[True,True,False,False,False])
        for rectangle in (None,[[0,0]],[[0,0],[1,1]],[[-1,0],[20,20]],[[0,0],[100,80]],[[0,0],[np.nan,20]]):
            with self.assertRaises(ValueError):projected_box_mask(xyz,matrix,100,80,rectangle)

    def test_rotated_ellipse_exact_intersection_tangency_and_center_outside(self):
        centers=np.array([[39,50],[36,36],[36,36],[37,50],[36.99,50],[50,50]])
        covariance=np.array([[[1,0],[0,1]],[[4,3.8],[3.8,4]],[[4,-3.8],[-3.8,4]],
                             [[1,0],[0,1]],[[1,0],[0,1]],[[1,0],[0,1]]])
        actual=ellipse_box_mask(centers,covariance,np.array([40,40]),np.array([60,60]))
        self.assertEqual(actual.tolist(),[True,True,False,True,False,True])

    def test_ellipse_projection_camera_rotation_scale_and_invalid_depth(self):
        projection=np.array([[1,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,0]],dtype=float)
        xyz=np.array([[-.62,0,2],[-.62,0,-2],[0,0,.1],[np.nan,0,2]])
        cov=np.tile([.0064,0,0,.00001,0,.00001],(4,1))
        rectangle=[[40,40],[60,60]]
        args=(np.eye(4),projection,100,100,[1,1],rectangle)
        self.assertEqual(projected_box_mask(xyz,projection,100,100,rectangle).tolist(),[False,False,True,False])
        self.assertEqual(projected_ellipse_mask(xyz,cov,*args).tolist(),[True,False,False,False])
        self.assertFalse(projected_ellipse_mask(xyz,cov*.35**2,*args).any())
        view=np.array([[0,1,0,0],[-1,0,0,0],[0,0,1,0],[0,0,0,1]],dtype=float)
        self.assertFalse(projected_ellipse_mask(np.array([[0,.62,2]]),cov[:1],view,view@projection,100,100,[1,1],rectangle).any())
        # An ellipse may enter the viewport while its center stays outside it.
        self.assertTrue(projected_ellipse_mask(np.array([[-2.06,0,2]]),cov[:1]*2,
            np.eye(4),projection,100,100,[1,1],[[0,40],[4,60]])[0])

    def test_default_ellipse_correction_selects_overlap_and_uses_frame_scale(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            e.objects[0]['hidden']=False
            base=torch.tensor([[.04,0,0,.0001,0,.0001],[.0001,0,0,.0001,0,.0001],
                               [.0001,0,0,.0001,0,.0001]])
            e.model.get_covariance=lambda scale:base*scale**2
            e.camera=lambda:SimpleNamespace(full_proj_transform=torch.eye(4),world_view_transform=torch.eye(4),
                                            FoVx=np.pi/2,FoVy=np.pi/2)
            rectangle=[[65,30],[70,50]]
            frame=e.frame('all')
            result=e.objects_action({'action':'correct_box','correction_mode':'point','frame_id':frame['frame_id'],
                                     'rectangle':rectangle})
            self.assertEqual(result['corrected_count'],0)
            result=e.objects_action({'action':'correct_box','frame_id':result['frame_id'],'rectangle':rectangle})
            self.assertEqual(result['correction_mode'],'ellipse')
            self.assertEqual(result['corrected_count'],1)
            np.testing.assert_array_equal(e.labels(),[0,2,1])
            e.objects_action({'action':'undo'})
            e.render_mode='splats'
            frame=e.frame('all')
            self.assertEqual(e.background_frame['gaussian_scale'],.35)
            result=e.objects_action({'action':'correct_box','frame_id':frame['frame_id'],'rectangle':rectangle})
            self.assertEqual(result['corrected_count'],0)
            np.testing.assert_array_equal(e.labels(),[1,2,1])
            for invalid in ('bad',None,[],1):
                with self.assertRaises(ValueError):
                    e.objects_action({'action':'correct_box','correction_mode':invalid,'frame_id':result['frame_id'],
                                      'rectangle':rectangle})

    def test_correction_partial_deletion_undo_redo_and_save(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            e.selection=torch.tensor([True,False,True])
            shown=e.frame('multi')
            before=e.instance_ids.copy()
            count=len(e.undo_stack)
            result=e.objects_action({'action':'background_box','frame_id':shown['frame_id'],'rectangle':[[40,30],[60,50]]})
            self.assertEqual(result['background_count'],1)
            np.testing.assert_array_equal(e.instance_ids,[0,2,1])
            self.assertEqual(e.selection.tolist(),[False,False,True])
            self.assertEqual(len(e.objects),2)
            self.assertEqual(result['objects'][0]['count'],1)
            self.assertTrue(e.ids_dirty)
            self.assertEqual(len(e.undo_stack),count+1)
            e.objects_action({'action':'undo'})
            np.testing.assert_array_equal(e.instance_ids,before)
            self.assertEqual(e.selection.tolist(),[True,False,True])
            e.objects_action({'action':'redo'})
            np.testing.assert_array_equal(e.instance_ids,[0,2,1])
            # A second box removes the rest of only the currently shown object.
            frame=e.frame('multi')
            result=e.objects_action({'action':'background_box','frame_id':frame['frame_id'],'rectangle':[[80,30],[95,50]]})
            np.testing.assert_array_equal(e.instance_ids,[0,2,0])
            self.assertEqual([o['number'] for o in e.objects],[2])
            self.assertEqual(e.next_object_id,3)
            e.objects_action({'action':'save_ids'})
            restored=test_knowledge.KnowledgeTest.engine(self,folder)
            np.testing.assert_array_equal(restored.instance_ids,[0,2,0])
            self.assertEqual([o['number'] for o in restored.objects],[2])

    def test_stale_frames_bad_box_and_empty_box_do_not_mutate(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            frame=e.frame('multi')
            labels=e.instance_ids.copy();history=len(e.undo_stack)
            for data in ({'frame_id':'old','rectangle':[[40,30],[60,50]]},
                         {'frame_id':frame['frame_id'],'rectangle':[[0,0],[1,1]]}):
                with self.assertRaises(ValueError):e.objects_action({'action':'background_box',**data})
                np.testing.assert_array_equal(e.instance_ids,labels)
                self.assertEqual(len(e.undo_stack),history)
            result=e.objects_action({'action':'background_box','frame_id':frame['frame_id'],'rectangle':[[1,1],[10,10]]})
            self.assertEqual(result['background_count'],0)
            self.assertEqual(len(e.undo_stack),history)
            np.testing.assert_array_equal(e.instance_ids,labels)
            with self.assertRaises(ValueError):
                e.objects_action({'action':'background_box','frame_id':frame['frame_id'],'rectangle':[[40,30],[60,50]]})
            e.invalidate()
            with self.assertRaises(ValueError):
                e.objects_action({'action':'background_box','frame_id':result['frame_id'],'rectangle':[[40,30],[60,50]]})

    def test_correct_to_existing_and_new_ids_history_and_persistence(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            original=e.history_state()
            frame=e.frame('multi')
            result=e.objects_action({'action':'correct_box','correction_mode':'point','target_id':2,'frame_id':frame['frame_id'],
                                     'rectangle':[[40,30],[60,50]]})
            self.assertEqual((result['corrected_count'],result['target_id']),(1,2))
            np.testing.assert_array_equal(e.labels(),[2,2,1])
            self.assertEqual([o['name'] for o in e.objects],['editable','unshown'])
            self.assertEqual(e.next_object_id,3)
            e.objects_action({'action':'undo'})
            np.testing.assert_array_equal(e.labels(),original[2])
            e.objects_action({'action':'redo'})
            frame=e.frame('multi')
            result=e.objects_action({'action':'correct_box','correction_mode':'point','target_id':7,'frame_id':frame['frame_id'],
                                     'rectangle':[[80,30],[95,50]]})
            np.testing.assert_array_equal(e.labels(),[2,2,7])
            self.assertEqual([o['number'] for o in e.objects],[2,7])
            self.assertEqual(e.objects[1]['name'],'物体 7')
            self.assertIn(e.objects[1]['id'],e.foreground_ids)
            self.assertEqual(e.next_object_id,8)
            e.objects_action({'action':'undo'})
            self.assertEqual([o['number'] for o in e.objects],[1,2])
            self.assertEqual(e.next_object_id,3)
            e.objects_action({'action':'redo'})
            e.objects_action({'action':'save_ids'})
            restored=test_knowledge.KnowledgeTest.engine(self,folder)
            np.testing.assert_array_equal(restored.labels(),[2,2,7])
            self.assertEqual(restored.next_object_id,8)
            self.assertEqual([o['name'] for o in restored.objects],['unshown','物体 7'])

    def test_correct_visible_background_only_and_reject_invalid_target(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            e.labels()[2]=0
            e.show_foreground=False
            frame=e.frame('all')
            history=len(e.undo_stack)
            for target in (-1,1.5,'2',True,None,2147483648):
                with self.assertRaises(ValueError):
                    e.objects_action({'action':'correct_box','correction_mode':'point','target_id':target,'frame_id':frame['frame_id'],
                                      'rectangle':[[0,0],[99,79]]})
            self.assertEqual(len(e.undo_stack),history)
            # Already-background points do not count as an edit or create history.
            result=e.objects_action({'action':'correct_box','correction_mode':'point','target_id':0,'frame_id':frame['frame_id'],
                                     'rectangle':[[0,0],[99,79]]})
            self.assertEqual(result['corrected_count'],0)
            self.assertEqual(len(e.undo_stack),history)
            result=e.objects_action({'action':'correct_box','correction_mode':'point','target_id':9,'frame_id':result['frame_id'],
                                     'rectangle':[[0,0],[99,79]]})
            np.testing.assert_array_equal(e.labels(),[1,2,9])
            self.assertEqual(result['corrected_count'],1)
            self.assertEqual(e.next_object_id,10)
            # Both existing hidden IDs remain untouched, including their metadata.
            self.assertTrue(all(obj['hidden'] for obj in e.objects[:2]))


    def test_preview_only_visible_front_cancel_confirm_and_stale(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            e.foreground_ids=None
            for o in e.objects:o['hidden']=False
            e.selection=torch.tensor([True,False,True])
            before=e.history_state();history=len(e.undo_stack);dirty=e.ids_dirty
            frame=e.frame('all')
            request=dict(action='preview_correction',correction_mode='point',target_id=7,
                         frame_id=frame['frame_id'],rectangle=[[40,30],[60,50]])
            result=e.objects_action(request)
            self.assertEqual(result['preview_count'],1)  # Rear ID 2 shares the center, but is occluded.
            np.testing.assert_array_equal(e.labels(),before[2])
            self.assertEqual(e.objects,before[1]);self.assertEqual(e.next_object_id,before[3])
            self.assertEqual(len(e.undo_stack),history);self.assertEqual(e.ids_dirty,dirty)
            self.assertEqual(e.selection.tolist(),before[0].tolist())
            token=result['preview_id']
            e.objects_action(dict(action='cancel_correction'))
            with self.assertRaises(ValueError):e.objects_action(dict(action='confirm_correction',preview_id=token))
            request['frame_id']=e.background_frame['id']
            preview=e.objects_action(request)
            result=e.objects_action(dict(action='confirm_correction',preview_id=preview['preview_id']))
            self.assertEqual(result['corrected_count'],1)
            np.testing.assert_array_equal(e.labels(),[7,2,1])
            self.assertEqual(len(e.undo_stack),history+1)
            with self.assertRaises(ValueError):e.objects_action(dict(action='confirm_correction',preview_id=preview['preview_id']))
            e.objects_action(dict(action='undo'))
            np.testing.assert_array_equal(e.labels(),before[2])
            e.objects_action(dict(action='redo'))
            np.testing.assert_array_equal(e.labels(),[7,2,1])
            e.objects_action(dict(action='save_ids'))
            restored=test_knowledge.KnowledgeTest.engine(self,folder)
            np.testing.assert_array_equal(restored.labels(),[7,2,1])
            # A foreground already at the target ID still occludes rear candidates.
            frame=e.frame('all');request.update(target_id=7,frame_id=frame['frame_id'])
            result=e.objects_action(request)
            self.assertEqual(result['preview_count'],0)
            e.invalidate()
            with self.assertRaises(ValueError):e.objects_action(dict(action='confirm_correction',preview_id=result['preview_id']))

    def test_preview_highlight_preserves_normal_colors_and_scene(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder);e.objects[0]['hidden']=False
            calls=[];render=e.render_simple
            def spy(*args,**kw):
                calls.append(kw)
                return render(*args,**kw)
            e.render_simple=spy
            frame=e.frame('all')
            result=e.objects_action(dict(action='preview_correction',correction_mode='point',target_id=7,
                frame_id=frame['frame_id'],rectangle=[[40,30],[60,50]]))
            colors=calls[-1]['override_color']
            np.testing.assert_allclose(colors[0].numpy(),[1,.8,.05],rtol=1e-6)
            self.assertEqual(colors[2].tolist(),[0,0,0])
            e.objects_action(dict(action='cancel_correction'))
            self.assertNotIn('override_color',calls[-1])
            self.assertEqual(e.labels().tolist(),[1,2,1])

    def test_point_owners_nearest_depth_and_hidden_layer_reveal(self):
        from backend.render_modes import point_owners
        from backend.background import visible_box_ids
        projection=torch.tensor([[1.,0,0,0],[0,1,0,0],[0,0,1,1],[0,0,0,0]])
        xyz=torch.tensor([[0.,0,2],[0,0,1],[0,0,-1],[float('nan'),0,1],[0,0,1]])
        owners=point_owners(xyz,projection,100,80).numpy()
        self.assertEqual(visible_box_ids(owners,[[40,30],[60,50]]).tolist(),[1])
        with TemporaryDirectory() as folder:
            e=self.engine(folder);e.foreground_ids=None
            for o in e.objects:o['hidden']=False
            frame=e.frame('all')
            self.assertEqual(e.correction_visibility(e.background_frame)[40,50],0)
            e.objects[0]['hidden']=True
            e.frame('all')
            self.assertEqual(e.correction_visibility(e.background_frame)[40,50],1)

    def test_click_accumulates_visible_gaussians_then_confirms_union(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder)
            e.foreground_ids=None
            for o in e.objects:o['hidden']=False
            e.selection=torch.tensor([True,False,True])
            before=e.history_state();history=len(e.undo_stack);dirty=e.ids_dirty
            def click(pixel, target=7):
                return e.objects_action(dict(action='preview_correction',frame_id=e.background_frame['id'],
                    target_id=target,correction_mode='ellipse',pixel=pixel))
            e.frame('all')
            first=click([50.2,40.8])
            self.assertEqual(first['preview_count'],1)
            self.assertEqual(e.correction_pending['indices'].tolist(),[0])  # Rear index 1 is not picked.
            duplicate=click([50,40])
            self.assertEqual(duplicate['preview_count'],1)
            second=click([90,40])
            self.assertEqual(second['preview_count'],2)
            self.assertEqual(e.correction_pending['indices'].tolist(),[0,2])
            empty=click([0,0]);self.assertEqual(empty['preview_count'],2)
            self.assertEqual(len(e.undo_stack),history);self.assertEqual(e.ids_dirty,dirty)
            self.assertEqual(e.next_object_id,before[3]);self.assertEqual(e.objects,before[1])
            np.testing.assert_array_equal(e.labels(),before[2])
            np.testing.assert_array_equal(e.selection.numpy(),before[0])
            for pixel in (None,[],[1],[float('nan'),0],[-1,0],[100,0],[0,80],[[1,2]]):
                with self.assertRaises(ValueError):click(pixel)
            with self.assertRaises(ValueError):click([50,40],target=8)
            self.assertEqual(e.correction_pending['indices'].tolist(),[0,2])
            with self.assertRaises(ValueError):
                e.objects_action(dict(action='confirm_correction',preview_id=first['preview_id']))
            result=e.objects_action(dict(action='confirm_correction',preview_id=empty['preview_id']))
            self.assertEqual(result['corrected_count'],2)
            self.assertEqual(e.labels().tolist(),[7,2,7])
            self.assertEqual(len(e.undo_stack),history+1)
            e.objects_action(dict(action='undo'))
            np.testing.assert_array_equal(e.labels(),before[2])
            e.objects_action(dict(action='redo'));e.objects_action(dict(action='save_ids'))
            restored=test_knowledge.KnowledgeTest.engine(self,folder)
            self.assertEqual(restored.labels().tolist(),[7,2,7])

    def test_click_target_does_not_pick_through_and_box_replaces_clicks(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder);e.foreground_ids=None
            for o in e.objects:o['hidden']=False
            e.frame('all')
            def click(pixel,target=7):
                return e.objects_action(dict(action='preview_correction',frame_id=e.background_frame['id'],
                    target_id=target,correction_mode='point',pixel=pixel))
            result=click([50,40],target=1)
            self.assertEqual(result['preview_count'],0)  # Target front still occludes rear ID 2.
            e.objects_action(dict(action='cancel_correction'))
            click([50,40]);click([90,40])
            result=e.objects_action(dict(action='preview_correction',frame_id=e.background_frame['id'],
                target_id=7,correction_mode='point',rectangle=[[40,30],[60,50]]))
            self.assertEqual(result['preview_count'],1)
            self.assertEqual(e.correction_pending['indices'].tolist(),[0])
            click([90,40]);self.assertEqual(e.correction_pending['indices'].tolist(),[0,2])
            e.objects_action(dict(action='cancel_correction'))
            self.assertIsNone(e.correction_pending)
            self.assertEqual(e.labels().tolist(),[1,2,1])
            e.objects[0]['hidden']=True;e.frame('all')
            result=click([50,40])
            self.assertEqual(e.correction_pending['indices'].tolist(),[1])

    def test_fractional_box_edges_use_same_pixels_as_click(self):
        from backend.background import visible_box_ids, visible_pixel_ids
        owners=np.full((80,100),-1,dtype=np.int64);owners[40,50]=0
        click=visible_pixel_ids(owners,[50.2,40.8])
        box=visible_box_ids(owners,[[50.2,39.8],[54.2,43.8]])
        np.testing.assert_array_equal(box,click)
        np.testing.assert_array_equal(visible_box_ids(owners,[[54.2,43.8],[50.2,39.8]]),click)

    def test_ellipse_box_matches_click_on_actual_rasterized_footprint(self):
        with TemporaryDirectory() as folder:
            e=self.engine(folder);e.objects[0]['hidden']=False
            e.model.get_covariance=lambda scale:torch.tensor([[.0001,0,0,.0001,0,.0001]]*3)
            e.camera=lambda:SimpleNamespace(full_proj_transform=torch.eye(4),world_view_transform=torch.eye(4),
                                            FoVx=np.pi/2,FoVy=np.pi/2)
            e.frame('all')
            click=e.objects_action(dict(action='preview_correction',correction_mode='ellipse',target_id=7,
                frame_id=e.background_frame['id'],pixel=[66,40]))
            clicked=e.correction_pending['indices'].copy()
            self.assertEqual(click['preview_count'],1)
            box=e.objects_action(dict(action='preview_correction',correction_mode='ellipse',target_id=7,
                frame_id=e.background_frame['id'],rectangle=[[65,38],[69,42]]))
            np.testing.assert_array_equal(e.correction_pending['indices'],clicked)
            self.assertEqual(box['preview_count'],1)


    def test_point_center_uses_canvas_edges_and_points_pixel_centers(self):
        # Native projected center (49.5,39.5) is canvas (50,40).
        xyz=np.array([[0.,0.,1.]])
        rectangle=[[49.8,39.8],[53,43]]
        self.assertTrue(projected_box_mask(xyz,np.eye(4),100,80,rectangle)[0])
        # Point render snaps to pixel (50,40), whose canvas center is (50.5,40.5).
        rectangle=[[50.2,40.2],[53,43]]
        self.assertFalse(projected_box_mask(xyz,np.eye(4),100,80,rectangle)[0])
        self.assertTrue(projected_box_mask(xyz,np.eye(4),100,80,rectangle,snap_to_pixel=True)[0])
