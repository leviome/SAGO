"""Mesh reconstruction contracts: real COLMAP conversion, world coordinates and cancellation."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
import torch

from backend.engine import Engine, Cancelled
from backend.mesh_distillation import MeshDistillation
from backend.mesh_export import generate_mesh, render_dataset, run_process, settings


def dataset(folder):
    folder=Path(folder);(folder/'images').mkdir(parents=True,exist_ok=True);(folder/'masks').mkdir(exist_ok=True)
    Image.new('RGB',(16,16),(50,100,200)).save(folder/'images/000.png')
    Image.new('L',(16,16),255).save(folder/'masks/000.png')
    sparse=folder/'sparse/0';sparse.mkdir(parents=True,exist_ok=True)
    (sparse/'cameras.txt').write_text('1 PINHOLE 16 16 10 10 8 8\n')
    (sparse/'images.txt').write_text('1 1 0 0 0 -10 -20 -25 1 000.png\n0 0 -1\n')
    (sparse/'points3D.txt').write_text('1 9 19 29 255 0 0 0\n2 11 21 31 0 255 0 0\n')


class MeshTest(unittest.TestCase):
    def config(self,root):
        root=Path(root);(root/'utils').mkdir();(root/'configs/nerf').mkdir(parents=True)
        (root/'utils/mlp_weights.txt').write_text('0')
        (root/'configs/nerf/workspace_mask002.json').write_text('{}')
        return dict(neus2_root=str(root),neus2_python=sys.executable,steps=100,size=128,views_per_orbit=8,resolution=64)

    def test_colmap_poses_masks_and_world_mesh_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);dataset(root/'data')
            pipeline=MeshDistillation(root/'data',root/'out',neus2_root=root,python_executable=sys.executable)
            pipeline.prepare_dataset()
            transform=json.loads(pipeline.transform_path.read_text())
            np.testing.assert_allclose(pipeline.center,[10,20,30])
            frame=transform['frames'][0];matrix=np.array(frame['transform_matrix'])
            np.testing.assert_allclose(matrix[:3,3]*pipeline.radius+pipeline.center,[10,20,25])
            np.testing.assert_allclose(np.array(frame['intrinsic_matrix'])[:3,:3],[[10,0,8],[0,10,8],[0,0,1]])
            with Image.open(pipeline.scene_dir/frame['file_path']) as image:self.assertEqual(image.mode,'RGBA');self.assertEqual(image.getpixel((0,0)),(50,100,200,255))
            raw=root/'raw.obj';raw.write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nv 8 8 8\nf 1 2 3\n')
            stats=pipeline.postprocess_mesh(raw)
            self.assertEqual((stats.vertices,stats.faces,stats.removed_vertices),(3,1,1))
            vertices=[list(map(float,line.split()[1:4])) for line in stats.path.read_text().splitlines() if line.startswith('v ')]
            np.testing.assert_allclose(vertices,np.array([[0,0,0],[1,0,0],[0,1,0]])*pipeline.radius+[10,20,30])
            raw.write_text('v 0 0 0\nf 1 2 3\n')
            with self.assertRaises(ValueError):pipeline.postprocess_mesh(raw)

    def test_render_isolates_saved_id_and_preserves_interaction(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine=Engine('unused.pt',tmp);engine.torch=torch
            engine.model=SimpleNamespace(get_xyz=torch.tensor([[0.,0.,0.],[1.,1.,1.],[20.,20.,20.]]),get_scaling=torch.full((3,3),.02))
            engine.view=dict(up=[1.,2.,3.],yaw=23.,fov=60.)
            before=copy.deepcopy(engine.view)
            def subset(model,mask):return SimpleNamespace(get_xyz=model.get_xyz[mask],get_scaling=model.get_scaling[mask])
            def camera(*args,**kwargs):return SimpleNamespace(world_view_transform=torch.eye(4),image_width=128,image_height=128)
            def render(cam,target,**kwargs):
                self.assertEqual(len(target.get_xyz),2)
                value=torch.zeros(3,128,128);value[:,20:100,30:90]=1 if kwargs.get('override_color') is not None else .2
                return dict(render=value,alpha=None)
            engine.utils=SimpleNamespace(masking_for_fg=subset,sphere2params=camera,save_cams_to_colmap=Mock())
            engine.render_simple=render
            count=render_dataset(engine,torch.tensor([True,True,False]),Path(tmp)/'data',dict(views=8,size=128),lambda *a:None,threading.Event())
            self.assertEqual(count,32);self.assertEqual(engine.view,before)
            with Image.open(Path(tmp)/'data/masks/000.png') as mask:
                self.assertEqual(mask.getpixel((50,50)),255);self.assertEqual(mask.getpixel((0,0)),0)
            self.assertEqual(engine.utils.save_cams_to_colmap.call_count,1)
            self.assertEqual(len(engine.utils.save_cams_to_colmap.call_args.args[0]),32)

    def test_process_cancel_stops_child_and_reports_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);event=threading.Event()
            timer=threading.Timer(.4,event.set);timer.start();started=time.monotonic()
            with self.assertRaises(Cancelled):run_process([sys.executable,'-c','import time;time.sleep(30)'],root,root/'cancel.log',lambda *a:None,event)
            timer.join();self.assertLess(time.monotonic()-started,4)
            with self.assertRaisesRegex(RuntimeError,'deliberate failure'):
                run_process([sys.executable,'-c',"raise RuntimeError('deliberate failure')"],root,root/'fail.log',lambda *a:None,threading.Event())

    def test_parallel_overlap_order_progress_and_failure_cancellation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'neus').mkdir();data=self.config(root/'neus')
            engine=SimpleNamespace(output=root/'exports',objects=[dict(id='a',number=1,name='A'),dict(id='b',number=2,name='B')],require_scene=lambda:None,object_mask=lambda n:torch.tensor([True]))
            main=threading.get_ident();barrier=threading.Barrier(2,timeout=3);threads=set();values=[]
            def render(e,mask,path,options,progress,cancel):
                self.assertEqual(threading.get_ident(),main);dataset(path);progress(100,'Rendered');return 32
            def reconstruct(pipeline,progress,cancel):
                threads.add(threading.get_ident());barrier.wait();progress(50,'Halfway')
                raw=pipeline.output_dir/'raw.obj';raw.write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n');return raw
            with patch('backend.mesh_export.run_process'),patch('backend.mesh_export.render_dataset',side_effect=render),patch('backend.mesh_export.run_neus2',side_effect=reconstruct):
                result=generate_mesh(engine,dict(data,ids=['b','a'],parallel_jobs=2),lambda n,m:values.append(n),threading.Event())
            self.assertEqual(len(threads),2);self.assertEqual(result['parallel_jobs'],2)
            self.assertEqual([i['number'] for i in result['items']],[1,2]);self.assertEqual(values,sorted(values));self.assertEqual(values[-1],100)
            barrier=threading.Barrier(2,timeout=3);sibling_stopped=threading.Event()
            def fail(pipeline,progress,cancel):
                barrier.wait()
                if pipeline.name=='ID_2':raise RuntimeError('parallel worker failed')
                self.assertTrue(cancel.wait(3));sibling_stopped.set();raise Cancelled('sibling stopped')
            with patch('backend.mesh_export.run_process'),patch('backend.mesh_export.render_dataset',side_effect=render),patch('backend.mesh_export.run_neus2',side_effect=fail):
                with self.assertRaisesRegex(RuntimeError,'parallel worker failed'):
                    generate_mesh(engine,dict(data,ids=['a','b'],parallel_jobs=2),lambda *a:None,threading.Event())
            self.assertTrue(sibling_stopped.is_set())
            failed=next(b for b in engine.output.glob('mesh-*') if not (b/'manifest.json').exists())
            self.assertEqual(list(failed.glob('ID_*/*.obj')),[])

    def test_user_cancel_propagates_to_every_parallel_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'neus').mkdir();data=self.config(root/'neus')
            engine=SimpleNamespace(output=root/'exports',objects=[dict(id='a',number=1,name='A'),dict(id='b',number=2,name='B')],require_scene=lambda:None,object_mask=lambda n:torch.tensor([True]))
            event=threading.Event();barrier=threading.Barrier(2,timeout=3);stopped=set()
            def render(e,mask,path,options,progress,cancel):dataset(path);return 32
            def worker(pipeline,progress,cancel):
                barrier.wait();event.set()
                while not cancel.is_set():cancel.wait(.01)
                stopped.add(pipeline.name);raise Cancelled('cancelled')
            with patch('backend.mesh_export.run_process'),patch('backend.mesh_export.render_dataset',side_effect=render),patch('backend.mesh_export.run_neus2',side_effect=worker):
                with self.assertRaises(Cancelled):generate_mesh(engine,dict(data,ids=['a','b']),lambda *a:None,event)
            self.assertEqual(stopped,{'ID_1','ID_2'});self.assertFalse(list(engine.output.glob('mesh-*/ID_*/*.obj')))

    def test_batch_saved_hidden_objects_and_no_partial_mesh_on_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'neus').mkdir();data=self.config(root/'neus')
            engine=SimpleNamespace(output=root/'exports',objects=[dict(id='a',number=1,name='A',hidden=True),dict(id='b',number=2,name='B',hidden=False)],require_scene=lambda:None,object_mask=lambda n:torch.tensor([n==1,n==2]))
            def render(e,mask,path,options,progress,cancel):dataset(path);return 32
            def reconstruct(pipeline,progress,cancel):
                raw=pipeline.output_dir/'raw.obj';raw.write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n');return raw
            with patch('backend.mesh_export.run_process'),patch('backend.mesh_export.render_dataset',side_effect=render),patch('backend.mesh_export.run_neus2',side_effect=reconstruct):
                result=generate_mesh(engine,dict(data,ids=['b','a','a']),lambda *a:None,threading.Event())
            self.assertEqual([i['number'] for i in result['items']],[1,2]);self.assertTrue(all(Path(i['saved_path']).is_file() for i in result['items']))
            def fail_second(pipeline,progress,cancel):
                if pipeline.name=='ID_2':raise RuntimeError('second object failed')
                return reconstruct(pipeline,progress,cancel)
            with patch('backend.mesh_export.run_process'),patch('backend.mesh_export.render_dataset',side_effect=render),patch('backend.mesh_export.run_neus2',side_effect=fail_second):
                with self.assertRaisesRegex(RuntimeError,'second object'):generate_mesh(engine,dict(data,ids=['a','b']),lambda *a:None,threading.Event())
            batches=list(engine.output.glob('mesh-*'));self.assertEqual(len(batches),2)
            self.assertEqual(sum(len(list(b.glob('ID_*/*.obj'))) for b in batches),2)
            for invalid in ([],['missing'],[1]):
                with self.assertRaises(ValueError):generate_mesh(engine,dict(data,ids=invalid),lambda *a:None,threading.Event())
            for key in ('steps','resolution','views_per_orbit','size','parallel_jobs'):
                with self.assertRaises(ValueError):settings(dict(data,**{key:True}))


if __name__=='__main__':unittest.main()
