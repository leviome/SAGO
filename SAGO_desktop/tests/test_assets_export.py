"""Asset folders contain complete splat parameters, ordinary colored points, and a mesh."""
import json
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
from plyfile import PlyData, PlyElement
import torch

from backend.assets_export import save_object_assets
from backend.engine import Cancelled
from backend.mesh_export import generate_mesh
from test_mesh_export import dataset


class AssetsTest(unittest.TestCase):
    def fixture(self,root):
        root=Path(root);neus=root/'neus';(neus/'utils').mkdir(parents=True);(neus/'configs/nerf').mkdir(parents=True)
        (neus/'utils/mlp_weights.txt').write_text('0');(neus/'configs/nerf/workspace_mask002.json').write_text('{}')
        xyz=torch.tensor([[10.,20.,30.],[11.,21.,31.],[99.,99.,99.]])
        dc=torch.tensor([[[0.,1.,-1.]],[[1.,0.,-1.]],[[0.,0.,0.]]])
        def subset(model,mask):
            target=SimpleNamespace(get_xyz=xyz[mask],_features_dc=dc[mask])
            def save(path):
                points=target.get_xyz.numpy();values=np.empty(len(points),dtype=[('x','f4'),('y','f4'),('z','f4'),('opacity','f4'),('scale_0','f4'),('f_dc_0','f4')])
                for a,key in enumerate(('x','y','z')):values[key]=points[:,a]
                values['opacity']=-2.;values['scale_0']=-3.;values['f_dc_0']=target._features_dc[:,0,0].numpy()
                PlyData([PlyElement.describe(values,'vertex')]).write(path)
            target.save_ply=save;return target
        engine=SimpleNamespace(output=root/'out',torch=torch,model=object(),utils=SimpleNamespace(masking_for_fg=subset),
            objects=[dict(id='a',number=1,name='Duck',hidden=True)],require_scene=lambda:None,object_mask=lambda number:torch.tensor([True,True,False]))
        options=dict(ids=['a'],include_assets=True,neus2_root=str(neus),neus2_python=sys.executable,steps=100,size=128,views_per_orbit=8,resolution=64)
        return engine,options

    def test_gaussian_parameters_and_standard_point_colors(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,_=self.fixture(tmp);folder=engine.output/'ID_1';folder.mkdir(parents=True)
            result=save_object_assets(engine,engine.object_mask(1),folder,threading.Event())
            splats=PlyData.read(folder/'gaussians.ply')['vertex'].data;points=PlyData.read(folder/'point_cloud.ply')['vertex'].data
            self.assertEqual(len(splats),2);self.assertEqual(len(points),2)
            self.assertIn('opacity',splats.dtype.names);self.assertNotIn('opacity',points.dtype.names)
            np.testing.assert_allclose(splats['opacity'],[-2,-2]);np.testing.assert_allclose(splats['scale_0'],[-3,-3])
            self.assertEqual(points.dtype.names,('x','y','z','red','green','blue'))
            np.testing.assert_allclose(points['x'],[10,11]);self.assertEqual(tuple(points[0][key] for key in ('red','green','blue')),(128,199,56))
            self.assertEqual(result['gaussian_count'],2)

    def test_asset_batch_archive_is_complete_and_excludes_work_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,options=self.fixture(tmp)
            def render(e,selection,path,options,progress,cancel):dataset(path);return 32
            def reconstruct(pipeline,progress,cancel):
                raw=pipeline.output_dir/'raw.obj';raw.write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n');return raw
            with patch('backend.mesh_export.run_process'),patch('backend.mesh_export.render_dataset',side_effect=render),patch('backend.mesh_export.run_neus2',side_effect=reconstruct):
                result=generate_mesh(engine,options,lambda *a:None,threading.Event())
            self.assertEqual(result['format'],'assets');item=result['items'][0]
            self.assertEqual({p.name for p in Path(item['directory']).iterdir()},{'gaussians.ply','point_cloud.ply','ID_1_Duck.obj'})
            self.assertEqual({f['kind'] for f in item['files']},{'gaussians','point_cloud','mesh'})
            with zipfile.ZipFile(result['archive_path']) as archive:
                self.assertEqual(set(archive.namelist()),{'ID_1/gaussians.ply','ID_1/point_cloud.ply','ID_1/ID_1_Duck.obj','manifest.json'})
                self.assertIsNone(archive.testzip());manifest=json.loads(archive.read('manifest.json'));self.assertEqual(manifest['items'][0]['gaussian_count'],2)
            self.assertTrue((Path(result['output_directory'])/'.work/ID_1/dataset').is_dir())

    def test_cancel_packaging_removes_partial_final_assets(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine,options=self.fixture(tmp);event=threading.Event()
            def render(e,selection,path,options,progress,cancel):dataset(path);return 32
            def reconstruct(pipeline,progress,cancel):
                raw=pipeline.output_dir/'raw.obj';raw.write_text('v 0 0 0\nv .1 0 0\nv 0 .1 0\nf 1 2 3\n');return raw
            def cancelled(*args):event.set();raise Cancelled('cancel packaging')
            with patch('backend.mesh_export.run_process'),patch('backend.mesh_export.render_dataset',side_effect=render),patch('backend.mesh_export.run_neus2',side_effect=reconstruct),patch('backend.assets_export.archive_assets',side_effect=cancelled):
                with self.assertRaises(Cancelled):generate_mesh(engine,options,lambda *a:None,event)
            batch=next(engine.output.glob('assets-*'));self.assertEqual(list((batch/'ID_1').iterdir()),[])
            self.assertFalse((batch/'assets.zip').exists());self.assertFalse((batch/'manifest.json').exists())


if __name__=='__main__':unittest.main()
