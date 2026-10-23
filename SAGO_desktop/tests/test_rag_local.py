"""No-API export validates saved PLY identity and leaves scene files untouched."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from plyfile import PlyData,PlyElement

from backend.knowledge import SceneKnowledge

spec=importlib.util.spec_from_file_location('rag_local',Path(__file__).resolve().parents[1]/'scripts/rag-local.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)


class LocalRAGTest(unittest.TestCase):
    def scene(self,folder):
        root=Path(folder);scene=root/'scene.ply'
        xyz=np.array([[0,0,0],[1,0,0],[4,0,2]],dtype='<f4')
        data=np.empty(3,dtype=[('x','f4'),('y','f4'),('z','f4')])
        for i,key in enumerate(('x','y','z')):data[key]=xyz[:,i]
        PlyData([PlyElement.describe(data,'vertex')]).write(scene)
        objects=[dict(id='a',number=1,name='Chair',hidden=True),dict(id='b',number=2,name='Lamp',hidden=False)]
        knowledge=SceneKnowledge(scene)
        knowledge.save_segmentation(np.array([1,1,2],dtype=np.int32),objects,3,hashlib.sha256(xyz.tobytes()).hexdigest())
        return scene,knowledge

    def test_export_no_network_no_source_writes_and_real_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,k=self.scene(folder);before=(scene.read_bytes(),k.path.read_bytes())
            with patch('urllib.request.build_opener',side_effect=AssertionError('network forbidden')):
                packet=module.export_scene(scene,Path(folder)/'result','What is nearest to ID 1?')
            self.assertEqual(before,(scene.read_bytes(),k.path.read_bytes()))
            document=json.loads(Path(packet['spatial_document']).read_text())
            self.assertEqual(packet['mode'],'codex_manual_no_api');self.assertEqual(len(document['objects']),2)
            self.assertEqual(document['objects'][0]['centroid'],[.5,0,0])
            self.assertEqual(document['objects'][0]['nearest_relations'][0]['to_id'],2)
            self.assertEqual(document['objects'][0]['semantic']['source'],'user_name')
            self.assertTrue((Path(folder)/'result/request.json').is_file())
            self.assertEqual(module.facts(document,[1])['facts'][0]['relations'][0]['centroid_distance'],np.linalg.norm([3.5,0,2]))

    def test_folder_record_selection_and_fingerprint_rejection(self):
        with tempfile.TemporaryDirectory() as folder:
            scene,k=self.scene(folder)
            self.assertTrue(module.export_scene(folder,Path(folder)/'result','count')['object_catalog'])
            data=k.read();data['scenes'][k.key]['segmentation']['gaussian_order_sha256']='wrong'
            k.path.write_text(json.dumps(data))
            with self.assertRaises(ValueError):module.export_scene(scene,Path(folder)/'result','count')


if __name__=='__main__':unittest.main()
