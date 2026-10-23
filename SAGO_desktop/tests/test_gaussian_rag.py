"""Spatial geometry, stale data, API tools, cancellation, persistence and credentials."""
import copy
import io
import json
import os
from pathlib import Path
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error

import numpy as np
import torch

from backend import gaussian_rag as rag
from backend.engine import Cancelled
import test_semantic as semantic_fixtures


class GaussianRAGTest(unittest.TestCase):
    def setUp(self):
        fixture=semantic_fixtures.SemanticTest();fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.engine=fixture.engine;self.cancel=threading.Event();self.progress=Mock()
        self.api=dict(base_url='https://example.invalid/v1',model='vision-test',key='secret-never-persist')
        self.engine.ground_plane=None

    def call(self,operation,**data):
        return rag.rag_action(self.engine,dict(operation=operation,**data),self.progress,self.cancel)

    def build(self):return self.call('build')

    def tool(self,ids,call_id='t1'):
        return dict(role='assistant',content=None,tool_calls=[dict(id=call_id,type='function',function=dict(name='get_spatial_facts',arguments=json.dumps(dict(ids=ids))))])

    def test_build_persists_scene_geometry_and_ids_without_api_or_secrets(self):
        e=self.engine;selection=e.selection.clone();view=copy.deepcopy(e.view);objects=copy.deepcopy(e.objects)
        e.knowledge.save_record(unrelated='keep')
        result=self.call('build',api=self.api)
        path=Path(result['saved_path']);document=json.loads(path.read_text())
        self.assertEqual(len(document['objects']),2)
        self.assertEqual(document['objects'][0]['centroid'],[.5,0.,0.])
        self.assertEqual(document['objects'][0]['gaussian_count'],2)
        self.assertEqual(document['objects'][0]['nearest_relations'][0]['to_id'],2)
        self.assertIn('metric scale unknown',document['units'])
        scene=e.knowledge.read()['scenes'][e.knowledge.key]
        self.assertEqual(scene['gaussian_rag'],document);self.assertEqual(scene['segmentation']['gaussian_ids'],[1,1,2])
        self.assertEqual(scene['unrelated'],'keep');self.assertNotIn(self.api['key'],e.knowledge.path.read_text()+path.read_text())
        self.assertFalse(e.ids_dirty);self.assertEqual(e.objects,objects);self.assertEqual(e.view,view)
        self.assertTrue(torch.equal(e.selection,selection))

    def test_geometry_relations_respect_up_and_are_not_physical_claims(self):
        self.engine.view['up']=[0,1,0]
        doc=rag.geometry_document(self.engine,self.cancel)
        a,b=doc['objects'];rel=rag.relation(a,b)
        self.assertAlmostEqual(rel['centroid_distance'],np.linalg.norm(np.array([90,90,90])-[.5,0,0]))
        self.assertEqual(rel['vertical_relation'],'below');self.assertFalse(rel['horizontal_box_overlap'])
        self.assertFalse(rel['spatial_boxes_overlap']);self.assertGreater(rel['spatial_box_gap_lower_bound'],0)
        self.assertNotIn('touching',rel);self.assertNotIn('supported_by',rel)
        backwards=rag.relation(b,a);self.assertEqual(backwards['vertical_relation'],'above')
        self.assertEqual(rag.relation(a,a)['spatial_box_contains_other'],True)
        facts=rag.facts(doc,[2,1]);self.assertEqual(facts['facts'][0]['object']['number'],2)

    def test_api_tools_use_authoritative_facts_and_return_citations(self):
        self.build()
        with patch.object(rag,'chat',side_effect=[self.tool([1]),dict(content='The nearest is Second [ID 2].')]) as chat:
            result=self.call('ask',question='What is nearest to ID 1?',api=self.api)
        self.assertEqual(result['referenced_ids'],[2]);self.assertEqual(result['retrieved_ids'],[1])
        messages=chat.call_args.args[1]
        returned=json.loads(messages[-1]['content'])
        self.assertEqual(returned['facts'][0]['relations'][0]['to_id'],2)
        self.assertNotIn(self.api['key'],json.dumps(messages))
        self.assertIn('NOT metres',messages[0]['content'])
        self.assertIsNone(self.engine.semantic_highlight)

    def test_model_cannot_cite_nonexistent_id_or_empty_answer(self):
        self.build()
        for content in ('Object [ID 999]',None,''):
            with patch.object(rag,'chat',return_value=dict(content=content)):
                with self.assertRaises(ValueError):self.call('ask',question='What?',api=self.api)

    def test_tool_errors_are_data_and_tool_round_limit_is_enforced(self):
        self.build()
        with patch.object(rag,'chat',side_effect=[self.tool([999]),dict(content='Insufficient evidence.')]) as chat:
            result=self.call('ask',question='Where is ID 999?',api=self.api)
        self.assertEqual(result['retrieved_ids'],[])
        self.assertIn('Unknown object ID',chat.call_args.args[1][-1]['content'])
        with patch.object(rag,'chat',return_value=self.tool([1])):
            with self.assertRaisesRegex(ValueError,'six API rounds'):self.call('ask',question='Where?',api=self.api)

    def test_stale_names_masks_ground_and_scene_reject_query_before_network(self):
        self.build()
        with patch.object(rag,'chat') as chat:
            self.engine.objects[0]['name']='renamed'
            with self.assertRaisesRegex(ValueError,'outdated'):self.call('ask',question='Where?',api=self.api)
            chat.assert_not_called()
        self.build();self.engine.instance_ids[:]=[1,2,2]
        with self.assertRaisesRegex(ValueError,'outdated'):rag.load_document(self.engine)
        self.build();self.engine.view['up']=[0,1,0]
        with self.assertRaisesRegex(ValueError,'outdated'):rag.load_document(self.engine)
        self.build();self.engine.gaussian_fingerprint='other scene'
        with self.assertRaisesRegex(ValueError,'outdated'):rag.load_document(self.engine)

    def test_hidden_state_is_not_geometry_change_and_separate_ply_has_no_index(self):
        self.build();self.engine.objects[0]['hidden']=False
        self.assertEqual(len(rag.load_document(self.engine)['objects']),2)
        from backend.knowledge import SceneKnowledge
        old=rag.spatial_path(self.engine)
        self.engine.knowledge=SceneKnowledge(self.engine.knowledge.path.parent/'another.ply')
        self.assertNotEqual(old,rag.spatial_path(self.engine))
        with self.assertRaisesRegex(ValueError,'first'):rag.load_document(self.engine)

    def test_optional_vision_uses_four_views_and_caches_descriptions(self):
        response=dict(content=json.dumps(dict(label='sphere',description='a red sphere',color='red')))
        with patch.object(rag,'chat',return_value=response) as chat:
            self.call('build',describe=True,api=self.api)
            self.assertEqual(chat.call_count,2)
            message=chat.call_args.args[1][0]
            self.assertEqual(sum(item['type']=='image_url' for item in message['content']),4)
            doc=rag.load_document(self.engine)
            self.assertEqual(doc['objects'][0]['semantic']['label'],'sphere')
            self.assertEqual(doc['objects'][0]['semantic']['source'],'api_vision')
            self.call('build',describe=True,api=self.api);self.assertEqual(chat.call_count,2)
            self.call('build');self.assertEqual(rag.load_document(self.engine)['objects'][0]['semantic']['label'],'sphere')
            self.engine.instance_ids[:]=[1,2,2]
            self.call('build');self.assertEqual(rag.load_document(self.engine)['objects'][0]['semantic']['source'],'user_name')

    def test_failed_or_cancelled_vision_keeps_previous_knowledge(self):
        self.build();before=self.engine.knowledge.path.read_bytes();e=self.engine;e.ids_dirty=True
        with patch.object(rag,'chat',return_value=dict(content='not JSON')):
            with self.assertRaisesRegex(ValueError,'valid object-description'):self.call('build',describe=True,api=self.api)
        self.assertEqual(e.knowledge.path.read_bytes(),before);self.assertTrue(e.ids_dirty)
        self.cancel.set()
        with self.assertRaises(Cancelled):self.call('build')
        self.assertEqual(e.knowledge.path.read_bytes(),before)

    def test_ground_heights_and_malformed_tool_calls(self):
        self.engine.ground_plane=dict(normal=[0,0,1],offset=-2)
        doc=rag.geometry_document(self.engine,self.cancel)
        self.assertEqual(doc['objects'][0]['ground_height']['min'],-2)
        self.assertEqual(doc['objects'][1]['ground_height']['max'],88)
        self.build()
        with patch.object(rag,'chat',return_value=dict(content=None,tool_calls=['invalid'])):
            with self.assertRaisesRegex(ValueError,'tool-call format'):self.call('ask',question='Where?',api=self.api)
        self.engine.view['up']=[0,0,0]
        with self.assertRaisesRegex(ValueError,'up vector'):rag.geometry_document(self.engine,self.cancel)

    def test_environment_configuration_and_test_without_scene(self):
        with patch.dict(os.environ,dict(SAGO_RAG_BASE_URL=self.api['base_url'],SAGO_RAG_MODEL=self.api['model'],SAGO_RAG_API_KEY=self.api['key'])):
            config=rag.api_options({});self.assertEqual(config['url'],'https://example.invalid/v1/chat/completions')
            e=self.engine;e.model=None;e.objects=[]
            with patch.object(rag,'chat',return_value=dict(content='OK')):
                self.assertTrue(self.call('test')['connected'])
        for api in (dict(self.api,base_url='ftp://bad'),dict(self.api,base_url='https://user:pass@example.com'),dict(self.api,key='a\nb')):
            with self.assertRaises(ValueError):rag.api_options(dict(api=api))

    def test_transport_json_tools_auth_and_sanitized_http_error(self):
        options=rag.api_options(dict(api=self.api));response=io.BytesIO(json.dumps(dict(choices=[dict(message=dict(content='OK'))])).encode())
        opener=Mock();opener.open.return_value=response
        with patch('urllib.request.build_opener',return_value=opener):
            self.assertEqual(rag.chat(options,[dict(role='user',content='hi')],self.cancel,[rag.TOOL])['content'],'OK')
        request=opener.open.call_args.args[0]
        self.assertEqual(request.get_header('Authorization'),'Bearer '+self.api['key'])
        self.assertEqual(json.loads(request.data)['tools'][0]['function']['name'],'get_spatial_facts')
        opener.open.side_effect=urllib.error.HTTPError(options['url'],401,'secret body',{},None)
        with patch('urllib.request.build_opener',return_value=opener):
            with self.assertRaisesRegex(ValueError,'HTTP 401') as caught:rag.chat(options,[],self.cancel)
        self.assertNotIn('secret body',str(caught.exception));self.assertNotIn(self.api['key'],str(caught.exception))

    def test_invalid_questions_and_tool_requests(self):
        self.build()
        for question in ('',None,'x'*4001):
            with self.assertRaises(ValueError):self.call('ask',question=question,api=self.api)
        doc=rag.load_document(self.engine)
        for ids in ([],[True],[999],None,[1]*21):
            with self.assertRaises(ValueError):rag.facts(doc,ids)
        self.engine.capture={}
        with self.assertRaisesRegex(ValueError,'Exit prompts'):self.call('build')


if __name__=='__main__':unittest.main()
