"""Identity and prompt validation regressions, without loading model weights."""
import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from PIL import Image

from backend.engine import Engine
from backend.sam3_worker import DroneSessions, seed


class Predictor:
    def __init__(self, tracked_id=7):
        self.requests = []
        self.tracked_id = tracked_id
        self.model = object()
        memory = []
        self.state = dict(input_batch=SimpleNamespace(img_batch=SimpleNamespace(
            tensors=torch.zeros((4, 3, 4, 6)))), memory=memory, alias=memory)
        self.masks = np.zeros((2, 4, 6), bool)
        self.masks[0, :, :3] = True
        self.masks[1, :, 3:] = True

    def handle_request(self, request):
        self.requests.append(request)
        if request['type'] == 'start_session':
            return {'session_id': 'test'}
        if request['type'] == 'add_prompt':
            return {'outputs': dict(out_obj_ids=np.array([7, 9]),
                    out_binary_masks=self.masks, out_probs=np.array([.8, .9]))}
        return {}

    def _get_session(self, session):
        return {'state': self.state}

    def handle_stream_request(self, request):
        yield dict(frame_index=1, outputs=dict(out_obj_ids=np.array([self.tracked_id]),
                                              out_binary_masks=self.masks[:1]))


class Sam3Test(unittest.TestCase):
    def test_text_and_spatial_hints_filter_instances_and_keep_raw_ids(self):
        p = Predictor()
        prompts = dict(text='toy', box=[], foreground=[(.1, .5)], background=[(.8, .5)])
        ids, masks, scores, all_ids = seed(p, 'test', prompts)
        self.assertEqual(ids.tolist(), [7])
        self.assertEqual(all_ids.tolist(), [7, 9])
        self.assertEqual(p.requests[0]['text'], 'toy')
        self.assertNotIn('points', p.requests[0])

    def test_point_only_uses_instance_prompt_and_box_uses_xywh(self):
        p = Predictor()
        seed(p, 'test', dict(text='', box=[], foreground=[(.1, .5)], background=[]))
        self.assertEqual(p.requests[-1]['obj_id'], 1)
        self.assertEqual(p.requests[-1]['point_labels'], [1])
        seed(p, 'test', dict(text='toy', box=[(.1, .2), (.8, .9)], foreground=[], background=[]))
        np.testing.assert_allclose(p.requests[-1]['bounding_boxes'], [[.1, .2, .7, .7]])

    def route_fixture(self, root, found=7):
        p = Predictor(tracked_id=found)
        image = root/'frame.png'
        Image.new('RGB', (6, 4)).save(image)
        np.save(root/'target.npy', p.masks[0])
        calls = []
        def prepare(state, image, model):
            return torch.ones((3, 4, 6))
        def advance(model, state, index):
            calls.append((index, list(state['memory'])))
            state['memory'].append(index)
            self.assertIs(state['memory'], state['alias'])
            return dict(out_obj_ids=np.array([p.tracked_id]), out_binary_masks=p.masks[:1])
        sessions = DroneSessions(p, prepare=prepare, advance=advance)
        route = sessions.start(str(image), dict(text='toy', box=[], foreground=[], background=[]),
                               str(root/'target.npy'))['route']
        return p, sessions, route, calls, image

    def test_incremental_steps_do_not_reseed_or_replay_and_branches_are_independent(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            p, sessions, route, calls, image = self.route_fixture(root)
            for side in (0, 1, 0, 1, 0, 1):
                sessions.step(route, side, str(image), str(root/'out.npz'))
                sessions.finish(route, side, True)
            self.assertEqual(calls, [(0, []), (1, [0]), (1, [0]), (2, [0, 1]), (2, [0, 1]), (3, [0, 1, 2]), (3, [0, 1, 2])])
            self.assertEqual(sum(r['type']=='start_session' for r in p.requests), 1)
            self.assertEqual(sum(r['type']=='add_prompt' for r in p.requests), 1)
            self.assertEqual(sessions.stats(route)['new_frames'], 6)
            self.assertEqual(sessions.stats(route)['seed_runs'], 1)
            sessions.close(route)
            self.assertFalse(sessions.routes)

    def test_plain_tensor_frame_storage_used_by_sam3(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, sessions, route, _, image = self.route_fixture(root)
            state = sessions.routes[route]['states'][0]
            state['input_batch'].img_batch = state['input_batch'].img_batch.tensors
            sessions.step(route, 0, str(image), str(root/'out.npz'))
            sessions.finish(route, 0, True)
            self.assertGreater(int(sessions.routes[route]['states'][0]['input_batch'].img_batch[1].sum()), 0)

    def test_reject_and_failed_step_leave_committed_memory_unchanged(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            _, sessions, route, calls, image = self.route_fixture(root)
            sessions.step(route, 0, str(image), str(root/'out.npz'))
            sessions.finish(route, 0, True)
            sessions.step(route, 0, str(image), str(root/'out.npz'))
            sessions.finish(route, 0, False)
            state = sessions.routes[route]['states'][0]
            self.assertEqual(state['memory'], [0, 1])
            self.assertEqual(int(state['input_batch'].img_batch.tensors[2].sum()), 0)
            advance = sessions.advance
            def failing(model, proposed, index):
                proposed['memory'].append(99)
                raise RuntimeError('inference failed')
            sessions.advance = failing
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                sessions.step(route, 0, str(image), str(root/'out.npz'))
            self.assertEqual(state['memory'], [0, 1])
            self.assertFalse(sessions.routes[route]['pending'])
            sessions.advance = advance
            sessions.step(route, 0, str(image), str(root/'out.npz'))
            sessions.finish(route, 0, True)
            self.assertEqual(calls, [(0, []), (1, [0]), (2, [0, 1]), (2, [0, 1])])

    def test_tracking_never_switches_to_other_instances(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for found, expected in ((7, 12), (9, 0)):
                _, sessions, route, _, image = self.route_fixture(root, found=found)
                sessions.step(route, 0, str(image), str(root/'out.npz'))
                with np.load(root/'out.npz') as output:
                    self.assertEqual(int(output['masks'].sum()), expected)
                sessions.close(route)  # also discards a pending, uncommitted step
                self.assertFalse(sessions.routes)

    def test_invalid_requests_fail_before_loading_models(self):
        with tempfile.TemporaryDirectory() as root:
            e = Engine('unused.pt', root)
            e.captured = lambda data: dict(rgb=np.zeros((4, 6, 3), np.uint8))
            for data in ({'model':'unknown'},
                         {'model':'sam3','text':1}, {'model':'sam3','text':'x'*257},
                         {'model':'sam3','text':'duck','foreground':[[float('nan'),0]]},
                         {'model':'sam3'}):
                with self.subTest(data=data), self.assertRaises(ValueError):
                    e.preview(data)
