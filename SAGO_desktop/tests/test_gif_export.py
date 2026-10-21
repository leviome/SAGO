"""Orbit GIF geometry, encoding, cancellation, and preservation of editing state."""
import copy
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from PIL import Image
import torch

from backend.engine import Engine, Cancelled
from backend.gif_export import orbit_geometry


class GifExportTest(unittest.TestCase):
    def fixture(self, folder):
        engine = Engine('unused.pt', folder)
        engine.torch = torch
        engine.model = SimpleNamespace(get_xyz=torch.tensor([[0., 0., 0.], [1., 0., 0.], [90., 90., 90.]]),
                                       get_scaling=torch.full((3, 3), .03))
        engine.selection = torch.tensor([True, True, False])
        engine.view = dict(up=[0., 0., 1.], yaw=15., pitch=40., fov=50., target=[5., 6., 7.], radius=20., size=768)
        engine.objects = [dict(id='test', number=1, name='Duck', hidden=True)]
        engine.instance_ids = np.array([1, 1, 0], dtype=np.int32)
        engine.capture = {'id': 'existing'}
        engine.render_mode = 'depth'
        engine.undo_stack = ['undo']
        def subset(model, selection):
            return SimpleNamespace(get_xyz=model.get_xyz[selection], get_scaling=model.get_scaling[selection])
        def camera(*args, **kwargs):
            return SimpleNamespace(world_view_transform=torch.eye(4), arguments=args, width=kwargs['width'])
        engine.utils = SimpleNamespace(masking_for_fg=Mock(side_effect=subset), sphere2params=Mock(side_effect=camera))
        calls = []
        def render(cam, model, **kwargs):
            calls.append((cam, model, kwargs))
            value = torch.ones(3, cam.width, cam.width)
            value[:, 16:96, 16:96] = torch.tensor([(len(calls) % 20)/20, .25, .75])[:, None, None]
            return {'render': value}
        engine.render_simple = Mock(side_effect=render)
        return engine, calls

    def test_geometry_fits_all_centers_and_extents_for_arbitrary_up(self):
        xyz = np.array([[0., 0., 0.], [3., 1., 2.]])
        scales = np.full_like(xyz, .2)
        center, up, radius = orbit_geometry(xyz, scales, [1., 2., 3.], 60.)
        np.testing.assert_allclose(center, xyz.mean(0))
        self.assertAlmostEqual(np.linalg.norm(up), 1.)
        self.assertTrue(np.all(np.linalg.norm(xyz-center, axis=1)+.6 < radius*.5))
        with self.assertRaises(ValueError):
            orbit_geometry(xyz, scales, [0, 0, 0], 60.)

    def test_sparse_giant_gaussians_do_not_dominate_framing(self):
        xyz = np.zeros((200, 3))
        xyz[:, 0] = np.linspace(-1, 1, 200)
        scaling = np.full_like(xyz, .02)
        scaling[-1] = 500
        _, _, radius = orbit_geometry(xyz, scaling, [0, 0, 1], 50.)
        self.assertLess(radius, 4.)

    def test_gif_frames_duration_target_and_state(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, calls = self.fixture(folder)
            before = copy.deepcopy((engine.view, engine.objects, engine.capture, engine.undo_stack))
            selection, labels = engine.selection.clone(), engine.labels().copy()
            progress = []
            result = engine.execute('gif', {'size':128, 'frames_per_orbit':4, 'fps':20,
                'render_mode':'points', 'show_background':False}, lambda n, message: progress.append((n, message)), threading.Event())
            self.assertEqual(result['frames'], 4)
            self.assertEqual(result['duration_seconds'], .2)
            self.assertEqual(len(calls), 4)
            for cam, model, options in calls:
                self.assertEqual(len(model.get_xyz), 2)
                self.assertEqual(options, {'bg': 'white'})
                np.testing.assert_allclose(cam.arguments[3], [.5, 0., 0.])
            np.testing.assert_allclose([call[0].arguments[2] for call in calls], np.full(4, np.deg2rad(60)))
            path = Path(folder)/result['download'].split('/')[-1]
            with Image.open(path) as gif:
                self.assertEqual(gif.size, (128, 128))
                self.assertEqual(gif.n_frames, 4)
                self.assertEqual(gif.info['loop'], 0)
                duration = 0
                for index in range(gif.n_frames):
                    gif.seek(index)
                    duration += gif.info['duration']
                    np.testing.assert_allclose(gif.convert('RGB').getpixel((0, 0)), [255, 255, 255])
                self.assertEqual(duration, 200)
            self.assertEqual(sorted(Path(folder).iterdir()), [path])
            self.assertEqual(before, (engine.view, engine.objects, engine.capture, engine.undo_stack))
            self.assertEqual(engine.render_mode, 'depth')
            self.assertTrue(engine.show_background)
            torch.testing.assert_close(engine.selection, selection)
            np.testing.assert_array_equal(engine.labels(), labels)
            self.assertEqual(progress[-1][0], 100)

    def test_custom_orbit_elevation_and_full_rotation(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, calls = self.fixture(folder)
            result = engine.execute('gif', {'size':128, 'frames_per_orbit':4,
                'elevation_degrees':-20.5}, lambda *args: None, threading.Event())
            self.assertEqual(result['elevation_degrees'], -20.5)
            np.testing.assert_allclose([c[0].arguments[2] for c in calls], np.full(4, np.deg2rad(-20.5)))
            np.testing.assert_allclose([c[0].arguments[1] for c in calls], np.deg2rad(15)+np.arange(4)*np.pi/2)

    def test_invalid_elevation_is_rejected_before_rendering(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, calls = self.fixture(folder)
            for value in (-90, 90, float('nan'), float('inf'), True, '60', None):
                with self.subTest(value=value), self.assertRaisesRegex(ValueError, '俯仰角'):
                    engine.execute('gif', {'elevation_degrees':value}, lambda *args: None, threading.Event())
            self.assertEqual(calls, [])

    def test_cancel_render_encode_and_encode_completion_cleanup(self):
        for stop in (1, 86, 99):
            with self.subTest(stop=stop), tempfile.TemporaryDirectory() as folder:
                engine, _ = self.fixture(folder)
                cancel = threading.Event()
                def progress(n, message):
                    if n >= stop:
                        cancel.set()
                if stop == 99:
                    original = Image.Image.save
                    def save(image, fp, *args, **kwargs):
                        original(image, fp, *args, **kwargs)
                        if str(fp).endswith('animation.gif'):
                            cancel.set()
                    from unittest.mock import patch
                    with patch.object(Image.Image, 'save', save), self.assertRaises(Cancelled):
                        engine.execute('gif', {'size':128, 'frames_per_orbit':4}, progress, cancel)
                else:
                    with self.assertRaises(Cancelled):
                        engine.execute('gif', {'size':128, 'frames_per_orbit':4}, progress, cancel)
                self.assertEqual(list(Path(folder).iterdir()), [])
                self.assertEqual(int(engine.selection.sum()), 2)

    def batch_fixture(self, folder):
        engine, calls = self.fixture(folder)
        engine.selection = None
        engine.instance_ids = np.array([1, 1, 2], dtype=np.int32)
        engine.objects.append(dict(id='second', number=2, name='桌子/../:?', hidden=False))
        engine.foreground_ids = set()  # The request, not stale server display state, defines the batch.
        engine.object_mask = lambda number: torch.from_numpy(engine.labels() == number)
        return engine, calls

    def test_checked_ids_are_independent_named_gifs_without_current_foreground(self):
        from urllib.parse import urlsplit, parse_qs
        with tempfile.TemporaryDirectory() as folder:
            engine, calls = self.batch_fixture(folder)
            before = copy.deepcopy((engine.view, engine.objects, engine.capture, engine.undo_stack))
            labels = engine.labels().copy()
            progress = []
            result = engine.execute('gif', {'source':'ids', 'ids':['second', 'test', 'second'],
                'size':128, 'frames_per_orbit':4}, lambda n, msg: progress.append((n, msg)), threading.Event())
            self.assertEqual(result['source'], 'ids')
            self.assertEqual([item['number'] for item in result['items']], [1, 2])
            self.assertEqual(len(calls), 8)
            for cam, model, options in calls[:4]:
                self.assertEqual(len(model.get_xyz), 2)
                np.testing.assert_allclose(cam.arguments[3], [.5, 0., 0.])
            for cam, model, options in calls[4:]:
                self.assertEqual(len(model.get_xyz), 1)
                np.testing.assert_allclose(cam.arguments[3], [90., 90., 90.])
            for item in result['items']:
                url = urlsplit(item['download'])
                path = Path(folder)/url.path.rsplit('/', 1)[-1]
                self.assertTrue(path.exists())
                self.assertEqual(Path(item['saved_path']), path.resolve())
                self.assertEqual(parse_qs(url.query)['filename'], [item['filename']])
                self.assertTrue(item['filename'].startswith('ID_'+str(item['number'])+'_'))
                self.assertFalse(any(ch in item['filename'] for ch in '/\\:?'))
                with Image.open(path) as image:
                    self.assertEqual(image.n_frames, 4)
            self.assertEqual(len(list(Path(folder).iterdir())), 2)
            self.assertEqual(before, (engine.view, engine.objects, engine.capture, engine.undo_stack))
            self.assertIsNone(engine.selection)
            np.testing.assert_array_equal(engine.labels(), labels)
            self.assertEqual([n for n, _ in progress], sorted(n for n, _ in progress))
            self.assertEqual(progress[-1][0], 100)
            self.assertTrue(any('ID 2' in msg for _, msg in progress))

    def test_all_objects_ignore_display_filters_and_requested_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, calls = self.batch_fixture(folder)
            engine.objects.reverse()
            result = engine.execute('gif', {'source':'all', 'ids':['missing'],
                'size':128, 'frames_per_orbit':4}, lambda *args: None, threading.Event())
            self.assertEqual(result['source'], 'all')
            self.assertEqual([item['number'] for item in result['items']], [1, 2])
            self.assertEqual(len(calls), 8)
            self.assertIsNone(engine.selection)
            self.assertEqual(engine.foreground_ids, set())
            self.assertTrue(next(obj for obj in engine.objects if obj['number'] == 1)['hidden'])

    def test_all_objects_require_saved_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, calls = self.fixture(folder)
            engine.objects = []
            with self.assertRaisesRegex(ValueError, '没有可导出的物体 ID'):
                engine.execute('gif', {'source':'all'}, lambda *args: None, threading.Event())
            self.assertEqual(calls, [])
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_batch_cancel_or_failure_removes_only_new_files(self):
        for failure in (False, True):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as folder:
                engine, calls = self.batch_fixture(folder)
                previous = Path(folder)/'previous.gif'
                previous.write_bytes(b'previous export')
                cancel = threading.Event()
                original = engine.render_simple.side_effect
                def render(*args, **kwargs):
                    if failure and len(calls) == 4:
                        raise RuntimeError('second object failed')
                    return original(*args, **kwargs)
                engine.render_simple.side_effect = render
                def progress(n, msg):
                    if not failure and n >= 60:
                        cancel.set()
                with self.assertRaises(RuntimeError if failure else Cancelled):
                    engine.execute('gif', {'source':'ids', 'ids':['test', 'second'],
                        'size':128, 'frames_per_orbit':4}, progress, cancel)
                self.assertEqual(list(Path(folder).iterdir()), [previous])
                self.assertEqual(previous.read_bytes(), b'previous export')
                self.assertIsNone(engine.selection)

    def test_batch_validates_sources_and_current_object_ids(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, _ = self.batch_fixture(folder)
            for data in ({'source':'bad'}, {'source':'ids'}, {'source':'ids','ids':[]},
                         {'source':'ids','ids':['missing']}, {'source':'ids','ids':[1]}):
                with self.assertRaises(ValueError):
                    engine.execute('gif', data, None, threading.Event())
            self.assertEqual(list(Path(folder).iterdir()), [])

    def test_empty_invalid_and_render_failure_leave_no_exports(self):
        with tempfile.TemporaryDirectory() as folder:
            engine, _ = self.fixture(folder)
            for data in ({'size':2048}, {'frames_per_orbit':1000}, {'fps':0}, {'size':True}, {'fps':24}):
                with self.assertRaises(ValueError):
                    engine.execute('gif', data, lambda *args: None, threading.Event())
            engine.selection.zero_()
            with self.assertRaisesRegex(ValueError, '非空选区'):
                engine.execute('gif', {}, None, threading.Event())
            engine.selection[:] = True
            engine.render_simple.side_effect = RuntimeError('test render failure')
            with self.assertRaisesRegex(RuntimeError, 'test render failure'):
                engine.execute('gif', {'size':128, 'frames_per_orbit':4}, lambda *args: None, threading.Event())
            self.assertEqual(list(Path(folder).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
