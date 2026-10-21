"""Web adapter around the vendored SAGO renderer, SAM2 and virtual drones.

Only the single GPU worker calls this class. No OpenGL or ImGui is required.
"""
import base64
import hashlib
import colorsys
import json
from collections import deque
from contextlib import ExitStack
from pathlib import Path
import uuid
import time

import numpy as np


class Cancelled(RuntimeError):
    pass


class Engine:
    def __init__(self, checkpoint, output):
        self.checkpoint = Path(checkpoint).expanduser()
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)
        self.model = self.tracker = self.capture = self.selection = None
        self.objects = []
        self.instance_ids = None
        self.foreground_ids = None
        self.id_colors = False
        self.show_background = self.show_foreground = True
        self.render_mode = 'rgb'
        self.background_frame = None
        self.correction_pending = None
        self.next_object_id = 1
        self.ids_dirty = self.ids_restored = False
        self.gaussian_fingerprint = None
        self.undo_stack, self.redo_stack = [], []
        self.scene_version = 0
        self.view = None
        self.scene_name = ''
        self.camera_entries = []
        self.ground_plane = self.ground_capture = None
        self.knowledge = None
        self.knowledge_warning = ''
        self.ground_restored = False
        self.drone_path = None
        self._bbox_selection = self._bbox = None
        self._bbox_up = None

    def initialize(self):
        if hasattr(self, 'torch'):
            return
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA 不可用，请在可访问 NVIDIA GPU 的 Python 环境启动。')
        from gs.scene.gaussian_model import GaussianModel
        from gs.gaussian_renderer import render_simple
        from sago_core import sago_utils as utils
        from sago_core.CrossViewTracker.cross_matcher import CrossMatch
        from sago_core.virtual_drone import VirtualDroneWithMemory
        self.torch, self.GaussianModel = torch, GaussianModel
        self.render_simple, self.utils = render_simple, utils
        self.CrossMatch, self.Drone = CrossMatch, VirtualDroneWithMemory

    @staticmethod
    def image_url(rgb):
        import cv2
        ok, encoded = cv2.imencode('.png', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError('图像编码失败')
        return 'data:image/png;base64,' + base64.b64encode(encoded).decode()

    def rgb(self, tensor):
        return (tensor.detach().permute(1, 2, 0).clamp(0, 1).cpu().numpy()*255).astype(np.uint8)

    def require_scene(self):
        if self.model is None:
            raise ValueError('请先加载场景')

    def labels(self):
        if self.instance_ids is None:
            self.instance_ids = np.zeros(len(self.model.get_xyz), dtype=np.int32)
        return self.instance_ids

    def object_mask(self, number):
        return self.torch.from_numpy(self.labels() == number).cuda()

    def active(self):
        hidden = [o['number'] for o in self.objects if o['hidden']]
        return self.torch.from_numpy(~np.isin(self.labels(), hidden)).cuda()

    @staticmethod
    def dimensions(view):
        aspect = float(view.get("aspect", 1.))
        size = view["size"]
        return (size, max(64, round(size/aspect))) if aspect >= 1 else (max(64, round(size*aspect)), size)

    def camera(self):
        v = self.view
        width, height = self.dimensions(v)
        cam = self.utils.sphere2params(v['radius'], np.deg2rad(v['yaw']), np.deg2rad(v['pitch']),
            self.torch.tensor(v['target'], dtype=self.torch.float32), self.torch.tensor(v['up'], dtype=self.torch.float32),
            np.deg2rad(v['fov']), width=width, height=height)
        # CustomCam's original camera_center is derived from the projection;
        # use the actual inverse view translation for correct SH evaluation.
        cam.camera_center = cam.world_view_transform.inverse()[3, :3]
        return cam

    def snapshot(self):
        return dict(scene=self.scene_name, version=self.scene_version, camera=self.view,
            cameras=[str(e.get('img_name', i)) for i, e in enumerate(self.camera_entries)],
            count=0 if self.model is None else len(self.model.get_xyz),
            selected=0 if self.selection is None else int(self.selection.sum()),
            objects=[dict(id=o['id'], number=o['number'], name=o['name'], hidden=o['hidden'], color='#'+''.join(f'{round(c*255):02x}' for c in self.object_color(o)), count=int(np.count_nonzero(self.labels() == o['number']))) for o in self.objects],
            canUndo=bool(self.undo_stack), canRedo=bool(self.redo_stack), ground=self.ground_plane,
            knowledge_path=str(self.knowledge.path) if self.knowledge is not None else None,
            knowledge_warning=self.knowledge_warning, ground_restored=self.ground_restored,
            ids_dirty=self.ids_dirty, ids_restored=self.ids_restored, next_object_id=self.next_object_id,
            render_mode=self.render_mode,
            display=dict(show_background=self.show_background, show_foreground=self.show_foreground,
                         id_colors=self.id_colors, foreground_ids=None if self.foreground_ids is None else sorted(self.foreground_ids)))

    def history_state(self):
        selection = None if self.selection is None else self.selection.cpu().numpy().copy()
        return selection, [dict(o) for o in self.objects], self.labels().copy(), self.next_object_id

    def save_history(self):
        self.undo_stack.append(self.history_state())
        self.undo_stack = self.undo_stack[-12:]
        self.redo_stack.clear()

    def history(self, direction):
        src, dst = (self.undo_stack, self.redo_stack) if direction == 'undo' else (self.redo_stack, self.undo_stack)
        if not src:
            return
        dst.append(self.history_state())
        selection, self.objects, self.instance_ids, self.next_object_id = src.pop()
        self.selection = None if selection is None else self.torch.from_numpy(selection).cuda()
        self.ids_dirty = True
        self.invalidate()

    def save_segmentation(self):
        if self.knowledge is None:
            raise ValueError('请先加载场景')
        try:
            self.knowledge.save_segmentation(self.labels(), self.objects, self.next_object_id, self.gaussian_fingerprint)
        except (OSError, ValueError, TypeError) as error:
            raise ValueError(f'分割结果未保存：{error}') from error
        self.ids_dirty = False

    def invalidate(self):
        self.scene_version += 1
        self.capture = None
        self.ground_capture = None
        self.background_frame = None
        self.correction_pending = None

    @staticmethod
    def object_color(obj):
        hue = int(hashlib.sha256(obj['id'].encode()).hexdigest()[:8], 16) / 2**32
        return colorsys.hsv_to_rgb(hue, .7, .95)

    def foreground_options(self, data):
        selected, colors = self.foreground_ids, self.id_colors
        for key in ('show_background', 'show_foreground', 'id_colors'):
            if key in data and type(data[key]) is not bool:
                raise ValueError('显示选项必须为布尔值')
        if 'foreground_ids' in data:
            ids = data['foreground_ids']
            if not isinstance(ids, list) or not all(isinstance(key, str) for key in ids):
                raise ValueError('多物体前景需要物体 ID 列表')
            if set(ids) - {o['id'] for o in self.objects}:
                raise ValueError('前景物体 ID 已变化，请重新选择')
            selected = set(ids)
        if 'id_colors' in data:
            if type(data['id_colors']) is not bool:
                raise ValueError('ID 着色选项必须为布尔值')
            colors = data['id_colors']
        self.foreground_ids, self.id_colors = selected, colors
        for key in ('show_background', 'show_foreground'):
            if key in data:
                setattr(self, key, data[key])

    def composed_colors(self, cam, objects, background):
        """Preserve view-dependent SH colors on background while tinting IDs."""
        torch = self.torch
        with torch.no_grad():
            colors = torch.zeros_like(self.model.get_xyz)
            if bool(background.any()):
                colors[background] = self.natural_colors(cam, background)
            for obj in objects:
                colors[self.object_mask(obj['number'])] = colors.new_tensor(self.object_color(obj))
        return colors

    def natural_colors(self, cam, mask):
        from gs.utils.sh_utils import eval_sh
        sh = self.model.get_features[mask].transpose(1, 2)
        direction = self.model.get_xyz[mask] - cam.camera_center
        direction = direction / direction.norm(dim=1, keepdim=True).clamp_min(1e-8)
        return self.torch.clamp_min(eval_sh(self.model.active_sh_degree, sh, direction) + .5, 0.)

    def frame(self, mode='all', highlight=None):
        self.require_scene()
        self.correction_pending = None
        render_options = {}
        cam = self.camera()
        if mode in ('all', 'multi'):
            legacy = mode == 'multi'
            objects = [o for o in self.objects if (legacy or (self.show_foreground and not o['hidden']))
                       and (self.foreground_ids is None or o['id'] in self.foreground_ids)]
            background = self.torch.from_numpy((self.labels() == 0) & (not legacy and self.show_background)).cuda()
            mask = self.torch.from_numpy(np.isin(self.labels(), [o['number'] for o in objects])).cuda() | background
            if self.id_colors and objects and self.render_mode != 'depth':
                render_options['override_color'] = self.composed_colors(cam, objects, background)
        else:
            mask = self.active()
            if self.selection is not None:
                if mode == 'foreground':
                    mask &= self.selection
                elif mode == 'background':
                    mask &= ~self.selection
        if highlight is not None and len(highlight):
            with self.torch.no_grad():
                colors = render_options.get('override_color')
                if colors is None:
                    colors = self.torch.zeros_like(self.model.get_xyz)
                    colors[mask] = self.natural_colors(cam, mask)
                else:
                    colors = colors.clone()
                colors[self.torch.from_numpy(highlight).cuda()] = colors.new_tensor([1., .8, .05])
                render_options['override_color'] = colors
        if bool(mask.any()):
            with self.torch.no_grad():
                if self.render_mode == 'points':
                    from backend.render_modes import point_image
                    width, height = self.dimensions(self.view)
                    colors = render_options.get('override_color')
                    colors = self.natural_colors(cam, mask) if colors is None else colors[mask]
                    rendered = point_image(self.model.get_xyz[mask], colors, cam.full_proj_transform, width, height)
                    image = self.rgb(rendered)
                else:
                    if self.render_mode == 'splats':
                        render_options.update(show_gau=True, scaling_modifier=.35)
                    rendered = self.render_simple(cam, self.model, extra_gau_mask=mask, **render_options)
                    if self.render_mode == 'depth':
                        from backend.render_modes import depth_rgb
                        image = depth_rgb(rendered['depth'])
                        if highlight is not None and len(highlight):
                            colors = self.torch.zeros_like(self.model.get_xyz)
                            colors[self.torch.from_numpy(highlight).cuda()] = 1.
                            overlay = self.render_simple(cam, self.model, extra_gau_mask=mask, override_color=colors)
                            weight = overlay['render'][0].detach().cpu().numpy().clip(0, 1)[..., None]
                            image = (image * (1-weight) + np.array([255, 204, 13]) * weight).astype(np.uint8)
                    else:
                        image = self.rgb(rendered['render'])
        else:
            width, height = self.dimensions(self.view)
            image = np.zeros((height, width, 3), dtype=np.uint8)
        overlays = self.visualization(cam)
        if mode == 'multi' or (mode == 'all' and self.id_colors):
            # A single-object box would be misleading while inspecting several objects.
            overlays = dict(overlays, bbox=None)
        result = dict(image=self.image_url(image), visualization=overlays, **self.snapshot())
        if mode in ('all', 'multi'):
            width, height = self.dimensions(self.view)
            token = uuid.uuid4().hex
            self.background_frame = dict(id=token, version=self.scene_version, cam=cam,
                                         width=width, height=height, numbers=[o['number'] for o in objects], mode=mode,
                                         gaussian_scale=.35 if self.render_mode == 'splats' else 1., render_mode=self.render_mode,
                                         include_background=mode == 'all' and self.show_background)
            result['frame_id'] = token
        return result

    def correction_candidates(self, data):
        from backend.background import projected_box_mask, visible_box_ids, visible_pixel_ids
        correction = data.get('correction_mode', 'point' if data.get('action') == 'background_box' else 'ellipse')
        if not isinstance(correction, str) or correction not in ('ellipse', 'point'):
            raise ValueError('请选择椭圆修正或点修正')
        target = data.get('target_id', 0)
        if type(target) is not int or not 0 <= target <= 2147483647:
            raise ValueError('目标 ID 必须为 0 到 2147483647 的整数')
        frame = self.background_frame
        if frame is None or frame['id'] != data.get('frame_id') or frame['version'] != self.scene_version:
            raise ValueError('画面已变化，请重新进入修正')
        numbers = frame['numbers'] + ([0] if frame.get('include_background') else [])
        candidates = np.flatnonzero(np.isin(self.labels(), numbers) & (self.labels() != target))
        if 'pixel' in data:
            if 'rectangle' in data:
                raise ValueError('点击和框选不能同时提交')
            visible = visible_pixel_ids(self.correction_visibility(frame), data['pixel'])
            indices = visible[np.isin(visible, candidates)]
            return indices, target, correction, frame
        # Use exactly the same rasterized pixel ownership as single-click picking.
        # A second 3-sigma ellipse test can disagree with the native alpha footprint,
        # particularly with Splats' opaque opacity, so it must not filter ellipse hits.
        visible = visible_box_ids(self.correction_visibility(frame), data.get('rectangle'))
        indices = visible[np.isin(visible, candidates)]
        if correction == 'point' and len(indices):
            xyz = self.model.get_xyz[self.torch.from_numpy(indices).cuda()].detach().cpu().numpy()
            indices = indices[projected_box_mask(xyz,
                frame['cam'].full_proj_transform.detach().cpu().numpy(), frame['width'], frame['height'],
                data.get('rectangle'), snap_to_pixel=frame['render_mode'] == 'points')]
        return indices, target, correction, frame

    def correction_visibility(self, frame):
        """Frontmost contributing Gaussian per pixel, with the displayed layer mask.

        Native alpha threshold and near-plane clipping apply. Do not use radii > 0:
        that only tests frustum overlap, and would select fully occluded Gaussians.
        """
        if 'owners' not in frame:
            mask = self.torch.from_numpy(np.isin(self.labels(),
                frame['numbers'] + ([0] if frame.get('include_background') else []))).cuda()
            with self.torch.no_grad():
                if frame['render_mode'] == 'points':
                    from backend.render_modes import point_owners
                    local = point_owners(self.model.get_xyz[mask], frame['cam'].full_proj_transform,
                                         frame['width'], frame['height'])
                    mapping = self.torch.where(mask)[0]
                    owners = self.torch.full_like(local, -1)
                    valid = local >= 0
                    owners[valid] = mapping[local[valid]]
                else:
                    result = self.render_simple(frame['cam'], self.model, extra_gau_mask=mask,
                        make_topk=True, scaling_modifier=frame['gaussian_scale'],
                        show_gau=frame['render_mode'] == 'splats')
                    owners = result['gs_per_pixel'][0].long()
                frame['owners'] = owners.detach().cpu().numpy()
        return frame['owners']

    def preview_correction(self, data):
        previous = self.correction_pending
        indices, target, correction, frame = self.correction_candidates(data)
        if 'pixel' in data and previous is not None:
            if (previous['version'] != self.scene_version or previous['frame_id'] != frame['id']
                    or previous['target'] != target or previous['correction'] != correction):
                raise ValueError('修正预览已变化，请取消选中后重试')
            indices = np.union1d(previous['indices'], indices)
        owners = frame.get('owners')
        result = self.frame(frame.get('mode', 'all'), highlight=indices)
        self.background_frame['owners'] = owners
        token = uuid.uuid4().hex
        self.correction_pending = dict(id=token, frame_id=result['frame_id'], version=self.scene_version,
            indices=indices, target=target, correction=correction, mode=frame.get('mode', 'all'))
        return dict(preview_id=token, preview_count=len(indices), target_id=target,
                    correction_mode=correction, **result)

    def confirm_correction(self, data):
        pending = self.correction_pending
        if (pending is None or pending['id'] != data.get('preview_id')
                or pending['version'] != self.scene_version or self.background_frame is None
                or pending['frame_id'] != self.background_frame['id']):
            raise ValueError('修正预览已失效，请重新选择')
        return self.apply_correction(pending['indices'], pending['target'], pending['correction'], pending['mode'])

    def correct_gaussians(self, data):
        # Legacy clients still use a one-step operation, with the same visibility filtering.
        indices, target, correction, frame = self.correction_candidates(data)
        return self.apply_correction(indices, target, correction, frame.get('mode', 'all'))

    def apply_correction(self, indices, target, correction, mode):
        if len(indices):
            self.save_history()
            self.labels()[indices] = target
            if target and not any(obj['number'] == target for obj in self.objects):
                key = uuid.uuid4().hex
                self.objects.append(dict(id=key, number=target, name=f'物体 {target}', hidden=False))
                if self.foreground_ids is not None:
                    self.foreground_ids.add(key)
                self.next_object_id = max(self.next_object_id, target + 1)
            remaining = set(np.unique(self.labels())) - {0}
            self.objects = [obj for obj in self.objects if obj['number'] in remaining]
            if self.selection is not None:
                selection = self.selection.clone()
                selection[self.torch.from_numpy(indices).cuda()] = False
                self.selection = selection
            self.ids_dirty = True
            self.invalidate()
        return dict(corrected_count=len(indices), target_id=target, correction_mode=correction,
                    background_count=len(indices) if target == 0 else 0, **self.frame(mode))

    def visualization(self, cam):
        """Small world-space overlays, projected by the browser using this exact camera."""
        up = tuple(self.view['up'])
        if self.selection is not self._bbox_selection or up != self._bbox_up:
            self._bbox_selection = self.selection
            self._bbox_up = up
            self._bbox = None
            if self.selection is not None and bool(self.selection.any()):
                from backend.visualization import upright_bbox
                xyz = self.model.get_xyz[self.selection].detach()
                self._bbox = upright_bbox(xyz, up)
        return dict(bbox=self._bbox, route=self.drone_path,
                    projection=cam.full_proj_transform.detach().cpu().tolist())

    def load(self, data):
        self.initialize()
        path = Path(data['path']).expanduser().resolve()
        if path.is_dir():
            candidates = sorted(path.glob('point_cloud/iteration_*/point_cloud.ply'),
                key=lambda p: int(p.parent.name.split('_')[-1]))
            candidates += list(path.glob('point_cloud/point_cloud.ply')) + list(path.glob('*.ply'))
            if not candidates:
                raise ValueError('目录内未找到 3DGS PLY')
            path = candidates[-1] if not candidates[0].parent.name.startswith('iteration_') else candidates[0]
            # Prefer the latest trained iteration.
            iterations = [p for p in candidates if p.parent.name.startswith('iteration_')]
            if iterations:
                path = iterations[-1]
        if not path.is_file() or path.suffix.lower() != '.ply':
            raise ValueError('请输入存在的 3DGS .ply 文件或场景目录')
        model = self.GaussianModel(3, True)
        with self.torch.no_grad():
            model.load_ply(str(path))
        xyz = model.get_xyz
        if not len(xyz) or not bool(self.torch.isfinite(xyz).all()):
            raise ValueError('场景为空或含无效坐标')
        center = xyz.median(dim=0).values
        radius = float(self.torch.quantile(self.torch.linalg.norm(xyz-center, dim=1), .9)) * 2.5
        self.model, self.scene_name = model, path.name
        self.selection, self.objects = None, []
        self.instance_ids = np.zeros(len(xyz), dtype=np.int32)
        self.foreground_ids = None
        self.id_colors = False
        self.show_background = self.show_foreground = True
        self.render_mode = 'rgb'
        self.next_object_id = 1
        self.ids_dirty = self.ids_restored = False
        self.gaussian_fingerprint = hashlib.sha256(xyz.detach().cpu().numpy().astype('<f4').tobytes()).hexdigest()
        self.drone_path = None
        self._bbox_selection = self._bbox = None
        self.undo_stack.clear(); self.redo_stack.clear()
        self.view = dict(yaw=0., pitch=0., radius=max(radius, .1), target=center.cpu().tolist(),
                         up=[0., -1., 0.], fov=50., size=768, aspect=self.valid_aspect(data.get('aspect', 1.)))
        self.camera_entries = []
        self.ground_plane = None
        from backend.knowledge import SceneKnowledge
        self.knowledge = SceneKnowledge(path)
        self.knowledge_warning = ''
        self.ground_restored = False
        for parent in list(path.parents)[:4]:
            camera_file = parent / 'cameras.json'
            if camera_file.is_file():
                try:
                    entries = json.loads(camera_file.read_text())
                    if isinstance(entries, list):
                        self.camera_entries = [e for e in entries if all(k in e for k in ('rotation', 'position', 'fx', 'width'))]
                except (ValueError, OSError, TypeError):
                    pass
                break
        if self.camera_entries:
            self.adopt_camera(0)
        try:
            saved_ground = self.knowledge.load_ground()
        except (ValueError, TypeError, OSError) as error:
            saved_ground = None
            self.knowledge_warning = f'无法读取地面知识 {self.knowledge.path}：{error}'
        if saved_ground is not None:
            self.orient_to_ground(saved_ground, persist=False)
            self.ground_restored = True
        try:
            saved_ids = self.knowledge.load_segmentation(len(xyz), self.gaussian_fingerprint)
            if saved_ids is not None:
                self.instance_ids, self.objects, self.next_object_id = saved_ids
                # Scene opening always starts with all objects visible.
                for obj in self.objects:
                    obj['hidden'] = False
                self.ids_restored = True
        except (OSError, ValueError, TypeError) as error:
            warning = f'无法读取三维分割知识：{error}'
            self.knowledge_warning = '；'.join(filter(None, [self.knowledge_warning, warning]))
        self.invalidate()
        return self.frame()

    def adopt_camera(self, index):
        if not 0 <= index < len(self.camera_entries):
            raise ValueError('相机编号无效')
        from gs.scene.cameras import CustomCam
        entry = self.camera_entries[index]
        rotation = self.torch.tensor(entry['rotation'], dtype=self.torch.float32)
        position = self.torch.tensor(entry['position'], dtype=self.torch.float32)
        distance = max(float(self.torch.linalg.norm(self.model.get_xyz.median(0).values.cpu()-position)), .1)
        target = position + rotation[:, 2] * distance
        up = -rotation[:, 1] if self.ground_plane is None else self.torch.tensor(
            self.ground_plane['normal'], dtype=self.torch.float32)
        c2w = self.torch.eye(4, device='cuda')
        c2w[:3, :3] = rotation.cuda(); c2w[:3, 3] = position.cuda()
        fov = 2*np.arctan(entry['width']/(2*entry['fx']))
        cam = CustomCam(self.view['size'], self.view['size'], fov, fov, c2w)
        radius, yaw, pitch = self.utils.params2sphere(cam, up, target)
        self.view.update(radius=float(radius), yaw=float(yaw)*180/np.pi, pitch=float(pitch)*180/np.pi,
            target=target.tolist(), up=up.tolist(), fov=float(np.rad2deg(fov)))

    @staticmethod
    def valid_aspect(value):
        aspect = float(value)
        if not np.isfinite(aspect) or not .2 <= aspect <= 5:
            raise ValueError("视口宽高比超出范围")
        return aspect

    def update_view(self, data):
        self.require_scene()
        if 'preset' in data:
            self.adopt_camera(int(data['preset']))
            self.invalidate()
            return self.frame(data.get('mode', 'all'))
        v = dict(self.view)
        if 'aspect' in data:
            v['aspect'] = self.valid_aspect(data['aspect'])
        for key in ('yaw', 'pitch', 'radius', 'fov', 'size', 'target', 'up'):
            if key in data:
                v[key] = data[key]
        if not (-85 <= v['pitch'] <= 85 and .001 <= v['radius'] <= 1e7 and 5 <= v['fov'] <= 150):
            raise ValueError('相机参数超出范围')
        if v['size'] not in (512, 768, 1024):
            raise ValueError('分辨率应为 512、768 或 1024')
        for key in ('target', 'up'):
            if len(v[key]) != 3 or not np.isfinite(v[key]).all():
                raise ValueError('坐标必须为三个有限数字')
        if np.linalg.norm(v['up']) < 1e-5:
            raise ValueError('上方向不能为零')
        if not all(np.isfinite(v[k]) for k in ('yaw', 'pitch', 'radius', 'fov')):
            raise ValueError('相机参数必须为有限数字')
        if 'up' in data and self.ground_plane is not None:
            unit = np.asarray(v['up'])/np.linalg.norm(v['up'])
            if not np.allclose(unit, self.ground_plane['normal'], atol=1e-5):
                self.ground_plane = None
        self.view = v
        self.invalidate()
        return self.frame(data.get('mode', 'all'))

    def freeze(self, foreground=False):
        self.require_scene()
        self.render_mode = 'rgb'
        cam = self.camera()
        scope = self.active()
        refining = foreground and self.selection is not None
        if refining:
            scope &= self.selection
            if not bool(scope.any()):
                raise ValueError('当前前景为空，请撤销或清空选区后重新提示')
        with self.torch.no_grad():
            rendered = self.render_simple(cam, self.model, extra_gau_mask=scope)
        rgb = self.rgb(rendered['render'])
        token = uuid.uuid4().hex
        self.capture = dict(id=token, version=self.scene_version, cam=cam, rgb=rgb,
                            depth=rendered['depth'], encoded=None, masks=None, gaussian_mask=scope.clone())
        return dict(capture=token, capture_mode='foreground' if refining else 'all', image=self.image_url(rgb), visualization=self.visualization(cam), **self.snapshot())

    def captured(self, data):
        c = self.capture
        if c is None or c['id'] != data.get('capture') or c['version'] != self.scene_version:
            raise ValueError('画面或场景已变化，请重新固定画面')
        return c

    def orient_to_ground(self, plane, persist=True):
        # Keep the same eye position and observation target; change the up basis.
        cam = self.camera()
        up = plane['normal']
        radius, yaw, pitch = self.utils.params2sphere(cam, up, self.view['target'])
        if persist and self.knowledge is not None:
            try:
                self.knowledge.save_ground(plane)
            except (OSError, ValueError, TypeError) as error:
                raise ValueError(f'地面未应用：无法保存 {self.knowledge.path}：{error}') from error
        self.knowledge_warning = ''
        self.ground_restored = False
        self.view.update(up=up, radius=float(radius), yaw=float(yaw)*180/np.pi,
                         pitch=float(np.clip(float(pitch)*180/np.pi, -85., 85.)))
        self.ground_plane = plane
        self.invalidate()

    def ground_action(self, data):
        from backend.ground import ground_plane, flip_plane
        self.require_scene()
        operation = data.get('operation')
        if operation == 'start':
            result = self.freeze()
            self.ground_capture, self.capture = self.capture, None
            self.ground_capture.update(points=[], pixels=[], plane=None)
            result.update(ground_capture=result.pop('capture'), ground_points=[], ground_pixels=[], ground_draft=None)
            return result
        if operation == 'flip' and self.ground_capture is None:
            if self.ground_plane is None:
                raise ValueError('请先用三点标定地面')
            self.orient_to_ground(flip_plane(self.ground_plane))
            return self.frame()
        c = self.ground_capture
        if c is None or c['id'] != data.get('capture') or c['version'] != self.scene_version:
            raise ValueError('地面标定画面已失效，请重新开始三点标定')
        if operation == 'cancel':
            self.ground_capture = None
            return self.frame()
        if operation == 'pick':
            if len(c['points']) >= 3:
                raise ValueError('已选三点，可应用地面或撤回重选')
            pixel = data.get('pixel')
            height, width = c['rgb'].shape[:2]
            if not isinstance(pixel, list) or len(pixel) != 2 or not all(
                    isinstance(x, (int, float)) and np.isfinite(x) and 0 <= x < bound
                    for x, bound in zip(pixel, (width, height))):
                raise ValueError('地面选点不在画面范围内')
            x, y = int(pixel[0]), int(pixel[1])
            depth = c['depth'].squeeze()[y, x]
            if not bool(self.torch.isfinite(depth)) or float(depth) <= 0:
                raise ValueError('该位置没有有效表面深度，请选择清晰可见的地面')
            screen = self.torch.tensor([[x, y]], dtype=self.torch.float32, device=depth.device)
            with self.torch.no_grad():
                world = self.utils.back_projection(screen, depth.reshape(1),
                    self.utils.load_3dgs_camera(c['cam'], device=str(depth.device)))[0].detach().cpu().numpy()
            if not np.isfinite(world).all():
                raise ValueError('无法恢复该点的三维坐标，请重新选点')
            if any(np.linalg.norm(world-np.asarray(p)) < 1e-8 for p in c['points']):
                raise ValueError('选点重合，请在地面上选择不同位置')
            proposed = c['points']+[world.tolist()]
            if len(proposed) == 3:
                camera_position = c['cam'].world_view_transform.inverse()[3, :3].detach().cpu().numpy()
                plane = ground_plane(proposed, camera_position, self.view['up'])
                c['plane'] = plane
            c['points'], c['pixels'] = proposed, c['pixels']+[[x, y]]
        elif operation in ('undo', 'reset'):
            c['points'] = c['points'][:-1] if operation == 'undo' else []
            c['pixels'] = c['pixels'][:-1] if operation == 'undo' else []
            c['plane'] = None
        elif operation == 'flip':
            if c['plane'] is None:
                raise ValueError('请先选择三个地面点')
            c['plane'] = flip_plane(c['plane'])
        elif operation == 'apply':
            if c['plane'] is None:
                raise ValueError('请先选择三个不共线的地面点')
            self.orient_to_ground(c['plane'])
            return self.frame()
        else:
            raise ValueError('无效地面标定操作')
        arrow = None
        if c['plane'] is not None and hasattr(c['cam'], 'full_proj_transform'):
            plane = c['plane']
            center, normal = np.asarray(plane['center']), np.asarray(plane['normal'])
            scale = np.linalg.norm(np.asarray(plane['points'])-center, axis=1).mean()*.6
            world = np.column_stack([np.stack([center, center+normal*scale]), np.ones(2)])
            clip = world @ c['cam'].full_proj_transform.detach().cpu().numpy()
            if np.isfinite(clip).all() and (clip[:, 3] > 1e-6).all():
                height, width = c['rgb'].shape[:2]
                arrow = (((clip[:, :2]/clip[:, 3:4]+1)*[width, height]-1)*.5).tolist()
        return dict(ground_capture=c['id'], ground_points=c['points'],
                    ground_pixels=c['pixels'], ground_draft=c['plane'], ground_arrow=arrow)

    def load_model(self, data):
        """Load a segmenter independently of scene capture or inference."""
        version = data.get('model', 'sam2')
        if version not in ('sam2', 'sam3', 'sam3.1'):
            raise ValueError('无效分割模型')
        self.initialize()
        started = time.perf_counter()
        reused = False
        grounding_seconds = 0.
        if version == 'sam2':
            if not self.checkpoint.is_file():
                raise ValueError(f'未找到 SAM2 权重：{self.checkpoint}；请用 --checkpoint 指定')
            if getattr(self, 'sam3_client', None) is not None:
                self.sam3_client.close()
            reused = self.tracker is not None
            if not reused:
                self.clear_model_cache()
                tracker = self.CrossMatch(model_path=str(self.checkpoint))
                tracker.sam.eval()
                self.torch.cuda.synchronize()
                self.tracker = tracker
            if data.get('text'):
                from backend.grounding import GroundingDetector
                if getattr(self, 'grounding', None) is None:
                    self.grounding = GroundingDetector()
                grounding_seconds = self.grounding.load()
                reused = reused and grounding_seconds == 0.
        else:
            if getattr(self, 'grounding', None) is not None:
                self.grounding.close()

            from backend.sam3_client import Sam3Client
            if self.tracker is not None:
                self.tracker = None
                self.clear_model_cache()
                import gc
                gc.collect()
                self.torch.cuda.empty_cache()
            if getattr(self, 'sam3_client', None) is None:
                self.sam3_client = Sam3Client(self.output)
            seconds = self.sam3_client.load(version)
            reused = seconds == 0.
            if not reused:
                self.clear_model_cache()
        return dict(model=version, reused=reused,
                    model_load_seconds=0. if reused else time.perf_counter()-started,
                    grounding_load_seconds=grounding_seconds)

    def clear_model_cache(self):
        if self.capture is not None:
            for key in ('encoded', 'preds', 'pointers', 'score', 'masks', 'sam2_candidates'):
                self.capture[key] = None

    def preview(self, data):
        c = self.captured(data)
        version = data.get('model', 'sam2')
        if version not in ('sam2', 'sam3', 'sam3.1'):
            raise ValueError('无效分割模型')
        text = data.get('text', '')
        if not isinstance(text, str) or len(text) > 256:
            raise ValueError('文本提示应为不超过 256 字的字符串')
        text = text.strip()
        height, width = c['rgb'].shape[:2]
        def points(key):
            result = data.get(key, [])
            if not isinstance(result, list) or len(result) > 100 or any(
                    not isinstance(p, (list, tuple)) or len(p) != 2 or
                    not all(isinstance(x, (int, float)) and np.isfinite(x) and 0 <= x < limit
                            for x, limit in zip(p, (width, height))) for p in result):
                raise ValueError('提示点不在画面范围内')
            return [(x/width, y/height) for x, y in result]
        fg, bg, box = points('foreground'), points('background'), points('box')
        if box and (len(box) != 2 or box[0][0] >= box[1][0] or box[0][1] >= box[1][1]):
            raise ValueError('框提示无效')
        if not fg and not box and not text:
            raise ValueError('请添加文本、前景点或框提示')
        # A failed re-preview must never leave old candidates applicable.
        c['masks'] = None
        model_seconds = self.load_model(dict(model=version, text=text))['model_load_seconds']
        with self.torch.no_grad():
            if version == 'sam2':
                self.torch.cuda.synchronize()
                started = time.perf_counter()
                if c['encoded'] is None:
                    c['encoded'] = self.tracker.image_encoder(c['rgb'][:, :, ::-1].copy())
                detections = self.grounding.detect(c['rgb'], text, fg, bg, box) if text else [dict(box=box)]
                if not detections:
                    raise ValueError('未找到匹配物体，请修改文本、提示点或框选范围')
                masks, candidates, labels, scores = [], [], [], []
                for instance, detection in enumerate(detections):
                    prompt = self.tracker.prompt_encoder(box=[detection['box']] if detection['box'] else [], fg_pts=fg, bg_pts=bg)
                    preds, pointers, score = self.tracker.decode_mask(c['encoded'], prompt)
                    for index in range(preds.shape[1]):
                        mask = self.tracker.make_hires_mask_uint8(preds[0, index], 0., height=height, width=width)
                        # Keep the exact prediction, pointer and object score together for 3D tracking.
                        candidates.append(dict(preds=preds[:, [index]], pointers=pointers[:, [index]], score=score))
                        masks.append(mask)
                        labels.append(f'目标 {instance+1} · Level {index}' if text else f'Level {index}')
                        scores.append(detection.get('score'))
                self.torch.cuda.synchronize()
                seconds = time.perf_counter() - started
                c.update(sam2_candidates=candidates, candidate_labels=labels, candidate_scores=scores,
                         preds=preds, pointers=pointers, score=score)
            else:
                prompts = dict(text=text, foreground=fg, background=bg, box=box)
                binary, scores, timing = self.sam3_client.infer([c['rgb']], prompts)
                if not len(binary):
                    raise ValueError('未找到匹配物体，请修改文本、点提示或框提示')
                masks = [self.torch.from_numpy(m.astype(np.uint8)*255).to('cuda:0') for m in binary]
                seconds = timing['sam_seconds']
                c.update(semantic_prompts=prompts, candidate_scores=scores.tolist())
            c.update(masks=masks, model=version, segmentation_seconds=seconds)
            urls = []
            for mask in masks:
                overlay = c['rgb'].copy()
                inside = mask.cpu().numpy() > 0
                overlay[inside] = (overlay[inside]*.5 + np.array([46, 211, 157])*.5).astype(np.uint8)
                urls.append(self.image_url(overlay))
        return dict(masks=urls, capture=c['id'], model=version, scores=c.get('candidate_scores'),
                    candidate_labels=c.get('candidate_labels') if version == 'sam2' else None,
                    timing=dict(model=version, sam_seconds=seconds, model_load_seconds=model_seconds))

    def save_drone_preview(self, run_id, index, bgr, mask):
        """Save the exact tracking input and mask; never render or infer twice."""
        import cv2
        folder = self.output / 'drone-previews' / run_id
        folder.mkdir(parents=True, exist_ok=True)
        binary = np.zeros(bgr.shape[:2], np.uint8) if mask is None else (mask.detach().cpu().numpy() > 0).astype(np.uint8)*255
        overlay = bgr.copy()
        inside = binary > 0
        overlay[inside] = (overlay[inside]*.5 + np.array([157, 211, 46])*.5).astype(np.uint8)
        urls = {}
        for kind, pixels in (('render', bgr), ('overlay', overlay), ('mask', binary)):
            name = f'{index}-{kind}.png'
            if not cv2.imwrite(str(folder/name), pixels):
                raise RuntimeError('无法保存 V-Drone 预览图')
            urls[kind] = f'/previews/{run_id}/{name}'
        return urls

    def apply(self, data, progress, cancel):
        if not data.get('drone'):
            raise ValueError('单视角应用已移除，请使用三维分割')
        c = self.captured(data)
        if c['masks'] is None:
            raise ValueError('请先预览分割')
        version = c.get('model', 'sam2')
        if data.get('model', version) != version:
            raise ValueError('分割模型已改变，请重新预览')
        level, padding = int(data.get('level', 0)), int(data.get('padding', 2))
        if not 0 <= level < len(c['masks']) or not 0 <= padding <= 40:
            raise ValueError('候选层或边缘扩展参数无效')
        operation = data.get('operation', 'set')
        if operation not in ('set', 'add', 'subtract', 'intersect'):
            raise ValueError('无效选区操作')
        mask = c['masks'][level]
        if not bool(mask.any()):
            raise ValueError('mask 为空，请调整提示')
        self.torch.cuda.synchronize()
        started = time.perf_counter()
        visualization_seconds = 0.
        run_id = uuid.uuid4().hex
        tracking_stats = None
        with self.torch.no_grad(), ExitStack() as cleanup:
            filt = self.utils.MaskShapedFrustumFilter(self.model, padding=padding, self_val=True,
                initial_cam=c['cam'], initial_mask=mask)
            result = filt.filtering(mask, c.get('gaussian_mask'), c['cam'], self_val=False, first_view_prompt=True) & self.active()
            if c.get('gaussian_mask') is not None:
                result &= c['gaussian_mask']
            reports = []
            if not bool(result.any()):
                raise ValueError('mask 未包含可见高斯')
            if data.get('drone'):
                semantic = version != 'sam2'
                progress(0, f'建立 {version.upper()} 跟踪')
                if semantic:
                    route_info = self.sam3_client.start_route(c['rgb'],
                        c['semantic_prompts'], target=mask.cpu().numpy() > 0)
                    semantic_route = route_info.pop('route')
                    tracking_stats = route_info
                    cleanup.callback(self.sam3_client.close_route, semantic_route)
                    memory = ptr = None
                else:
                    candidate = c['sam2_candidates'][level] if c.get('sam2_candidates') else dict(
                        preds=c['preds'][:, [level]], pointers=c['pointers'][:, [level]], score=c['score'])
                    memory = (self.tracker.sam.memory_encoder(c['encoded'][0], candidate['preds'],
                        candidate['score'], is_prompt_encoding=True),)
                    ptr = (candidate['pointers'],)
                depth = c['depth'].squeeze()
                surface_mask = mask.bool() & self.torch.isfinite(depth) & (depth > 0)
                if not bool(surface_mask.any()):
                    raise ValueError('首视角选区没有有效表面深度，请调整视角或提示')
                center = self.utils.mask_average_location(surface_mask, depth, self.utils.load_3dgs_camera(c['cam'], device='cuda:0')).cpu()
                if not bool(self.torch.isfinite(center).all()):
                    raise ValueError('首视角表面中心无效，请调整视角或提示')
                drones = [self.Drone(up_vector=self.view['up'], initial_view=c['cam'], look_at=center,
                    memory=memory, ptr=ptr, drone_id=i) for i in range(2)]
                for drone in drones:
                    drone.debug_verbose = False
                for angle in (45, 90, 135):
                    for side, sign in enumerate((1, -1)):
                        if cancel.is_set():
                            raise Cancelled('已取消巡航，未提交选区')
                        fg = self.utils.masking_for_fg(self.model, result)
                        # s0 -> s1 uses only the visible surface. Each later step
                        # uses the foreground retained by the preceding view.
                        center_source = 'surface' if not reports else 'foreground'
                        if reports:
                            center = fg.get_xyz.mean(0).cpu()
                        input_count = int(result.sum())
                        progress(len(reports), f'巡航视角 {sign*angle:+}°')
                        drone = drones[side]
                        pose = drone.return_camera_pose(sign*np.deg2rad(angle), np.pi/6, center=center)
                        if semantic:
                            # A video session has a fixed image size and intrinsics.
                            pose = self.utils.sphere2params(drone.distance,
                                drone.initial_yaw+sign*np.deg2rad(angle), np.pi/6, center,
                                drone.up_vector, c['cam'].FoVy, like_cam=c['cam'], device='cuda:0')
                        pose.camera_center = pose.world_view_transform.inverse()[3, :3]
                        old = deque(drone.prev_mems), deque(drone.prev_ptrs)
                        assets = {}
                        def on_preview(view, tracked_mask):
                            nonlocal visualization_seconds
                            self.torch.cuda.synchronize()
                            preview_started = time.perf_counter()
                            assets.update(self.save_drone_preview(run_id, len(reports), view, tracked_mask))
                            visualization_seconds += time.perf_counter() - preview_started
                        if semantic:
                            rendered = self.render_simple(pose, fg, bg_color=drone.background_color)
                            rgb = self.rgb(rendered['render'])
                            binary, step_report = self.sam3_client.step_route(semantic_route, side, rgb)
                            tracked = self.torch.from_numpy(binary[0].astype(np.uint8)*255).to('cuda:0')
                            on_preview(rgb[:, :, ::-1].copy(), tracked)
                        else:
                            tracked = drone.tracking_in_new_view(fg, pose, self.tracker, prompt_aug=False, preview_callback=on_preview)
                        proposed = filt.filtering(tracked, result, pose)
                        accepted = proposed is not None and bool(proposed.any())
                        if accepted:
                            accepted = not bool(self.torch.equal(proposed, result))
                            # Native filter retains the old candidate on coverage rejection.
                            result = proposed
                        if not accepted:
                            drone.prev_mems, drone.prev_ptrs = old
                        if semantic:
                            tracking_stats = self.sam3_client.finish_step(semantic_route, side, accepted)
                        from backend.visualization import camera_frustum
                        reports.append(dict(yaw=sign*angle, drone=side, changed=accepted, count=int(result.sum()),
                            position=pose.camera_center.detach().cpu().tolist(),
                            frustum=camera_frustum(pose),
                            center=center.tolist(), center_source=center_source, input_count=input_count,
                            lost=tracked is None or not bool(tracked.any()), **assets))
                        progress(len(reports), f'完成 {len(reports)}/6，剩余 {int(result.sum()):,} 个高斯', list(reports))
            if cancel.is_set():
                raise Cancelled('已取消，未提交选区')
            self.save_history()
            if data.get('drone'):
                xyz = self.model.get_xyz[result].detach()
                size = float(self.torch.linalg.vector_norm(xyz.amax(0)-xyz.amin(0)))
                for report in reports:
                    distance = np.linalg.norm(np.asarray(report['position'])-report['center'])
                    report['frustum']['depth'] = float(np.clip(size*.3, distance*.025, distance*.2))
                initial_frustum = camera_frustum(c['cam'])
                distance = np.linalg.norm(np.asarray(initial_frustum['position'])-reports[0]['center'])
                initial_frustum['depth'] = float(np.clip(size*.3, distance*.025, distance*.2))
                self.drone_path = dict(origin=initial_frustum['position'], origin_frustum=initial_frustum, views=reports)
            if self.selection is None or operation == 'set':
                self.selection = result
            elif operation == 'add':
                self.selection = self.selection | result
            elif operation == 'subtract':
                self.selection = self.selection & ~result
            else:
                self.selection = self.selection & result
            self.invalidate()
        self.torch.cuda.synchronize()
        seconds = time.perf_counter() - started - visualization_seconds
        return dict(route=reports, model=version, tracking_stats=tracking_stats, timing=dict(model=version, preview_seconds=visualization_seconds, refine_seconds=seconds, total_seconds=seconds+c.get('segmentation_seconds', 0.)), **self.frame('foreground'))

    def objects_action(self, data):
        self.require_scene()
        action = data['action']
        if action in ('undo', 'redo'):
            self.history(action)
        elif action == 'save_ids':
            self.save_segmentation()
        elif action == 'preview_correction':
            return self.preview_correction(data)
        elif action == 'confirm_correction':
            return self.confirm_correction(data)
        elif action == 'cancel_correction':
            return self.frame('all')
        elif action in ('background_box', 'correct_box'):
            return self.correct_gaussians(data)
        elif action == 'rename':
            obj = next((o for o in self.objects if o['id'] == data.get('id')), None)
            if obj is None:
                raise ValueError('物体不存在')
            name = data.get('name')
            if not isinstance(name, str) or not name.strip():
                raise ValueError('物体名称不能为空')
            name = name.strip()
            if len(name) > 80:
                raise ValueError('物体名称不能超过 80 个字符')
            if name != obj['name']:
                self.save_history()
                obj['name'] = name
                self.ids_dirty = True
            return self.snapshot()
        elif action == 'renumber':
            objects = sorted(self.objects, key=lambda o: o['number'])
            if not objects:
                raise ValueError('暂无可排序的物体 ID')
            numbers = np.asarray([o['number'] for o in objects], dtype=np.int32)
            if not np.array_equal(numbers, np.arange(1, len(objects)+1)) or self.next_object_id != len(objects)+1:
                self.save_history()
                positive = self.labels() > 0
                labels = np.zeros_like(self.labels())
                labels[positive] = np.searchsorted(numbers, self.labels()[positive]).astype(np.int32)+1
                self.instance_ids = labels
                for number, obj in enumerate(objects, 1):
                    obj['number'] = number
                self.objects = objects
                self.next_object_id = len(objects)+1
                self.ids_dirty = True
                self.invalidate()
        elif action == 'merge':
            ids = data.get('ids')
            if not isinstance(ids, list) or len(ids) != 2 or not all(isinstance(i, str) for i in ids) or ids[0] == ids[1]:
                raise ValueError('请选择两个不同的物体 ID')
            objects = [next((o for o in self.objects if o['id'] == key), None) for key in ids]
            if any(o is None for o in objects):
                raise ValueError('物体不存在，请重新选择 ID')
            target, source = sorted(objects, key=lambda o: o['number'])
            self.save_history()
            self.labels()[self.labels() == source['number']] = target['number']
            self.objects.remove(source)
            target['hidden'] = False
            self.selection = self.object_mask(target['number'])
            self.ids_dirty = True
            self.invalidate()
        elif action == 'clear':
            self.save_history(); self.selection = None; self.invalidate()
        elif action in ('save', 'add_id'):
            if self.selection is None or not bool(self.selection.any()):
                raise ValueError('当前选区为空')
            assigned = self.selection.detach().cpu().numpy().astype(bool) & (self.labels() == 0)
            if not assigned.any():
                raise ValueError('选区已属于现有 ID，没有可新增的高斯')
            number = self.next_object_id
            if number > 2147483647:
                raise ValueError('物体 ID 已超出 int32 范围')
            name = str(data.get('name', '')).strip()[:80] or f'物体 {number}'
            self.save_history()
            key = uuid.uuid4().hex
            self.objects.append(dict(id=key, number=number, name=name, hidden=action == 'add_id'))
            if self.foreground_ids is not None:
                self.foreground_ids.add(key)
            self.labels()[assigned] = number
            self.next_object_id += 1
            self.ids_dirty = True
            if action == 'add_id':
                self.selection = None
            self.invalidate()
        elif action in ('select', 'hide', 'delete'):
            obj = next((o for o in self.objects if o['id'] == data.get('id')), None)
            if obj is None:
                raise ValueError('物体不存在')
            self.save_history()
            if action == 'select':
                self.selection = self.object_mask(obj['number'])
            elif action == 'hide':
                obj['hidden'] = not obj['hidden']
            else:
                self.labels()[self.labels() == obj['number']] = 0
                self.objects.remove(obj)
            if action != 'select':
                self.ids_dirty = True
            self.invalidate()
        else:
            raise ValueError('未知物体操作')
        return self.frame()

    def export(self, data):
        self.require_scene()
        kind = data.get('kind', 'ply')
        name = uuid.uuid4().hex
        with self.torch.no_grad():
            if kind in ('ply', 'mask'):
                if self.selection is None or not bool(self.selection.any()):
                    raise ValueError('请先创建非空选区')
                if kind == 'ply':
                    path = self.output / (name + '.ply')
                    self.utils.masking_for_fg(self.model, self.selection, save_name=str(path))
                else:
                    path = self.output / (name + '.npy')
                    np.save(path, self.selection.cpu().numpy())
            elif kind == 'mask2d':
                import cv2
                c = self.captured(data)
                level = int(data.get('level', 0))
                if c['masks'] is None or not 0 <= level < len(c['masks']):
                    raise ValueError('请先预览有效的二维 mask')
                path = self.output / (name + '.png')
                cv2.imwrite(str(path), c['masks'][level].cpu().numpy())
            elif kind == 'image':
                import cv2
                path = self.output / (name + '.png')
                url = self.frame(data.get('mode', 'all'))['image']
                path.write_bytes(base64.b64decode(url.split(',', 1)[1]))
            else:
                raise ValueError('未知导出格式')
        return dict(download='/downloads/' + path.name)

    def execute(self, action, data, progress, cancel):
        if action == 'gif':
            from backend.gif_export import generate_gif
            return generate_gif(self, data, progress, cancel)
        if 'render_mode' in data:
            style = data['render_mode']
            if not isinstance(style, str) or style not in ('rgb', 'splats', 'points', 'depth'):
                raise ValueError('未知渲染模式')
            if style != self.render_mode and (self.capture is not None or self.ground_capture is not None):
                raise ValueError('请先退出分割提示或地面标定，再切换渲染模式')
            self.render_mode = style
        if action in ('frame', 'view', 'export'):
            self.foreground_options(data)
        if (action in ('frame', 'capture') or (action == 'ground' and data.get('operation') == 'start')) and self.view is not None and 'aspect' in data:
            aspect = self.valid_aspect(data['aspect'])
            if abs(aspect-self.view.get('aspect', 1.)) > .002:
                self.view['aspect'] = aspect
                self.invalidate()
        if action == 'ground': return self.ground_action(data)
        if action == 'load_model':
            progress(0, f"正在加载 {str(data.get('model', 'sam2')).upper()}…")
            return self.load_model(data)
        if action == 'load': return self.load(data)
        if action == 'view': return self.update_view(data)
        if action == 'frame': return self.frame(data.get('mode', 'all'))
        if action == 'capture': return self.freeze(foreground=True)
        if action == 'preview': return self.preview(data)
        if action == 'apply': return self.apply(data, progress, cancel)
        if action == 'objects': return self.objects_action(data)
        if action == 'export': return self.export(data)
        raise ValueError('未知操作')
