"""Per-scene ground and segmentation metadata, merged and atomically saved beside scene data."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def validate_plane(plane):
    try:
        points = np.asarray(plane['points'], dtype=float)
        normal = np.asarray(plane['normal'], dtype=float)
        offset = float(plane['offset'])
        if points.shape != (3, 3) or normal.shape != (3,) or not np.isfinite(points).all() or not np.isfinite(normal).all() or not np.isfinite(offset):
            raise ValueError()
        norm = np.linalg.norm(normal)
        first, second = points[1]-points[0], points[2]-points[0]
        cross = np.cross(first, second)
        length = np.linalg.norm(cross)
        if norm < 1e-8 or length <= .01*np.linalg.norm(first)*np.linalg.norm(second) or length < 1e-16:
            raise ValueError()
        normal, offset = normal/norm, offset/norm
        scale = max(np.linalg.norm(first), np.linalg.norm(second), 1e-6)
        if abs(np.dot(normal, cross/length)) < .99999 or np.max(np.abs(points @ normal+offset)) > 1e-4*scale:
            raise ValueError()
        return dict(points=points.tolist(), normal=normal.tolist(), offset=float(offset), center=points.mean(0).tolist())
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError('地平面记录无效或与三点坐标不一致') from error


class SceneKnowledge:
    def __init__(self, ply):
        ply = Path(ply).resolve()
        folder = ply.parent
        if folder.name.startswith('iteration_') and folder.parent.name == 'point_cloud':
            folder = folder.parent.parent
        elif folder.name == 'point_cloud':
            folder = folder.parent
        self.path = folder/'SAGO_knowledge.json'
        self.key = ply.relative_to(folder).as_posix()

    def read(self):
        if not self.path.exists():
            return {}
        with self.path.open(encoding='utf-8') as stream:
            data = json.load(stream)
        if not isinstance(data, dict) or not isinstance(data.get('scenes', {}), dict):
            raise ValueError('知识文件根节点和 scenes 必须为对象')
        return data

    def load_ground(self):
        record = self.read().get('scenes', {}).get(self.key)
        if record is None:
            return None
        if not isinstance(record, dict):
            raise ValueError('场景知识记录必须为对象')
        if record.get('ground_plane') is None:
            return None
        plane = validate_plane(record['ground_plane'])
        if 'up_vector' in record:
            up = np.asarray(record['up_vector'], dtype=float)
            if up.shape != (3,) or not np.isfinite(up).all() or not np.allclose(up, plane['normal'], atol=1e-5):
                raise ValueError('向上向量与地平面法线不一致')
        return plane

    def save_ground(self, plane):
        plane = validate_plane(plane)
        self.save_record(ground_plane=plane, up_vector=plane['normal'])
        return plane

    def save_record(self, **fields):
        data = self.read()  # Never overwrite an unreadable existing knowledge file.
        data.setdefault('schema_version', 1)
        scenes = data.setdefault('scenes', {})
        previous = scenes.get(self.key, {})
        if not isinstance(previous, dict):
            raise ValueError('场景知识记录必须为对象')
        scenes[self.key] = dict(previous, **fields, updated_at=datetime.now(timezone.utc).isoformat())
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.path.parent,
                    prefix='.SAGO_knowledge-', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(data, stream, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def validate_segmentation(record, count, fingerprint):
        try:
            if type(record['gaussian_count']) is not int or record['gaussian_count'] != count or record['gaussian_order_sha256'] != fingerprint:
                raise ValueError('高斯数量或原始顺序与保存记录不匹配')
            values = record['gaussian_ids']
            if not isinstance(values, list) or len(values) != count:
                raise ValueError('ID 向量长度与高斯数量不一致')
            if any(type(v) is not int or not 0 <= v <= 2147483647 for v in values):
                raise ValueError('ID 向量必须为非负 int32 整数')
            labels = np.asarray(values, dtype=np.int32)
            objects = record['objects']
            if not isinstance(objects, list):
                raise ValueError('物体记录必须为列表')
            numbers, keys = set(), set()
            for obj in objects:
                number = obj['number']
                if type(number) is not int or number <= 0 or number > 2147483647 or number in numbers:
                    raise ValueError('物体 ID 必须为唯一正整数')
                if not isinstance(obj['id'], str) or not obj['id'] or obj['id'] in keys:
                    raise ValueError('物体记录标识无效或重复')
                if not isinstance(obj['name'], str) or len(obj['name']) > 80 or type(obj['hidden']) is not bool:
                    raise ValueError('物体名称或显示状态无效')
                numbers.add(number)
                keys.add(obj['id'])
            if numbers != set(np.unique(labels)) - {0}:
                raise ValueError('ID 向量与物体记录不一致')
            next_id = record['next_id']
            if type(next_id) is not int or not max(numbers, default=0) < next_id <= 2147483648:
                raise ValueError('下一个物体 ID 无效')
            return labels, [dict(id=o['id'], number=o['number'], name=o['name'], hidden=o['hidden']) for o in objects], next_id
        except (KeyError, TypeError, OverflowError) as error:
            raise ValueError('三维分割记录格式无效') from error

    def load_segmentation(self, count, fingerprint):
        scene = self.read().get('scenes', {}).get(self.key, {})
        if not isinstance(scene, dict):
            raise ValueError('场景知识记录必须为对象')
        record = scene.get('segmentation')
        return None if record is None else self.validate_segmentation(record, count, fingerprint)

    def save_segmentation(self, labels, objects, next_id, fingerprint):
        record = dict(gaussian_count=len(labels), gaussian_order_sha256=fingerprint,
                      order='original_ply_vertex_order', gaussian_ids=labels.tolist(),
                      objects=[dict(o) for o in objects], next_id=next_id)
        self.validate_segmentation(record, len(labels), fingerprint)
        self.save_record(segmentation=record)
