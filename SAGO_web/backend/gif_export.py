"""Foreground orbit GIFs, rendered without touching the interactive camera."""
import math
import os
import re
from pathlib import Path
import tempfile
import time
import uuid
from urllib.parse import quote

import numpy as np
from PIL import Image


def orbit_geometry(xyz, scaling, up, fov_degrees):
    """Fit the main foreground including Gaussian extents, tolerating sparse floaters."""
    xyz, scaling, up = (np.asarray(value, dtype=np.float64) for value in (xyz, scaling, up))
    if (xyz.ndim != 2 or xyz.shape[1] != 3 or not len(xyz) or not np.isfinite(xyz).all()
            or scaling.shape != xyz.shape or not np.isfinite(scaling).all() or (scaling < 0).any()
            or up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8
            or not np.isfinite(fov_degrees) or not 5 <= fov_degrees <= 150):
        raise ValueError('无法为前景计算有效的环绕相机')
    center = xyz.mean(axis=0)
    # A few giant/stray Gaussians should not reduce the actual object to a dot.
    # This affects framing only; every selected Gaussian is still rendered.
    bound = max(float(np.quantile(np.linalg.norm(xyz-center, axis=1) + 3*scaling.max(axis=1), .99)), 1e-4)
    distance = bound / math.sin(math.radians(fov_degrees)/2) * 1.15
    return center, up/np.linalg.norm(up), distance


def _generate_one(engine, data, progress, cancel, selection):
    from backend.engine import Cancelled

    def check_cancel():
        if cancel.is_set():
            raise Cancelled('GIF 生成已取消')

    def integer(name, default, low, high):
        value = data.get(name, default)
        if type(value) is not int or not low <= value <= high:
            raise ValueError('无效的 GIF 参数')
        return value

    engine.require_scene()
    if selection is None or not bool(selection.any()):
        raise ValueError('请先创建非空选区')
    size = integer('size', 512, 128, 768)
    steps = integer('frames_per_orbit', 72, 4, 144)
    elevation = data.get('elevation_degrees', 60)
    if type(elevation) not in (int, float) or not math.isfinite(elevation) or not -89 <= elevation <= 89:
        raise ValueError('GIF 俯仰角应在 -89° 到 89° 之间')
    fps = integer('fps', 20, 1, 50)
    if fps not in (10, 20, 25, 50):
        raise ValueError('无效的 GIF 参数')
    started = time.perf_counter()
    check_cancel()
    progress(0, '正在准备前景 GIF')
    engine.output.mkdir(parents=True, exist_ok=True)
    name = uuid.uuid4().hex + '.gif'
    destination = engine.output/name
    quantized = []
    # Temporary frames and incomplete GIFs disappear on cancellation or failure.
    with tempfile.TemporaryDirectory(prefix='gif-', dir=engine.output) as temporary:
        folder = Path(temporary)
        with engine.torch.no_grad():
            target = engine.utils.masking_for_fg(engine.model, selection.clone())
            center, up, distance = orbit_geometry(target.get_xyz.detach().cpu().numpy(),
                target.get_scaling.detach().cpu().numpy(), engine.view['up'], engine.view['fov'])
            center = engine.torch.tensor(center, dtype=engine.torch.float32)
            up = engine.torch.tensor(up, dtype=engine.torch.float32)
            thumbnails = []
            total = steps
            for index in range(total):
                check_cancel()
                cam = engine.utils.sphere2params(distance,
                    math.radians(engine.view['yaw']) + 2*math.pi*(index % steps)/steps,
                    math.radians(elevation), center, up,
                    math.radians(engine.view['fov']), width=size, height=size)
                cam.camera_center = cam.world_view_transform.inverse()[3, :3]
                rendered = engine.render_simple(cam, target, bg='white')
                frame = Image.fromarray(engine.rgb(rendered['render']))
                frame.save(folder/f'{index:04d}.png')
                thumbnails.append(frame.resize((32, 32), Image.Resampling.BILINEAR))
                frame.close()
                progress(round((index+1)/total*85), f'GIF 渲染 {index+1}/{total}')
            del rendered, target
        check_cancel()
        # One palette sampled from every view avoids per-frame color flicker.
        columns = 16
        sample = Image.new('RGB', (columns*32, math.ceil(total/columns)*32), 'white')
        for index, thumbnail in enumerate(thumbnails):
            sample.paste(thumbnail, ((index % columns)*32, (index//columns)*32))
            thumbnail.close()
        palette = sample.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
        sample.close()
        colors = palette.getpalette()
        colors += [0] * (768 - len(colors))
        colors[255*3:256*3] = [255, 255, 255]
        palette.putpalette(colors)
        try:
            for index in range(total):
                check_cancel()
                with Image.open(folder/f'{index:04d}.png') as frame:
                    rgb = frame.convert('RGB')
                    encoded = rgb.quantize(palette=palette, dither=Image.Dither.FLOYDSTEINBERG)
                    # Palette training/nearest-color caching must not tint white backgrounds.
                    white_pixels = Image.fromarray((np.all(np.asarray(rgb) == 255, axis=2)*255).astype(np.uint8))
                    encoded.paste(255, mask=white_pixels)
                    white_pixels.close()
                    rgb.close()
                    quantized.append(encoded)
                progress(85 + round((index+1)/total*13), f'GIF 编码 {index+1}/{total}')
            check_cancel()
            pending = folder/'animation.gif'
            duration = 1000 // fps
            quantized[0].save(pending, format='GIF', save_all=True, append_images=quantized[1:],
                duration=duration, loop=0, optimize=False, disposal=2, background=255)
            check_cancel()
            os.replace(pending, destination)
        finally:
            palette.close()
            for frame in quantized:
                frame.close()
    progress(100, '前景 GIF 已生成')
    return dict(download='/downloads/'+name, saved_path=str(destination.resolve()), frames=total, fps=fps, size=size,
                duration_seconds=total/fps, elevation_degrees=elevation, generation_seconds=time.perf_counter()-started)


def generate_gif(engine, data, progress, cancel):
    """Export the current selection, all objects, or explicitly requested objects."""
    from backend.engine import Cancelled

    engine.require_scene()
    source = data.get('source', 'current')
    if source not in ('current', 'all', 'ids'):
        raise ValueError('无效的 GIF 来源')
    if source == 'current':
        result = _generate_one(engine, data, progress, cancel, engine.selection)
        return dict(result, source=source, items=[dict(result, filename='Current_foreground.gif')])
    if source == 'all':
        objects = sorted(engine.objects, key=lambda obj: obj['number'])
        if not objects:
            raise ValueError('没有可导出的物体 ID')
    else:
        requested = data.get('ids')
        if not isinstance(requested, list) or not requested or any(not isinstance(key, str) for key in requested):
            raise ValueError('请在三维可视化中勾选至少一个物体 ID')
        by_id = {obj['id']: obj for obj in engine.objects}
        if any(key not in by_id for key in requested):
            raise ValueError('物体 ID 已变化，请重新选择 GIF 来源')
        objects = sorted((by_id[key] for key in set(requested)), key=lambda obj: obj['number'])
    labels = engine.labels()
    if any(not np.any(labels == obj['number']) for obj in objects):
        raise ValueError('所选物体 ID 没有可导出的高斯')
    started = time.perf_counter()
    items, completed = [], []
    try:
        for index, obj in enumerate(objects):
            if cancel.is_set():
                raise Cancelled('GIF 生成已取消')
            def report(value, message):
                progress(round((index + value/100)/len(objects)*100),
                    f"GIF 物体 {index+1}/{len(objects)} · ID {obj['number']} · {value}%")
            # Object exports include hidden objects, independently of the current selection.
            selection = engine.object_mask(obj['number'])
            result = _generate_one(engine, data, report, cancel, selection)
            storage_name = result['download'].rsplit('/', 1)[-1]
            completed.append(engine.output/storage_name)
            stem = re.sub(r'[\x00-\x1f<>:"/\\|?*]', '_', obj['name']).strip(' .')[:80] or 'Object'
            filename = f"ID_{obj['number']}_{stem}.gif"
            items.append(dict(result, id=obj['id'], number=obj['number'], name=obj['name'], filename=filename,
                download=result['download']+'?filename='+quote(filename, safe='')))
        if cancel.is_set():
            raise Cancelled('GIF 生成已取消')
    except BaseException:
        # A batch is committed only when every requested object succeeds.
        for file in completed:
            file.unlink(missing_ok=True)
        raise
    progress(100, f'已生成 {len(items)} 个物体 GIF')
    return dict(source=source, items=items, generation_seconds=time.perf_counter()-started)
