"""Full Gaussian attributes and conventional XYZ/RGB points for a saved object."""
from pathlib import Path
from urllib.parse import quote
import zipfile

import numpy as np
from plyfile import PlyData, PlyElement


def file_record(output, path, kind):
    path = Path(path)
    relative = path.relative_to(output).as_posix()
    return dict(kind=kind, filename=path.name, saved_path=str(path.resolve()),
        download='/downloads/'+quote(relative, safe='/')+'?filename='+quote(path.name, safe=''))


def save_object_assets(engine, selection, folder, cancel):
    from backend.mesh_export import check_cancel
    check_cancel(cancel)
    with engine.torch.no_grad():
        target = engine.utils.masking_for_fg(engine.model, selection.clone())
        target.save_ply(str(folder/'gaussians.ply'))
        check_cancel(cancel)
        xyz = target.get_xyz.detach().cpu().numpy()
        dc = target._features_dc.detach().cpu().numpy()[:, 0, :]
        # The degree-zero spherical harmonic is view-independent RGB.
        rgb = np.rint(np.clip(.5 + .28209479177387814*dc, 0, 1)*255).astype(np.uint8)
        if not np.isfinite(xyz).all() or not np.isfinite(dc).all():
            raise ValueError('Invalid object coordinates or colors for asset export')
        vertices = np.empty(len(xyz), dtype=[(key, '<f4') for key in ('x','y','z')]+
            [(key, 'u1') for key in ('red','green','blue')])
        for axis, key in enumerate(('x','y','z')):
            vertices[key] = xyz[:, axis]
        for channel, key in enumerate(('red','green','blue')):
            vertices[key] = rgb[:, channel]
        PlyData([PlyElement.describe(vertices, 'vertex')], text=False).write(str(folder/'point_cloud.ply'))
    check_cancel(cancel)
    return dict(gaussian_count=len(xyz), files=[file_record(engine.output, folder/'gaussians.ply', 'gaussians'),
        file_record(engine.output, folder/'point_cloud.ply', 'point_cloud')])


def archive_assets(batch, items, cancel):
    from backend.mesh_export import check_cancel
    destination = batch/'assets.zip'
    pending = batch/'assets.zip.tmp'
    try:
        with zipfile.ZipFile(pending, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
            for item in items:
                for file in item['files']:
                    check_cancel(cancel)
                    path = Path(file['saved_path'])
                    # Stream large Gaussian files and check cancellation between chunks.
                    with path.open('rb') as source, archive.open(path.relative_to(batch).as_posix(), 'w', force_zip64=True) as output:
                        while chunk := source.read(1024*1024):
                            check_cancel(cancel)
                            output.write(chunk)
            check_cancel(cancel)
            archive.write(batch/'manifest.json', 'manifest.json')
        check_cancel(cancel)
        pending.replace(destination)
    finally:
        pending.unlink(missing_ok=True)
    return destination
