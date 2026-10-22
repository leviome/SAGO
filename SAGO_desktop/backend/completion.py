"""Expand a foreground by Gaussian-center containment in an upright volume."""
import cv2
import torch
from backend.visualization import upright_bbox


@torch.no_grad()
def complete_foreground(xyz, selection, up, shape):
    if shape not in ('cube', 'cylinder'):
        raise ValueError('Unknown completion shape')
    if selection is None or not bool(selection.any()):
        raise ValueError('当前选区为空')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or selection.shape != xyz.shape[:1] or selection.dtype != torch.bool:
        raise ValueError('Invalid foreground shape')
    seed = xyz[selection].detach().to(dtype=torch.float64)
    vertical = torch.as_tensor(up, dtype=seed.dtype, device=seed.device).clone()
    if vertical.shape != (3,) or not bool(torch.isfinite(vertical).all()) or float(vertical.norm()) < 1e-8:
        raise ValueError('Invalid up vector')
    if not bool(torch.isfinite(seed).all()):
        raise ValueError('Foreground coordinates must be finite')
    vertical /= vertical.norm()
    if shape == 'cube':
        box = upright_bbox(seed, vertical)
        axes = seed.new_tensor(box['axes'])
        center = seed.new_tensor(box['center'])
        half = seed.new_tensor(box['size']) / 2
        volume = dict(shape=shape, **box)
        extent = float(half.max())
    else:
        reference = torch.eye(3, dtype=seed.dtype, device=seed.device)[vertical.abs().argmin()]
        horizontal = torch.linalg.cross(reference, vertical)
        horizontal /= horizontal.norm()
        tangent = torch.linalg.cross(vertical, horizontal)
        axes = torch.stack([horizontal, tangent, vertical])
        origin = seed.mean(0)
        local = (seed-origin) @ axes.T
        flat = local[:, :2]
        scale = float(flat.abs().max())
        if scale:
            hull = cv2.convexHull((flat / scale).cpu().float().numpy())
            circle, _ = cv2.minEnclosingCircle(hull)
            circle = seed.new_tensor(circle) * scale
        else:
            circle = seed.new_zeros(2)
        # Recompute the radius in float64: OpenCV adds a fixed padding otherwise.
        radius = float(torch.linalg.vector_norm(flat-circle, dim=1).max())
        low, high = local[:, 2].amin(), local[:, 2].amax()
        height = float(high-low)
        center = origin + torch.cat([circle, ((low+high)/2).reshape(1)]) @ axes
        volume = dict(shape=shape, center=center.cpu().tolist(), axis=vertical.cpu().tolist(), radius=radius, height=height)
        extent = max(radius, height/2)
    tolerance = extent * 1e-7 + torch.finfo(seed.dtype).eps * max(1., float(seed.abs().max())) * 32
    result = selection.clone()
    # Evaluate the entire scene, including occluded centers, without an N-by-3 float64 temporary.
    for start in range(0, len(xyz), 262144):
        local = (xyz[start:start+262144].detach().to(dtype=seed.dtype)-center) @ axes.T
        finite = torch.isfinite(local).all(dim=1)
        if shape == 'cube':
            inside = (local.abs() <= half+tolerance).all(dim=1)
        else:
            inside = (local[:, :2].square().sum(dim=1) <= (radius+tolerance)**2) & (local[:, 2].abs() <= height/2+tolerance)
        result[start:start+262144] |= inside & finite
    return result, volume
