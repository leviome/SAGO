"""World-space geometry for display only; never changes segmentation masks."""
import math

import torch


def upright_bbox(xyz, up):
    """Minimum horizontal-area rectangle extruded along the supplied up vector."""
    import cv2

    if not len(xyz):
        return None
    xyz = xyz.detach()
    vertical = torch.as_tensor(up, dtype=xyz.dtype, device=xyz.device)
    vertical = vertical / torch.linalg.vector_norm(vertical)
    reference = torch.eye(3, dtype=xyz.dtype, device=xyz.device)[vertical.abs().argmin()]
    horizontal = torch.linalg.cross(reference, vertical)
    horizontal /= torch.linalg.vector_norm(horizontal)
    tangent = torch.linalg.cross(vertical, horizontal)
    origin = xyz.mean(0)
    centered = xyz - origin
    flat = centered @ torch.stack([horizontal, tangent], dim=1)
    # Convex hull + rotating calipers; the vertical axis remains constrained.
    hull = cv2.convexHull(flat.cpu().float().numpy())
    angle = math.radians(cv2.minAreaRect(hull)[2])
    x = math.cos(angle)*horizontal + math.sin(angle)*tangent
    y = torch.linalg.cross(vertical, x)
    axes = torch.stack([x, y, vertical])
    local = centered @ axes.T
    low, high = local.amin(0), local.amax(0)
    corners = torch.stack([torch.stack([high[j] if i & (1 << j) else low[j]
                                      for j in range(3)]) for i in range(8)]) @ axes + origin
    return dict(corners=corners.cpu().tolist(), axes=axes.cpu().tolist(),
                size=(high-low).cpu().tolist(), center=((low+high)*.5 @ axes+origin).cpu().tolist())


def camera_frustum(cam):
    """Use the actual render extrinsics; camera axes are right, down, forward."""
    inverse = cam.world_view_transform.inverse().detach().cpu().numpy()
    return dict(position=inverse[3, :3].tolist(), right=inverse[0, :3].tolist(),
                down=inverse[1, :3].tolist(), forward=inverse[2, :3].tolist(),
                tan_half_fov=[math.tan(cam.FoVx*.5), math.tan(cam.FoVy*.5)])
