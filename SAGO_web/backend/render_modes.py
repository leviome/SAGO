"""Display-only depth coloring and a z-buffer for Gaussian center points."""
import numpy as np
import torch


def depth_rgb(depth):
    import cv2
    values = depth.detach().cpu().numpy() if torch.is_tensor(depth) else np.asarray(depth)
    values = np.asarray(values, dtype=np.float32)
    if values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if values.ndim != 2:
        raise ValueError('深度图必须为二维图像')
    valid = np.isfinite(values) & (values > 0)
    image = np.zeros((*values.shape, 3), dtype=np.uint8)
    if valid.any():
        near, far = np.percentile(values[valid], [2, 98])
        normalized = np.full(values.shape, .5, dtype=np.float32)
        if far > near:
            normalized[valid] = np.clip((values[valid] - near) / (far - near), 0, 1)
        gray = ((1 - normalized) * 255).astype(np.uint8)
        colored = cv2.cvtColor(cv2.applyColorMap(gray, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
        image[valid] = colored[valid]
    return image


def point_image(xyz, colors, projection, width, height):
    """One-pixel centers, nearest depth wins; equal-depth ties are deterministic."""
    image = xyz.new_zeros((width * height, 3))
    clip = xyz @ projection[:3, :] + projection[3, :]
    valid = torch.isfinite(clip).all(dim=1) & (clip[:, 3] > 1e-6) & (clip[:, 2] >= 0)
    clip, colors = clip[valid], colors[valid]
    if len(clip):
        screen = (clip[:, :2] / clip[:, 3:4] + 1) * xyz.new_tensor([width, height]) * .5 - .5
        inside = (screen[:, 0] >= -.5) & (screen[:, 0] < width-.5) & (screen[:, 1] >= -.5) & (screen[:, 1] < height-.5)
        pixels = torch.floor(screen[inside] + .5).long()
        depth, colors = clip[inside, 3], colors[inside]
        flat = pixels[:, 1] * width + pixels[:, 0]
        zbuffer = xyz.new_full((width * height,), float('inf'))
        zbuffer.scatter_reduce_(0, flat, depth, reduce='amin', include_self=True)
        order = torch.arange(len(flat), device=xyz.device)
        nearest = torch.where(depth == zbuffer[flat], order, len(flat))
        winners = torch.full((width * height,), len(flat), device=xyz.device, dtype=torch.long)
        winners.scatter_reduce_(0, flat, nearest, reduce='amin', include_self=True)
        shown = winners < len(flat)
        image[shown] = colors[winners[shown]]
    return image.reshape(height, width, 3).permute(2, 0, 1)


def point_owners(xyz, projection, width, height):
    """Original point indices at visible pixels; same z-buffer and ties as point_image."""
    owners = torch.full((width * height,), -1, dtype=torch.long, device=xyz.device)
    clip = xyz @ projection[:3, :] + projection[3, :]
    valid = torch.isfinite(clip).all(dim=1) & (clip[:, 3] > 1e-6) & (clip[:, 2] >= 0)
    indices = torch.where(valid)[0]
    clip = clip[valid]
    if len(clip):
        screen = (clip[:, :2] / clip[:, 3:4] + 1) * xyz.new_tensor([width, height]) * .5 - .5
        inside = (screen[:, 0] >= -.5) & (screen[:, 0] < width-.5) & (screen[:, 1] >= -.5) & (screen[:, 1] < height-.5)
        pixels = torch.floor(screen[inside] + .5).long()
        depth, indices = clip[inside, 3], indices[inside]
        flat = pixels[:, 1] * width + pixels[:, 0]
        zbuffer = xyz.new_full((width * height,), float('inf'))
        zbuffer.scatter_reduce_(0, flat, depth, reduce='amin', include_self=True)
        nearest = torch.where(depth == zbuffer[flat], indices, len(xyz))
        winners = torch.full_like(owners, len(xyz))
        winners.scatter_reduce_(0, flat, nearest, reduce='amin', include_self=True)
        owners = torch.where(winners < len(xyz), winners, -1)
    return owners.reshape(height, width)
