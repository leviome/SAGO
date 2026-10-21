"""Screen-space center and projected-ellipse selection for Gaussian ID correction."""
import numpy as np


def validated_box(width, height, rectangle):
    box = np.asarray(rectangle, dtype=float)
    if box.shape != (2, 2) or not np.isfinite(box).all():
        raise ValueError('修正需要有效的矩形框')
    low, high = box.min(axis=0), box.max(axis=0)
    if width < 1 or height < 1 or np.any(low < 0) or high[0] > width-1 or high[1] > height-1 or np.any(high-low < 2):
        raise ValueError('矩形框过小或超出当前画面')
    return low, high


def projected_box_mask(xyz, projection, width, height, rectangle, *, snap_to_pixel=False):
    low, high = validated_box(width, height, rectangle)
    xyz = np.asarray(xyz, dtype=np.float32)
    projection = np.asarray(projection, dtype=np.float32)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or projection.shape != (4, 4) or not np.isfinite(projection).all():
        raise ValueError('高斯位置或相机投影无效')
    clip = xyz @ projection[:3, :] + projection[3, :]
    valid = np.isfinite(clip).all(axis=1) & (clip[:, 3] > 1e-6) & (clip[:, 2] >= 0)
    screen = np.full((len(xyz), 2), np.nan, dtype=np.float32)
    # DOM/canvas coordinates start at pixel edges. Native Gaussian centers use
    # integer pixel centers, offset by -0.5; convert back to canvas coordinates.
    screen[valid] = (clip[valid, :2]/clip[valid, 3:4]+1)*np.array([width, height], dtype=np.float32)*.5
    if snap_to_pixel:
        screen[valid] = np.floor(screen[valid]) + .5
    return valid & (screen >= low).all(axis=1) & (screen <= high).all(axis=1)


def ellipse_box_mask(centers, covariance, low, high):
    """Exact rectangle intersection with a rotated 3-sigma ellipse, not its AABB."""
    centers = np.asarray(centers, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    xx, xy, yy = covariance[:, 0, 0], covariance[:, 0, 1], covariance[:, 1, 1]
    det = xx * yy - xy * xy
    valid = np.isfinite(centers).all(1) & np.isfinite(covariance).all((1, 2)) & (xx > 0) & (yy > 0) & (det > 0)
    radius = 3 * np.sqrt(np.maximum(np.stack([xx, yy], axis=1), 0))
    valid &= (centers + radius >= low).all(1) & (centers - radius <= high).all(1)
    result = np.zeros(len(centers), dtype=bool)
    indices = np.flatnonzero(valid)
    if not len(indices):
        return result
    lo, hi = low - centers[indices], high - centers[indices]
    # Minimize the ellipse's quadratic form over each of the four rectangle edges.
    a, b, c = yy[indices] / det[indices], -xy[indices] / det[indices], xx[indices] / det[indices]
    minimum = np.where(((lo <= 0) & (hi >= 0)).all(1), 0., np.inf)
    for x in (lo[:, 0], hi[:, 0]):
        y = np.clip(-b * x / c, lo[:, 1], hi[:, 1])
        minimum = np.minimum(minimum, a*x*x + 2*b*x*y + c*y*y)
    for y in (lo[:, 1], hi[:, 1]):
        x = np.clip(-b * y / a, lo[:, 0], hi[:, 0])
        minimum = np.minimum(minimum, a*x*x + 2*b*x*y + c*y*y)
    result[indices] = minimum <= 9. + 1e-6
    return result


def projected_ellipse_mask(xyz, covariance, view, projection, width, height, tan_fov, rectangle):
    """Match EWA camera covariance, FoV clamping and the renderer's pixel low-pass."""
    low, high = validated_box(width, height, rectangle)
    xyz, covariance = np.asarray(xyz, dtype=np.float32), np.asarray(covariance, dtype=np.float32)
    view, projection = np.asarray(view, dtype=np.float32), np.asarray(projection, dtype=np.float32)
    tan_fov = np.asarray(tan_fov, dtype=np.float32)
    if (xyz.shape != (len(xyz), 3) or covariance.shape != (len(xyz), 6) or view.shape != (4, 4)
            or projection.shape != (4, 4) or tan_fov.shape != (2,) or not np.isfinite(tan_fov).all()
            or np.any(tan_fov <= 0) or not np.isfinite(view).all() or not np.isfinite(projection).all()):
        raise ValueError('高斯协方差或相机投影无效')
    result = np.zeros(len(xyz), dtype=bool)
    focal = np.array([width, height], dtype=np.float32) / (2 * tan_fov)
    for start in range(0, len(xyz), 65536):
        points, packed = xyz[start:start+65536], covariance[start:start+65536]
        camera = points @ view[:3, :3] + view[3, :3]
        clip = points @ projection[:3, :] + projection[3, :]
        valid = (np.isfinite(camera).all(1) & np.isfinite(clip).all(1) & np.isfinite(packed).all(1)
                 & (camera[:, 2] > .2) & (clip[:, 3] > 1e-6) & (clip[:, 2] >= 0))
        indices = np.flatnonzero(valid)
        if not len(indices):
            continue
        cam, cov = camera[indices], packed[indices]
        ratio = np.clip(cam[:, :2] / cam[:, 2:3], -1.3*tan_fov, 1.3*tan_fov)
        jacobian = np.zeros((len(indices), 2, 3), dtype=np.float32)
        jacobian[:, 0, 0], jacobian[:, 1, 1] = focal[0]/cam[:, 2], focal[1]/cam[:, 2]
        jacobian[:, :, 2] = -focal * ratio / cam[:, 2:3]
        transform = jacobian @ view[:3, :3].T
        world_cov = np.empty((len(indices), 3, 3), dtype=np.float32)
        world_cov[:, 0, 0], world_cov[:, 1, 1], world_cov[:, 2, 2] = cov[:, 0], cov[:, 3], cov[:, 5]
        world_cov[:, 0, 1] = world_cov[:, 1, 0] = cov[:, 1]
        world_cov[:, 0, 2] = world_cov[:, 2, 0] = cov[:, 2]
        world_cov[:, 1, 2] = world_cov[:, 2, 1] = cov[:, 4]
        screen_cov = transform @ world_cov @ transform.transpose(0, 2, 1)
        screen_cov[:, 0, 0] += .3
        screen_cov[:, 1, 1] += .3
        screen = (clip[indices, :2]/clip[indices, 3:4]+1)*np.array([width, height])*.5-.5
        result[start+indices] = ellipse_box_mask(screen, screen_cov, low, high)
    return result


def visible_box_ids(owners, rectangle):
    """Select pixel owners inside a validated drag box; -1 means empty pixel."""
    owners = np.asarray(owners)
    height, width = owners.shape
    lower, upper = validated_box(width, height, rectangle)
    # A pixel is a unit cell in canvas coordinates, not a mathematical point.
    # Include the same boundary cell that a click at the box edge would hit.
    x0, y0 = np.floor(lower).astype(int)
    x1, y1 = np.floor(upper).astype(int)
    ids = np.unique(owners[y0:y1+1, x0:x1+1])
    return ids[ids >= 0]


def visible_pixel_ids(owners, pixel):
    """Hit the one visible Gaussian at a canvas pixel, without picking through it."""
    owners = np.asarray(owners)
    height, width = owners.shape
    point = np.asarray(pixel, dtype=float)
    if (point.shape != (2,) or not np.isfinite(point).all() or np.any(point < 0)
            or point[0] > width-1 or point[1] > height-1):
        raise ValueError('请选择画面内的有效像素')
    x, y = np.floor(point).astype(int)
    index = int(owners[y, x])
    return np.array([index], dtype=np.int64) if index >= 0 else np.empty(0, dtype=np.int64)
