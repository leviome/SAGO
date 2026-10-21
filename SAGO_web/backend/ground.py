"""Three surface samples define a world-space plane and oriented unit normal."""
import numpy as np


def ground_plane(points, camera_position, previous_up):
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (3, 3) or not np.isfinite(points).all():
        raise ValueError('地面需要三个有效的三维点')
    first, second = points[1]-points[0], points[2]-points[0]
    a, b = np.linalg.norm(first), np.linalg.norm(second)
    cross = np.cross(first, second)
    length = np.linalg.norm(cross)
    if min(a, b) <= 1e-8 or length <= .01*a*b:
        raise ValueError('三点重合或接近共线，请撤回并选择分散的地面点')
    normal = cross/length
    center = points.mean(axis=0)
    facing = float(np.dot(normal, np.asarray(camera_position)-center))
    if abs(facing) < 1e-7*max(a, b):
        facing = float(np.dot(normal, previous_up))
    if facing < 0:
        normal = -normal
    return dict(points=points.tolist(), normal=normal.tolist(),
                offset=float(-np.dot(normal, points[0])), center=center.tolist())


def flip_plane(plane):
    return dict(plane, normal=[-x for x in plane['normal']], offset=-plane['offset'])
