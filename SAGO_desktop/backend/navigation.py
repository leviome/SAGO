"""Camera-relative translation shared by pointer panning and keyboard roaming."""
import math
import numpy as np


def translated_target(view, camera_to_world, data):
    """Translate eye and target together; preserve viewing angles and radius."""
    move = np.asarray(data.get('move', [0, 0]), dtype=float)
    pan = np.asarray(data.get('pan', [0, 0]), dtype=float)
    if any(v.shape != (2,) or not np.isfinite(v).all() or (np.abs(v) > 1).any() for v in (move, pan)):
        raise ValueError('Navigation requires two finite values between -1 and 1')
    axes = np.asarray(camera_to_world, dtype=float)
    # Gaussian camera matrices store right, down, forward in inverse rows.
    right, down, forward = axes[:3, :3]
    delta = float(view['radius']) * (move[0] * right + move[1] * forward)
    scale = 2 * float(view['radius']) * math.tan(math.radians(view['fov']) / 2)
    delta -= scale * (pan[0] * view.get('aspect', 1) * right + pan[1] * down)
    return (np.asarray(view['target'], dtype=float) + delta).tolist()
