"""Choose a contrasting drone background from the unpadded prompt mask."""
import numpy as np

from backend.render_modes import validate_background_color


def choose_drone_background(rgb, mask, enabled=True, manual=(0, 0, 0)):
    """Use the mean sRGB luminance of the one-pixel inner boundary, once per run.

    Only foreground pixels from the original mask are sampled. A constant-zero
    erosion border also handles full-image masks and objects touching the frame.
    The input image and mask are never modified or dilated.
    """
    if type(enabled) is not bool:
        raise ValueError('Adversarial background must be a boolean')
    if not enabled:
        return dict(enabled=False, color=validate_background_color(manual),
                    edge_luminance=None, edge_pixels=0)
    import cv2
    image = np.asarray(rgb)
    binary = np.asarray(mask) > 0
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or binary.shape != image.shape[:2]:
        raise ValueError('Drone background requires an RGB byte image and a matching 2D mask')
    if not binary.any():
        raise ValueError('Cannot choose a drone background from an empty mask')
    eroded = cv2.erode(binary.astype(np.uint8), np.ones((3, 3), np.uint8),
                       borderType=cv2.BORDER_CONSTANT, borderValue=0)
    boundary = binary & (eroded == 0)
    pixels = image[boundary].astype(np.float64) / 255.
    luminance = float((pixels @ np.array([.2126, .7152, .0722])).mean())
    color = [1., 1., 1.] if luminance < .5 else [0., 0., 0.]
    return dict(enabled=True, color=color, edge_luminance=luminance,
                edge_pixels=int(boundary.sum()))
