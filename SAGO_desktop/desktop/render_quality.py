"""Choose display pixels independently from frozen segmentation coordinates."""
import math


def target_size(width, height, device_pixel_ratio=1., selected=None):
    if selected is not None:
        return int(selected)
    pixels=max(width,height)*device_pixel_ratio
    return max(512,min(2560,math.ceil(pixels/64)*64))
