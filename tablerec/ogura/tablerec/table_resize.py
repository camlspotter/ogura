"""Shared shrink-only geometry and aligned soft supervision resizing."""
import numpy as np
from PIL import Image


def capped_size(size, max_side):
    if max_side is not None and (type(max_side) is not int or max_side<=0):
        raise ValueError('max_side must be a positive integer or None')
    if max_side is None or max(size)<=max_side:
        return tuple(size)
    ratio=max_side/max(size)
    return tuple(max(1,round(length*ratio)) for length in size)


def resize_masks(values, size, *, conservative=False):
    """Area-average masks; conservative evaluation selects fully covered pixels."""
    masks=[]
    for mask in values:
        image=Image.fromarray(np.asarray(mask,dtype=np.float32),mode='F')
        resized=np.array(image.resize(size,Image.Resampling.BOX),dtype=np.float32)
        masks.append(resized>=.999 if conservative else resized)
    return np.stack(masks)


def resize_image(image, max_side):
    size=capped_size(image.size,max_side)
    return image if size==image.size else image.resize(size,Image.Resampling.LANCZOS)
