"""Display-only colors for object IDs returned by scene RAG."""
import colorsys
import copy

import torch


def object_color(obj_id):
    """Stable, distinct hues for sparse object IDs, in RGB [0, 1]."""
    return colorsys.hsv_to_rgb((obj_id * 0.618033988749895) % 1, 0.8, 1.0)


def extract_object_masks(rag_mask, object_ids):
    if rag_mask is None or rag_mask.ndim != 1:
        raise ValueError("RAG highlight requires one label per Gaussian")
    masks = {}
    for obj_id in dict.fromkeys(object_ids):
        if type(obj_id) is not int or obj_id <= 0:
            raise ValueError("RAG highlight IDs must be positive integers")
        mask = rag_mask == obj_id
        if not mask.any().item():
            raise ValueError(f"Object {obj_id} is absent from the current RAG mask")
        masks[obj_id] = mask
    return masks


@torch.no_grad()
def highlighted_gaussians(gs, masks):
    """Return a rendering copy; leave scene features and geometry untouched."""
    if not masks:
        return gs
    for mask in masks.values():
        if mask.dtype != torch.bool or mask.shape != (len(gs.get_xyz),):
            raise ValueError("RAG highlight mask no longer matches the rendered scene; rebuild RAG")
    return _colored_copy(gs, masks.items())


@torch.no_grad()
def highlighted_rag_objects(gs, labels):
    """Color all positive labels, constructing only one object mask at a time."""
    if labels.shape != (len(gs.get_xyz),):
        raise ValueError("RAG mask no longer matches the rendered scene")
    ids = torch.unique(labels[labels > 0]).tolist()
    if not ids:
        return gs
    return _colored_copy(gs, ((int(obj_id), labels == obj_id) for obj_id in ids))


@torch.no_grad()
def _colored_copy(gs, masks):
    display = copy.copy(gs)
    display._features_dc = gs._features_dc.clone()
    display._features_rest = gs._features_rest.clone()
    for obj_id, mask in masks:
        mask = mask.to(display._features_dc.device)
        rgb = display._features_dc.new_tensor(object_color(obj_id))
        display._features_dc[mask] = (rgb - 0.5) / 0.28209479177387814
        display._features_rest[mask.to(display._features_rest.device)] = 0
    return display
