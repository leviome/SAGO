import math
import os.path as osp

import camlab
import cv2
import json
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import ImageDraw
from camlab import load_3dgs_camera
import torchvision
from sympy.sets.handlers import union

from backend.camera_math import create_cam2world_matrix, get_origin, normalize_vecs
from gs.gaussian_renderer import render_simple
from gs.scene import GaussianModel
from gs.scene.cameras import CustomCam




def unified_render(
        viewpoint_camera, pc: GaussianModel,
        device="cuda:0",
        bg_color=torch.Tensor([0., 0., 0.]),
        scaling_modifier=1.0, bg="black",
        make_topk=False,
        gaussian_mask=None, extra_gau_mask=None,
        show_gau=False, mutlicolor=True, render_fg_only=False,
        override_color=None, debug=False, valid=None, look_at=None,
        obj_rm=False,
        show_box=False,
        render_mask=False,
        render_particles=False,
        up=torch.tensor([0.0, 1.0, 0.0]),
        render_mode="3dgs",
):
    # The optional 2DGS renderer is only imported when explicitly requested.
    if render_mode == "2dgs":
        from gs2d.gaussian_renderer import render_simple as render2dgs
        return render2dgs(viewpoint_camera, pc, bg_color=bg_color)
    render_fn_map = {
        "3dgs": render_simple,
    }
    render_fn = render_fn_map.get(render_mode)
    if render_fn is None:
        return None

    kwargs = dict(
        viewpoint_camera=viewpoint_camera,
        pc=pc,
        device=device,
        bg_color=bg_color,
        scaling_modifier=scaling_modifier,
        bg=bg,
        make_topk=make_topk,
        gaussian_mask=gaussian_mask,
        extra_gau_mask=extra_gau_mask,
        show_gau=show_gau,
        mutlicolor=mutlicolor,
        render_fg_only=render_fg_only,
        override_color=override_color,
        debug=debug,
        valid=valid,
        look_at=look_at,
        obj_rm=obj_rm,
        show_box=show_box,
        render_mask=render_mask,
        render_particles=render_particles,
        up=up
    )
    return render_fn(**kwargs)


def clear_mask_scatters(mask):
    is_numpy = isinstance(mask, np.ndarray)
    if is_numpy:
        mask_t = torch.from_numpy(mask)
    elif torch.is_tensor(mask):
        mask_t = mask
    else:
        raise TypeError("mask must be a torch.Tensor or np.ndarray")

    original_dtype = mask_t.dtype
    original_dim = mask_t.dim()

    if original_dim == 2:
        mask_2d = mask_t > 0
    elif original_dim == 3:
        # Support [1, H, W], [C, H, W], or [H, W, C]
        if mask_t.shape[0] in (1, 3):
            mask_2d = (mask_t > 0).any(dim=0)
        elif mask_t.shape[-1] in (1, 3):
            mask_2d = (mask_t > 0).any(dim=-1)
        else:
            raise ValueError("Unsupported 3D mask shape")
    else:
        raise ValueError("mask must be 2D or 3D")

    if not mask_2d.any():
        out = torch.zeros_like(mask_t)
        return out.cpu().numpy() if is_numpy else out

    unvisited = mask_2d.clone()
    largest_comp = torch.zeros_like(mask_2d)
    largest_size = 0

    while unvisited.any():
        seed = torch.nonzero(unvisited, as_tuple=False)[0]
        frontier = torch.zeros_like(unvisited)
        frontier[seed[0], seed[1]] = True

        comp = torch.zeros_like(unvisited)
        unvisited[seed[0], seed[1]] = False

        while frontier.any():
            comp |= frontier
            dilated = F.max_pool2d(
                frontier.float().unsqueeze(0).unsqueeze(0),
                kernel_size=3,
                stride=1,
                padding=1
            ).squeeze(0).squeeze(0) > 0
            new_frontier = dilated & unvisited
            unvisited[new_frontier] = False
            frontier = new_frontier

        comp_size = int(comp.sum().item())
        if comp_size > largest_size:
            largest_size = comp_size
            largest_comp = comp

    if original_dim == 2:
        out = largest_comp
    else:
        if mask_t.shape[0] in (1, 3):
            out = largest_comp.unsqueeze(0)
            if mask_t.shape[0] != 1:
                out = out.expand(mask_t.shape[0], -1, -1)
        else:
            out = largest_comp.unsqueeze(-1)
            if mask_t.shape[-1] != 1:
                out = out.expand(-1, -1, mask_t.shape[-1])

    if original_dtype != torch.bool:
        out = out.to(dtype=original_dtype)

    return out.cpu().numpy() if is_numpy else out


class Memory:
    def __init__(self, memory, ptr, prev_mems, prev_ptrs):
        self.memory = memory
        self.ptr = ptr
        self.prev_mems = prev_mems
        self.prev_ptrs = prev_ptrs


def render2img(rendered):
    view = torch.clamp(rendered.permute(1, 2, 0),
                       min=0.0, max=1.0).detach().cpu().numpy() * 255
    view = view.astype(np.uint8)
    view = cv2.cvtColor(view, cv2.COLOR_RGB2BGR)
    return view


import struct
from pathlib import Path


def _write_uint64(f, value):
    f.write(struct.pack("<Q", int(value)))


def _write_int32(f, value):
    f.write(struct.pack("<i", int(value)))


def _write_int64(f, value):
    f.write(struct.pack("<q", int(value)))


def _write_double(f, value):
    f.write(struct.pack("<d", float(value)))


def _write_string(f, value):
    if value is None:
        value = ""
    if not isinstance(value, str):
        value = str(value)
    f.write(value.encode("utf-8") + b"\x00")


def _qvec_from_w2c(R_w2c):
    """
    COLMAP images.bin needs qvec of world-to-camera rotation.
    Input:
        R_w2c: torch.Tensor or np.ndarray, shape [3, 3]
    Output:
        np.ndarray shape [4], [qw, qx, qy, qz]
    """
    if torch.is_tensor(R_w2c):
        R = R_w2c.detach().cpu().float()
    else:
        R = torch.as_tensor(R_w2c, dtype=torch.float32)

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = torch.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = torch.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = torch.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = torch.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    q = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)
    q = q / torch.linalg.norm(q)
    return q.cpu().numpy()


def _get_cam_w2c_and_center(cam):
    """
    Try to robustly extract w2c and camera center from CustomCam.
    """
    if hasattr(cam, "world_view_transform"):
        w2c = cam.world_view_transform.T
    elif hasattr(cam, "world_view_transform"):
        w2c = cam.world_view_transform
    elif hasattr(cam, "extr"):
        # If extr is cam2world, convert to w2c
        extr = cam.extr
        if torch.is_tensor(extr):
            extr = extr.detach().clone()
        else:
            extr = torch.as_tensor(extr)
        c2w = extr
        w2c = torch.linalg.inv(c2w)
    else:
        raise AttributeError("Cannot find world_view_transform or extr in CustomCam")

    if torch.is_tensor(w2c):
        w2c = w2c.detach().cpu()
    else:
        w2c = torch.as_tensor(w2c, dtype=torch.float32)

    R = w2c[:3, :3]
    t = w2c[:3, 3]
    center = (-R.T @ t)
    return w2c, center


def _infer_cam_name(cam, cam_id):
    for attr in ["image_name", "name", "img_name"]:
        if hasattr(cam, attr):
            v = getattr(cam, attr)
            if v:
                return str(v)
    return f"{cam_id:03d}.png"


def _cam_intrinsics(cam):
    width = int(getattr(cam, "image_width"))
    height = int(getattr(cam, "image_height"))
    fovx = float(getattr(cam, "FoVx"))
    fovy = float(getattr(cam, "FoVy"))

    fx = 0.5 * width / np.tan(fovx / 2.0)
    fy = 0.5 * height / np.tan(fovy / 2.0)
    cx = width / 2.0
    cy = height / 2.0
    return width, height, fx, fy, cx, cy


def _get_gaussian_rgb(gaussians):
    """
    Convert SH DC features to a rough RGB for COLMAP point cloud.
    """
    dc = gaussians._features_dc
    if not torch.is_tensor(dc):
        dc = torch.as_tensor(dc)

    # Common shape: [N, 1, 3]
    if dc.dim() == 3:
        rgb = dc[:, 0, :]
    elif dc.dim() == 2:
        rgb = dc
    else:
        raise ValueError(f"Unexpected _features_dc shape: {tuple(dc.shape)}")

    rgb = (rgb + 1.0) / 2.0
    rgb = torch.clamp(rgb, 0.0, 1.0)
    rgb = (rgb * 255.0).to(torch.uint8)
    return rgb.cpu().numpy()


def save_cams_to_colmap(cams, gaussians, out_dir="sparse/0"):
    """
    Save CustomCam list and GaussianModel point cloud to COLMAP binary format.

    Outputs:
        out_dir/cameras.bin
        out_dir/images.bin
        out_dir/points3D.bin
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cameras_bin = out_dir / "cameras.bin"
    images_bin = out_dir / "images.bin"
    points3d_bin = out_dir / "points3D.bin"

    # ---------- cameras.bin ----------
    # Use one COLMAP camera per input cam. If intrinsics are different,
    # each cam gets its own camera_id.
    camera_records = []
    for cam_id, cam in enumerate(cams, start=1):
        width, height, fx, fy, cx, cy = _cam_intrinsics(cam)

        # COLMAP PINHOLE: model_id = 1, params = [fx, fy, cx, cy]
        camera_records.append({
            "camera_id": cam_id,
            "model_id": 1,  # PINHOLE
            "width": width,
            "height": height,
            "params": [fx, fy, cx, cy],
        })

    with open(cameras_bin, "wb") as f:
        _write_uint64(f, len(camera_records))
        for rec in camera_records:
            _write_int32(f, rec["camera_id"])
            _write_int32(f, rec["model_id"])
            _write_uint64(f, rec["width"])
            _write_uint64(f, rec["height"])
            for p in rec["params"]:
                _write_double(f, p)

    # ---------- images.bin ----------
    with open(images_bin, "wb") as f:
        _write_uint64(f, len(cams))
        for image_id, cam in enumerate(cams, start=1):
            w2c, _ = _get_cam_w2c_and_center(cam)
            R = w2c[:3, :3]
            t = w2c[:3, 3]

            qvec = _qvec_from_w2c(R)
            name = _infer_cam_name(cam, image_id - 1)
            camera_id = image_id  # one image -> one camera

            _write_int32(f, image_id)
            for v in qvec:
                _write_double(f, v)
            for v in t.cpu().numpy().tolist():
                _write_double(f, v)
            _write_int32(f, camera_id)
            _write_string(f, name)

            # No 2D points in this export, so write 0 correspondences
            _write_uint64(f, 0)

    # ---------- points3D.bin ----------
    xyz = gaussians.get_xyz
    if not torch.is_tensor(xyz):
        xyz = torch.as_tensor(xyz)
    xyz = xyz.detach().cpu().float().numpy()

    rgb = _get_gaussian_rgb(gaussians)

    with open(points3d_bin, "wb") as f:
        _write_uint64(f, xyz.shape[0])

        for pid in range(xyz.shape[0]):
            _write_int64(f, pid + 1)
            _write_double(f, float(xyz[pid, 0]))
            _write_double(f, float(xyz[pid, 1]))
            _write_double(f, float(xyz[pid, 2]))

            r, g, b = rgb[pid].tolist()
            f.write(struct.pack("<BBB", r, g, b))

            _write_double(f, 0.0)  # error
            _write_uint64(f, 0)     # track length = 0

    print(f"Saved COLMAP files to: {out_dir}")
    print(f"  - {cameras_bin}")
    print(f"  - {images_bin}")
    print(f"  - {points3d_bin}")


def save_cams_to_json(cams, filename="cameras.json"):
    """
    Save a list of CustomCam objects to a JSON file.

    Args:
        cams: List of CustomCam objects
        filename: Output JSON file path
    """
    cameras_data = []

    for cam_id, cam in enumerate(cams):
        # Get the camera matrix (cam2world transform)
        c2w = cam.world_view_transform.T  # Convert from world_view_transform

        # Extract rotation matrix and translation
        rotation = c2w[:3, :3].cpu().numpy() if torch.is_tensor(c2w[:3, :3]) else c2w[:3, :3]
        position = c2w[:3, 3].cpu().numpy() if torch.is_tensor(c2w[:3, 3]) else c2w[:3, 3]

        # Compute focal lengths from FoV
        fx = 0.5 * cam.image_width / np.tan(cam.FoVx / 2)
        fy = 0.5 * cam.image_height / np.tan(cam.FoVy / 2)

        # Create camera entry
        camera_entry = {
            "id": cam_id,
            "img_name": cam.image_name if hasattr(cam, 'image_name') and cam.image_name else f"frame_{cam_id:05d}",
            "width": cam.image_width,
            "height": cam.image_height,
            "position": position.tolist(),
            "rotation": rotation.tolist(),
            "fy": float(fy),
            "fx": float(fx)
        }

        cameras_data.append(camera_entry)

    # Save to JSON file
    with open(filename, 'w') as f:
        json.dump(cameras_data, f)

    print(f"Saved {len(cams)} cameras to {filename}")


def mask_average_location(mask, depth, cam):
    nonzero_indices = torch.nonzero(mask)
    depths = depth[nonzero_indices[:, 0], nonzero_indices[:, 1]]
    points2d = nonzero_indices[:, [1, 0]].to(dtype=torch.float32)
    pts3d = back_projection(points2d, depths, cam)
    lookat_point = pts3d.mean(dim=0).cpu()
    return lookat_point


def apply_offset_to_pc1_xyz(xyz, offset):
    if offset is None:
        return xyz
    off = torch.as_tensor(offset, device=xyz.device, dtype=xyz.dtype)
    if off.numel() != 3:
        raise ValueError("offset must have 3 elements")
    return xyz + off.reshape(1, 3)


def merge_gaussians(pc1, pc2, offset=None):
    if pc1 is None and pc2 is None:
        return None
    if pc1 is None:
        return merge_gaussians(pc2, None)
    if pc2 is None:
        pc_merge = GaussianModel(pc1.max_sh_degree, True)
        pc_merge.precise_init(
            active_sh_degree=pc1.active_sh_degree,
            xyz=apply_offset_to_pc1_xyz(pc1.get_xyz.clone(), offset),
            features_dc=pc1._features_dc.clone(),
            features_rest=pc1._features_rest.clone(),
            scaling=pc1._scaling.clone(),
            rotation=pc1._rotation.clone(),
            opacity=pc1._opacity.clone(),
            max_radii2D=pc1.max_radii2D.clone(),
            xyz_gradient_accum=pc1.xyz_gradient_accum.clone() if torch.is_tensor(pc1.xyz_gradient_accum) else None,
            denom=pc1.denom.clone() if torch.is_tensor(pc1.denom) else pc1.denom,
            spatial_lr_scale=pc1.spatial_lr_scale,
            sh_degree=pc1.max_sh_degree,
        )
        return pc_merge

    if not isinstance(pc1, GaussianModel) or not isinstance(pc2, GaussianModel):
        raise TypeError("pc1 and pc2 must be GaussianModel instances")

    if pc1.active_sh_degree != pc2.active_sh_degree:
        raise ValueError("pc1 and pc2 must have the same active_sh_degree")

    if pc1.max_sh_degree != pc2.max_sh_degree:
        raise ValueError("pc1 and pc2 must have the same max_sh_degree")

    if pc1.disable_xyz_log_activation != pc2.disable_xyz_log_activation:
        raise ValueError("pc1 and pc2 must share disable_xyz_log_activation setting")

    if pc1._features_dc.shape[1:] != pc2._features_dc.shape[1:]:
        raise ValueError("pc1 and pc2 _features_dc shapes are incompatible")
    if pc1._features_rest.shape[1:] != pc2._features_rest.shape[1:]:
        raise ValueError("pc1 and pc2 _features_rest shapes are incompatible")
    if pc1._scaling.shape[1:] != pc2._scaling.shape[1:]:
        raise ValueError("pc1 and pc2 _scaling shapes are incompatible")
    if pc1._rotation.shape[1:] != pc2._rotation.shape[1:]:
        raise ValueError("pc1 and pc2 _rotation shapes are incompatible")
    if pc1._opacity.shape[1:] != pc2._opacity.shape[1:]:
        raise ValueError("pc1 and pc2 _opacity shapes are incompatible")

    merged_xyz = torch.cat((apply_offset_to_pc1_xyz(pc1.get_xyz, offset), pc2.get_xyz), dim=0)
    merged_features_dc = torch.cat((pc1._features_dc, pc2._features_dc), dim=0)
    merged_features_rest = torch.cat((pc1._features_rest, pc2._features_rest), dim=0)
    merged_scaling = torch.cat((pc1._scaling, pc2._scaling), dim=0)
    merged_rotation = torch.cat((pc1._rotation, pc2._rotation), dim=0)
    merged_opacity = torch.cat((pc1._opacity, pc2._opacity), dim=0)

    # Keep bookkeeping tensors aligned with merged Gaussian count.
    if torch.is_tensor(pc1.max_radii2D) and torch.is_tensor(pc2.max_radii2D) and \
            pc1.max_radii2D.numel() and pc2.max_radii2D.numel():
        merged_max_radii2D = torch.cat((pc1.max_radii2D, pc2.max_radii2D), dim=0)
    else:
        merged_max_radii2D = torch.zeros(merged_xyz.shape[0], device=merged_xyz.device, dtype=merged_xyz.dtype)

    if torch.is_tensor(pc1.denom) and torch.is_tensor(pc2.denom) and \
            pc1.denom.numel() and pc2.denom.numel():
        merged_denom = torch.cat((pc1.denom, pc2.denom), dim=0)
    else:
        merged_denom = torch.zeros((merged_xyz.shape[0], 1), device=merged_xyz.device, dtype=merged_xyz.dtype)

    if torch.is_tensor(pc1.xyz_gradient_accum) and torch.is_tensor(pc2.xyz_gradient_accum) and \
            pc1.xyz_gradient_accum.numel() and pc2.xyz_gradient_accum.numel():
        merged_xyz_grad_accum = torch.cat((pc1.xyz_gradient_accum, pc2.xyz_gradient_accum), dim=0)
    else:
        merged_xyz_grad_accum = None

    pc_merge = GaussianModel(pc1.max_sh_degree, True)
    pc_merge.precise_init(
        active_sh_degree=pc1.active_sh_degree,
        xyz=merged_xyz,
        features_dc=merged_features_dc,
        features_rest=merged_features_rest,
        scaling=merged_scaling,
        rotation=merged_rotation,
        opacity=merged_opacity,
        max_radii2D=merged_max_radii2D,
        xyz_gradient_accum=merged_xyz_grad_accum,
        denom=merged_denom,
        spatial_lr_scale=pc1.spatial_lr_scale,
        sh_degree=pc1.max_sh_degree,
    )

    return pc_merge


def ground_rect_gaussians(gaussians, up_vec, center_point=(0, 0, 0), save_name=None, save_glb=False,
                          center_as_origin=False):
    def _quat_mul(q1, q2):
        w1, x1, y1, z1 = q1.unbind(dim=-1)
        w2, x2, y2, z2 = q2.unbind(dim=-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ], dim=-1)

    if gaussians is None:
        raise ValueError("gaussians is None")

    xyz = gaussians.get_xyz
    device, dtype = xyz.device, xyz.dtype

    up = -1 * torch.as_tensor(up_vec, device=device, dtype=dtype)
    if up.numel() != 3:
        raise ValueError("up_vec must have 3 elements")
    up = up.reshape(3)
    up_norm = torch.norm(up)
    if up_norm < 1e-8:
        raise ValueError("up_vec norm is too small")
    up = up / up_norm

    world_up = torch.tensor([0.0, 1.0, 0.0], device=device, dtype=dtype)
    dot = torch.clamp(torch.dot(up, world_up), -1.0, 1.0)

    if dot > 1.0 - 1e-6:
        rot_mat = torch.eye(3, device=device, dtype=dtype)
        align_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
    elif dot < -1.0 + 1e-6:
        # 180-degree flip: choose any axis perpendicular to up.
        aux = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=dtype)
        if torch.abs(torch.dot(up, aux)) > 0.9:
            aux = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=dtype)
        axis = torch.cross(up, aux, dim=0)
        axis = axis / torch.norm(axis)
        angle = torch.tensor(math.pi, device=device, dtype=dtype)
        K = skew_symmetric(axis).to(device=device, dtype=dtype)
        rot_mat = torch.eye(3, device=device, dtype=dtype) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        align_q = torch.cat([torch.cos(angle * 0.5).reshape(1), axis * torch.sin(angle * 0.5)])
    else:
        axis = torch.cross(up, world_up, dim=0)
        axis = axis / torch.norm(axis)
        angle = torch.acos(dot)
        K = skew_symmetric(axis).to(device=device, dtype=dtype)
        rot_mat = torch.eye(3, device=device, dtype=dtype) + torch.sin(angle) * K + (1 - torch.cos(angle)) * (K @ K)
        align_q = torch.cat([torch.cos(angle * 0.5).reshape(1), axis * torch.sin(angle * 0.5)])

    align_q = F.normalize(align_q.unsqueeze(0), dim=-1).squeeze(0)

    center = torch.as_tensor(center_point, device=device, dtype=dtype)
    if center.numel() != 3:
        raise ValueError("center_point must have 3 elements")
    center = center.reshape(1, 3)

    # center_as_origin=True returns coordinates in the rotated local frame centered at center_point.
    if center_as_origin:
        rectified_xyz = torch.matmul(xyz - center, rot_mat.T)
    else:
        rectified_xyz = torch.matmul(xyz - center, rot_mat.T) + center

    old_q = gaussians._rotation.to(device=device, dtype=dtype)
    align_q_expand = align_q.unsqueeze(0).expand(old_q.shape[0], -1)
    rectified_rotation = _quat_mul(align_q_expand, old_q)
    rectified_rotation = F.normalize(rectified_rotation, dim=-1)

    rectified_gaussians = GaussianModel(gaussians.max_sh_degree, True)
    rectified_gaussians.precise_init(
        active_sh_degree=gaussians.active_sh_degree,
        xyz=rectified_xyz,
        features_dc=gaussians._features_dc.clone(),
        features_rest=gaussians._features_rest.clone(),
        scaling=gaussians._scaling.clone(),
        rotation=rectified_rotation,
        opacity=gaussians._opacity.clone(),
        max_radii2D=gaussians.max_radii2D.clone() if torch.is_tensor(gaussians.max_radii2D) else gaussians.max_radii2D,
        xyz_gradient_accum=gaussians.xyz_gradient_accum.clone() if torch.is_tensor(
            gaussians.xyz_gradient_accum) else None,
        denom=gaussians.denom.clone() if torch.is_tensor(gaussians.denom) else gaussians.denom,
        spatial_lr_scale=gaussians.spatial_lr_scale,
        sh_degree=gaussians.max_sh_degree,
    )

    if save_name:
        rectified_gaussians.save_ply(save_name)

    if save_glb:
        points = rectified_gaussians.get_xyz.cpu().numpy()
        rgb = (rectified_gaussians._features_dc.squeeze(1).cpu().numpy() + 1) / 2
        rgb = np.clip(rgb, 0, 1)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
        glb_name = save_name.replace('.ply', '.glb') if save_name else 'object.glb'
        o3d.io.write_triangle_mesh(glb_name, mesh, write_ascii=False)

    return rectified_gaussians


def masking_for_fg(gaussians, gau_mask,
                   save_name=None, for_bg=False, save_glb=False,
                   offset=None, scale_factor=1.0):
    if gau_mask is None:
        return gaussians
    gau_mask = gau_mask if not for_bg else ~gau_mask
    fg = GaussianModel(0, True)
    new_xyz = gaussians.get_xyz[gau_mask] * scale_factor
    new_scaling = gaussians._scaling[gau_mask]
    device = new_xyz.device
    log_factor = torch.log(torch.tensor(scale_factor,
                                        dtype=torch.float32, device=device))
    fg.precise_init(gaussians.active_sh_degree,
                    xyz=apply_offset_to_pc1_xyz(new_xyz.clone(), offset),
                    features_dc=gaussians._features_dc[gau_mask],
                    features_rest=gaussians._features_rest[gau_mask],
                    scaling=new_scaling + log_factor,
                    rotation=gaussians._rotation[gau_mask],
                    opacity=gaussians._opacity[gau_mask],
                    max_radii2D=gaussians.max_radii2D,
                    xyz_gradient_accum=None,
                    denom=gaussians.denom,
                    spatial_lr_scale=gaussians.spatial_lr_scale)
    if save_name:
        fg.save_ply(save_name)
    if save_glb:
        points = fg.get_xyz.cpu().numpy()
        rgb = (fg._features_dc.squeeze(1).cpu().numpy() + 1) / 2
        rgb = np.clip(rgb, 0, 1)
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(points)
        mesh.vertex_colors = o3d.utility.Vector3dVector(rgb)
        glb_name = save_name.replace('.ply', '.glb') if save_name else 'object.glb'
        o3d.io.write_triangle_mesh(glb_name, mesh, write_ascii=False)
    return fg


def viz_mask(image, mask, only_fg=False, box=None, ratio=1, points=None, red_points=None,
             save_name=None, prefix="", contour=False, to_alpha=False, text=None):
    if box is not None:
        points = None
        red_points = None
    if contour:
        # Draw contour on a copy
        # img = image.copy()
        # blank = np.ones_like(img) * 255
        # contours, _ = cv2.findContours(mask.astype(np.uint8),
        #                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = image.copy()
        # create transparent BG (BGRA) instead of white background
        blank = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        contours, _ = cv2.findContours(mask.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # draw contours into BGRA blank with full alpha
        # blank = cv2.drawContours(blank, contours, -1, (0, 85, 255, 255), 10)
        blank = cv2.drawContours(img, contours, -1, (0, 85, 255, 255), 10)
        img = blank

    else:
        # color = np.array([30 / 255, 0 / 255, 100 / 255, 0.3])
        # if mask.ndim == 3:
        #     mask_for_color = mask[0]
        # else:
        #     mask_for_color = mask
        # h, w = mask_for_color.shape[-2:]
        # mask_image = mask_for_color.reshape(h, w, 1) * color.reshape(1, 1, -1)
        # img = mask_image[:, :, :3] * image
        # indx = np.where(img == 0)
        # img[indx] = image[indx]
        # improved mask visualization: smooth alpha blend overlay (BGR)
        color = np.array([30, 0, 170], dtype=np.uint8)  # BGR color
        alpha = 0.7 if not only_fg else 0.0
        if mask.ndim == 3:
            mask_for_color = mask[0]
        else:
            mask_for_color = mask
        h, w = mask_for_color.shape[-2:]

        # normalize and prepare arrays
        mask_bin = (mask_for_color > 0).astype(np.float32)
        img_float = image.astype(np.float32) / 255.0
        overlay = np.ones_like(img_float, dtype=np.float32)
        overlay[..., 0] = color[0] / 255.0
        overlay[..., 1] = color[1] / 255.0
        overlay[..., 2] = color[2] / 255.0

        mask_exp = mask_bin.reshape(h, w, 1)
        blended = img_float * (1.0 - mask_exp * alpha) + overlay * (mask_exp * alpha)

        img = (np.clip(blended, 0.0, 1.0) * 255).astype(np.uint8)

        if points:
            for p in points:
                img = cv2.circle(img, (int(p[0]), int(p[1])), 12, (0, 255, 0), 10)
        if red_points:
            for p in red_points:
                img = cv2.circle(img, (int(p[0]), int(p[1])), 12, (255, 0, 0), 10)

        if box:
            (x1, y1), (x2, y2) = box
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        if ratio != 1:
            h0, w0, _ = image.shape
            img = cv2.resize(img, (int(w0 / ratio), int(h0 / ratio)))

    if to_alpha:
        # Create RGBA with transparent background outside mask, preserving contour if drawn
        if mask.ndim == 3:
            mask_bin = mask[0]
        else:
            mask_bin = mask
        mask_bin = (mask_bin > 0).astype(np.uint8)

        if img.shape[2] == 3:
            rgba = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        else:
            rgba = img.copy()

        rgba[..., 3] = mask_bin * 255
        bg_idx = mask_bin == 0
        rgba[bg_idx] = (0, 0, 0, 0)
        if ratio != 1:
            h0, w0 = mask_bin.shape[-2:]
            rgba = cv2.resize(rgba, (int(w0 / ratio), int(h0 / ratio)), interpolation=cv2.INTER_LINEAR)
        img = rgba

    if text is not None:
        cv2.putText(img, str(text), (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    2.0, (0, 0, 255), 3, cv2.LINE_AA)
    if save_name is not None:
        save_name = osp.join(prefix, save_name)
        cv2.imwrite(save_name, img)
    return img


@torch.no_grad()
def render2mask_torch(img_tensor):
    view = torch.clamp(img_tensor.permute(1, 2, 0), min=0.0, max=1.0)

    if view.dim() == 3 and view.shape[-1] >= 3:
        gray = 0.114 * view[..., 0] + 0.587 * view[..., 1] + 0.299 * view[..., 2]
    else:
        gray = view.squeeze(-1) if view.dim() == 3 else view

    border = torch.cat((gray[0, :], gray[-1, :], gray[:, 0], gray[:, -1]), dim=0)
    bg_is_white = border.mean() > 0.5

    if bg_is_white:
        mask = (gray < 0.9).to(torch.uint8)
    else:
        mask = (gray > 0.1).to(torch.uint8)

    return mask.detach().cpu().numpy() * 255


def render_mask(cam, pc, mask_3d, save_name=None, with_bg=False):
    if not with_bg:
        mask_renders = render_simple(viewpoint_camera=cam,
                                     pc=masking_for_fg(pc, mask_3d),
                                     render_mask=True)
        rendered_mask = mask_renders["mask"] > 0.3
    else:
        mask_renders = render_simple(viewpoint_camera=cam, pc=pc,
                                     gaussian_mask=[mask_3d],
                                     render_mask=True)

        tensor = mask_renders["mask"]
        target = torch.tensor([1.4713, 0.4999, 0.4999],
                              device=tensor.device,
                              dtype=tensor.dtype).view(3, 1, 1)

        rendered_mask = (torch.abs(tensor - target)
                         <= 0.5).all(dim=0, keepdim=True).expand(3, -1, -1)

    if save_name is not None:
        torchvision.utils.save_image(rendered_mask.float(), save_name)
    return rendered_mask


def render2mask(img):
    arr = np.asarray(img)
    if arr.ndim == 3:
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    else:
        gray = arr

    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    _, bin_mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    def border_fg_count(m):
        return (
                np.count_nonzero(m[0, :])
                + np.count_nonzero(m[-1, :])
                + np.count_nonzero(m[:, 0])
                + np.count_nonzero(m[:, -1])
        )

    cand_a = bin_mask
    cand_b = cv2.bitwise_not(bin_mask)
    mask = cand_a if border_fg_count(cand_a) <= border_fg_count(cand_b) else cand_b

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask, dtype=np.uint8)

    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def mIOU_torch(mask1, mask2, ioi=False):
    if mask1 is None or mask2 is None:
        return 0.0

    m1 = torch.as_tensor(mask1)
    m2 = torch.as_tensor(mask2, device=m1.device)

    if m1.shape != m2.shape:
        raise ValueError("mask1 and mask2 must have the same shape")

    m1 = m1 > 0
    m2 = m2 > 0

    inter = torch.logical_and(m1, m2).sum(dtype=torch.float32)
    if not ioi:
        union = torch.logical_or(m1, m2).sum(dtype=torch.float32)
    else:
        union = m1.sum(dtype=torch.float32)

    return float((inter / union).item()) if union.item() > 0 else 1.0


def mIOU(mask1, mask2, ioi=False):
    if mask1 is None or mask2 is None:
        return 0.0

    m1 = np.asarray(mask1)
    m2 = np.asarray(mask2)

    if m1.shape != m2.shape:
        raise ValueError("mask1 and mask2 must have the same shape")

    m1 = m1 > 0
    m2 = m2 > 0

    inter = np.logical_and(m1, m2).sum(dtype=np.float64)
    if not ioi:
        union = np.logical_or(m1, m2).sum(dtype=np.float64)
    else:
        union = np.sum(m1, dtype=np.float64)

    return float(inter / union) if union > 0 else 1.0


@torch.no_grad()
def initial_miou_validation(mask_3d, gaussians, initial_cam, initial_mask, with_box=False):
    fg_set = masking_for_fg(gaussians, mask_3d)
    renders = render_simple(initial_cam, fg_set)
    initial_view = renders["render"] > 0.0
    ioi = 0
    initial_box = mask2box_torch(initial_mask)
    new_box = mask2box_torch(initial_view[0, :, :])

    if with_box and new_box is not None and initial_box is not None:
        (ix1, iy1), (ix2, iy2) = initial_box
        (nx1, ny1), (nx2, ny2) = new_box

        # intersection
        x1_inter = max(ix1, nx1)
        y1_inter = max(iy1, ny1)
        x2_inter = min(ix2, nx2)
        y2_inter = min(iy2, ny2)

        inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
        initial_area = (ix2 - ix1) * (iy2 - iy1)

        ioi = inter_area / initial_area if initial_area > 0 else 0

    pred_mask = initial_view
    if pred_mask.dim() == 3:
        pred_mask = pred_mask.any(dim=0)
    pred_mask = pred_mask > 0

    if isinstance(initial_mask, np.ndarray):
        gt_mask = torch.from_numpy(initial_mask).to(device=pred_mask.device)
    else:
        gt_mask = initial_mask.to(device=pred_mask.device)

    if gt_mask.dim() == 3:
        gt_mask = gt_mask.any(dim=0)
    gt_mask = gt_mask > 0

    inter = torch.logical_and(pred_mask, gt_mask).sum(dtype=torch.float32)
    union = torch.sum(gt_mask, dtype=torch.float32)
    miou = inter / union if union.item() > 0 else torch.tensor(1.0, device=pred_mask.device)
    if with_box:
        return float(miou.item()), ioi

    return float(miou.item())


def distinct_background(img, mask):
    device, dtype = img.device, img.dtype
    if img.dim() != 3:
        raise ValueError(r"img must be a 3D tensor \([C,H,W] or [H,W,C]\).")

    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask).to(device=device)
    elif torch.is_tensor(mask):
        mask = mask.to(device=device)
    else:
        raise TypeError("mask must be a numpy array or torch tensor.")

    # Normalize image layout to HWC for indexing
    if img.shape[0] == 3 and img.shape[-1] != 3:
        img_hwc = img.permute(1, 2, 0)
    elif img.shape[-1] == 3:
        img_hwc = img
    else:
        raise ValueError("img must have 3 color channels.")

    # Normalize mask to HW
    if mask.dim() == 3:
        if mask.shape[0] in (1, 3) and mask.shape[1] == img_hwc.shape[0] and mask.shape[2] == img_hwc.shape[1]:
            mask_2d = (mask > 0).any(dim=0)
        elif mask.shape[-1] in (1, 3) and mask.shape[0] == img_hwc.shape[0] and mask.shape[1] == img_hwc.shape[1]:
            mask_2d = (mask > 0).any(dim=-1)
        else:
            raise ValueError("mask shape is incompatible with img.")
    elif mask.dim() == 2:
        mask_2d = mask > 0
    else:
        raise ValueError("mask must be a 2D or 3D tensor.")

    # Fallback when no foreground pixel is selected
    if not mask_2d.any().item():
        white_val = 1.0 if img.is_floating_point() else float(torch.iinfo(dtype).max)
        return torch.tensor([white_val, white_val, white_val], device=device, dtype=dtype)

    # Use contour mean instead of full foreground mean
    mask_f = mask_2d.to(torch.float32).unsqueeze(0).unsqueeze(0)
    kernel = torch.ones((1, 1, 3, 3), device=device, dtype=torch.float32)
    eroded = (F.conv2d(mask_f, kernel, padding=1) == 9).squeeze(0).squeeze(0)
    contour_mask = mask_2d & (~eroded)

    if not contour_mask.any().item():
        contour_mask = mask_2d

    contour_pixels = img_hwc[contour_mask].to(torch.float32)
    contour_mean = contour_pixels.mean(dim=0)
    # print("contour_mean:", contour_mean)

    white_val = 1.0 if img.is_floating_point() else float(torch.iinfo(dtype).max)
    black_color_f = torch.tensor([0.0, 0.0, 0.0], device=device, dtype=torch.float32)
    white_color_f = torch.tensor([white_val, white_val, white_val], device=device, dtype=torch.float32)

    # Pick the background color that is more different from contour color
    contrast_black = torch.norm(contour_mean - black_color_f, p=2)
    contrast_white = torch.norm(contour_mean - white_color_f, p=2)

    bg_color_f = white_color_f if contrast_white >= contrast_black else black_color_f
    return bg_color_f.to(dtype=dtype)


def get_area(gaussians, mask_3d, up_vector, ratio=1.0):
    xyz = gaussians.get_xyz
    device, dtype = xyz.device, xyz.dtype

    if mask_3d is None:
        raise ValueError("mask_3d is None.")
    if mask_3d.dtype != torch.bool:
        mask_3d = mask_3d > 0

    pts = xyz[mask_3d]
    if pts.numel() == 0:
        print("mask_3d selects no points.")
        return None, None, None

    # Center of the selected object
    pos3d = pts.mean(dim=0)

    # Cylinder axis direction
    up = torch.as_tensor(up_vector, device=device, dtype=dtype)
    up_norm = torch.norm(up)
    if up_norm < 1e-8:
        raise ValueError("up_vector norm is too small.")
    up = up / up_norm

    # Decompose points into axial and radial components
    rel = pts - pos3d.unsqueeze(0)
    axial_scalar = (rel * up.unsqueeze(0)).sum(dim=1)  # signed distance along axis
    axial = axial_scalar.unsqueeze(1) * up.unsqueeze(0)
    radial = rel - axial

    # Radius = max perpendicular distance to axis
    radius = torch.sqrt((radial * radial).sum(dim=1)).max()

    # Height = extent along axis
    height = axial_scalar.max() - axial_scalar.min()

    return pos3d, radius * ratio, height * ratio


def area_tracking(gaussians, pos3d, radius, up_vector, height=None):
    xyz = gaussians.get_xyz
    device, dtype = xyz.device, xyz.dtype
    up_vector = torch.as_tensor(up_vector, device=device, dtype=dtype)

    center = pos3d.to(device=device, dtype=dtype)
    up = up_vector

    up_norm = torch.norm(up)
    if up_norm < 1e-8:
        raise ValueError("up_vector norm is too small.")

    up = up / up_norm
    rel = xyz - center.unsqueeze(0)

    # Cylinder axis projection
    axial_scalar = (rel * up.unsqueeze(0)).sum(dim=1)
    axial = axial_scalar.unsqueeze(1) * up.unsqueeze(0)
    radial = rel - axial

    radius_sq = torch.as_tensor(radius, device=device, dtype=dtype) ** 2
    radial_mask = (radial * radial).sum(dim=1) <= radius_sq

    # If height is provided, use a finite cylinder centered at pos3d.
    if height is not None and height > 0.0:
        h = torch.as_tensor(height, device=device, dtype=dtype)
        half_h = h * 0.5
        height_mask = torch.abs(axial_scalar) <= half_h
        mask_3d = radial_mask & height_mask
    else:
        mask_3d = radial_mask

    return mask_3d


def launch_bidirectional_drones(tracker, drone_0, drone_1, mask_filter,
                                yaw, pitch, fg_set, mask_3d,
                                memory_pkgs=None,
                                step_num=0, background_aug=True,
                                save_2d_mask=False,
                                lookat_point=None):
    _mask_3d = mask_3d.clone() if mask_3d is not None else None

    cam_pose_0 = drone_0.return_camera_pose(yaw=yaw, pitch=pitch, center=lookat_point)
    cam_pose_1 = drone_1.return_camera_pose(yaw=-yaw, pitch=pitch, center=lookat_point)

    mask_0 = drone_0.tracking_in_new_view(fg_set, cam_pose_0, step_num=step_num, prompt_aug=background_aug,
                                          tracker=tracker, memory_pkgs=memory_pkgs)
    mask_1 = drone_1.tracking_in_new_view(fg_set, cam_pose_1, step_num=step_num, prompt_aug=background_aug,
                                          tracker=tracker, memory_pkgs=memory_pkgs)

    if save_2d_mask:
        image_save("mask_drone_0.png", mask_0.cpu().numpy())
        image_save("mask_drone_1.png", mask_1.cpu().numpy())

    _mask_3d = mask_filter.filtering(mask_0, _mask_3d, cam_pose_0)
    _mask_3d = mask_filter.filtering(mask_1, _mask_3d, cam_pose_1)

    return _mask_3d


class MaskShapedFrustumFilter:
    def __init__(self, gaussians, padding=3, device="cuda:0",
                 self_val=False, initial_cam=None, initial_mask=None):
        self.padding = padding
        self.gaussians = gaussians
        self.device = device

        self.self_val = self_val
        # self.self_val = False
        if self.self_val:
            assert initial_cam is not None
            assert initial_mask is not None
            self.initial_cam = initial_cam
            self.initial_mask = initial_mask

    def filtering(self, mask_2d, mask_3d, cam, padding=None, self_val=True, first_view_prompt=False):
        padding = self.padding if padding is None else padding
        internal_mask_3d = mask_3d.clone() if mask_3d is not None else None

        if mask_2d is None:
            print("lost!")
            return mask_3d

        mask_2d = mask_2d > 0.0
        mask_2d = swell_mask(mask_2d, padding)
        cam = load_3dgs_camera(cam, device=self.device)

        keep_out_screen = False
        if torch.cat((mask_2d[0], mask_2d[-1], mask_2d[:, 0], mask_2d[:, -1])).any() or \
                not mask_2d.any():
            keep_out_screen = True
        if first_view_prompt: keep_out_screen = False
        _3d_mask = cam.prune_gaussians_by_mask(self.gaussians, mask_2d, keep_out_screen)

        if internal_mask_3d is None:
            internal_mask_3d = _3d_mask
        else:
            internal_mask_3d &= _3d_mask
        if self.self_val and self_val:
            mask_torch = render_mask(self.initial_cam, self.gaussians, internal_mask_3d)
            miou = mIOU_torch(self.initial_mask, mask_torch[0], ioi=True)

            if miou < 0.8:
                # print("reject")
                return mask_3d

        return internal_mask_3d


def mask_shaped_frustum_filtering(mask_2d, mask_3d, gaussians,
                                  cam, device="cuda:0", padding=0,
                                  self_val=False, initial_cam=None, initial_mask=None):
    if mask_2d is None:
        print("lost!")
        return mask_3d
    try:
        mask_2d = mask_2d > 0.0
        mask_2d = swell_mask(mask_2d, padding)
        cam = load_3dgs_camera(cam, device=device)

        keep_out_screen = False
        if torch.cat((mask_2d[0], mask_2d[-1], mask_2d[:, 0], mask_2d[:, -1])).any() or \
                not mask_2d.any():
            keep_out_screen = True
        _3d_mask = cam.prune_gaussians_by_mask(gaussians, mask_2d, keep_out_screen)

        if mask_3d is None:
            mask_3d = _3d_mask
        else:
            mask_3d &= _3d_mask
        if self_val:
            assert initial_cam is not None, "Self-validation needs initial camera pose!"
            assert initial_mask is not None, "Self-validation needs initial 2D mask!"
            val_score, ioi = initial_miou_validation(mask_3d, gaussians,
                                                     initial_cam=initial_cam, initial_mask=initial_mask,
                                                     with_box=True)
    except TypeError:
        pass
    return mask_3d


def image_save(save_name, img, prefix=""):
    save_name = osp.join(prefix, save_name)
    cv2.imwrite(save_name, img)


def skew_symmetric(v):
    return torch.tensor([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])


def rotate_coordinates(p, up_vector):
    """
    旋转向量 p，使得默认的 [0,1,0] 对应到 up_vector
    """
    up_vector = up_vector / torch.norm(up_vector)
    default_up = torch.tensor([0., 1., 0.])

    # 计算旋转轴和角度
    axis = torch.cross(default_up, up_vector)
    norm_axis = torch.norm(axis)
    if norm_axis < 1e-6:
        # up_vector == default_up 或相反
        if torch.dot(default_up, up_vector) > 0:
            R = torch.eye(3)
        else:
            # 旋转 180 度
            R = torch.diag(torch.tensor([1., -1., 1.]))
    else:
        axis = axis / norm_axis
        angle = torch.acos(torch.clamp(torch.dot(default_up, up_vector), -1.0, 1.0))
        K = skew_symmetric(axis)
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K
    return R @ p


def inverse_rotate_coordinates(p, up_vector):
    """
    逆向旋转：R^T
    """
    up_vector = up_vector / torch.norm(up_vector)
    default_up = torch.tensor([0., 1., 0.])

    axis = torch.cross(default_up, up_vector)
    norm_axis = torch.norm(axis)
    if norm_axis < 1e-6:
        if torch.dot(default_up, up_vector) > 0:
            R = torch.eye(3)
        else:
            R = torch.diag(torch.tensor([1., -1., 1.]))
    else:
        axis = axis / norm_axis
        angle = torch.acos(torch.clamp(torch.dot(default_up, up_vector), -1.0, 1.0))
        K = skew_symmetric(axis)
        R = torch.eye(3) + torch.sin(angle) * K + (1 - torch.cos(angle)) * K @ K

    return R.T @ p


def vector_angle(point_a, point_b, vector_u):
    # 计算连线向量 V = B - A
    vector_v = point_b - point_a

    # 计算点积
    dot_product = torch.dot(vector_v, vector_u)

    # 计算模长乘积
    norms_product = torch.norm(vector_v) * torch.norm(vector_u)

    # 避免数值不稳定（确保余弦值在有效范围内）
    cos_theta = torch.clamp(dot_product / norms_product, -1.0, 1.0)

    # 计算夹角（弧度）
    theta_rad = torch.arccos(cos_theta)
    print(theta_rad * 180 / math.pi)

    return theta_rad.item()


def find_up_vector(points):
    if type(points) == str:
        pcd = o3d.io.read_point_cloud(points)
    else:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=0.01,  # 距离阈值，根据你的点云密度调整
        ransac_n=3,  # 最小样本数
        num_iterations=1000  # 迭代次数
    )
    return -1 * plane_model[0:3]


def mask2box_torch(mask, expand=0):
    if mask.dim() == 3:
        # For HWC masks, check if any channel has a non-zero value
        mask = torch.any(mask, dim=2)

    if mask.dim() == 2:
        h, w = mask.shape
    else:
        # This case should ideally not be hit if input is HW or HWC
        # but as a fallback
        h, w, *_ = mask.shape

    nonzero_indices = torch.nonzero(mask, as_tuple=False)
    if len(nonzero_indices) == 0:
        return None

    min_y, min_x = torch.min(nonzero_indices, dim=0).values
    max_y, max_x = torch.max(nonzero_indices, dim=0).values

    bbox = [(max(min_x.item() - expand, 0),
             max(min_y.item() - expand, 0)),
            (min(max_x.item() + expand, w),
             min(max_y.item() + expand, h))]

    return bbox


def cut_gaussian(gaussians, save_name=None):
    print("*" * 50)

    # --- 1. 获取核心数据 ---
    xyz = gaussians.get_xyz  # (N, 3)
    scaling = gaussians._scaling  # (N, 3)
    rotation = gaussians._rotation  # (N, 4) wxyz

    N = xyz.shape[0]
    print(f"  原始高斯数量 N: {N}")

    # --- 2. 核心数学运算 (Vectorized) ---

    # 确定主轴 (k) 和最大尺度 (s_max)
    # scaling shape: (N, 3) -> [w, h, l]
    _, k = torch.max(torch.abs(scaling), dim=1, keepdim=True)  # max_scale (N, 1), k (N, 1)
    max_scale = scaling.gather(1, k)

    # 计算新的尺度 (s_new): 主轴减半
    new_scaling_base = scaling.clone()

    # 使用 advanced indexing 设置主轴元素
    row_indices = torch.arange(N, device=scaling.device)
    k_flattened = k.squeeze(1)  # (N,)

    # 设置主轴上的新尺度: s'_k = 0.5 * s_k
    new_scaling_base[row_indices, k_flattened] *= 0.5  # (N, 3)

    print(f"  [Math] 计算完成新的 {N} 个高斯的尺度。")

    # 计算局部位移 d_local: 0.5 * s_max 沿着主轴 e_k
    d_local = torch.zeros_like(xyz)  # (N, 3)

    # d_local 只有在主轴 k 上的分量为 max_scale * 0.5
    d_local[row_indices, k_flattened] = torch.abs(max_scale.squeeze(1) * 0.5)

    # 计算世界位移 Δx: R @ d_local (使用 PyTorch 批量旋转)
    delta_x = batch_rotate_vector(rotation, d_local)  # (N, 3)

    print(f"  [Math] 计算完成 {N} 个高斯的世界位移 Δx。")

    # 计算新的中心点 x_1, x_2
    new_xyz_1 = xyz + 0.1 * delta_x
    new_xyz_2 = xyz - 0.1 * delta_x

    # --- 3. 构造新的高斯数组（2N 个元素） ---

    # 拼接 XYZ: [x_1, x_2, ..., x_N, x'_1, x'_2, ..., x'_N]
    final_xyz = torch.cat([new_xyz_1, new_xyz_2], dim=0)

    # 拼接 SCALING: [s'_1, s'_2, ..., s'_N, s'_1, s'_2, ..., s'_N]
    final_scaling = torch.cat([new_scaling_base, new_scaling_base], dim=0)

    # 拼接其他继承属性 (Clone: 复制两次，保持不变)
    final_features_dc = torch.cat([gaussians._features_dc, gaussians._features_dc], dim=0)
    final_features_rest = torch.cat([gaussians._features_rest, gaussians._features_rest], dim=0)
    final_rotation = torch.cat([gaussians._rotation, gaussians._rotation], dim=0)
    final_opacity = torch.cat([gaussians._opacity, gaussians._opacity], dim=0)

    # --- 4. 初始化新的 GaussianModel ---

    sub_gaussians = GaussianModel(gaussians.active_sh_degree, True)

    sub_gaussians.precise_init(
        active_sh_degree=gaussians.active_sh_degree,
        xyz=final_xyz,
        features_dc=final_features_dc,
        features_rest=final_features_rest,
        scaling=final_scaling,
        rotation=final_rotation,
        opacity=final_opacity,
        max_radii2D=gaussians.max_radii2D,
        xyz_gradient_accum=None,  # 重置或保持 None
        denom=gaussians.denom,
        spatial_lr_scale=gaussians.spatial_lr_scale
    )

    if save_name is not None:
        sub_gaussians.save_ply(save_name)

    print("*" * 50)
    return sub_gaussians


def quaternion_to_rotation_matrix(q_wxyz: torch.Tensor) -> torch.Tensor:
    """
    Converts a batch of quaternions (wxyz) to a batch of 3x3 rotation matrices.
    q_wxyz shape: (N, 4)
    Returns: (N, 3, 3)
    """
    # 确保张量是二维的 (N, 4)
    if q_wxyz.dim() == 1:
        q_wxyz = q_wxyz.unsqueeze(0)

    # w, x, y, z are now (N,)
    w, x, y, z = q_wxyz.unbind(-1)

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    # 构建旋转矩阵 R (N, 9)
    R_flat = torch.stack([
        1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
        2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
        2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=-1)

    # Reshape to (N, 3, 3)
    R = R_flat.view(-1, 3, 3)

    return R


def batch_rotate_vector(q_wxyz: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotates a batch of vectors v using a batch of quaternions q_wxyz.
    v' = R @ v
    q_wxyz shape: (N, 4)
    v shape: (N, 3)
    Returns: (N, 3)
    """
    R = quaternion_to_rotation_matrix(q_wxyz)  # (N, 3, 3)
    # v.unsqueeze(-1) changes (N, 3) to (N, 3, 1)
    # torch.bmm performs batch matrix multiplication: (N, 3, 3) @ (N, 3, 1) -> (N, 3, 1)
    rotated_v = torch.bmm(R, v.unsqueeze(-1)).squeeze(-1)
    return rotated_v


def swell_mask(mask, padding=10):
    mask_float = mask.float()
    kernel = torch.ones((1, 1, padding * 2 + 1, padding * 2 + 1)).to(mask_float)
    expanded_mask = F.conv2d(mask_float.unsqueeze(0).unsqueeze(0), kernel, padding=padding)
    expanded_mask = (expanded_mask > 0)
    expanded_mask = expanded_mask.squeeze(0).squeeze(0)
    return expanded_mask


def prune_ply(gaussians: GaussianModel, roi_index,
              save_name="widgets/sam_v2/blank/ply/object.ply"):
    small_gaussian = GaussianModel(3, True)
    small_gaussian.precise_init(gaussians.active_sh_degree,
                                xyz=gaussians._xyz[roi_index],
                                features_dc=gaussians._features_dc[roi_index],
                                features_rest=gaussians._features_rest[roi_index],
                                scaling=gaussians._scaling[roi_index],
                                rotation=gaussians._rotation[roi_index],
                                opacity=gaussians._opacity[roi_index],
                                max_radii2D=gaussians.max_radii2D,
                                xyz_gradient_accum=None,
                                denom=gaussians.denom,
                                spatial_lr_scale=gaussians.spatial_lr_scale)
    small_gaussian.pick = roi_index
    small_gaussian.save_ply(save_name)
    return small_gaussian


def back_projection(points2d, depth, cam: camlab.CameraObjTensor):
    """
    points2d: [N, 2]
    depth: [N]
    cam: camlab.CameraObjTensor
    return: [N, 3]
    """
    points2d = points2d.to(cam.device)
    depth = -depth.to(cam.device)
    return cam.cam2world_batch(cam.screen2cam_batch(points2d, depth))


def inverse_get_origin(camera_origins, lookat_position, up_vector):
    # 计算相对位置
    relative_pos = camera_origins - lookat_position

    # 计算 radius
    radius = torch.linalg.norm(relative_pos)

    # 处理 radius 接近零的情况
    if radius < 1e-5:
        radius = 1
        # raise ValueError("Camera position is too close to lookat_position, cannot compute spherical coordinates.")

    # 如果 up_vector 为零，直接计算球面坐标
    if torch.equal(up_vector, torch.zeros_like(up_vector)):
        x, y, z = relative_pos[0], relative_pos[1], relative_pos[2]
        vertical_mean = torch.arccos(y / radius)
        sin_v = torch.sin(vertical_mean)
        if sin_v < 1e-5:  # 极点附近，horizontal_mean 不稳定
            horizontal_mean = torch.tensor(0.0, device=relative_pos.device)
        else:
            horizontal_mean = math.pi - torch.atan2(z, x)
        return horizontal_mean, vertical_mean, radius

    # 归一化 up_vector
    unit_vector = normalize_vecs(up_vector)

    # 计算旋转矩阵的逆
    base_vector = torch.tensor([0.0, -1.0, 0.0], device=relative_pos.device)
    theta = torch.arccos(torch.dot(unit_vector, base_vector))

    if theta == 0:
        rotation_matrix = torch.eye(3, device=relative_pos.device)
    elif theta == torch.pi:
        rotation_matrix = -torch.eye(3, device=relative_pos.device)
        rotation_matrix[0, 0] = 1
    else:
        k = torch.cross(base_vector, unit_vector, dim=0)
        k /= torch.linalg.norm(k)
        K = torch.tensor([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], device=relative_pos.device)
        rotation_matrix = (
                torch.eye(3, device=relative_pos.device) + torch.sin(theta) * K + (1 - torch.cos(theta)) * torch.matmul(
            K, K)
        )

    # 计算逆旋转矩阵（旋转矩阵的逆为其转置）
    inv_rotation_matrix = rotation_matrix.T

    # 应用逆旋转到 relative_pos
    unrotated_pos = torch.matmul(inv_rotation_matrix, relative_pos[..., None])[..., 0]

    # 从未旋转的坐标计算球面坐标
    x, y, z = unrotated_pos[0], unrotated_pos[1], unrotated_pos[2]
    vertical_mean = torch.arccos(torch.clamp(y / radius, -1.0, 1.0))
    sin_v = torch.sin(vertical_mean)

    if sin_v < 1e-5:  # 极点附近，horizontal_mean 不稳定
        horizontal_mean = torch.tensor(0.0, device=relative_pos.device)
    else:
        horizontal_mean = math.pi - torch.atan2(z, x)

    # 确保 vertical_mean 在 [1e-5, π - 1e-5] 范围内以匹配原函数
    vertical_mean = torch.clamp(vertical_mean, 1e-5, math.pi - 1e-5)

    return horizontal_mean - math.pi / 2, vertical_mean - math.pi / 2, radius


def get_camera_pos_3d(cam):
    ext = cam.world_view_transform.T
    R = ext[:3, :3]
    T = ext[:3, 3]
    pos3d = (-R.T @ T).detach()
    return pos3d


def params2sphere(cam, up_vector, lookat_point):
    ext = cam.world_view_transform.T
    R = ext[:3, :3]
    T = ext[:3, 3]
    cam_center = (-R.T @ T).detach()

    lookat_point = torch.as_tensor(lookat_point, dtype=torch.float32, device=cam_center.device)

    up = torch.as_tensor(up_vector, dtype=cam_center.dtype, device=cam_center.device)

    yaw, pitch, radius = inverse_get_origin(cam_center, lookat_point, up)

    return radius, yaw, pitch


def sphere2params(radius, yaw, pitch, lookat_point, up_vector, fov_rad=np.pi / 3,
                  like_cam=None, offset=None,
                  width=1024, height=1024, device="cuda:0", dtype=torch.float32):
    lookat_point = torch.zeros(3, device=device, dtype=dtype) if lookat_point is None else lookat_point
    if offset is not None:
        lookat_point = lookat_point + torch.as_tensor(offset, dtype=dtype, device=lookat_point.device)
    cam_pos = get_origin(
        yaw + np.pi / 2,
        pitch + np.pi / 2,
        radius,
        lookat_point,
        up_vector=up_vector)

    forward = normalize_vecs(lookat_point - cam_pos)
    cam_params = create_cam2world_matrix(forward, cam_pos, up_vector)[0].to(device)
    if like_cam is not None:
        width = like_cam.image_width
        height = like_cam.image_height
        fovx = like_cam.FoVx
        fovy = like_cam.FoVy
    else:
        fovx = 2 * math.atan(math.tan(fov_rad / 2) * width / height)
        fovy = fov_rad

    render_cam = CustomCam(width, height, fovy=fovy, fovx=fovx, extr=cam_params.to(dtype))

    return render_cam


def draw_3d_box(p: torch.Tensor,
                points_2d: torch.Tensor,
                show_points_color: bool = False,
                edge_color: tuple = (255, 255, 255),
                edge_width: int = 5,
                point_radius: int = 8):
    """
    p:          [3, H, W] GPU 张量（float32 [0,1] 或 uint8 [0,255]）
    points_2d:  [8, 2] PyTorch Tensor，2D 像素坐标（CPU 或 GPU 均可）
    返回:       新的 GPU 张量，已画好 3D 线框 + 8 个不同颜色的顶点圆点
    """
    # 1. 转成 PIL 图像
    if p.dtype == torch.float32:
        img = (p * 255).clamp(0, 255).to(torch.uint8).cpu()
    else:
        img = p.cpu()

    pil_img = T.ToPILImage()(img)
    draw = ImageDraw.Draw(pil_img, "RGBA")

    # 2. 8 个顶点对应的 8 种不同颜色
    colors = [
        (255, 0, 0),  # Red
        (255, 165, 0),  # Orange
        (255, 255, 0),  # Yellow
        (0, 255, 0),  # Green
        (0, 255, 255),  # Cyan
        (0, 0, 255),  # Blue
        (128, 0, 128),  # Purple
        (255, 0, 255)  # Magenta
    ]

    # 3. 画 12 条 3D 线框
    edges = [
        (0, 1), (0, 3), (2, 3), (2, 1),  # 底面
        (4, 5), (4, 7), (6, 7), (6, 5),  # 顶面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 垂直边
    ]

    # 直接处理 Tensor（支持 GPU/CPU）
    pts = points_2d.cpu().numpy().astype(int)

    for i, j in edges:
        draw.line([tuple(pts[i]), tuple(pts[j])],
                  fill=edge_color + (255,),
                  width=edge_width)

    if show_points_color:
        # 4. 在 8 个顶点画不同颜色的圆点
        for i in range(8):
            x, y = pts[i]
            r = point_radius
            # 黑色描边
            draw.ellipse([x - r - 2, y - r - 2, x + r + 2, y + r + 2], fill=(0, 0, 0, 180))
            # 彩色圆点
            draw.ellipse([x - r, y - r, x + r, y + r], fill=colors[i] + (255,))

    # 5. 转回 GPU 张量
    drawn = T.ToTensor()(pil_img).to(p.device)
    if p.dtype == torch.float32:
        return drawn
    else:
        return (drawn * 255).to(torch.uint8)


def slerp(cam0, cam1, num_frames=50):
    def _unwrap_cam(cam):
        if isinstance(cam, dict):
            if "3dgs" not in cam:
                raise KeyError("camera dict must contain key '3dgs'")
            return cam["3dgs"], True
        return cam, False

    def _mat_to_quat_wxyz(rot):
        trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
        if trace > 0:
            s = torch.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (rot[2, 1] - rot[1, 2]) / s
            qy = (rot[0, 2] - rot[2, 0]) / s
            qz = (rot[1, 0] - rot[0, 1]) / s
        elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
            s = torch.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            qw = (rot[2, 1] - rot[1, 2]) / s
            qx = 0.25 * s
            qy = (rot[0, 1] + rot[1, 0]) / s
            qz = (rot[0, 2] + rot[2, 0]) / s
        elif rot[1, 1] > rot[2, 2]:
            s = torch.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            qw = (rot[0, 2] - rot[2, 0]) / s
            qx = (rot[0, 1] + rot[1, 0]) / s
            qy = 0.25 * s
            qz = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = torch.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            qw = (rot[1, 0] - rot[0, 1]) / s
            qx = (rot[0, 2] + rot[2, 0]) / s
            qy = (rot[1, 2] + rot[2, 1]) / s
            qz = 0.25 * s
        q = torch.tensor([qw, qx, qy, qz], device=rot.device, dtype=rot.dtype)
        return F.normalize(q.unsqueeze(0), dim=-1).squeeze(0)

    def _slerp_quat(q0, q1, t):
        q0 = F.normalize(q0.unsqueeze(0), dim=-1).squeeze(0)
        q1 = F.normalize(q1.unsqueeze(0), dim=-1).squeeze(0)

        dot = torch.dot(q0, q1)
        if dot < 0.0:
            q1 = -q1
            dot = -dot

        dot = torch.clamp(dot, -1.0, 1.0)
        if dot > 0.9995:
            q = q0 + t * (q1 - q0)
            return F.normalize(q.unsqueeze(0), dim=-1).squeeze(0)

        theta_0 = torch.acos(dot)
        sin_theta_0 = torch.sin(theta_0)
        theta = theta_0 * t
        sin_theta = torch.sin(theta)

        s0 = torch.sin(theta_0 - theta) / sin_theta_0
        s1 = sin_theta / sin_theta_0
        q = s0 * q0 + s1 * q1
        return F.normalize(q.unsqueeze(0), dim=-1).squeeze(0)

    cam0_3dgs, cam0_is_dict = _unwrap_cam(cam0)
    cam1_3dgs, cam1_is_dict = _unwrap_cam(cam1)
    if cam0_is_dict != cam1_is_dict:
        raise ValueError("cam0 and cam1 should have the same type (both dict or both camera object)")

    if num_frames <= 1:
        return [cam0]

    ext0 = cam0_3dgs.world_view_transform.T
    ext1 = cam1_3dgs.world_view_transform.T

    r0_w2c, t0 = ext0[:3, :3], ext0[:3, 3]
    r1_w2c, t1 = ext1[:3, :3], ext1[:3, 3]

    r0_c2w = r0_w2c.T
    r1_c2w = r1_w2c.T
    c0 = -r0_w2c.T @ t0
    c1 = -r1_w2c.T @ t1

    q0 = _mat_to_quat_wxyz(r0_c2w)
    q1 = _mat_to_quat_wxyz(r1_c2w)

    device = ext0.device
    dtype = ext0.dtype

    width0, height0 = cam0_3dgs.image_width, cam0_3dgs.image_height
    width1, height1 = cam1_3dgs.image_width, cam1_3dgs.image_height
    fovx0, fovy0 = cam0_3dgs.FoVx, cam0_3dgs.FoVy
    fovx1, fovy1 = cam1_3dgs.FoVx, cam1_3dgs.FoVy

    cams = []
    ts = torch.linspace(0.0, 1.0, steps=num_frames, device=device, dtype=dtype)
    for t in ts:
        qi = _slerp_quat(q0, q1, t)
        ri = quaternion_to_rotation_matrix(qi.unsqueeze(0)).squeeze(0).to(device=device, dtype=dtype)
        ci = (1.0 - t) * c0 + t * c1

        cam2world = torch.eye(4, device=device, dtype=dtype)
        cam2world[:3, :3] = ri
        cam2world[:3, 3] = ci

        width_i = int(round((1.0 - float(t)) * width0 + float(t) * width1))
        height_i = int(round((1.0 - float(t)) * height0 + float(t) * height1))
        fovx_i = (1.0 - float(t)) * fovx0 + float(t) * fovx1
        fovy_i = (1.0 - float(t)) * fovy0 + float(t) * fovy1

        cam_i = CustomCam(width_i, height_i, fovy=fovy_i, fovx=fovx_i, extr=cam2world)

        if cam0_is_dict:
            cams.append({"3dgs": cam_i})
        else:
            cams.append(cam_i)

    return cams
