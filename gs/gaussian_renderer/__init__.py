#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math

# from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
from diff_gaussian_rasterization_mask import GaussianRasterizer as GaussianRasterizerMask
from diff_gaussian_rasterization import GaussianRasterizer as GaussianRasterizerTopK
from diff_gaussian_rasterization_depth import GaussianRasterizationSettings, GaussianRasterizer

from gs.scene.gaussian_model import GaussianModel
from gs.utils.sh_utils import eval_sh
from sago_core.utils.read_camera import get_upright_3d_bbox_corners


def render(viewpoint_camera, pc: GaussianModel, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
           override_color=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype,
                                          requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D=means3D,
        means2D=means2D,
        shs=shs,
        colors_precomp=colors_precomp,
        opacities=opacity,
        scales=scales,
        rotations=rotations,
        cov3D_precomp=cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii}


zero = torch.zeros(16, 3, device="cuda")

_color_rgbs = [
    (3.544, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 3.544), (3.544, 0.5, 0.772),
    (3.5, 0.6, 1.544), (0.0, 0.5, 3.544), (2, 0.8, 3), (0.0, 2, 3.544),
    (3.544, 1.772, 0.0), (1.772, 0.0, 2.0), (1.772, 1.0, 0.0), (1.772, 3.544, 1.772),
    (3.544, 1, 1), (0.5, 1.8, 3.544), (1, 1, 0.0), (0.0, 1.0, 0.544),
    (0.0, 1.5, 1.772), (2, 0.5, 0.0), (1, 0.5, 3.544), (1, 0.544, 0.544),
    (0.5, 1.5, 0.3), (3, 1.5, 1.5), (0.5, 1, 0.2), (3.0, 1, 1.0),
    (1, 2, 3), (1.2, 0.5, 3.3), (1.2, 2.5, 3.3), (3.0, 3.0, 0.0),
    (0.0, 3.0, 3.0), (3.0, 0.0, 3.0), (1.5, 0.0, 0.0), (0.0, 1.5, 0.0),
    (0.0, 0.0, 1.5), (1.5, 1.5, 0.0), (0.0, 1.5, 1.5), (1.5, 0.0, 1.5),
    (1.5, 1.5, 1.5), (3.5, 0.0, 1.75), (0.0, 3.5, 1.75), (1.75, 3.5, 0.0),
    (3.5, 1.75, 0.0), (1.75, 0.0, 3.5), (0.0, 1.75, 3.5), (2.5, 1.0, 0.5),
    (0.5, 2.5, 1.0), (1.0, 0.5, 2.5), (2.5, 0.5, 1.0), (1.0, 2.5, 0.5),
    (0.5, 1.0, 2.5), (3.2, 1.2, 0.8), (0.8, 3.2, 1.2), (1.2, 0.8, 3.2),
    (3.2, 0.8, 1.2), (1.2, 3.2, 0.8), (0.8, 1.2, 3.2), (0.2, 0.8, 1.5),
    (1.5, 0.2, 0.8), (0.8, 1.5, 0.2), (0.2, 1.5, 0.8), (0.8, 0.2, 1.5),
    (1.5, 0.8, 0.2), (2.8, 2.0, 1.0), (1.0, 2.8, 2.0), (2.0, 1.0, 2.8),
    (2.8, 1.0, 2.0), (2.0, 2.8, 1.0), (1.0, 2.0, 2.8), (0.7, 1.4, 2.1),
    (2.1, 0.7, 1.4), (1.4, 2.1, 0.7), (0.7, 2.1, 1.4), (1.4, 0.7, 2.1),
    (2.1, 1.4, 0.7), (3.4, 0.2, 1.8), (1.8, 3.4, 0.2), (0.2, 1.8, 3.4),
    (3.4, 1.8, 0.2), (0.2, 3.4, 1.8), (1.8, 0.2, 3.4), (1.0, 1.0, 1.0),
    (2.0, 2.0, 2.0), (3.0, 3.0, 3.0), (0.5, 0.5, 0.5), (1.3, 2.6, 0.4),
    (0.4, 1.3, 2.6), (2.6, 0.4, 1.3), (1.3, 0.4, 2.6), (2.6, 1.3, 0.4),
    (0.4, 2.6, 1.3), (2.2, 0.9, 3.1), (3.1, 2.2, 0.9), (0.9, 3.1, 2.2),
    (2.2, 3.1, 0.9), (0.9, 2.2, 3.1), (3.1, 0.9, 2.2), (0.1, 0.1, 0.1),
]

color_library = []
for rgb in _color_rgbs:
    t = zero.clone()
    t[0] = torch.tensor(rgb, dtype=zero.dtype, device=zero.device)
    color_library.append(t)


def render_simple(viewpoint_camera, pc: GaussianModel,
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
                  rag_mask=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    up = torch.FloatTensor(up).to(device)
    x = y = -1
    if look_at is not None:
        # [0.38, 1.88, -0.55]
        from camlab import load_3dgs_camera
        p_tensor = torch.tensor([look_at]).to(device)
        cam = load_3dgs_camera(viewpoint_camera, device=device)
        p0, _ = cam.world2screen(p_tensor, to_int=True)
        x, y = p0[0].cpu().numpy()
    if bg == "white":
        bg_color = torch.Tensor([1., 1., 1.])
    bg_color = bg_color.to(device)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform.float(),
        projmatrix=viewpoint_camera.full_proj_transform.float(),
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center.float(),
        prefiltered=False,
        debug=debug
    )
    if make_topk:
        rasterizer = GaussianRasterizerTopK(raster_settings=raster_settings)
    else:
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    if valid is None:
        opacity = pc.get_opacity
    else:
        opacity = torch.clamp(pc.get_opacity - valid.unsqueeze(1),
                              0.0, 1.0)

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    scales = pc.get_scaling
    rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    convert_shs_python = False
    colors_precomp = None
    if override_color is None:
        if convert_shs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree + 1) ** 2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    if extra_gau_mask is not None:
        opacity = torch.clamp(opacity - ~extra_gau_mask.unsqueeze(1) * 1.0,
                              0.0, 1.0)

    bbox_3d_proj_list = []
    if gaussian_mask is not None and len(gaussian_mask) > 0 and gaussian_mask[0] is not None:
        if show_box:
            from camlab import load_3dgs_camera
            if len(gaussian_mask[0]) == len(means3D):
                for gm in gaussian_mask:
                    fg_mean = means3D[gm]
                    bbox_3d = get_upright_3d_bbox_corners(fg_mean, up)
                    cam_cl = load_3dgs_camera(viewpoint_camera, device=str(device))
                    pp, _ = cam_cl.world2screen(bbox_3d)
                    bbox_3d_proj = pp.float()  # .cpu().numpy()
                    bbox_3d_proj_list.append(bbox_3d_proj)

        if render_fg_only:
            _gaussian_mask = gaussian_mask[0]
            _bg_gau = ~_gaussian_mask
            opacity = torch.clamp(opacity - _bg_gau.unsqueeze(1) * 1.0,
                                  0.0, 1.0)
        else:
            if mutlicolor:
                # shs_ = torch.zeros_like(shs)
                shs_ = shs.clone()
                for cid, g_mask in enumerate(gaussian_mask):
                    mask_render = color_library[cid]
                    try:
                        # shs_[g_mask] += mask_render
                        shs_[g_mask] = mask_render
                    except IndexError:
                        pass
            else:
                shs_ = shs

            # remove fg
            if obj_rm:
                try:
                    for cid, g_mask in enumerate(gaussian_mask):
                        opacity = torch.clamp(opacity - g_mask.unsqueeze(1) * 1.0,
                                              0.0, 1.0)
                except RuntimeError:
                    pass


            shs = shs_

    if rag_mask is not None:
        bg_part = (rag_mask > 0).to(rag_mask.dtype)
        opacity = torch.clamp(opacity - bg_part.unsqueeze(1) * 1.0,
                              0.0, 1.0)

    if show_gau:
        # Keep the visibility mask: opaque ellipses must not reveal hidden IDs.
        opacity = torch.where(opacity > 0, torch.full_like(opacity, 10), torch.zeros_like(opacity))
        # shs = torch.ones_like(shs) * 0.3
        # print(opacity.shape)

    # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
    # rendered_image, radii = rasterizer(
    if render_mask:
        mask_shs = torch.zeros_like(shs)
        # opacity = torch.ones_like(opacity)
        if gaussian_mask is not None:
            mask_shs = torch.zeros_like(shs)
            # opacity[~gaussian_mask[0]] *= 0.1
            mask_shs[gaussian_mask[0]] = color_library[0]

        rendered_image, radii, rendered_depth, weight_map = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=mask_shs,
            colors_precomp=None,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        return {"mask": rendered_image}

    if render_particles:
        particle_scales = torch.ones_like(scales) * 0.01
        # particle_scales = scales * 1e-2
        scales = particle_scales

    if not make_topk:
        rendered_image, radii, rendered_depth, weight_map = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        rendered_alpha = None
        rendered_depth = rendered_depth[None]
        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "alpha": rendered_alpha,
            "depth": rendered_depth,
            "look_at": [x, y],
            # "3d_box": bbox_3d,
            "3d_box_proj_list": bbox_3d_proj_list,
        }
    else:
        (rendered_image, radii, rendered_depth, rendered_alpha, proj_means_2D,
         conic_2D, conic_2D_inv, gs_per_pixel, weight_per_gs_pixel, x_mu) = rasterizer(
            means3D=means3D,
            means2D=means2D,
            shs=shs,
            colors_precomp=colors_precomp,
            opacities=opacity,
            scales=scales,
            rotations=rotations,
            cov3D_precomp=cov3D_precomp)
        return {
            "render": rendered_image,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
            "alpha": rendered_alpha,
            "depth": rendered_depth,
            "look_at": [x, y],
            "gs_per_pixel": gs_per_pixel,
            # "3d_box": bbox_3d,
            "3d_box_proj_list": bbox_3d_proj_list,
        }
