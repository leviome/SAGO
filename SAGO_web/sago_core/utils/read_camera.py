import os

import torch
from PIL import Image
import json
import numpy as np
import cv2
import struct
import os.path as osp
import math
from gs.scene.cameras import SimpleCam, CameraOri


def look_at_p(cam, P):
    R = cam.R
    T = cam.T
    # P_camera = R @ (P - T)
    P_camera = R @ P + T
    # P_camera = - R.T @ T
    z_new = P_camera / np.linalg.norm(P_camera)
    ref = np.array([1, 0, 0]) if abs(z_new[0]) < 0.9 else np.array([0, 1, 0])
    # ref = np.array([0, 1, 0])

    # 计算y轴
    y_new = np.cross(z_new, ref)
    y_new = y_new / np.linalg.norm(y_new)

    # 计算x轴
    x_new = np.cross(y_new, z_new)

    # 构建新的旋转矩阵
    R_new = np.vstack([x_new, y_new, z_new]).T

    return CameraOri(colmap_id=cam.colmap_id, R=R_new, T=cam.T,
                     FoVx=cam.FoVx, FoVy=cam.FoVy,
                     image=None, gt_alpha_mask=None,
                     image_width=cam.width, image_height=cam.height,
                     image_name=cam.image_name, uid=1, data_device="cuda:0")


def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2]])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def read_intrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::WriteCamerasBinary(const std::string& path)
        void Reconstruction::ReadCamerasBinary(const std::string& path)
    """
    cameras = {}
    with open(path_to_model_file, "rb") as fid:
        num_cameras = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_properties = read_next_bytes(
                fid, num_bytes=24, format_char_sequence="iiQQ")
            camera_id = camera_properties[0]
            model_id = camera_properties[1]
            width = camera_properties[2]
            height = camera_properties[3]
            num_params = 3
            params = read_next_bytes(fid, num_bytes=8 * num_params,
                                     format_char_sequence="d" * num_params)

            # SIMPLE PINHOLE
            focal_length_x = params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
    return [width, height, FovX, FovY]


def read_extrinsics_binary(path_to_model_file):
    """
    see: src/base/reconstruction.cc
        void Reconstruction::ReadImagesBinary(const std::string& path)
        void Reconstruction::WriteImagesBinary(const std::string& path)
    """
    exts = []
    with open(path_to_model_file, "rb") as fid:
        num_reg_images = read_next_bytes(fid, 8, "Q")[0]
        for _ in range(num_reg_images):
            ext_info = dict()
            binary_image_properties = read_next_bytes(
                fid, num_bytes=64, format_char_sequence="idddddddi")
            image_id = binary_image_properties[0]
            qvec = np.array(binary_image_properties[1:5])
            tvec = np.array(binary_image_properties[5:8])
            camera_id = binary_image_properties[8]
            image_name = ""
            current_char = read_next_bytes(fid, 1, "c")[0]
            while current_char != b"\x00":  # look for the ASCII 0 entry
                image_name += current_char.decode("utf-8")
                current_char = read_next_bytes(fid, 1, "c")[0]
            num_points2D = read_next_bytes(fid, num_bytes=8,
                                           format_char_sequence="Q")[0]
            x_y_id_s = read_next_bytes(fid, num_bytes=24 * num_points2D,
                                       format_char_sequence="ddq" * num_points2D)
            xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                   tuple(map(float, x_y_id_s[1::3]))])
            point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
            # print(image_name)
            R = np.transpose(qvec2rotmat(qvec))
            T = np.array(tvec)
            ext_info["image_name"] = image_name
            ext_info["R"] = R
            ext_info["T"] = T
            exts.append(ext_info)
    return exts


def read_cam_gs(folder="/data/OpenSource/LERF_mask/figurines/sparse/0/"):
    exts = read_extrinsics_binary(osp.join(folder, "images.bin"))
    intrs = read_intrinsics_binary(osp.join(folder, "cameras.bin"))
    # print(exts)
    # print(intrs)
    exts = sorted(exts.copy(), key=lambda x: x["image_name"])
    cams = []
    width, height, FovX, FovY = intrs
    for ext in exts:
        R = ext["R"]
        T = ext["T"]
        cam = CameraOri(colmap_id=0, R=R, T=T,
                        FoVx=FovX, FoVy=FovY,
                        image=None, gt_alpha_mask=None,
                        image_width=width, image_height=height,
                        image_name=ext["image_name"], uid=1, data_device="cuda:0")

        cams.append(cam)
    # print(cams[0].image_name)
    # print(cams[0].R)
    # print(cams[0].T)
    # print(cams[0].FoVx)
    # print(cams[0].FoVy)
    # print(cams[0].image_width)
    # print(cams[0].image_height)

    return cams


def JSON_to_camera(camera_entry, resolution=None,
                   distance_coef=1.0, resolution_coef=1.0):
    # id = camera_entry['id']
    try:
        rotation = np.array(camera_entry["rotation"])
        pos = np.array(camera_entry["position"])
        w2c = np.eye(4)
        w2c[:3, 3] = pos
        w2c[:3, :3] = rotation
    except KeyError:
        w2c = np.array(camera_entry["w2c"])
    Rt = np.linalg.inv(w2c)
    R = Rt[:3, :3].transpose()
    T = Rt[:3, 3]

    width = int(camera_entry['width'] * resolution_coef)
    height = int(camera_entry['height'] * resolution_coef)

    if resolution is not None:
        width, height = resolution
    try:
        fovx = focal2fov(camera_entry['fx'], width)
        fovy = focal2fov(camera_entry['fx'], height)  # SIMPLE HOLE
    except KeyError:
        fovx = focal2fov(camera_entry['focal'][0], width)
        fovy = focal2fov(camera_entry['focal'][1], height)

    return CameraOri(colmap_id=0, R=R, T=T * distance_coef,
                     FoVx=fovx, FoVy=fovy,
                     image=None, gt_alpha_mask=None,
                     image_width=width, image_height=height,
                     # image_name="",
                     image_name=camera_entry["img_name"],
                     uid=1, data_device="cuda:0")


def read_json_cam(json_file,
                  resolution=None, distance_coef=1.0, resolution_coef=1.0, for_dict=False):
    with open(json_file, "r") as f:
        cam_dict = json.loads(f.read())
    # if 'train' in cam_dict.keys():
    #     cam_dict = cam_dict['train']
    if not for_dict:
        cams = [JSON_to_camera(cam, resolution, distance_coef, resolution_coef)
                for cam in cam_dict]
        cams = sorted(cams.copy(), key=lambda x: x.image_name)
    else:
        cams = dict()
        for cam in cam_dict:
            cam = JSON_to_camera(cam, resolution, distance_coef, resolution_coef)
            k = cam.image_name.split('.')[0].split('cam')[-1]
            cams[f"cam_{k}"] = cam

    return cams


def get_upright_3d_bbox_corners(
        points: torch.Tensor,  # [N, 3] 你的点云（GPU 或 CPU）
        up_vector: torch.Tensor = None  # 可不传，默认使用 [0, -1, 0]
) -> torch.Tensor:
    """
    返回在**原始坐标系**下的 8 个顶点坐标
    - Box 高度方向严格与真实向上对齐
    - Box 中心 与 点云质心 完全重合
    """
    device = points.device
    dtype = points.dtype

    # 默认使用你场景的向上向量 [0, -1, 0]
    if up_vector is None:
        up_vector = torch.tensor([0.0, -1.0, 0.0], device=device, dtype=torch.float)

    # 1. 计算点云质心
    centroid = points.mean(dim=0)

    # 2. 归一化向上向量
    up = up_vector.float().to(device).flatten()
    up = up / torch.norm(up)

    # 3. 构造正交坐标系（新 Z 轴 = up）
    # 为了数值更稳定，针对 [0,-1,0] 选择合适的临时向量
    temp = torch.tensor([1.0, 0.0, 0.0], device=device, dtype=torch.float)
    x_axis = torch.cross(temp, up, dim=0)
    x_axis = x_axis / torch.norm(x_axis)
    y_axis = torch.cross(up, x_axis, dim=0)

    R = torch.stack([x_axis, y_axis, up], dim=1)  # [3, 3]

    # 4. 平移到质心 + 旋转到 aligned 坐标系
    points_aligned = (points - centroid) @ R

    # 5. 计算 AABB
    min_coords = points_aligned.min(dim=0)[0]
    max_coords = points_aligned.max(dim=0)[0]

    # 6. 生成 8 个角点（aligned 坐标系下，中心在原点）
    corners_aligned = torch.tensor([
        [min_coords[0], min_coords[1], min_coords[2]],  # 底面
        [max_coords[0], min_coords[1], min_coords[2]],
        [max_coords[0], max_coords[1], min_coords[2]],
        [min_coords[0], max_coords[1], min_coords[2]],

        [min_coords[0], min_coords[1], max_coords[2]],  # 顶面
        [max_coords[0], min_coords[1], max_coords[2]],
        [max_coords[0], max_coords[1], max_coords[2]],
        [min_coords[0], max_coords[1], max_coords[2]],
    ], device=device, dtype=dtype)

    # 7. 旋转回原始坐标系 + 平移回质心
    corners_original = corners_aligned @ R.t() + centroid

    return corners_original


def viz_depth(depth, save_name="depth_color.png", colormap=cv2.COLORMAP_JET):
    # visualize depth with a colormap and save overlays
    depth_np = depth.cpu().numpy()[0]
    depth_np = np.nan_to_num(depth_np, nan=0.0, posinf=0.0, neginf=0.0)

    # normalize and invert for nicer near-bright visualization
    dmin, dmax = float(np.nanmin(depth_np)), float(np.nanmax(depth_np))
    if dmax - dmin > 1e-6:
        depth_norm = (depth_np - dmin) / (dmax - dmin)
    else:
        depth_norm = np.zeros_like(depth_np, dtype=np.float32)
    depth_norm = 1.0 - depth_norm
    depth_uint8 = (np.clip(depth_norm, 0.0, 1.0) * 255).astype(np.uint8)

    # colorize using chosen colormap, convert to RGB to match view_0
    depth_color = cv2.applyColorMap(depth_uint8, colormap)
    depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

    # save visualizations
    cv2.imwrite(save_name, cv2.cvtColor(depth_color, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    read_cam_gs()
    read_json_cam()
