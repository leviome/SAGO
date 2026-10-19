"""Export Gaussian object geometry as UTF-8 JSON-formatted TXT for RAG.

Standalone: python -m sago_core.utils.rag_geometry --up -0.06 -0.96 -0.26
Only NumPy, Pillow and (for the CLI) plyfile are needed; no GPU or CLIP required.
"""
import argparse
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def coordinate_basis(up):
    up = np.asarray(up, dtype=np.float64)
    if up.shape != (3,) or not np.isfinite(up).all() or np.linalg.norm(up) < 1e-8:
        raise ValueError("up must be a finite, nonzero 3-vector")
    up = up / np.linalg.norm(up)
    reference = np.eye(3)[np.argmin(np.abs(up))]
    x = np.cross(up, reference)
    x /= np.linalg.norm(x)
    y = np.cross(up, x)
    return np.column_stack((x, y, up))


def box_info(points):
    low, high = points.min(axis=0), points.max(axis=0)
    extent = high - low
    return {"min": low.tolist(), "max": high.tolist(),
            "center": ((low + high) / 2).tolist(), "size": extent.tolist(),
            "volume": float(np.prod(extent)),
            "corners": [list(p) for p in itertools.product(*zip(low, high))]}


def object_geometry(obj_id, points, basis, opacity=None, rgb=None, scales=None):
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Object {obj_id}: expected nonempty Nx3 points")
    if not np.isfinite(points).all():
        raise ValueError(f"Object {obj_id}: nonfinite coordinates")
    aligned = points @ basis
    world_box, aligned_box = box_info(points), box_info(aligned)
    centroid = points.mean(axis=0)
    robust_limits = np.percentile(aligned, [1, 99], axis=0)
    obj = {
        "obj_id": int(obj_id), "semantic_label": None,
        "gaussian_count": len(points),
        "centroid_world": centroid.tolist(),
        "centroid_aligned": (centroid @ basis).tolist(),
        "aabb_world": world_box,
        "bbox_up_aligned": aligned_box,
        "bbox_up_aligned_corners_world": (np.array(aligned_box["corners"]) @ basis.T).tolist(),
        "robust_bbox_aligned_percentile_1_99": box_info(robust_limits),
        "position_std_aligned": aligned.std(axis=0).tolist(),
        "radius_about_centroid": float(np.linalg.norm(points - centroid, axis=1).max()),
        "bev_bbox_aligned_xyxy": [*aligned_box["min"][:2], *aligned_box["max"][:2]],
        "height_range_aligned": [aligned_box["min"][2], aligned_box["max"][2]],
        "height": aligned_box["size"][2],
        "bev_bbox_area": float(np.prod(aligned_box["size"][:2])),
    }
    if opacity is not None:
        opacity = np.asarray(opacity, dtype=float).reshape(-1)
        if len(opacity) != len(points) or not np.isfinite(opacity).all() or np.any((opacity < 0) | (opacity > 1)):
            raise ValueError("opacity must contain one finite probability per Gaussian")
        obj["opacity_statistics"] = {"mean": float(opacity.mean()),
                                     "percentiles_5_50_95": np.percentile(opacity, [5, 50, 95]).tolist()}
        if opacity.sum() > 0:
            weighted = np.average(points, axis=0, weights=opacity)
            obj["opacity_weighted_centroid_world"] = weighted.tolist()
            obj["opacity_weighted_centroid_aligned"] = (weighted @ basis).tolist()
    if rgb is not None:
        rgb = np.asarray(rgb, dtype=float)
        if rgb.shape != points.shape or not np.isfinite(rgb).all():
            raise ValueError("rgb must be finite Nx3")
        obj["appearance_dc_rgb_0_1"] = {
            "mean": rgb.mean(axis=0).tolist(),
            "median": np.median(rgb, axis=0).tolist(),
            "note": "SH DC 基色统计，不包含视角相关颜色，不是语义颜色标签。"}
    if scales is not None:
        scales = np.asarray(scales, dtype=float)
        if scales.shape != points.shape or not np.isfinite(scales).all() or (scales < 0).any():
            raise ValueError("scales must be finite, nonnegative Nx3")
        obj["gaussian_axis_scale_median"] = np.median(scales, axis=0).tolist()
    return obj


def pair_relation(a, b, tolerance):
    low_a, high_a = np.array(a["bbox_up_aligned"]["min"]), np.array(a["bbox_up_aligned"]["max"])
    low_b, high_b = np.array(b["bbox_up_aligned"]["min"]), np.array(b["bbox_up_aligned"]["max"])
    delta = np.array(b["centroid_aligned"]) - a["centroid_aligned"]
    overlap = np.maximum(0, np.minimum(high_a, high_b) - np.maximum(low_a, low_b))
    gaps = np.maximum(0, np.maximum(low_a - high_b, low_b - high_a))
    area_intersection = float(np.prod(overlap[:2]))
    area_a, area_b = a["bev_bbox_area"], b["bev_bbox_area"]
    union = area_a + area_b - area_intersection
    volume_intersection = float(np.prod(overlap))
    volume_union = a["bbox_up_aligned"]["volume"] + b["bbox_up_aligned"]["volume"] - volume_intersection
    facts = []
    for axis, name in enumerate(("aligned_x", "aligned_y", "up")):
        if delta[axis] > tolerance:
            facts.append(f"b_centroid_positive_{name}_of_a")
        elif delta[axis] < -tolerance:
            facts.append(f"b_centroid_negative_{name}_of_a")
    if low_b[2] > high_a[2] + tolerance:
        facts.append("b_bbox_entirely_above_a_bbox")
    if low_a[2] > high_b[2] + tolerance:
        facts.append("a_bbox_entirely_above_b_bbox")
    if area_intersection > 0:
        facts.append("bev_bboxes_overlap")
    if np.all(low_a <= low_b) and np.all(high_a >= high_b):
        facts.append("a_bbox_contains_b_bbox")
    if np.all(low_b <= low_a) and np.all(high_b >= high_a):
        facts.append("b_bbox_contains_a_bbox")
    candidates = []
    if area_intersection > 0:
        if abs(low_b[2] - high_a[2]) <= tolerance:
            candidates.append("a_may_support_b")
        if abs(low_a[2] - high_b[2]) <= tolerance:
            candidates.append("b_may_support_a")
    return {
        "a": a["obj_id"], "b": b["obj_id"],
        "delta_centroid_b_minus_a_aligned": delta.tolist(),
        "centroid_distance_3d": float(np.linalg.norm(delta)),
        "centroid_distance_bev": float(np.linalg.norm(delta[:2])),
        "centroid_height_difference_b_minus_a": float(delta[2]),
        "bbox_gap_distance_lower_bound": float(np.linalg.norm(gaps)),
        "bbox_gap_by_axis": gaps.tolist(),
        "b_bottom_minus_a_top": float(low_b[2] - high_a[2]),
        "a_bottom_minus_b_top": float(low_a[2] - high_b[2]),
        "bev_bbox_intersection_area": area_intersection,
        "bev_bbox_iou": area_intersection / union if union > 0 else 0.0,
        "bev_overlap_fraction_a": area_intersection / area_a if area_a > 0 else 0.0,
        "bev_overlap_fraction_b": area_intersection / area_b if area_b > 0 else 0.0,
        "aligned_bbox_iou_3d": volume_intersection / volume_union if volume_union > 0 else 0.0,
        "geometry_facts": facts,
        "unverified_candidates": candidates,
    }


def write_txt(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def export_geometry(objects, up, output_dir="rag/knowledge", asset_root="rag",
                    scene_source=None, up_source="provided", image_size=1024, tolerance=None):
    """objects yields dictionaries with obj_id, points and optional opacity/rgb/scales."""
    if image_size < 2:
        raise ValueError("image_size must be >= 2")
    basis = coordinate_basis(up)
    records = [object_geometry(basis=basis, **obj) for obj in objects]
    records.sort(key=lambda x: x["obj_id"])
    if not records or len({r["obj_id"] for r in records}) != len(records):
        raise ValueError("Expected at least one object with unique IDs")
    lower = np.min([r["bbox_up_aligned"]["min"] for r in records], axis=0)
    upper = np.max([r["bbox_up_aligned"]["max"] for r in records], axis=0)
    diagonal = float(np.linalg.norm(upper - lower))
    if tolerance is None:
        tolerance = max(diagonal * 0.01, 1e-6)
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("tolerance must be finite and nonnegative")
    center_xy = (lower[:2] + upper[:2]) / 2
    span = max(float(np.max(upper[:2] - lower[:2])) * 1.1, 1e-6)
    bev_min = center_xy - span / 2
    bev_max = center_xy + span / 2
    pixel_scale = (image_size - 1) / span

    def project(xy):
        return [(xy[0] - bev_min[0]) * pixel_scale, (bev_max[1] - xy[1]) * pixel_scale]

    asset_root = Path(asset_root).resolve()
    for record in records:
        obj_id = record["obj_id"]
        record["coordinate_context"] = {
            "scene_metadata": "scene.txt", "up_source": up_source,
            "basis_columns_x_y_up_in_world": basis.tolist(),
            "units": "原始场景单位，未校准",
            "bbox_definition": "高斯中心范围，非实体表面；aligned z 沿 up；BEV 为共享正交投影。",
        }
        x0, y0, x1, y1 = record["bev_bbox_aligned_xyxy"]
        record["bev_bbox_pixel_xyxy"] = [*project([x0, y1]), *project([x1, y0])]
        record["bev_centroid_pixel_uv"] = project(record["centroid_aligned"][:2])
        record["centroid_height_above_exported_scene_min"] = record["centroid_aligned"][2] - lower[2]
        record["assets"] = {
            "ply": str(asset_root / "ply" / f"object_{obj_id}.ply"),
            "embedding": str(asset_root / "embeddings" / f"object_{obj_id}.pt"),
            "views": [str(asset_root / "views" / f"object_{obj_id}" / f"view_{angle:03d}.png")
                      for angle in (0, 90, 180, 270)],
        }
        record["asset_exists_at_export"] = {
            "ply": Path(record["assets"]["ply"]).is_file(),
            "embedding": Path(record["assets"]["embedding"]).is_file(),
            "views": [Path(p).is_file() for p in record["assets"]["views"]],
        }

    relations = [pair_relation(a, b, tolerance) for a, b in itertools.combinations(records, 2)]
    for record in records:
        related = [r for r in relations if record["obj_id"] in (r["a"], r["b"])]
        record["nearest_objects_by_centroid"] = [
            {"obj_id": r["b"] if r["a"] == record["obj_id"] else r["a"],
             "distance_3d": r["centroid_distance_3d"], "distance_bev": r["centroid_distance_bev"]}
            for r in sorted(related, key=lambda r: r["centroid_distance_3d"])[:5]]
        record["relation_pairs"] = [[r["a"], r["b"]] for r in related]

    scene = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scene_source": scene_source, "object_ids": [r["obj_id"] for r in records],
        "object_count": len(records), "gaussian_count": sum(r["gaussian_count"] for r in records),
        "distance_units": "原始场景单位，未校准；不能当作米。",
        "coordinate_frame": {
            "origin_world": [0, 0, 0], "up_source": up_source,
            "up_vector_normalized": basis[:, 2].tolist(),
            "basis_columns_x_y_up_in_world": basis.tolist(),
            "world_to_aligned": "row-vector p_aligned = p_world @ basis; z follows up_vector",
            "aligned_to_world": "row-vector p_world = p_aligned @ basis.T",
            "horizontal_axes": "x = normalize(cross(up, least-parallel world axis)); y = cross(up,x). 与四视图轨道一致。",
            "direction_caution": "x/y 是固定参考轴，不代表物体自身左右、前后或东南西北；上下由传入 up 的符号决定。",
        },
        "exported_object_bounds_aligned": {"min": lower.tolist(), "max": upper.tolist()},
        "bev_projection": {
            "type": "orthographic_bbox_schematic", "image": "bev_layout.png",
            "width": image_size, "height": image_size,
            "aligned_xy_min": bev_min.tolist(), "aligned_xy_max": bev_max.tolist(),
            "pixels_per_scene_unit": pixel_scale,
            "mapping": "u=(x-xmin)*scale; v=(ymax-y)*scale; xyxy=[left,top,right,bottom]; pixel centers 0..size-1",
        },
        "relation_tolerance": tolerance,
        "relation_tolerance_rule": "默认 0.01 * 导出物体整体包围盒对角线，可显式设置。",
        "embedding_contract": "assets.embedding 指向既有 .pt，包含 obj_id/model_name/view_angles/embeddings；读取文件确认实际模型，文本需使用匹配的 CLIP。",
        "limitations": [
            "包围盒基于全部高斯中心，不包含高斯椭球延伸；不是实体表面或分割轮廓。",
            "robust bbox 是沿各轴 1%-99% 分位数范围，仅用于抗离群参考；关系使用完整 bbox。",
            "bbox center 与 centroid 不同；centroid 是高斯中心的算术平均，不是质量中心。",
            "BEV bbox 是共享正交投影的矩形，不是透视截图检测框；重叠不证明碰撞。",
            "bbox gap 是点集距离下界；bbox 包含不等于物体包含；支撑候选未经物理或表面验证。",
            "高度零点为原始世界原点；scene_min 不是已识别的地面或桌面。",
            "不推测类别、物体名称、朝向、可见性或遮挡。semantic_label 留空，后续可加入有来源的语义。",
            "仅描述此次导出的正 ID 物体，未包含背景；ID 与当前分割标签绑定，重分割后需重新导出。",
            "读取对象仅以本次 scene.txt 的 object_ids 为准；文件路径存在不保证其内容来自同一版本。",
        ],
        "retrieval_guidance": "先用 CLIP 检索候选 ID，再读取对应 objects/object_ID.txt；关系查询读取 relations.txt 中含该 ID 的条目；回答引用 ID、字段和坐标系，推测关系须说明不确定性。",
    }
    output = Path(output_dir)
    (output / "objects").mkdir(parents=True, exist_ok=True)
    write_txt(output / "scene.txt", scene)
    write_txt(output / "objects.txt", records)
    write_txt(output / "relations.txt", {"pair_convention": "a,b 对应物体 ID；delta 为 b-a；每个无序对只保存一次。",
                                         "pairs": relations})
    for record in records:
        write_txt(output / "objects" / f"object_{record['obj_id']}.txt", record)
    from PIL import Image, ImageDraw
    image = Image.new("RGB", (image_size, image_size), "white")
    draw = ImageDraw.Draw(image)
    for record in records:
        obj_id = record["obj_id"]
        color = tuple(40 + (obj_id * multiplier) % 170 for multiplier in (37, 71, 113))
        draw.rectangle(record["bev_bbox_pixel_xyxy"], outline=color, width=2)
        u, v = record["bev_centroid_pixel_uv"]
        draw.ellipse((u-2, v-2, u+2, v+2), fill=color)
        draw.text((u+3, v+3), str(obj_id), fill=color)
    image.save(output / "bev_layout.png")
    return scene


def read_ply_objects(ply_dir):
    from plyfile import PlyData
    for path in sorted(Path(ply_dir).glob("object_*.ply")):
        obj_id = int(path.stem.removeprefix("object_"))
        if obj_id <= 0:
            continue
        data = PlyData.read(path)["vertex"].data
        names = set(data.dtype.names)
        obj = {"obj_id": obj_id, "points": np.column_stack([data[k] for k in ("x", "y", "z")])}
        if "opacity" in names:
            obj["opacity"] = 1 / (1 + np.exp(-np.clip(data["opacity"].astype(float), -80, 80)))
        if all(f"f_dc_{i}" in names for i in range(3)):
            obj["rgb"] = np.clip(0.5 + 0.28209479177387814 * np.column_stack(
                [data[f"f_dc_{i}"] for i in range(3)]), 0, 1)
        if all(f"scale_{i}" in names for i in range(3)):
            obj["scales"] = np.exp(np.column_stack([data[f"scale_{i}"] for i in range(3)]).astype(float))
        yield obj


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", default="rag")
    parser.add_argument("--output", default="rag/knowledge")
    parser.add_argument("--up", nargs=3, type=float, required=True)
    parser.add_argument("--up-source", default="CLI --up")
    parser.add_argument("--scene-source")
    parser.add_argument("--tolerance", type=float)
    args = parser.parse_args()
    scene = export_geometry(read_ply_objects(Path(args.asset_root) / "ply"), args.up,
                            args.output, args.asset_root, args.scene_source, args.up_source,
                            tolerance=args.tolerance)
    print(f"Exported {scene['object_count']} objects to {args.output}")


if __name__ == "__main__":
    main()
