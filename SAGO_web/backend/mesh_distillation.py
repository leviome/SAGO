#!/usr/bin/env python3
"""COLMAP-to-NeuS2 surface reconstruction in one self-contained Python file.

The :class:`MeshDistillation` pipeline:

1. reads a COLMAP binary or text sparse model;
2. converts camera poses to the NeuS2 transform format;
3. combines RGB images and masks into RGBA training images;
4. trains NeuS2 and extracts a marching-cubes OBJ;
5. converts the OBJ back to COLMAP world coordinates; and
6. removes unreferenced vertices and validates the result.

Adapted from SAGO addon/mesh/mesh_distillation.py.
NeuS2 and its Python environment are runtime dependencies. Dataset preparation and mesh post-processing are self-contained;
training uses SAGO's headless NeuS2 worker.
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

import numpy as np
from PIL import Image


CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
}


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class ColmapImage:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


@dataclass(frozen=True)
class MeshStats:
    path: Path
    vertices: int
    faces: int
    removed_vertices: int
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]


class MeshDistillation:
    """Prepare a COLMAP scene, reconstruct it with NeuS2, and export an OBJ.

    Parameters are intentionally explicit so the class can be imported into a
    larger pipeline. Calling :meth:`run` executes the complete process.
    """

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        *,
        name: str | None = None,
        neus2_root: str | Path,
        python_executable: str | Path,
        network: str | Path = "workspace_mask002.json",
        sparse_dir: str | Path | None = None,
        steps: int = 15_000,
        marching_cubes_resolution: int = 512,
        aabb_scale: float = 1.0,
        scale: float = 0.5,
        offset: Iterable[float] = (0.5, 0.5, 0.5),
        output_obj: str | Path | None = None,
        output_space: str = "colmap",
        clean_mesh: bool = True,
    ) -> None:
        self.input_dir = Path(input_dir).expanduser().resolve()
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.neus2_root = Path(neus2_root).expanduser().resolve()
        self.sparse_dir = Path(sparse_dir).expanduser().resolve() if sparse_dir else None
        self.name = name or self.input_dir.name
        if not self.name or Path(self.name).name != self.name:
            raise ValueError("name must be a single non-empty path component")

        self.python_executable = Path(python_executable).expanduser().resolve()

        network_path = Path(network).expanduser()
        if not network_path.is_absolute():
            network_path = self.neus2_root / "configs" / "nerf" / network_path
        self.network = network_path.resolve()

        self.steps = int(steps)
        self.marching_cubes_resolution = int(marching_cubes_resolution)
        self.aabb_scale = float(aabb_scale)
        self.scale = float(scale)
        self.offset = [float(value) for value in offset]
        self.output_space = output_space
        self.clean_mesh = bool(clean_mesh)

        if self.steps <= 0:
            raise ValueError("steps must be positive")
        if self.marching_cubes_resolution <= 0:
            raise ValueError("marching_cubes_resolution must be positive")
        if len(self.offset) != 3:
            raise ValueError("offset must contain exactly three values")
        if output_space not in {"colmap", "neus2"}:
            raise ValueError("output_space must be 'colmap' or 'neus2'")

        suffix = "colmap" if output_space == "colmap" else "neus2"
        default_obj = self.output_dir / f"{self.name}_{suffix}.obj"
        self.output_obj = Path(output_obj).expanduser().resolve() if output_obj else default_obj
        self.scene_dir = self.output_dir / "prepared" / self.name
        self.transform_path = self.scene_dir / "transform.json"
        self.normalization_path = self.scene_dir / "normalization.json"
        self.log_path = self.output_dir / "logs" / self.name / "train.log"

        self.center: np.ndarray | None = None
        self.radius: float | None = None

    @staticmethod
    def _read_bytes(stream: BinaryIO, size: int, fmt: str) -> tuple:
        data = stream.read(size)
        if len(data) != size:
            raise EOFError("Unexpected end of COLMAP binary file")
        return struct.unpack("<" + fmt, data)

    @classmethod
    def _read_cameras_binary(cls, path: Path) -> dict[int, Camera]:
        cameras: dict[int, Camera] = {}
        with path.open("rb") as stream:
            count = cls._read_bytes(stream, 8, "Q")[0]
            for _ in range(count):
                camera_id, model_id, width, height = cls._read_bytes(stream, 24, "iiQQ")
                if model_id not in CAMERA_MODELS:
                    raise ValueError(f"Unsupported COLMAP camera model id: {model_id}")
                model, parameter_count = CAMERA_MODELS[model_id]
                params = np.asarray(
                    cls._read_bytes(stream, parameter_count * 8, "d" * parameter_count),
                    dtype=np.float64,
                )
                cameras[camera_id] = Camera(camera_id, model, width, height, params)
        return cameras

    @classmethod
    def _read_images_binary(cls, path: Path) -> dict[int, ColmapImage]:
        images: dict[int, ColmapImage] = {}
        with path.open("rb") as stream:
            count = cls._read_bytes(stream, 8, "Q")[0]
            for _ in range(count):
                values = cls._read_bytes(stream, 64, "idddddddi")
                name_bytes = bytearray()
                while True:
                    byte = stream.read(1)
                    if byte == b"\0":
                        break
                    if not byte:
                        raise EOFError("Unexpected end while reading a COLMAP image name")
                    name_bytes.extend(byte)
                points_count = cls._read_bytes(stream, 8, "Q")[0]
                stream.seek(points_count * 24, os.SEEK_CUR)
                image_id = int(values[0])
                images[image_id] = ColmapImage(
                    image_id=image_id,
                    qvec=np.asarray(values[1:5], dtype=np.float64),
                    tvec=np.asarray(values[5:8], dtype=np.float64),
                    camera_id=int(values[8]),
                    name=name_bytes.decode("utf-8"),
                )
        return images

    @classmethod
    def _read_points_binary(cls, path: Path) -> np.ndarray:
        points: list[tuple[float, float, float]] = []
        with path.open("rb") as stream:
            count = cls._read_bytes(stream, 8, "Q")[0]
            for _ in range(count):
                values = cls._read_bytes(stream, 43, "QdddBBBd")
                points.append((values[1], values[2], values[3]))
                track_length = cls._read_bytes(stream, 8, "Q")[0]
                stream.seek(track_length * 8, os.SEEK_CUR)
        return np.asarray(points, dtype=np.float64)

    @staticmethod
    def _data_lines(path: Path) -> list[str]:
        return [
            line
            for raw in path.read_text().splitlines()
            if (line := raw.strip()) and not line.startswith("#")
        ]

    @classmethod
    def _read_cameras_text(cls, path: Path) -> dict[int, Camera]:
        cameras: dict[int, Camera] = {}
        for line in cls._data_lines(path):
            values = line.split()
            camera_id = int(values[0])
            cameras[camera_id] = Camera(
                camera_id,
                values[1],
                int(values[2]),
                int(values[3]),
                np.asarray([float(value) for value in values[4:]], dtype=np.float64),
            )
        return cameras

    @classmethod
    def _read_images_text(cls, path: Path) -> dict[int, ColmapImage]:
        lines = cls._data_lines(path)
        if len(lines) % 2:
            raise ValueError(f"Malformed COLMAP images text file: {path}")
        images: dict[int, ColmapImage] = {}
        for line in lines[::2]:
            values = line.split()
            image_id = int(values[0])
            images[image_id] = ColmapImage(
                image_id,
                np.asarray([float(value) for value in values[1:5]], dtype=np.float64),
                np.asarray([float(value) for value in values[5:8]], dtype=np.float64),
                int(values[8]),
                values[9],
            )
        return images

    @classmethod
    def _read_points_text(cls, path: Path) -> np.ndarray:
        return np.asarray(
            [[float(value) for value in line.split()[1:4]] for line in cls._data_lines(path)],
            dtype=np.float64,
        )

    def _find_sparse_model(self) -> Path:
        if self.sparse_dir is not None:
            candidates = [self.sparse_dir]
        else:
            candidates = [
                self.input_dir / "sparse_txt",
                self.input_dir / "sparse" / "0",
                self.input_dir / "sparse",
                self.input_dir / "colmap" / "sparse" / "0",
                self.input_dir / "colmap" / "sparse",
            ]
        for candidate in candidates:
            if (candidate / "cameras.bin").exists() or (candidate / "cameras.txt").exists():
                return candidate.resolve()
        raise FileNotFoundError(f"No COLMAP sparse model found below {self.input_dir}")

    def _load_colmap_model(
        self, sparse_model: Path
    ) -> tuple[dict[int, Camera], dict[int, ColmapImage], np.ndarray]:
        if (sparse_model / "cameras.bin").exists():
            return (
                self._read_cameras_binary(sparse_model / "cameras.bin"),
                self._read_images_binary(sparse_model / "images.bin"),
                self._read_points_binary(sparse_model / "points3D.bin"),
            )
        return (
            self._read_cameras_text(sparse_model / "cameras.txt"),
            self._read_images_text(sparse_model / "images.txt"),
            self._read_points_text(sparse_model / "points3D.txt"),
        )

    @staticmethod
    def _qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
        w, x, y, z = qvec
        return np.asarray(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _intrinsic_matrix(camera: Camera) -> np.ndarray:
        params = camera.params
        if camera.model in {
            "SIMPLE_PINHOLE",
            "SIMPLE_RADIAL",
            "RADIAL",
            "SIMPLE_RADIAL_FISHEYE",
            "RADIAL_FISHEYE",
        }:
            fx = fy = params[0]
            cx, cy = params[1:3]
        elif camera.model in {
            "PINHOLE",
            "OPENCV",
            "OPENCV_FISHEYE",
            "FULL_OPENCV",
            "FOV",
            "THIN_PRISM_FISHEYE",
        }:
            fx, fy, cx, cy = params[:4]
        else:
            raise ValueError(f"Unsupported COLMAP camera model: {camera.model}")
        return np.asarray(
            [[fx, 0, cx, 0], [0, fy, cy, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        )

    @staticmethod
    def _find_image(images_dir: Path, colmap_name: str) -> Path:
        direct = images_dir / colmap_name
        basename = images_dir / Path(colmap_name).name
        for candidate in (direct, basename):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"COLMAP image not found: {colmap_name}")

    @staticmethod
    def _find_mask(masks_dir: Path, image_path: Path) -> Path | None:
        candidates = (
            image_path.name,
            f"mask_{image_path.stem}{image_path.suffix}",
            f"{image_path.stem}_mask{image_path.suffix}",
            f"{image_path.stem}.png",
            f"mask_{image_path.stem}.png",
            f"{image_path.stem}_mask.png",
        )
        return next((masks_dir / name for name in candidates if (masks_dir / name).exists()), None)

    def _write_rgba(self, image_path: Path, masks_dir: Path, destination: Path) -> str:
        destination.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(image_path) as source:
            if source.mode == "RGBA":
                source.save(destination)
                return "image-alpha"
            rgb = source.convert("RGB")
        mask_path = self._find_mask(masks_dir, image_path)
        if mask_path is None:
            alpha = Image.new("L", rgb.size, 255)
            alpha_source = "opaque-alpha"
        else:
            with Image.open(mask_path) as mask:
                alpha = mask.convert("L")
                if alpha.size != rgb.size:
                    alpha = alpha.resize(rgb.size, Image.Resampling.NEAREST)
            alpha_source = "mask-alpha"
        Image.merge("RGBA", (*rgb.split(), alpha)).save(destination)
        return alpha_source

    def prepare_dataset(self) -> Path:
        """Create NeuS2 RGBA images and ``transform.json`` from COLMAP data."""
        images_dir = self.input_dir / "images"
        masks_dir = self.input_dir / "masks"
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {images_dir}")

        sparse_model = self._find_sparse_model()
        cameras, images, points = self._load_colmap_model(sparse_model)
        if not cameras or not images or points.size == 0:
            raise ValueError("COLMAP model must contain cameras, registered images, and 3D points")

        center = (points.max(axis=0) + points.min(axis=0)) * 0.5
        radius = float(np.linalg.norm(points - center, axis=1).max() * 1.1)
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError("Computed an invalid COLMAP normalization radius")
        self.center, self.radius = center, radius

        frames: list[dict] = []
        alpha_sources: dict[str, int] = {}
        for colmap_image in sorted(images.values(), key=lambda item: item.name):
            camera = cameras[colmap_image.camera_id]
            source = self._find_image(images_dir, colmap_image.name)
            # Alpha-bearing training inputs must use a format that supports RGBA.
            destination_name = Path(colmap_image.name).stem + ".png"
            destination = self.scene_dir / "images" / destination_name
            alpha_source = self._write_rgba(source, masks_dir, destination)
            alpha_sources[alpha_source] = alpha_sources.get(alpha_source, 0) + 1

            rotation_w2c = self._qvec_to_rotation(colmap_image.qvec)
            camera_center = -rotation_w2c.T @ colmap_image.tvec
            transform = np.eye(4, dtype=np.float64)
            transform[:3, :3] = rotation_w2c.T
            transform[:3, 3] = (camera_center - center) / radius
            frames.append(
                {
                    "file_path": f"images/{destination.name}",
                    "transform_matrix": transform.tolist(),
                    "intrinsic_matrix": self._intrinsic_matrix(camera).tolist(),
                }
            )

        first_camera = cameras[next(iter(cameras))]
        transforms = {
            "from_na": True,
            "w": int(first_camera.width),
            "h": int(first_camera.height),
            "aabb_scale": self.aabb_scale,
            "scale": self.scale,
            "offset": self.offset,
            "frames": frames,
        }
        self.scene_dir.mkdir(parents=True, exist_ok=True)
        self.transform_path.write_text(json.dumps(transforms, indent=2) + "\n")
        self.normalization_path.write_text(
            json.dumps(
                {
                    "input_dir": str(self.input_dir),
                    "sparse_dir": str(sparse_model),
                    "center": center.tolist(),
                    "radius": radius,
                    "neus2_to_colmap_world": "world_xyz = neus2_xyz * radius + center",
                    "colmap_world_to_neus2": "neus2_xyz = (world_xyz - center) / radius",
                    "alpha_sources": alpha_sources,
                },
                indent=2,
            )
            + "\n"
        )
        print(
            f"Prepared {len(frames)} frames; center={center.round(6).tolist()}, "
            f"radius={radius:.9g}, alpha={alpha_sources}"
        )
        return self.transform_path

    def train_and_extract(self) -> Path:
        from backend.mesh_export import run_neus2
        return run_neus2(self, lambda value, message: None, None)

    @staticmethod
    def _face_vertex_index(token: str, vertex_count: int) -> int:
        index = int(token.split("/", 1)[0])
        if index == 0:
            raise ValueError("OBJ vertex indices cannot be zero")
        return index if index > 0 else vertex_count + index + 1

    def postprocess_mesh(self, source_obj: str | Path) -> MeshStats:
        """Transform, clean, validate, and atomically write the final OBJ."""
        source_obj = Path(source_obj)
        lines = source_obj.read_text(errors="replace").splitlines()
        vertex_count = sum(line.startswith("v ") for line in lines)
        if vertex_count == 0:
            raise ValueError(f"OBJ contains no vertices: {source_obj}")

        used: set[int] = set()
        face_count = 0
        for line in lines:
            if not line.startswith("f "):
                continue
            face_count += 1
            for token in line.split()[1:]:
                used.add(self._face_vertex_index(token, vertex_count))
        if face_count == 0 or not used:
            raise ValueError(f"OBJ contains no valid faces: {source_obj}")
        if min(used) < 1 or max(used) > vertex_count:
            raise ValueError("OBJ face references an out-of-range vertex")

        kept = sorted(used) if self.clean_mesh else list(range(1, vertex_count + 1))
        remap = {old: new for new, old in enumerate(kept, start=1)}
        self.output_obj.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.output_obj.with_name(self.output_obj.name + ".tmp")
        bounds_min = np.full(3, np.inf)
        bounds_max = np.full(3, -np.inf)
        current_vertex = 0

        if self.output_space == "colmap" and (self.center is None or self.radius is None):
            normalization = json.loads(self.normalization_path.read_text())
            self.center = np.asarray(normalization["center"], dtype=np.float64)
            self.radius = float(normalization["radius"])

        with temporary.open("w") as output:
            output.write("# MeshDistillation NeuS2 reconstruction\n")
            output.write(f"# output_space={self.output_space}\n")
            for line in lines:
                if line.startswith("v "):
                    current_vertex += 1
                    if current_vertex not in remap:
                        continue
                    values = line.split()
                    xyz = np.asarray([float(value) for value in values[1:4]], dtype=np.float64)
                    if self.output_space == "colmap":
                        xyz = xyz * float(self.radius) + np.asarray(self.center)
                    if not np.isfinite(xyz).all():
                        raise ValueError("OBJ contains a non-finite transformed vertex")
                    bounds_min = np.minimum(bounds_min, xyz)
                    bounds_max = np.maximum(bounds_max, xyz)
                    extras = " " + " ".join(values[4:]) if len(values) > 4 else ""
                    output.write("v " + " ".join(f"{value:.9g}" for value in xyz) + extras + "\n")
                elif line.startswith("f "):
                    rewritten: list[str] = []
                    for token in line.split()[1:]:
                        fields = token.split("/")
                        old_index = self._face_vertex_index(token, vertex_count)
                        fields[0] = str(remap[old_index])
                        rewritten.append("/".join(fields))
                    output.write("f " + " ".join(rewritten) + "\n")
                else:
                    output.write(line + "\n")
        temporary.replace(self.output_obj)

        return MeshStats(
            path=self.output_obj,
            vertices=len(kept),
            faces=face_count,
            removed_vertices=vertex_count - len(kept),
            bounds_min=tuple(float(value) for value in bounds_min),
            bounds_max=tuple(float(value) for value in bounds_max),
        )

    def run(self, *, prepare_only: bool = False) -> MeshStats | Path:
        """Execute the complete reconstruction pipeline."""
        self.prepare_dataset()
        if prepare_only:
            return self.transform_path
        raw_mesh = self.train_and_extract()
        stats = self.postprocess_mesh(raw_mesh)
        print(
            f"Mesh: {stats.path}\n"
            f"vertices={stats.vertices}, faces={stats.faces}, "
            f"removed_vertices={stats.removed_vertices}\n"
            f"bounds=[{stats.bounds_min}, {stats.bounds_max}]"
        )
        return stats

