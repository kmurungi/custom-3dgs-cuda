from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import re
import shutil
import struct
import subprocess

import torch as t

from gaussian_splatting.scene.camera import camera
from gaussian_splatting.scene.gaussian import gaussian


CAMERA_MODEL_PARAMS = {
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
class ColmapPaths:
    """Files and directories produced by a COLMAP reconstruction."""

    workspace: Path
    database: Path
    sparse: Path
    images: Path


def _run(command: list[str]) -> None:
    """Run a COLMAP command and surface a useful error on failure."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        details = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            f"COLMAP command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{details}"
        )


@lru_cache(maxsize=None)
def _subcommand_options(executable: str, subcommand: str) -> frozenset[str]:
    """Collect the option names a COLMAP subcommand accepts."""
    result = subprocess.run(
        [executable, subcommand, "-h"], capture_output=True, text=True
    )
    output = result.stdout + result.stderr
    return frozenset(match.group(1) for match in re.finditer(r"--([\w.]+)", output))


def _gpu_args(
    executable: str,
    subcommand: str,
    prefixes: tuple[str, ...],
    use_gpu: bool,
) -> list[str]:
    """Build the GPU flag for a subcommand, handling COLMAP option renames.

    COLMAP 4.x renamed ``SiftExtraction``/``SiftMatching`` to
    ``FeatureExtraction``/``FeatureMatching``.
    """
    options = _subcommand_options(executable, subcommand)
    for prefix in prefixes:
        option = f"{prefix}.use_gpu"
        if option in options:
            return [f"--{option}", "1" if use_gpu else "0"]
    return []


def run_colmap(
    data_path: str | Path,
    output_path: str | Path | None = None,
    *,
    colmap_command: str = "colmap",
    camera_model: str = "PINHOLE",
    use_gpu: bool = True,
) -> ColmapPaths:
    """Create a sparse COLMAP reconstruction from a directory of images.

    Args:
        data_path: Directory containing the input images.
        output_path: COLMAP workspace. Defaults to ``<data_path>_colmap``.
        colmap_command: COLMAP executable name or full path.
        camera_model: Camera model passed to COLMAP's feature extractor.
        use_gpu: Whether feature extraction and matching should use CUDA.

    Returns:
        Paths to the database, sparse model, and undistorted images.
    """
    image_path = Path(data_path).expanduser().resolve()
    if not image_path.is_dir():
        raise NotADirectoryError(f"Image directory does not exist: {image_path}")
    if not any(path.is_file() for path in image_path.iterdir()):
        raise ValueError(f"Image directory is empty: {image_path}")

    executable = shutil.which(colmap_command)
    if executable is None:
        command_path = Path(colmap_command).expanduser()
        if not command_path.is_file():
            raise FileNotFoundError(
                "COLMAP was not found. Install COLMAP and add it to PATH, or "
                "pass its executable path with colmap_command."
            )
        executable = str(command_path.resolve())

    # Kept outside image_path: COLMAP scans the image directory recursively.
    workspace = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else image_path.parent / f"{image_path.name}_colmap"
    )
    database_path = workspace / "database.db"
    sparse_path = workspace / "sparse"
    undistorted_path = workspace / "undistorted"
    sparse_path.mkdir(parents=True, exist_ok=True)
    undistorted_path.mkdir(parents=True, exist_ok=True)

    _run(
        [
            executable,
            "feature_extractor",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--ImageReader.camera_model",
            camera_model,
        ]
        + _gpu_args(executable, "feature_extractor", ("FeatureExtraction", "SiftExtraction"), use_gpu)
    )
    _run(
        [
            executable,
            "exhaustive_matcher",
            "--database_path",
            str(database_path),
        ]
        + _gpu_args(executable, "exhaustive_matcher", ("FeatureMatching", "SiftMatching"), use_gpu)
    )
    _run(
        [
            executable,
            "mapper",
            "--database_path",
            str(database_path),
            "--image_path",
            str(image_path),
            "--output_path",
            str(sparse_path),
        ]
    )

    model_path = sparse_path / "0"
    if not model_path.is_dir():
        raise RuntimeError(
            "COLMAP did not create a sparse model. Check image overlap and "
            f"the reconstruction logs in {workspace}."
        )

    _run(
        [
            executable,
            "image_undistorter",
            "--image_path",
            str(image_path),
            "--input_path",
            str(model_path),
            "--output_path",
            str(undistorted_path),
            "--output_type",
            "COLMAP",
        ]
    )

    return ColmapPaths(
        workspace=workspace,
        database=database_path,
        sparse=undistorted_path / "sparse",
        images=undistorted_path / "images",
    )


def _resolve_model_dir(sparse_path: Path) -> Path:
    """Find the directory that actually contains cameras/images/points3D."""
    sparse_path = sparse_path.expanduser().resolve()
    candidates = [sparse_path, sparse_path / "0"]
    for candidate in candidates:
        if (candidate / "cameras.bin").is_file() or (candidate / "cameras.txt").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find COLMAP cameras.bin/txt under {sparse_path}"
    )


def _read_next_bytes(fid, num_bytes: int, format_char_sequence: str, endian_character: str = "<"):
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


def _qvec_to_rotmat(qvec: t.Tensor) -> t.Tensor:
    """Convert COLMAP quaternion (w, x, y, z) to a 3x3 rotation matrix."""
    w, x, y, z = qvec
    return t.tensor(
        [
            [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
            [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
            [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
        ],
        dtype=t.float32,
    )


def _intrinsics_from_params(model_name: str, params: list[float]) -> tuple[float, float, float, float]:
    if model_name in {"SIMPLE_PINHOLE", "SIMPLE_RADIAL", "SIMPLE_RADIAL_FISHEYE", "RADIAL", "RADIAL_FISHEYE", "FOV"}:
        f, cx, cy = params[:3]
        return float(f), float(f), float(cx), float(cy)
    if model_name in {"PINHOLE", "OPENCV", "OPENCV_FISHEYE", "FULL_OPENCV", "THIN_PRISM_FISHEYE"}:
        fx, fy, cx, cy = params[:4]
        return float(fx), float(fy), float(cx), float(cy)
    raise ValueError(f"Unsupported COLMAP camera model: {model_name}")


def _read_cameras_text(path: Path) -> dict[int, dict]:
    cameras = {}
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            camera_id = int(tokens[0])
            model_name = tokens[1]
            width, height = int(tokens[2]), int(tokens[3])
            params = [float(value) for value in tokens[4:]]
            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_cameras_binary(path: Path) -> dict[int, dict]:
    cameras = {}
    with path.open("rb") as file:
        num_cameras = _read_next_bytes(file, 8, "Q")[0]
        for _ in range(num_cameras):
            camera_id, model_id, width, height = _read_next_bytes(file, 24, "iiQQ")
            model_name, num_params = CAMERA_MODEL_PARAMS[model_id]
            params = list(_read_next_bytes(file, 8 * num_params, "d" * num_params))
            cameras[camera_id] = {
                "model": model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def _read_images_text(path: Path) -> dict[int, dict]:
    images = {}
    with path.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip() and not line.startswith("#")]

    index = 0
    while index < len(lines):
        tokens = lines[index].split()
        image_id = int(tokens[0])
        qvec = [float(value) for value in tokens[1:5]]
        tvec = [float(value) for value in tokens[5:8]]
        camera_id = int(tokens[8])
        name = tokens[9]
        images[image_id] = {
            "qvec": qvec,
            "tvec": tvec,
            "camera_id": camera_id,
            "name": name,
        }
        index += 2  # skip points2D line
    return images


def _read_images_binary(path: Path) -> dict[int, dict]:
    images = {}
    with path.open("rb") as file:
        num_images = _read_next_bytes(file, 8, "Q")[0]
        for _ in range(num_images):
            image_id = _read_next_bytes(file, 4, "i")[0]
            qvec = list(_read_next_bytes(file, 32, "dddd"))
            tvec = list(_read_next_bytes(file, 24, "ddd"))
            camera_id = _read_next_bytes(file, 4, "i")[0]

            name_chars = []
            while True:
                char = file.read(1)
                if char == b"\x00":
                    break
                name_chars.append(char.decode("utf-8"))
            name = "".join(name_chars)

            num_points2d = _read_next_bytes(file, 8, "Q")[0]
            file.read(24 * num_points2d)  # x, y, point3D_id

            images[image_id] = {
                "qvec": qvec,
                "tvec": tvec,
                "camera_id": camera_id,
                "name": name,
            }
    return images


def _read_points3d_text(path: Path) -> tuple[t.Tensor, t.Tensor]:
    xyz_list = []
    rgb_list = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tokens = line.split()
            xyz_list.append([float(tokens[1]), float(tokens[2]), float(tokens[3])])
            rgb_list.append([float(tokens[4]), float(tokens[5]), float(tokens[6])])

    if not xyz_list:
        raise ValueError(f"No 3D points found in {path}")

    xyz = t.tensor(xyz_list, dtype=t.float32)
    rgb = t.tensor(rgb_list, dtype=t.float32) / 255.0
    return xyz, rgb


def _read_points3d_binary(path: Path) -> tuple[t.Tensor, t.Tensor]:
    xyz_list = []
    rgb_list = []
    with path.open("rb") as file:
        num_points = _read_next_bytes(file, 8, "Q")[0]
        for _ in range(num_points):
            _point_id = _read_next_bytes(file, 8, "Q")[0]
            xyz = list(_read_next_bytes(file, 24, "ddd"))
            rgb = list(_read_next_bytes(file, 3, "BBB"))
            _error = _read_next_bytes(file, 8, "d")[0]
            track_length = _read_next_bytes(file, 8, "Q")[0]
            file.read(8 * track_length)  # image_id, point2D_idx
            xyz_list.append(xyz)
            rgb_list.append(rgb)

    if not xyz_list:
        raise ValueError(f"No 3D points found in {path}")

    xyz = t.tensor(xyz_list, dtype=t.float32)
    rgb = t.tensor(rgb_list, dtype=t.float32) / 255.0
    return xyz, rgb


def _read_cameras(model_dir: Path) -> dict[int, dict]:
    binary = model_dir / "cameras.bin"
    text = model_dir / "cameras.txt"
    if binary.is_file():
        return _read_cameras_binary(binary)
    if text.is_file():
        return _read_cameras_text(text)
    raise FileNotFoundError(f"Missing cameras.bin/txt in {model_dir}")


def _read_images(model_dir: Path) -> dict[int, dict]:
    binary = model_dir / "images.bin"
    text = model_dir / "images.txt"
    if binary.is_file():
        return _read_images_binary(binary)
    if text.is_file():
        return _read_images_text(text)
    raise FileNotFoundError(f"Missing images.bin/txt in {model_dir}")


def _read_points3d(model_dir: Path) -> tuple[t.Tensor, t.Tensor]:
    binary = model_dir / "points3D.bin"
    text = model_dir / "points3D.txt"
    if binary.is_file():
        return _read_points3d_binary(binary)
    if text.is_file():
        return _read_points3d_text(text)
    raise FileNotFoundError(f"Missing points3D.bin/txt in {model_dir}")


def _init_scales(xyz: t.Tensor, k_neighbors: int = 3) -> t.Tensor:
    """Isotropic scale from mean distance to k nearest neighbors."""
    num_points = xyz.shape[0]
    if num_points == 1:
        return t.full((1, 3), 0.01, dtype=t.float32)

    k = min(k_neighbors, num_points - 1)
    distances = t.cdist(xyz, xyz)
    distances.fill_diagonal_(float("inf"))
    nearest, _ = t.topk(distances, k=k, largest=False, dim=1)
    mean_dist = nearest.mean(dim=1).clamp_min(1e-7)
    return mean_dist.unsqueeze(1).repeat(1, 3)


def parse_colmap(
    sparse_path: str | Path,
    images_path: str | Path | None = None,
) -> tuple[gaussian, camera]:
    """Parse a COLMAP sparse model into ``gaussian`` and ``camera`` objects.

    Args:
        sparse_path: Directory containing ``cameras`` / ``images`` / ``points3D``.
        images_path: Optional directory of undistorted images. Stored on the
            returned camera object as ``image_paths``.

    Returns:
        Initialized ``(gaussians, cameras)`` pair.
    """
    model_dir = _resolve_model_dir(Path(sparse_path))
    colmap_cameras = _read_cameras(model_dir)
    colmap_images = _read_images(model_dir)
    xyz, rgb = _read_points3d(model_dir)

    if not colmap_images:
        raise ValueError(f"No reconstructed images found in {model_dir}")

    image_items = sorted(colmap_images.items(), key=lambda item: item[0])
    num_cameras = len(image_items)
    cameras = camera(num_cameras)

    image_names: list[str] = []
    image_paths: list[Path | None] = []
    images_root = Path(images_path).expanduser().resolve() if images_path is not None else None

    for index, (_image_id, image_data) in enumerate(image_items):
        camera_data = colmap_cameras[image_data["camera_id"]]
        fx, fy, cx, cy = _intrinsics_from_params(camera_data["model"], camera_data["params"])

        qvec = t.tensor(image_data["qvec"], dtype=t.float32)
        tvec = t.tensor(image_data["tvec"], dtype=t.float32)

        cameras.R[index] = _qvec_to_rotmat(qvec)
        cameras.t[index] = tvec
        cameras.fx[index] = fx
        cameras.fy[index] = fy
        cameras.cx[index] = cx
        cameras.cy[index] = cy

        name = image_data["name"]
        image_names.append(name)
        image_paths.append(images_root / name if images_root is not None else None)

    cameras.image_names = image_names
    cameras.image_paths = image_paths

    num_gaussians = xyz.shape[0]
    gaussians = gaussian(num_gaussians)
    gaussians.mu = xyz
    gaussians.q = t.zeros(num_gaussians, 4, dtype=t.float32)
    gaussians.q[:, 0] = 1.0  # identity quaternion (w, x, y, z)
    gaussians.s = _init_scales(xyz)
    gaussians.alpha = t.full((num_gaussians, 1), 0.1, dtype=t.float32)
    gaussians.A = rgb
    gaussians.k_j = t.zeros(num_gaussians, 15, 3, dtype=t.float32)

    return gaussians, cameras


def load_colmap(
    data_path: str | Path,
    output_path: str | Path | None = None,
    *,
    colmap_command: str = "colmap",
    camera_model: str = "PINHOLE",
    use_gpu: bool = True,
) -> tuple[gaussian, camera]:
    """Run COLMAP on an image directory and initialize scene objects."""
    paths = run_colmap(
        data_path,
        output_path,
        colmap_command=colmap_command,
        camera_model=camera_model,
        use_gpu=use_gpu,
    )
    return parse_colmap(paths.sparse, paths.images)
