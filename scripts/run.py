"""Render a trained checkpoint and export Gaussian attributes for visualization."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image
from plyfile import PlyData, PlyElement
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussian_splatting.rasterizer.rasterize import rasterize
from gaussian_splatting.scene.gaussian import gaussian
from gaussian_splatting.utils.dataloader import parse_colmap


def resolve_checkpoint(path: Path | None) -> Path:
    """Prefer an explicit path, else final output, else newest checkpoint."""
    if path is not None:
        checkpoint = path.expanduser()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    final_path = Path("output") / "gaussians_final.pt"
    if final_path.is_file():
        return final_path

    checkpoint_dir = Path("checkpoints")
    candidates = sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            "No checkpoint found. Expected output/gaussians_final.pt or "
            "checkpoints/epoch_*.pt (or pass --checkpoint)."
        )
    return candidates[-1]


def resolve_colmap_workspace(dataset: Path, workspace: Path | None) -> Path:
    if workspace is not None:
        return workspace.expanduser()
    dataset = dataset.expanduser()
    default = dataset.parent / f"{dataset.name}_colmap"
    if not default.is_dir():
        raise FileNotFoundError(
            f"COLMAP workspace not found: {default}. Pass --colmap-workspace."
        )
    return default


def load_gaussians(checkpoint_path: Path, device: t.device) -> gaussian:
    state = t.load(checkpoint_path, map_location="cpu", weights_only=True)
    n = int(state["N"])
    gaussians = gaussian(n)
    for key in ("mu", "q", "s", "alpha", "A", "k_j"):
        setattr(gaussians, key, state[key].to(device=device, dtype=t.float32).contiguous())
    gaussians.N = n
    return gaussians


def save_ply(
    gaussians: gaussian,
    out_path: Path,
    *,
    max_sh_degree: int = 3,
) -> None:
    """
    Export Gaussians as an official Inria 3DGS ``point_cloud.ply``.

    SuperSplat / Jawset / SIBR expect **pre-activation** parameters:
      - ``opacity``: logit (viewer applies sigmoid)
      - ``scale_*``: log-scale (viewer applies exp)
      - ``rot_*``: unit quaternion (w, x, y, z)
      - ``f_rest_*``: channel-major SH rest coeffs, shape flattened from
        ``(N, 3, (max_sh_degree + 1)^2 - 1)``

    Our trainer already stores opacity logits and log-scales; this exporter
    normalizes quaternions and reorders SH to match Inria's transpose+flatten.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rest = (max_sh_degree + 1) ** 2 - 1  # degree 3 → 15

    with t.no_grad():
        xyz = gaussians.mu.detach().float().cpu()
        n = int(xyz.shape[0])

        # --- SH DC: (N, 3) → f_dc_0..2 (same as Inria flatten of (N, 1, 3)) ---
        f_dc = gaussians.A.detach().float().cpu().reshape(n, 3)

        # --- SH rest: (N, K, 3) → (N, 3, K) → (N, 3*K) channel-major ---
        # Inria: features_rest.transpose(1, 2).flatten(start_dim=1)
        rest = gaussians.k_j.detach().float().cpu()
        if rest.ndim != 3 or rest.shape[-1] != 3:
            raise ValueError(f"Expected k_j shape (N, K, 3), got {tuple(rest.shape)}")
        if rest.shape[1] < n_rest:
            pad = t.zeros(n, n_rest - rest.shape[1], 3, dtype=rest.dtype)
            rest = t.cat([rest, pad], dim=1)
        elif rest.shape[1] > n_rest:
            rest = rest[:, :n_rest, :]
        # (N, K, 3) → (N, 3, K) → (N, 3K): all R, then all G, then all B
        f_rest = rest.transpose(1, 2).reshape(n, 3 * n_rest).contiguous()

        # --- Opacity: keep logits (viewer: sigmoid). Support legacy (0,1) checkpoints. ---
        alpha = gaussians.alpha.detach().float().cpu().reshape(n, 1)
        if bool(((alpha > 0.0) & (alpha < 1.0)).all()) and float(alpha.min()) > 1e-6:
            # Legacy raw probabilities → Inria logits
            alpha = alpha.clamp(1e-6, 1.0 - 1e-6)
            opacity = t.log(alpha / (1.0 - alpha))
        else:
            opacity = alpha

        # --- Scales: keep log-scales (viewer: exp) ---
        scale = gaussians.s.detach().float().cpu().reshape(n, 3)

        # --- Rotations: L2-normalize quaternions (w, x, y, z) ---
        quat = gaussians.q.detach().float().cpu().reshape(n, 4)
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    xyz_np = xyz.numpy().astype(np.float32)
    normals = np.zeros_like(xyz_np, dtype=np.float32)
    f_dc_np = f_dc.numpy().astype(np.float32)
    f_rest_np = f_rest.numpy().astype(np.float32)
    opacity_np = opacity.numpy().astype(np.float32)
    scale_np = scale.numpy().astype(np.float32)
    rot_np = quat.numpy().astype(np.float32)

    # Exact Inria attribute order / names
    dtype_fields: list[tuple[str, str]] = [
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("nx", "f4"),
        ("ny", "f4"),
        ("nz", "f4"),
    ]
    dtype_fields += [(f"f_dc_{i}", "f4") for i in range(f_dc_np.shape[1])]
    dtype_fields += [(f"f_rest_{i}", "f4") for i in range(f_rest_np.shape[1])]
    dtype_fields.append(("opacity", "f4"))
    dtype_fields += [(f"scale_{i}", "f4") for i in range(scale_np.shape[1])]
    dtype_fields += [(f"rot_{i}", "f4") for i in range(rot_np.shape[1])]

    vertex = np.empty(n, dtype=dtype_fields)
    vertex[:] = list(
        map(
            tuple,
            np.concatenate(
                (xyz_np, normals, f_dc_np, f_rest_np, opacity_np, scale_np, rot_np),
                axis=1,
            ),
        )
    )

    el = PlyElement.describe(vertex, "vertex")
    PlyData([el]).write(str(out_path))
    print(
        f"[export] Wrote {n} Gaussians (SH degree {max_sh_degree}, "
        f"f_rest={f_rest_np.shape[1]}) -> {out_path}"
    )


def export_attributes(gaussians: gaussian, out_path: Path) -> None:
    """Backward-compatible alias for :func:`save_ply`."""
    save_ply(gaussians, out_path)


def tensor_to_uint8(image: t.Tensor) -> np.ndarray:
    arr = image.detach().float().clamp(0.0, 1.0).cpu().numpy()
    return (arr * 255.0 + 0.5).astype(np.uint8)


def load_gt_image(image_path: Path) -> np.ndarray | None:
    if image_path is None or not Path(image_path).is_file():
        return None
    return np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)


def render_views(
    gaussians: gaussian,
    cameras,
    render_dir: Path,
    device: t.device,
    *,
    max_views: int | None,
    side_by_side: bool,
) -> None:
    render_dir.mkdir(parents=True, exist_ok=True)
    views = cameras.sample_camera_view()
    if max_views is not None:
        views = views[: max(0, max_views)]

    for view in tqdm(views, desc="Rendering"):
        for key in ("R", "T", "fx", "fy", "cx", "cy"):
            view[key] = view[key].to(device)

        gt = load_gt_image(view.get("image_path"))
        if gt is not None:
            height, width = gt.shape[:2]
        else:
            # Fall back to COLMAP principal-point inference of resolution.
            width = int(round(float(view["cx"].item()) * 2.0))
            height = int(round(float(view["cy"].item()) * 2.0))
        view["height"], view["width"] = height, width

        with t.no_grad():
            rendered = rasterize(gaussians, view)

        pred = tensor_to_uint8(rendered)
        name = Path(view.get("image_name") or "view.png").stem

        if side_by_side and gt is not None:
            if gt.shape[:2] != pred.shape[:2]:
                gt = np.asarray(Image.fromarray(gt).resize((pred.shape[1], pred.shape[0])))
            panel = np.concatenate([gt, pred], axis=1)
            Image.fromarray(panel).save(render_dir / f"{name}_compare.png")
        else:
            Image.fromarray(pred).save(render_dir / f"{name}_pred.png")

    print(f"[render] Saved {len(views)} image(s) -> {render_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a checkpoint and export Gaussian attributes"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=Path,
        default=Path("Horse"),
        help="original image folder (used to locate <name>_colmap)",
    )
    parser.add_argument(
        "--colmap-workspace",
        type=Path,
        default=None,
        help="COLMAP workspace (default: <dataset>_colmap)",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=Path,
        default=None,
        help="checkpoint .pt (default: output/gaussians_final.pt or newest epoch_*.pt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("output") / "renders",
        help="directory for rendered images and attribute export",
    )
    parser.add_argument(
        "--max-views",
        type=int,
        default=8,
        help="max camera views to render (default: 8; use 0 for all)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="save prediction only (no GT | pred side-by-side)",
    )
    parser.add_argument(
        "--export-only",
        action="store_true",
        help="only export Gaussian attributes, skip rendering",
    )
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="only render images, skip attribute export",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    checkpoint = resolve_checkpoint(args.checkpoint)
    workspace = resolve_colmap_workspace(args.dataset, args.colmap_workspace)
    sparse = workspace / "undistorted" / "sparse"
    images = workspace / "undistorted" / "images"
    if not sparse.is_dir():
        raise FileNotFoundError(f"Missing sparse model: {sparse}")

    out_dir = args.output.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run] Checkpoint: {checkpoint}")
    print(f"[run] COLMAP:     {workspace}")
    print(f"[run] Device:     {device}")

    gaussians = load_gaussians(checkpoint, device)

    if not args.render_only:
        save_ply(gaussians, out_dir / "point_cloud.ply")

    if not args.export_only:
        _, cameras = parse_colmap(sparse, images)
        max_views = None if args.max_views == 0 else args.max_views
        render_views(
            gaussians,
            cameras,
            out_dir,
            device,
            max_views=max_views,
            side_by_side=not args.no_compare,
        )


if __name__ == "__main__":
    main()
