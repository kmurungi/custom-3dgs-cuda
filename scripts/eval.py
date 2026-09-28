"""Score a trained checkpoint against COLMAP views (mean L1 and PSNR)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gaussian_splatting.rasterizer.rasterize import rasterize
from gaussian_splatting.utils.dataloader import parse_colmap

import run as render_cli


def load_gt_image(image_path: Path, device: t.device) -> t.Tensor:
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return t.from_numpy(arr).to(device)


def psnr_from_mse(mse: t.Tensor) -> float:
    mse_val = float(mse.detach())
    if mse_val <= 0.0:
        return float("inf")
    return float(10.0 * t.log10(t.tensor(1.0 / mse_val)))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint with mean L1 and PSNR on COLMAP views"
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
        "--max-views",
        type=int,
        default=0,
        help="max camera views to score (default: 0 = all)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not t.cuda.is_available():
        raise RuntimeError(
            "Evaluation requires a CUDA GPU. The rasterizer rejects CPU tensors."
        )
    device = t.device("cuda")

    checkpoint = render_cli.resolve_checkpoint(args.checkpoint)
    workspace = render_cli.resolve_colmap_workspace(args.dataset, args.colmap_workspace)
    sparse = workspace / "undistorted" / "sparse"
    images = workspace / "undistorted" / "images"
    if not sparse.is_dir():
        raise FileNotFoundError(f"Missing sparse model: {sparse}")

    print(f"[eval] Checkpoint: {checkpoint}")
    print(f"[eval] COLMAP:     {workspace}")

    gaussians = render_cli.load_gaussians(checkpoint, device)
    _, cameras = parse_colmap(sparse, images)
    views = cameras.sample_camera_view()
    if args.max_views > 0:
        views = views[: args.max_views]
    if not views:
        raise RuntimeError(f"No camera views found under {images}")

    l1_sum = 0.0
    psnr_sum = 0.0
    n_scored = 0

    for view in tqdm(views, desc="Evaluating"):
        gt_path = view.get("image_path")
        if gt_path is None or not Path(gt_path).is_file():
            continue
        gt = load_gt_image(Path(gt_path), device)
        height, width = int(gt.shape[0]), int(gt.shape[1])
        for key in ("R", "T", "fx", "fy", "cx", "cy"):
            view[key] = view[key].to(device)
        view["height"], view["width"] = height, width

        with t.no_grad():
            rendered = rasterize(gaussians, view).clamp(0.0, 1.0)
        if rendered.shape != gt.shape:
            raise RuntimeError(
                f"Shape mismatch for {gt_path}: "
                f"render {tuple(rendered.shape)} vs gt {tuple(gt.shape)}"
            )

        diff = rendered - gt
        l1_sum += float(t.mean(t.abs(diff)))
        psnr_sum += psnr_from_mse(t.mean(diff ** 2))
        n_scored += 1

    if n_scored == 0:
        raise RuntimeError(
            f"No ground-truth images could be loaded from {images}."
        )

    print(
        f"[eval] Views: {n_scored} | "
        f"L1: {l1_sum / n_scored:.6f} | "
        f"PSNR: {psnr_sum / n_scored:.3f} dB"
    )


if __name__ == "__main__":
    main()
