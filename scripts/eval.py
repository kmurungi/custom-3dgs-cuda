"""Score a trained checkpoint against COLMAP views (mean L1, PSNR, and SSIM)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch as t
from PIL import Image
from torchmetrics.functional.image import structural_similarity_index_measure as ssim
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


def to_nchw(img_hwc: t.Tensor) -> t.Tensor:
    """Convert HWC float image in [0, 1] to NCHW for torchmetrics."""
    return img_hwc.permute(2, 0, 1).unsqueeze(0).clamp(0.0, 1.0)


def psnr_from_mse(mse: t.Tensor) -> float:
    mse_val = float(mse.detach())
    if mse_val <= 0.0:
        return float("inf")
    return float(10.0 * t.log10(t.tensor(1.0 / mse_val)))


def center_crop(img_hwc: t.Tensor, fraction: float) -> t.Tensor:
    """Keep the centered ``fraction`` of height and width (subject-focused metrics)."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"--crop-fraction must be in (0, 1], got {fraction}")
    h, w = int(img_hwc.shape[0]), int(img_hwc.shape[1])
    ch = max(1, int(round(h * fraction)))
    cw = max(1, int(round(w * fraction)))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return img_hwc[y0 : y0 + ch, x0 : x0 + cw]


def score_pair(pred: t.Tensor, gt: t.Tensor) -> tuple[float, float, float]:
    """Return (L1, PSNR, SSIM) for a single HWC pair in [0, 1]."""
    diff = pred - gt
    l1 = float(t.mean(t.abs(diff)))
    psnr = psnr_from_mse(t.mean(diff ** 2))
    ssim_val = float(ssim(to_nchw(pred), to_nchw(gt), data_range=1.0))
    return l1, psnr, ssim_val


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint with mean L1, PSNR, and SSIM on COLMAP views"
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
    parser.add_argument(
        "--crop-fraction",
        type=float,
        default=0.6,
        help="center-crop fraction for subject metrics (default: 0.6 = middle 60%%)",
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
    print(f"[eval] Crop:       center {args.crop_fraction:.0%}")

    gaussians = render_cli.load_gaussians(checkpoint, device)
    _, cameras = parse_colmap(sparse, images)
    views = cameras.sample_camera_view()
    if args.max_views > 0:
        views = views[: args.max_views]
    if not views:
        raise RuntimeError(f"No camera views found under {images}")

    full = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
    crop = {"l1": 0.0, "psnr": 0.0, "ssim": 0.0}
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

        l1, psnr, ssim_val = score_pair(rendered, gt)
        full["l1"] += l1
        full["psnr"] += psnr
        full["ssim"] += ssim_val

        pred_c = center_crop(rendered, args.crop_fraction)
        gt_c = center_crop(gt, args.crop_fraction)
        l1_c, psnr_c, ssim_c = score_pair(pred_c, gt_c)
        crop["l1"] += l1_c
        crop["psnr"] += psnr_c
        crop["ssim"] += ssim_c

        n_scored += 1

    if n_scored == 0:
        raise RuntimeError(
            f"No ground-truth images could be loaded from {images}."
        )

    print(
        f"[eval] full  | Views: {n_scored} | "
        f"L1: {full['l1'] / n_scored:.6f} | "
        f"PSNR: {full['psnr'] / n_scored:.3f} dB | "
        f"SSIM: {full['ssim'] / n_scored:.3f}"
    )
    print(
        f"[eval] crop  | Views: {n_scored} | "
        f"L1: {crop['l1'] / n_scored:.6f} | "
        f"PSNR: {crop['psnr'] / n_scored:.3f} dB | "
        f"SSIM: {crop['ssim'] / n_scored:.3f} "
        f"(center {args.crop_fraction:.0%})"
    )


if __name__ == "__main__":
    main()
