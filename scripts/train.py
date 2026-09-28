import argparse
from pathlib import Path
import math
import random
import sys

# Allow `python scripts/train.py` without installing the Python package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch as t
from PIL import Image
from tqdm import tqdm

from gaussian_splatting.rasterizer.rasterize import RasterizeFunction, rasterize
from gaussian_splatting.utils.dataloader import load_colmap, parse_colmap
from gaussian_splatting.utils.optim import (
    get_mu_lr,
    get_positional_lr_scheduler,
    get_scale_lr,
    setup_optimizer,
)
from gaussian_splatting.utils.saveweights import save_weights
from gaussian_splatting.utils.adc import (
    accumulate_densification_stats,
    adaptive_density_control,
    clamp_log_scales,
    ensure_densification_state,
    scene_extent_from_cameras,
    update_max_radii2d,
)


def load_gt_image(image_path, device: t.device) -> t.Tensor:
    """Load an RGB image as float tensor in [0, 1] with shape (H, W, 3)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return t.from_numpy(arr).to(device)


def enable_gaussian_grads(gaussians) -> None:
    for tensor in (gaussians.mu, gaussians.q, gaussians.s, gaussians.alpha, gaussians.A, gaussians.k_j):
        tensor.requires_grad_(True)


def _workspace_ready(workspace: Path) -> bool:
    return (workspace / "undistorted" / "sparse").is_dir() and (
        workspace / "undistorted" / "images"
    ).is_dir()


def load_scene(dataset: str | Path, colmap_workspace: str | Path | None = None):
    """Load gaussians/cameras from a precomputed COLMAP workspace, or run COLMAP."""
    dataset_path = Path(dataset).expanduser()
    workspace = (
        Path(colmap_workspace).expanduser()
        if colmap_workspace is not None
        else dataset_path.parent / f"{dataset_path.name}_colmap"
    )

    if colmap_workspace is not None or _workspace_ready(workspace):
        if not _workspace_ready(workspace):
            raise FileNotFoundError(
                f"COLMAP workspace incomplete: {workspace}. "
                "Expected undistorted/sparse and undistorted/images."
            )
        sparse = workspace / "undistorted" / "sparse"
        images = workspace / "undistorted" / "images"
        print(f"[train] Using precomputed COLMAP: {workspace}", flush=True)
        return parse_colmap(sparse, images)

    print(f"[train] No complete COLMAP workspace at {workspace}; running COLMAP.", flush=True)
    return load_colmap(dataset_path)


def train(args):
    """
    Training loop Custom Gaussian Splatting
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    gaussians, cameras = load_scene(args.dataset, args.colmap_workspace)
    # Move trainable params to device and enable grads
    for name in ("mu", "q", "s", "alpha", "A", "k_j"):
        setattr(gaussians, name, getattr(gaussians, name).to(device))
    cameras.R = cameras.R.to(device)
    cameras.t = cameras.t.to(device)
    enable_gaussian_grads(gaussians)

    images = cameras.sample_camera_view()
    for view in images:
        for key in ("R", "T", "fx", "fy", "cx", "cy"):
            view[key] = view[key].to(device)
        if view.get("image_path") is None:
            raise RuntimeError(
                "Camera view is missing image_path. "
                "Ensure load_colmap/parse_colmap attached image paths."
            )
        view["rgb"] = load_gt_image(view["image_path"], device)
        view["height"], view["width"] = int(view["rgb"].shape[0]), int(view["rgb"].shape[1])

    # Fixed scene extent from camera centers (never recomputed from active points).
    fixed_scene_extent = scene_extent_from_cameras(cameras)
    print(
        f"[train] Gaussians: {gaussians.N} | views: {len(images)} | "
        f"scene extent: {fixed_scene_extent:.4f}",
        flush=True,
    )

    lr_init = 1.6e-4 * fixed_scene_extent
    lr_final = 1.6e-6 * fixed_scene_extent
    scale_lr_init = 5e-3
    scale_lr_final = 5e-5
    learning_rates = {
        "mu": lr_init,
        "q": 1e-3,
        "s": scale_lr_init,
        "alpha": 5e-2,
        "A": 2.5e-3,
        "k_j": 6.25e-4,
    }
    optimizer = setup_optimizer(gaussians, lrs=learning_rates)
    ensure_densification_state(gaussians)
    clamp_log_scales(gaussians, fixed_scene_extent)

    # Typical image width for screen-space prune threshold.
    image_width = max(int(v["width"]) for v in images)
    max_screen_size = image_width * 0.1

    checkpoint_every = max(1, args.epochs // max(1, args.checkpoints))
    TOTAL_STEPS = args.epochs * len(images)
    iteration = 0
    stats_window = {"cloned": 0, "split": 0, "pruned": 0}
    pbar = tqdm(total=TOTAL_STEPS, desc="Training 3DGS")
    densify_until = int(0.7 * TOTAL_STEPS)
    mu_scheduler = get_positional_lr_scheduler(
        optimizer,
        lr_init=lr_init,
        lr_final=lr_final,
        scale_lr_init=scale_lr_init,
        scale_lr_final=scale_lr_final,
        lr_delay_mult=0.01,
        max_steps=TOTAL_STEPS,
    )

    for epoch in range(args.epochs):
        random.shuffle(images)
        for image in images:
            gt_img = image["rgb"]

            optimizer.zero_grad(set_to_none=True)
            rendered_img = rasterize(gaussians, image)
            update_max_radii2d(gaussians, RasterizeFunction.last_radii2d)
            loss = t.mean(t.abs(gt_img - rendered_img))  # plain L1 for first training loop
            loss.backward()
            mean2d_grad = RasterizeFunction.last_mean2d_grad
            img_h, img_w = int(image["height"]), int(image["width"])

            # Stop structural edits past 70% of total steps.
            allow_densify = iteration < densify_until
            # Opacity reset every 3,000 steps, only while densifying.
            do_reset = (iteration % 3000 == 0) and (iteration > 0) and allow_densify

            accumulate_densification_stats(
                gaussians,
                mean2d_grad=mean2d_grad,
                width=img_w,
                height=img_h,
            )
            optimizer.step()
            mu_scheduler.step()
            # Opacity is logits: keep activated opacity in ~[1e-4, 0.99]
            gaussians.alpha.data.clamp_(
                math.log(1e-4 / (1.0 - 1e-4)),
                math.log(0.99 / (1.0 - 0.99)),
            )
            clamp_log_scales(gaussians, fixed_scene_extent)

            if iteration % 100 == 0:
                gaussians, optimizer, stats = adaptive_density_control(
                    gaussians,
                    optimizer,
                    grad_threshold=0.00005,
                    percent_dense=0.01,
                    min_opacity=0.01,
                    scene_extent=fixed_scene_extent,
                    max_screen_size=max_screen_size,
                    reset_opacity=do_reset,
                    max_gaussians=800_000,
                    allow_densify=allow_densify,
                )
                for key in stats_window:
                    stats_window[key] += stats[key]

            iteration += 1
            current_mu_lr = get_mu_lr(optimizer)
            current_scale_lr = get_scale_lr(optimizer)
            if iteration % 500 == 0:
                print(
                    f"[Step {iteration}] Active Gaussians: {gaussians.N} | "
                    f"Cloned: {stats_window['cloned']} | "
                    f"Split: {stats_window['split']} | "
                    f"Pruned: {stats_window['pruned']} | "
                    f"mu_lr={current_mu_lr:.3e} | scale_lr={current_scale_lr:.3e}",
                    flush=True,
                )
                stats_window = {"cloned": 0, "split": 0, "pruned": 0}

            pbar.update(1)
            pbar.set_postfix(
                loss=float(loss.detach()),
                epoch=epoch,
                N=gaussians.N,
                mu_lr=f"{current_mu_lr:.2e}",
                s_lr=f"{current_scale_lr:.2e}",
            )

        if epoch % checkpoint_every == 0:
            save_weights(gaussians, Path(args.checkpoint_path) / f"epoch_{epoch:04d}.pt")

    pbar.close()
    save_weights(gaussians, Path(args.output) / "gaussians_final.pt")


def parse_arguments():
    parser = argparse.ArgumentParser(description="3D Gaussian Splatting training")
    parser.add_argument("-d", "--dataset", type=str, required=True, help="image directory")
    parser.add_argument(
        "--colmap-workspace",
        type=str,
        default=None,
        help="precomputed COLMAP workspace (default: <dataset>_colmap when that workspace is complete)",
    )
    parser.add_argument("-e", "--epochs", type=int, default=100, help="epochs")
    parser.add_argument("-o", "--output", type=str, default="output", help="folder for gaussians_final.pt")
    parser.add_argument("-c", "--checkpoints", type=int, default=5, help="number of checkpoints")
    parser.add_argument(
        "-cp",
        "--checkpoint_path",
        type=str,
        default="checkpoints",
        help="folder for epoch checkpoints",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
