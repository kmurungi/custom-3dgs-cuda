import argparse
from pathlib import Path
import random

import numpy as np
import torch as t
from PIL import Image
from tqdm import tqdm

from gaussian_splatting.rasterizer.rasterize import rasterize
from gaussian_splatting.utils.dataloader import load_colmap
from gaussian_splatting.utils.loss import calculate_loss
from gaussian_splatting.utils.optim import setup_optimizer
from gaussian_splatting.utils.saveweights import save_weights
from gaussian_splatting.utils.adc import (
    accumulate_densification_stats,
    adaptive_density_control,
    ensure_densification_state,
)


def is_refinement_iteration(i: int) -> bool:
    return 500 <= i <= 15000 and i % 100 == 0


def is_opacity_reset_iteration(i: int) -> bool:
    return 500 <= i <= 15000 and i > 0 and i % 3000 == 0


def load_gt_image(image_path, device: t.device) -> t.Tensor:
    """Load an RGB image as float tensor in [0, 1] with shape (H, W, 3)."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return t.from_numpy(arr).to(device)


def enable_gaussian_grads(gaussians) -> None:
    for tensor in (gaussians.mu, gaussians.q, gaussians.s, gaussians.alpha, gaussians.A, gaussians.k_j):
        tensor.requires_grad_(True)


def train(args):
    """
    Training loop Custom Gaussian Splatting
    """
    device = t.device("cuda" if t.cuda.is_available() else "cpu")

    gaussians, cameras = load_colmap(args.dataset)
    # Move trainable params to device and enable grads
    for name in ("mu", "q", "s", "alpha", "A", "k_j"):
        setattr(gaussians, name, getattr(gaussians, name).to(device))
    enable_gaussian_grads(gaussians)

    images = cameras.sample_camera_view()
    for view in images:
        for key in ("R", "T", "fx", "fy", "cx", "cy"):
            view[key] = view[key].to(device)

    learning_rates = {
        "mu": args.lr * 10.0,
        "q": args.lr,
        "s": args.lr,
        "alpha": args.lr,
        "A": args.lr,
        "k_j": args.lr,
    }
    optimizer = setup_optimizer(gaussians, lrs=learning_rates, lr=args.lr)
    ensure_densification_state(gaussians)

    checkpoint_every = max(1, args.epochs // max(1, args.checkpoints))
    total_iterations = args.epochs * len(images)
    iteration = 0
    pbar = tqdm(total=total_iterations, desc="Training 3DGS")

    for epoch in range(args.epochs):
        random.shuffle(images)
        for image in images:
            if image.get("image_path") is None:
                raise RuntimeError(
                    "Camera view is missing image_path. "
                    "Ensure load_colmap/parse_colmap attached image paths."
                )

            gt_img = load_gt_image(image["image_path"], device)
            image["height"], image["width"] = int(gt_img.shape[0]), int(gt_img.shape[1])
            image["rgb"] = gt_img

            optimizer.zero_grad(set_to_none=True)
            rendered_img = rasterize(gaussians, image)
            loss = calculate_loss(gt_img, rendered_img)
            loss.backward()
            accumulate_densification_stats(gaussians)
            optimizer.step()

            if is_refinement_iteration(iteration):
                gaussians, optimizer = adaptive_density_control(
                    gaussians,
                    optimizer,
                    reset_opacity=is_opacity_reset_iteration(iteration),
                )

            iteration += 1
            pbar.update(1)
            pbar.set_postfix(
                loss=float(loss.detach()),
                epoch=epoch,
                N=gaussians.N,
            )

        if epoch % checkpoint_every == 0:
            save_weights(gaussians, Path(args.checkpoint_path) / f"epoch_{epoch:04d}.pt")

    pbar.close()
    save_weights(gaussians, Path(args.output) / "gaussians_final.pt")


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="3D Gaussian Splatting Training & De-Lighting Engine"
    )
    parser.add_argument("-d", "--dataset", type=str, required=True, help="path for loading data")
    parser.add_argument("--delight", action="store_true", help="activate delighting")
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate")
    parser.add_argument("-e", "--epochs", type=int, default=100, help="epochs")
    parser.add_argument("-o", "--output", type=str, default="./output", help="output folder")
    parser.add_argument("-c", "--checkpoints", type=int, default=5, help="number of checkpoints")
    parser.add_argument("-cp", "--checkpoint_path", type=str, default="./checkpoints")
    parser.add_argument("--benchmark", action="store_true", help="save benchmark")
    return parser.parse_args()


def main():
    args = parse_arguments()
    Path(args.output).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_path).mkdir(parents=True, exist_ok=True)
    train(args)


if __name__ == "__main__":
    main()
