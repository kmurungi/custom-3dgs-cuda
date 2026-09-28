"""Time the CUDA rasterizer forward and backward passes.

Warm up for 100 iterations, then time each measured iteration with
``torch.cuda.Event(enable_timing=True)``. Forward latency is one rendered
frame. Rendering throughput is ``1000 / forward_ms``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch as t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gaussian_splatting.rasterizer.rasterize import rasterize
from gaussian_splatting.scene.gaussian import gaussian


RESOLUTIONS = ((1280, 720), (1920, 1080))
PARAM_NAMES = ("mu", "q", "s", "alpha", "A", "k_j")


def _params(gaussians: gaussian) -> tuple[t.Tensor, ...]:
    return tuple(getattr(gaussians, name) for name in PARAM_NAMES)


def make_gaussians(n: int, device: t.device, seed: int) -> gaussian:
    """Synthetic cloud in front of an identity camera, log-scale ~0.05."""
    gen = t.Generator(device="cpu")
    gen.manual_seed(seed)

    z = t.empty(n, 1).uniform_(2.0, 6.0, generator=gen)
    x = t.empty(n, 1).uniform_(-0.35, 0.35, generator=gen) * z
    y = t.empty(n, 1).uniform_(-0.25, 0.25, generator=gen) * z

    gaussians = gaussian(n)
    gaussians.mu = t.cat((x, y, z), dim=1).to(device=device, dtype=t.float32)
    gaussians.q = t.zeros(n, 4, device=device, dtype=t.float32)
    gaussians.q[:, 0] = 1.0
    gaussians.s = t.full((n, 3), -3.0, device=device, dtype=t.float32)
    gaussians.alpha = t.zeros(n, 1, device=device, dtype=t.float32)
    gaussians.A = t.zeros(n, 3, device=device, dtype=t.float32)
    gaussians.k_j = t.zeros(n, 15, 3, device=device, dtype=t.float32)
    for tensor in _params(gaussians):
        tensor.requires_grad_(True)
    return gaussians


def make_view(width: int, height: int, device: t.device) -> dict:
    """Pinhole camera at the origin looking down +Z. Focal length equals width."""
    return {
        "R": t.eye(3, device=device, dtype=t.float32),
        "T": t.zeros(3, device=device, dtype=t.float32),
        "fx": float(width),
        "fy": float(width),
        "cx": width * 0.5,
        "cy": height * 0.5,
        "height": height,
        "width": width,
    }


def _clear_grads(gaussians: gaussian) -> None:
    for tensor in _params(gaussians):
        tensor.grad = None


def warmup(gaussians: gaussian, view: dict, steps: int) -> None:
    for _ in range(steps):
        _clear_grads(gaussians)
        image = rasterize(gaussians, view)
        image.sum().backward()
    t.cuda.synchronize()


def measure(gaussians: gaussian, view: dict, steps: int) -> tuple[float, float]:
    """Mean forward and backward GPU time in milliseconds."""
    forward_events: list[tuple[t.cuda.Event, t.cuda.Event]] = []
    backward_events: list[tuple[t.cuda.Event, t.cuda.Event]] = []

    for _ in range(steps):
        _clear_grads(gaussians)
        start_f = t.cuda.Event(enable_timing=True)
        end_f = t.cuda.Event(enable_timing=True)
        start_b = t.cuda.Event(enable_timing=True)
        end_b = t.cuda.Event(enable_timing=True)

        start_f.record()
        image = rasterize(gaussians, view)
        end_f.record()

        loss = image.sum()
        start_b.record()
        loss.backward()
        end_b.record()

        forward_events.append((start_f, end_f))
        backward_events.append((start_b, end_b))

    t.cuda.synchronize()
    forward_ms = [start.elapsed_time(end) for start, end in forward_events]
    backward_ms = [start.elapsed_time(end) for start, end in backward_events]
    return (
        sum(forward_ms) / len(forward_ms),
        sum(backward_ms) / len(backward_ms),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark CUDA rasterizer forward/backward latency and FPS"
    )
    parser.add_argument(
        "-n",
        "--gaussians",
        type=int,
        default=100_000,
        help="number of synthetic Gaussians (default: 100000)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=100,
        help="untimed warmup iterations per resolution (default: 100)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=100,
        help="timed iterations per resolution (default: 100)",
    )
    parser.add_argument("--seed", type=int, default=0, help="CPU seed for the point cloud")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.gaussians < 1:
        raise ValueError("--gaussians must be positive")
    if args.warmup < 1 or args.iters < 1:
        raise ValueError("--warmup and --iters must be positive")
    if not t.cuda.is_available():
        raise RuntimeError("Benchmark requires a CUDA GPU.")

    device = t.device("cuda")
    gaussians = make_gaussians(args.gaussians, device, args.seed)
    print(
        f"[bench] Device: {t.cuda.get_device_name(device)} | "
        f"Gaussians: {args.gaussians} | warmup: {args.warmup} | iters: {args.iters}",
        flush=True,
    )
    print(
        f"{'resolution':<14}{'forward_ms':>14}{'backward_ms':>14}{'fps':>10}",
        flush=True,
    )

    for width, height in RESOLUTIONS:
        view = make_view(width, height, device)
        warmup(gaussians, view, args.warmup)
        forward_ms, backward_ms = measure(gaussians, view, args.iters)
        fps = 1000.0 / forward_ms
        label = f"{width}x{height}"
        print(
            f"{label:<14}{forward_ms:14.3f}{backward_ms:14.3f}{fps:10.1f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
