"""Finite-difference checks for the custom CUDA backward kernels.

Kernels are float32, so ``gradcheck`` uses a larger step than the double-
precision default. Inputs sit in a smooth region: in front of the camera,
quaternions away from zero, colors inside the spherical-harmonic clamp, and
rasterized samples away from tile edges and the alpha cutoff.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch.autograd import gradcheck

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

crc = pytest.importorskip("custom_rasterizer_cuda")

from gaussian_splatting.rasterizer.rasterize import RasterizeFunction


# float32 finite differences: step 1e-3, absolute slack 1e-3, relative 1e-2.
_GRADCHECK = dict(
    eps=1e-3,
    atol=1e-3,
    rtol=1e-2,
    nondet_tol=1e-4,
    check_batched_grad=False,
    raise_exception=True,
)


@pytest.fixture
def device() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for custom rasterizer gradcheck")
    return torch.device("cuda")


def _leaf(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().clone().requires_grad_(True)


def _check(func, inputs: tuple[torch.Tensor, ...]) -> None:
    assert gradcheck(func, inputs, **_GRADCHECK)


def test_projection_backward_matches_finite_differences(device: torch.device) -> None:
    """∂(mean₂D, Σ₂) / ∂(μ, q, log-scale) against the projection backward kernel."""
    rotation = torch.eye(3, device=device, dtype=torch.float32)
    translation = torch.zeros(3, device=device, dtype=torch.float32)
    fx = fy = 200.0
    cx = cy = 100.0

    mu = _leaf(
        torch.tensor(
            [[0.30, -0.20, 3.5], [-0.25, 0.15, 5.0]],
            device=device,
            dtype=torch.float32,
        )
    )
    quat = torch.tensor([1.0, 0.15, -0.05, 0.02], device=device, dtype=torch.float32)
    quat = quat / quat.norm()
    q = _leaf(quat.expand(2, 4).clone())
    s = _leaf(
        torch.tensor(
            [[-2.2, -2.0, -1.8], [-2.5, -2.1, -2.3]],
            device=device,
            dtype=torch.float32,
        )
    )

    class Project(torch.autograd.Function):
        @staticmethod
        def forward(ctx, means, quats, scales):
            n = means.shape[0]
            means = means.contiguous()
            quats = quats.contiguous()
            scales = scales.contiguous()
            mean_2d, cov_2d, _depths = crc.project(
                n, means, quats, scales, rotation, translation, fx, fy, cx, cy
            )
            # Depth is a sort key. The projection backward does not return ∂L/∂depth.
            ctx.save_for_backward(means, quats, scales)
            return mean_2d, cov_2d

        @staticmethod
        def backward(ctx, grad_mean, grad_cov):
            means, quats, scales = ctx.saved_tensors
            return crc.backwards_projection(
                grad_mean.contiguous(),
                grad_cov.contiguous(),
                means,
                quats,
                scales,
                rotation,
                translation,
                fx,
                fy,
                cx,
                cy,
            )

    mean_2d, cov_2d = Project.apply(mu, q, s)
    assert torch.isfinite(mean_2d).all() and torch.isfinite(cov_2d).all()
    assert float(mean_2d.detach().abs().sum()) > 1.0
    assert float(cov_2d.detach().abs().sum()) > 0.1
    _check(Project.apply, (mu, q, s))


def test_sh_backward_matches_finite_differences(device: torch.device) -> None:
    """∂color / ∂(μ, albedo, kⱼ) against the spherical-harmonic backward kernel."""
    camera = torch.zeros(3, device=device, dtype=torch.float32)
    mu = _leaf(
        torch.tensor(
            [[0.40, -0.30, 2.5], [-0.20, 0.50, 3.0]],
            device=device,
            dtype=torch.float32,
        )
    )
    albedo = _leaf(torch.zeros(2, 3, device=device, dtype=torch.float32))
    k_j = _leaf(
        torch.linspace(-0.005, 0.005, 2 * 15 * 3, device=device, dtype=torch.float32).reshape(
            2, 15, 3
        )
    )

    class SphericalHarmonics(torch.autograd.Function):
        @staticmethod
        def forward(ctx, means, alb, coeff):
            n = means.shape[0]
            means = means.contiguous()
            alb = alb.contiguous()
            coeff = coeff.contiguous()
            colors = crc.SH(n, camera, means, alb, coeff)
            ctx.save_for_backward(means, alb, coeff)
            return colors

        @staticmethod
        def backward(ctx, grad_colors):
            means, alb, coeff = ctx.saved_tensors
            albedo_grad, coeff_grad, mu_grad = crc.backwards_SH(
                grad_colors.contiguous(),
                means,
                alb,
                coeff,
                camera,
            )
            return mu_grad, albedo_grad, coeff_grad

    colors = SphericalHarmonics.apply(mu, albedo, k_j)
    assert torch.isfinite(colors).all()
    # Stay inside the forward clamp so the backward mask is the identity.
    assert float(colors.detach().min()) > 0.05
    assert float(colors.detach().max()) < 0.95
    _check(SphericalHarmonics.apply, (mu, albedo, k_j))


def test_rasterize_backward_matches_finite_differences(device: torch.device) -> None:
    """∂image / ∂(mean₂D, Σ₂, color, α) on pixels away from the alpha cutoff."""
    height = width = 16
    # One isotropic Gaussian at the center of the single 16×16 tile.
    # 3σ radius stays inside the tile under a 1e-3 finite-difference step.
    mean_2d = _leaf(torch.tensor([[8.0, 8.0]], device=device, dtype=torch.float32))
    cov_2d = _leaf(torch.tensor([[4.0, 0.0, 4.0]], device=device, dtype=torch.float32))
    colors = _leaf(torch.tensor([[0.25, 0.45, 0.70]], device=device, dtype=torch.float32))
    alpha = _leaf(torch.tensor([[0.6]], device=device, dtype=torch.float32))
    depths = torch.tensor([2.0], device=device, dtype=torch.float32)

    class Raster(torch.autograd.Function):
        @staticmethod
        def forward(ctx, means, cov, rgb, opacity):
            n = means.shape[0]
            means = means.contiguous()
            cov = cov.contiguous()
            rgb = rgb.contiguous()
            opacity = opacity.contiguous()
            image, sorted_ids, tile_ranges, final_t, n_contrib = crc.rasterize(
                n, depths, means, cov, rgb, opacity, height, width
            )
            ctx.save_for_backward(
                means, cov, rgb, opacity, sorted_ids, tile_ranges, final_t, n_contrib
            )
            ctx.num_gaussians = n
            return image

        @staticmethod
        def backward(ctx, grad_image):
            means, cov, rgb, opacity, sorted_ids, tile_ranges, final_t, n_contrib = (
                ctx.saved_tensors
            )
            return crc.backwards_rasterization(
                grad_image.contiguous(),
                means,
                cov,
                rgb,
                opacity,
                sorted_ids,
                tile_ranges,
                final_t,
                n_contrib,
                ctx.num_gaussians,
                height,
                width,
            )

    def raster_center(means, cov, rgb, opacity):
        image = Raster.apply(means, cov, rgb, opacity)
        return image[6:10, 6:10].sum()

    center = raster_center(mean_2d, cov_2d, colors, alpha)
    assert torch.isfinite(center)
    assert float(center.detach()) > 1e-2
    _check(raster_center, (mean_2d, cov_2d, colors, alpha))


def test_full_pipeline_backward_matches_finite_differences(device: torch.device) -> None:
    """Training backward (project + SH + blend) on a one-Gaussian crop."""
    height = width = 16
    rotation = torch.eye(3, device=device, dtype=torch.float32)
    translation = torch.zeros(3, device=device, dtype=torch.float32)
    # Projects near pixel (8.2, 7.84), the middle of the only tile.
    mu = _leaf(torch.tensor([[0.05, -0.04, 4.0]], device=device, dtype=torch.float32))
    quat = torch.tensor([1.0, 0.1, -0.05, 0.02], device=device, dtype=torch.float32)
    quat = quat / quat.norm()
    q = _leaf(quat.reshape(1, 4))
    s = _leaf(torch.full((1, 3), -2.0, device=device, dtype=torch.float32))
    alpha = _leaf(torch.tensor([[0.5]], device=device, dtype=torch.float32))
    albedo = _leaf(torch.zeros(1, 3, device=device, dtype=torch.float32))
    k_j = _leaf(
        torch.linspace(-0.005, 0.005, 15 * 3, device=device, dtype=torch.float32).reshape(
            1, 15, 3
        )
    )

    def pipeline(means, quats, scales, opacity, alb, coeff):
        image = RasterizeFunction.apply(
            means,
            quats,
            scales,
            opacity,
            alb,
            coeff,
            rotation,
            translation,
            16.0,
            16.0,
            8.0,
            8.0,
            height,
            width,
        )
        return image[7:9, 7:9].sum()

    center = pipeline(mu, q, s, alpha, albedo, k_j)
    assert torch.isfinite(center)
    assert float(center.detach()) > 1e-3
    _check(pipeline, (mu, q, s, alpha, albedo, k_j))
