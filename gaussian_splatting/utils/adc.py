from __future__ import annotations
import torch as t
from gaussian_splatting.scene.gaussian import gaussian

def scene_extent_from_means(mu: t.Tensor) -> float:
    """Axis-aligned bounding-box diagonal of Gaussian means"""
    mins = mu.min(dim=0).values
    maxs = mu.max(dim=0).values
    extent = float((maxs - mins).norm().item())
    return max(extent, 1e-3)


def ensure_densification_state(gaussians: gaussian) -> None:
    """Allocate / resize per-Gaussian densification accumulators"""
    n = gaussians.mu.shape[0]
    device = gaussians.mu.device
    dtype = gaussians.mu.dtype

    if (
        not hasattr(gaussians, "xyz_grad_accum")
        or gaussians.xyz_grad_accum.shape[0] != n
    ):
        gaussians.xyz_grad_accum = t.zeros(n, 1, device=device, dtype=dtype)
        gaussians.denom = t.zeros(n, 1, device=device, dtype=dtype)


def accumulate_densification_stats(gaussians: gaussian) -> None:
    """
    Accumulate mean positional gradient magnitude for densification
    """
    if gaussians.mu.grad is None:
        return

    ensure_densification_state(gaussians)
    grad_norm = t.norm(gaussians.mu.grad.detach(), dim=-1, keepdim=True)
    gaussians.xyz_grad_accum = gaussians.xyz_grad_accum + grad_norm
    gaussians.denom = gaussians.denom + 1.0


def _activated_scales(s: t.Tensor) -> t.Tensor:
    """Match CUDA projection: scales are stored in log-space"""
    return t.exp(s)


def _build_rotation(q: t.Tensor) -> t.Tensor:
    """Quaternion (w, x, y, z) -> rotation matrices (N, 3, 3)"""
    q = t.nn.functional.normalize(q, dim=-1)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r00 = 1 - 2 * (y * y + z * z)
    r01 = 2 * (x * y - w * z)
    r02 = 2 * (x * z + w * y)
    r10 = 2 * (x * y + w * z)
    r11 = 1 - 2 * (x * x + z * z)
    r12 = 2 * (y * z - w * x)
    r20 = 2 * (x * z - w * y)
    r21 = 2 * (y * z + w * x)
    r22 = 1 - 2 * (x * x + y * y)
    return t.stack(
        [
            t.stack([r00, r01, r02], dim=-1),
            t.stack([r10, r11, r12], dim=-1),
            t.stack([r20, r21, r22], dim=-1),
        ],
        dim=1,
    )


def _prune_optimizer(optimizer: t.optim.Optimizer, keep_mask: t.Tensor) -> None:
    """Keep optimizer params / Adam state rows selected by keep_mask"""
    for group in optimizer.param_groups:
        stored = optimizer.state.get(group["params"][0], None)
        old = group["params"][0]
        new = old[keep_mask].detach().requires_grad_(True)
        group["params"][0] = new

        if stored is not None:
            new_state = {}
            for key, value in stored.items():
                if t.is_tensor(value) and value.shape[:1] == old.shape[:1]:
                    new_state[key] = value[keep_mask]
                else:
                    new_state[key] = value
            del optimizer.state[old]
            optimizer.state[new] = new_state


def _cat_optimizer(
    optimizer: t.optim.Optimizer, extension: dict[str, t.Tensor]
) -> None:
    """Append new Gaussian rows and zero-init Adam moments for them"""
    name_to_ext = extension
    for group in optimizer.param_groups:
        name = group["name"]
        # Map optimizer display names back to attribute keys
        key = {
            "mu": "mu",
            "q": "q",
            "Scale": "s",
            "Opacity": "alpha",
            "Albedo Coefficient": "A",
            "Illumination Coefficients": "k_j",
        }.get(name, name)

        ext = name_to_ext[key]
        stored = optimizer.state.get(group["params"][0], None)
        old = group["params"][0]
        new = t.cat([old, ext], dim=0).detach().requires_grad_(True)
        group["params"][0] = new

        if stored is not None:
            new_state = {}
            for key_s, value in stored.items():
                if t.is_tensor(value) and value.shape[:1] == old.shape[:1]:
                    pad = t.zeros(
                        (ext.shape[0], *value.shape[1:]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    new_state[key_s] = t.cat([value, pad], dim=0)
                else:
                    new_state[key_s] = value
            del optimizer.state[old]
            optimizer.state[new] = new_state


def _sync_gaussians_from_optimizer(
    gaussians: gaussian, optimizer: t.optim.Optimizer
) -> None:
    """Point gaussian attributes at the live optimizer parameter tensors"""
    name_map = {
        "mu": "mu",
        "q": "q",
        "Scale": "s",
        "Opacity": "alpha",
        "Albedo Coefficient": "A",
        "Illumination Coefficients": "k_j",
    }
    for group in optimizer.param_groups:
        attr = name_map[group["name"]]
        setattr(gaussians, attr, group["params"][0])
    gaussians.N = int(gaussians.mu.shape[0])


def _extend_densification_state(gaussians: gaussian, n_new: int) -> None:
    if n_new <= 0:
        return
    ensure_densification_state(gaussians)
    device = gaussians.mu.device
    dtype = gaussians.mu.dtype
    gaussians.xyz_grad_accum = t.cat(
        [gaussians.xyz_grad_accum, t.zeros(n_new, 1, device=device, dtype=dtype)],
        dim=0,
    )
    gaussians.denom = t.cat(
        [gaussians.denom, t.zeros(n_new, 1, device=device, dtype=dtype)],
        dim=0,
    )


def _prune_points(
    gaussians: gaussian, optimizer: t.optim.Optimizer, keep_mask: t.Tensor
) -> None:
    _prune_optimizer(optimizer, keep_mask)
    _sync_gaussians_from_optimizer(gaussians, optimizer)

    if hasattr(gaussians, "xyz_grad_accum") and gaussians.xyz_grad_accum.shape[0] == keep_mask.shape[0]:
        gaussians.xyz_grad_accum = gaussians.xyz_grad_accum[keep_mask]
        gaussians.denom = gaussians.denom[keep_mask]
    else:
        ensure_densification_state(gaussians)


def _densify_and_clone(
    gaussians: gaussian,
    optimizer: t.optim.Optimizer,
    grads: t.Tensor,
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
) -> None:
    """Duplicate under-reconstructed small Gaussians and nudge them along the grad"""
    scales = _activated_scales(gaussians.s)
    selected = (grads.squeeze(-1) >= grad_threshold) & (
        scales.max(dim=1).values <= percent_dense * scene_extent
    )
    if not selected.any():
        return

    extension = {
        "mu": gaussians.mu[selected],
        "q": gaussians.q[selected],
        "s": gaussians.s[selected],
        "alpha": gaussians.alpha[selected],
        "A": gaussians.A[selected],
        "k_j": gaussians.k_j[selected],
    }
    _cat_optimizer(optimizer, extension)
    _sync_gaussians_from_optimizer(gaussians, optimizer)
    _extend_densification_state(gaussians, int(selected.sum().item()))


def _densify_and_split(
    gaussians: gaussian,
    optimizer: t.optim.Optimizer,
    grads: t.Tensor,
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
    n_split: int = 2,
    scale_factor: float = 1.6,
) -> None:
    """Replace oversized high-gradient Gaussians with smaller children"""
    n_init = gaussians.mu.shape[0]
    # Newly cloned rows have no densification history — pad with zeros
    padded = t.zeros(n_init, device=grads.device, dtype=grads.dtype)
    padded[: grads.shape[0]] = grads.squeeze(-1)

    scales = _activated_scales(gaussians.s)
    selected = (padded >= grad_threshold) & (
        scales.max(dim=1).values > percent_dense * scene_extent
    )
    if not selected.any():
        return

    stds = scales[selected].repeat(n_split, 1)
    means = t.zeros_like(stds)
    samples = t.normal(mean=means, std=stds)
    rots = _build_rotation(gaussians.q[selected]).repeat(n_split, 1, 1)
    new_mu = t.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.mu[
        selected
    ].repeat(n_split, 1)
    new_s = t.log(_activated_scales(gaussians.s[selected]).repeat(n_split, 1) / scale_factor)
    new_q = gaussians.q[selected].repeat(n_split, 1)
    new_alpha = gaussians.alpha[selected].repeat(n_split, 1)
    new_A = gaussians.A[selected].repeat(n_split, 1)
    new_k_j = gaussians.k_j[selected].repeat(n_split, 1, 1)

    extension = {
        "mu": new_mu,
        "q": new_q,
        "s": new_s,
        "alpha": new_alpha,
        "A": new_A,
        "k_j": new_k_j,
    }
    _cat_optimizer(optimizer, extension)
    _sync_gaussians_from_optimizer(gaussians, optimizer)

    n_new = n_split * int(selected.sum().item())
    _extend_densification_state(gaussians, n_new)

    # Drop the parents that were split 
    prune_mask = t.cat(
        [
            selected,
            t.zeros(n_new, dtype=t.bool, device=selected.device),
        ],
        dim=0,
    )
    _prune_points(gaussians, optimizer, ~prune_mask)


def _reset_opacity(gaussians: gaussian, optimizer: t.optim.Optimizer, value: float = 0.01) -> None:
    """Cap opacities (used periodically in the original 3DGS schedule)"""
    for group in optimizer.param_groups:
        if group["name"] != "Opacity":
            continue
        old = group["params"][0]
        new = t.clamp_max(old, value).detach().requires_grad_(True)
        stored = optimizer.state.get(old, None)
        group["params"][0] = new
        if stored is not None:
            del optimizer.state[old]
            optimizer.state[new] = {
                k: (t.zeros_like(v) if t.is_tensor(v) and v.shape == old.shape else v)
                for k, v in stored.items()
            }
        gaussians.alpha = new
        break


def adaptive_density_control(
    gaussians: gaussian,
    optimizer: t.optim.Optimizer,
    *,
    grad_threshold: float = 0.0002,
    percent_dense: float = 0.01,
    min_opacity: float = 0.005,
    max_screen_size: float | None = None,
    scene_extent: float | None = None,
    reset_opacity: bool = False,
) -> tuple[gaussian, t.optim.Optimizer]:
    """
    Densify under-reconstructed regions and prune transparent / oversized Gaussians
    """
    ensure_densification_state(gaussians)

    if scene_extent is None:
        scene_extent = scene_extent_from_means(gaussians.mu.detach())

    denom = gaussians.denom.clamp_min(1.0)
    grads = gaussians.xyz_grad_accum / denom
    grads = t.nan_to_num(grads, nan=0.0)

    _densify_and_clone(
        gaussians, optimizer, grads, grad_threshold, scene_extent, percent_dense
    )
    _densify_and_split(
        gaussians, optimizer, grads, grad_threshold, scene_extent, percent_dense
    )

    prune = (gaussians.alpha.squeeze(-1) < min_opacity)
    scales = _activated_scales(gaussians.s)
    # World-space size prune 
    prune = prune | (scales.max(dim=1).values > 0.1 * scene_extent)
    if max_screen_size is not None and hasattr(gaussians, "max_radii2D"):
        prune = prune | (gaussians.max_radii2D > max_screen_size)

    if prune.any() and (~prune).any():
        _prune_points(gaussians, optimizer, ~prune)
    elif prune.all():
        # keep the most opaque gausisans
        keep_n = max(1, gaussians.N // 10)
        keep_idx = t.topk(gaussians.alpha.squeeze(-1), k=keep_n).indices
        keep = t.zeros(gaussians.N, dtype=t.bool, device=gaussians.mu.device)
        keep[keep_idx] = True
        _prune_points(gaussians, optimizer, keep)

    if reset_opacity:
        _reset_opacity(gaussians, optimizer)

    # Reset densification stats for the next interval.
    n = gaussians.mu.shape[0]
    gaussians.xyz_grad_accum = t.zeros(n, 1, device=gaussians.mu.device, dtype=gaussians.mu.dtype)
    gaussians.denom = t.zeros(n, 1, device=gaussians.mu.device, dtype=gaussians.mu.dtype)

    return gaussians, optimizer
