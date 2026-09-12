from __future__ import annotations

import math
from typing import Any

import torch as t
from torch.nn import Parameter

from gaussian_splatting.scene.camera import camera as CameraBatch
from gaussian_splatting.scene.gaussian import gaussian

_GROUP_TO_ATTR = {
    "mu": "mu",
    "q": "q",
    "Scale": "s",
    "Opacity": "alpha",
    "Albedo Coefficient": "A",
    "Illumination Coefficients": "k_j",
}


def scene_extent_from_means(mu: t.Tensor) -> float:
    mins = mu.min(dim=0).values
    maxs = mu.max(dim=0).values
    extent = float((maxs - mins).norm().item())
    return max(extent, 1e-3)


def scene_extent_from_cameras(cameras: CameraBatch) -> float:
    centers = -t.bmm(cameras.R.transpose(1, 2), cameras.t.unsqueeze(-1)).squeeze(-1)
    scene_center = centers.mean(dim=0)
    extent = float((centers - scene_center).norm(dim=-1).max().item()) * 1.1
    return max(extent, 1e-3)


def ensure_densification_state(gaussians: gaussian) -> None:
    """Allocate densify buffers, or resize them to match ``mu`` (preserve prefix)."""
    n = int(gaussians.mu.shape[0])
    device = gaussians.mu.device
    dtype = gaussians.mu.dtype

    def _fit(buf: t.Tensor | None, rest: tuple[int, ...]) -> t.Tensor:
        if buf is None:
            return t.zeros((n, *rest), device=device, dtype=dtype)
        if buf.shape[0] == n:
            return buf
        out = t.zeros((n, *rest), device=device, dtype=dtype)
        m = min(int(buf.shape[0]), n)
        if m > 0:
            out[:m] = buf[:m]
        return out

    gaussians.xyz_grad_accum = _fit(getattr(gaussians, "xyz_grad_accum", None), (1,))
    gaussians.denom = _fit(getattr(gaussians, "denom", None), (1,))
    gaussians.max_radii2D = _fit(getattr(gaussians, "max_radii2D", None), ()).view(n)


def accumulate_densification_stats(
    gaussians: gaussian,
    mean2d_grad: t.Tensor | None = None,
    *,
    width: int | None = None,
    height: int | None = None,
) -> None:
    if mean2d_grad is None or mean2d_grad.shape[0] != gaussians.mu.shape[0]:
        return

    ensure_densification_state(gaussians)
    grad_xy = mean2d_grad[:, :2].detach()

    if width is not None and height is not None and width > 0 and height > 0:
        scale = t.tensor([width / 2.0, height / 2.0], device=grad_xy.device, dtype=grad_xy.dtype)
        grad_xy = grad_xy * scale

    grad_norm = t.norm(grad_xy, dim=-1, keepdim=True)
    visible = grad_norm.squeeze(-1) > 0
    if not visible.any():
        return
    gaussians.xyz_grad_accum[visible] += grad_norm[visible]
    gaussians.denom[visible] += 1.0


def update_max_radii2d(gaussians: gaussian, radii: t.Tensor | None) -> None:
    if radii is None or radii.shape[0] != gaussians.mu.shape[0]:
        return
    ensure_densification_state(gaussians)
    gaussians.max_radii2D = t.maximum(gaussians.max_radii2D, radii.detach())


def clamp_log_scales(gaussians: gaussian, scene_extent: float) -> None:
    max_log = math.log(max(0.1 * scene_extent, 1e-7))
    gaussians.s.data.clamp_(max=max_log)


def _activated_scales(s: t.Tensor) -> t.Tensor:
    return t.exp(s)


def _activated_opacity(alpha: t.Tensor) -> t.Tensor:
    return t.sigmoid(alpha)


def _build_rotation(q: t.Tensor) -> t.Tensor:
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
    return t.stack([
        t.stack([r00, r01, r02], dim=-1),
        t.stack([r10, r11, r12], dim=-1),
        t.stack([r20, r21, r22], dim=-1),
    ], dim=1)


def update_optimizer_state(
    optimizer: t.optim.Optimizer,
    action_type: str,
    *,
    keep_mask: t.Tensor | None = None,
    extension: dict[str, t.Tensor] | None = None,
) -> None:
    action = action_type.lower()
    for group in optimizer.param_groups:
        attr = _GROUP_TO_ATTR[group["name"]]
        old = group["params"][0]
        stored = optimizer.state.get(old, None)

        if action == "prune":
            if keep_mask is None:
                raise ValueError("prune requires keep_mask")
            new_tensor = old[keep_mask].detach()
        elif action in ("clone", "split_append"):
            if extension is None:
                raise ValueError(f"{action} requires extension")
            new_tensor = t.cat([old.detach(), extension[attr].detach()], dim=0)
        else:
            raise ValueError(f"Unknown action_type: {action_type}")

        new_param = Parameter(new_tensor)
        group["params"][0] = new_param

        if stored is None:
            continue

        new_state: dict[str, Any] = {}
        for key, value in stored.items():
            if t.is_tensor(value) and value.shape[:1] == old.shape[:1]:
                if action == "prune":
                    new_state[key] = value[keep_mask]
                else:
                    pad = t.zeros(
                        (extension[attr].shape[0], *value.shape[1:]),
                        device=value.device,
                        dtype=value.dtype,
                    )
                    new_state[key] = t.cat([value, pad], dim=0)
            else:
                new_state[key] = value
        del optimizer.state[old]
        optimizer.state[new_param] = new_state


def _sync_gaussians_from_optimizer(gaussians: gaussian, optimizer: t.optim.Optimizer) -> None:
    for group in optimizer.param_groups:
        setattr(gaussians, _GROUP_TO_ATTR[group["name"]], group["params"][0])
    gaussians.N = int(gaussians.mu.shape[0])


def _extend_densification_state(gaussians: gaussian, n_new: int) -> None:
    """Pad densify buffers after clone/split append (mu already grown by n_new)."""
    if n_new <= 0:
        return
    # mu was already expanded; fit buffers to current N (do NOT allocate-then-cat).
    ensure_densification_state(gaussians)


def _prune_points(gaussians: gaussian, optimizer: t.optim.Optimizer, keep_mask: t.Tensor) -> int:
    n_before = int(gaussians.mu.shape[0])
    n_keep = int(keep_mask.sum().item())
    pruned = n_before - n_keep
    if pruned <= 0:
        return 0

    update_optimizer_state(optimizer, "prune", keep_mask=keep_mask)
    _sync_gaussians_from_optimizer(gaussians, optimizer)

    # Prefer masked keep when buffers still match pre-prune length; else refit.
    if (
        hasattr(gaussians, "xyz_grad_accum")
        and gaussians.xyz_grad_accum.shape[0] == n_before
        and gaussians.denom.shape[0] == n_before
        and gaussians.max_radii2D.shape[0] == n_before
    ):
        gaussians.xyz_grad_accum = gaussians.xyz_grad_accum[keep_mask]
        gaussians.denom = gaussians.denom[keep_mask]
        gaussians.max_radii2D = gaussians.max_radii2D[keep_mask]
    else:
        ensure_densification_state(gaussians)
    return pruned


def _densify_and_clone(
    gaussians: gaussian,
    optimizer: t.optim.Optimizer,
    grads: t.Tensor,
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
) -> int:
    scales = _activated_scales(gaussians.s)
    selected = (grads.squeeze(-1) >= grad_threshold) & (
        scales.max(dim=1).values <= percent_dense * scene_extent
    )
    n_clone = int(selected.sum().item())
    if n_clone == 0:
        return 0

    extension = {
        "mu": gaussians.mu[selected],
        "q": gaussians.q[selected],
        "s": gaussians.s[selected],
        "alpha": gaussians.alpha[selected],
        "A": gaussians.A[selected],
        "k_j": gaussians.k_j[selected],
    }
    update_optimizer_state(optimizer, "clone", extension=extension)
    _sync_gaussians_from_optimizer(gaussians, optimizer)
    _extend_densification_state(gaussians, n_clone)
    return n_clone


def _densify_and_split(
    gaussians: gaussian,
    optimizer: t.optim.Optimizer,
    grads: t.Tensor,
    grad_threshold: float,
    scene_extent: float,
    percent_dense: float,
    n_split: int = 2,
    scale_factor: float = 1.6,
) -> int:
    n_current = gaussians.mu.shape[0]

    if grads.ndim == 1:
        grads = grads.unsqueeze(-1)
    if grads.shape[0] < n_current:
        pad_size = n_current - grads.shape[0]
        grads = t.cat(
            [grads, t.zeros(pad_size, grads.shape[1], device=grads.device, dtype=grads.dtype)],
            dim=0,
        )

    scales = _activated_scales(gaussians.s)
    selected = (grads.squeeze(-1) >= grad_threshold) & (
        scales.max(dim=1).values > percent_dense * scene_extent
    )
    n_selected = int(selected.sum().item())
    if n_selected == 0:
        return 0

    stds = scales[selected].repeat(n_split, 1)
    samples = t.normal(mean=t.zeros_like(stds), std=stds)
    rots = _build_rotation(gaussians.q[selected]).repeat(n_split, 1, 1)
    new_mu = t.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + gaussians.mu[selected].repeat(
        n_split, 1
    )
    new_s = t.log(_activated_scales(gaussians.s[selected]).repeat(n_split, 1) / scale_factor)
    extension = {
        "mu": new_mu,
        "q": gaussians.q[selected].repeat(n_split, 1),
        "s": new_s,
        "alpha": gaussians.alpha[selected].repeat(n_split, 1),
        "A": gaussians.A[selected].repeat(n_split, 1),
        "k_j": gaussians.k_j[selected].repeat(n_split, 1, 1),
    }
    update_optimizer_state(optimizer, "split_append", extension=extension)
    _sync_gaussians_from_optimizer(gaussians, optimizer)

    n_new = n_split * n_selected
    _extend_densification_state(gaussians, n_new)

    prune_mask = t.cat(
        [selected, t.zeros(n_new, dtype=t.bool, device=selected.device)],
        dim=0,
    )
    _prune_points(gaussians, optimizer, ~prune_mask)
    return n_selected


def _reset_opacity(gaussians: gaussian, optimizer: t.optim.Optimizer, value: float = 0.01) -> None:
    logit_val = math.log(value / (1.0 - value))
    for group in optimizer.param_groups:
        if group["name"] != "Opacity":
            continue
        old = group["params"][0]
        new = Parameter(t.clamp_max(old.detach(), logit_val))
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
    grad_threshold: float = 0.00005,
    percent_dense: float = 0.01,
    min_opacity: float = 0.01,
    max_screen_size: float | None = None,
    scene_extent: float | None = None,
    reset_opacity: bool = False,
    max_gaussians: int = 800_000,
    allow_densify: bool = True,
) -> tuple[gaussian, t.optim.Optimizer, dict[str, int]]:
    """
    Refined Adaptive Density Control.

    1. Pixel-space densify gate (default ``grad_threshold=5e-5``) for fine detail.
    2. ``percent_dense=0.01`` forces oversized Gaussians to split.
    3. Prunes ``sigmoid(alpha) < min_opacity`` (default 0.01) and
       ``exp(s).max > 0.1 * scene_extent``.
    4. When ``allow_densify`` is False, skips clone/split/prune/opacity-reset.
    """
    ensure_densification_state(gaussians)

    if scene_extent is None:
        raise ValueError("scene_extent must be provided.")

    denom = gaussians.denom.clamp_min(1.0)
    grads = t.nan_to_num(gaussians.xyz_grad_accum / denom, nan=0.0)

    n_before = int(gaussians.mu.shape[0])
    n_cloned = 0
    n_split = 0

    # Execute clone/split only if allowed and under VRAM cap
    if allow_densify and n_before < max_gaussians:
        n_cloned = _densify_and_clone(
            gaussians, optimizer, grads, grad_threshold, scene_extent, percent_dense
        )
        n_split = _densify_and_split(
            gaussians, optimizer, grads, grad_threshold, scene_extent, percent_dense
        )

    n_pruned = 0
    # Freeze topology past densify window: no prune / opacity reset (avoids late collapse).
    if allow_densify:
        ensure_densification_state(gaussians)
        scales = _activated_scales(gaussians.s)
        opacity = _activated_opacity(gaussians.alpha.squeeze(-1))

        # Strip semi-transparent fog + giant world-space blobs.
        prune = opacity < min_opacity
        prune = prune | (scales.max(dim=1).values > 0.1 * scene_extent)

        if max_screen_size is not None:
            prune = prune | (gaussians.max_radii2D > max_screen_size)


        if prune.any() and (~prune).any():
            n_pruned = _prune_points(gaussians, optimizer, ~prune)
        elif prune.all():
            keep_n = max(1, gaussians.N // 10)
            keep_idx = t.topk(opacity, k=keep_n).indices
            keep = t.zeros(gaussians.N, dtype=t.bool, device=gaussians.mu.device)
            keep[keep_idx] = True
            n_pruned = _prune_points(gaussians, optimizer, keep)

        if reset_opacity:
            _reset_opacity(gaussians, optimizer, value=0.01)


    n = gaussians.mu.shape[0]
    gaussians.xyz_grad_accum = t.zeros(n, 1, device=gaussians.mu.device, dtype=gaussians.mu.dtype)
    gaussians.denom = t.zeros(n, 1, device=gaussians.mu.device, dtype=gaussians.mu.dtype)
    gaussians.max_radii2D.zero_()

    stats = {"cloned": n_cloned, "split": n_split, "pruned": n_pruned}
    return gaussians, optimizer, stats