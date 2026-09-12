from __future__ import annotations

import math

import torch as t
from torch.nn import Parameter
from torch.optim.lr_scheduler import LambdaLR

from gaussian_splatting.scene.gaussian import gaussian


def setup_optimizer(gaussians: gaussian, lrs: dict, lr: float = 0.01):
    """
    Wrap trainable Gaussian tensors as Parameters and return an Adam optimizer.
    """
    gaussians.mu = Parameter(gaussians.mu.detach())
    gaussians.q = Parameter(gaussians.q.detach())
    gaussians.s = Parameter(gaussians.s.detach())
    gaussians.alpha = Parameter(gaussians.alpha.detach())
    gaussians.A = Parameter(gaussians.A.detach())
    gaussians.k_j = Parameter(gaussians.k_j.detach())

    params = [
        {"params": [gaussians.mu], "lr": lrs["mu"], "name": "mu"},
        {"params": [gaussians.q], "lr": lrs["q"], "name": "q"},
        {"params": [gaussians.s], "lr": lrs["s"], "name": "Scale"},
        {"params": [gaussians.alpha], "lr": lrs["alpha"], "name": "Opacity"},
        {"params": [gaussians.A], "lr": lrs["A"], "name": "Albedo Coefficient"},
        {"params": [gaussians.k_j], "lr": lrs["k_j"], "name": "Illumination Coefficients"},
    ]
    return t.optim.Adam(params, lr=lr, eps=1e-15)


def _expon_lr_lambda(
    lr_init: float,
    lr_final: float,
    *,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 0.01,
    max_steps: int = 30_000,
):
    """Inria ``get_expon_lr_func`` as a LambdaLR multiplier (relative to lr_init)."""

    def _lambda(step: int) -> float:
        if lr_init <= 0.0:
            return 0.0
        if lr_delay_steps > 0:
            delay_rate = lr_delay_mult + (1.0 - lr_delay_mult) * math.sin(
                0.5 * math.pi * min(max(step / float(lr_delay_steps), 0.0), 1.0)
            )
        else:
            delay_rate = 1.0
        t_frac = min(max(step / float(max_steps), 0.0), 1.0)
        log_lerp = math.exp(
            math.log(lr_init) * (1.0 - t_frac) + math.log(lr_final) * t_frac
        )
        return delay_rate * log_lerp / lr_init

    return _lambda


def get_positional_lr_scheduler(
    optimizer: t.optim.Optimizer,
    *,
    lr_init: float = 1.6e-4,
    lr_final: float = 1.6e-6,
    scale_lr_init: float = 5e-3,
    scale_lr_final: float = 5e-5,
    lr_delay_steps: int = 0,
    lr_delay_mult: float = 0.01,
    max_steps: int = 30_000,
) -> LambdaLR:
    """
    Exponential decay for ``mu`` (position) and ``Scale`` Adam groups.

    Position: ``lr_init`` → ``lr_final`` over ``max_steps``.
    Scale: ``scale_lr_init`` (default 0.005) → ``scale_lr_final``.
    Optional warm-up via ``lr_delay_steps`` / ``lr_delay_mult`` (half-sine).
    Other param groups keep a constant multiplier of 1.
    """
    if lr_init <= 0.0 or lr_final <= 0.0:
        raise ValueError("lr_init and lr_final must be positive")
    if scale_lr_init <= 0.0 or scale_lr_final <= 0.0:
        raise ValueError("scale_lr_init and scale_lr_final must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be >= 1")

    name_to_idx = {g.get("name"): i for i, g in enumerate(optimizer.param_groups)}
    if "mu" not in name_to_idx:
        raise ValueError("optimizer has no param group named 'mu'")
    if "Scale" not in name_to_idx:
        raise ValueError("optimizer has no param group named 'Scale'")

    mu_idx = name_to_idx["mu"]
    scale_idx = name_to_idx["Scale"]

    optimizer.param_groups[mu_idx]["lr"] = lr_init
    optimizer.param_groups[mu_idx]["initial_lr"] = lr_init
    optimizer.param_groups[scale_idx]["lr"] = scale_lr_init
    optimizer.param_groups[scale_idx]["initial_lr"] = scale_lr_init
    for i, group in enumerate(optimizer.param_groups):
        if i in (mu_idx, scale_idx):
            continue
        group.setdefault("initial_lr", group["lr"])

    mu_fn = _expon_lr_lambda(
        lr_init,
        lr_final,
        lr_delay_steps=lr_delay_steps,
        lr_delay_mult=lr_delay_mult,
        max_steps=max_steps,
    )
    scale_fn = _expon_lr_lambda(
        scale_lr_init,
        scale_lr_final,
        lr_delay_steps=lr_delay_steps,
        lr_delay_mult=lr_delay_mult,
        max_steps=max_steps,
    )

    lambdas = [(lambda _step: 1.0) for _ in optimizer.param_groups]
    lambdas[mu_idx] = mu_fn
    lambdas[scale_idx] = scale_fn
    return LambdaLR(optimizer, lr_lambda=lambdas)


def get_mu_lr(optimizer: t.optim.Optimizer) -> float:
    """Current Adam learning rate for the positional (mu) param group."""
    for group in optimizer.param_groups:
        if group.get("name") == "mu":
            return float(group["lr"])
    return float("nan")


def get_scale_lr(optimizer: t.optim.Optimizer) -> float:
    """Current Adam learning rate for the Scale param group."""
    for group in optimizer.param_groups:
        if group.get("name") == "Scale":
            return float(group["lr"])
    return float("nan")
