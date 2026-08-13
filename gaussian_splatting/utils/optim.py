import torch as t
from gaussian_splatting.scene.gaussian import gaussian


def setup_optimizer(gaussians: gaussian, lrs: dict, lr: float = 0.01):
    """
    Adds gaussian weights to be optimized and returns an Adam optimizer.
    """
    params = [
        {"params": [gaussians.mu], "lr": lrs["mu"], "name": "mu"},
        {"params": [gaussians.q], "lr": lrs["q"], "name": "q"},
        {"params": [gaussians.s], "lr": lrs["s"], "name": "Scale"},
        {"params": [gaussians.alpha], "lr": lrs["alpha"], "name": "Opacity"},
        {"params": [gaussians.A], "lr": lrs["A"], "name": "Albedo Coefficient"},
        {"params": [gaussians.k_j], "lr": lrs["k_j"], "name": "Illumination Coefficients"},
    ]
    return t.optim.Adam(params, lr=lr)
