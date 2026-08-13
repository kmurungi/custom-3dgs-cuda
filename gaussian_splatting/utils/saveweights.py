from pathlib import Path
import torch as t


def save_weights(gaussians, path) -> None:
    """Save Gaussian parameters to a .pt checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    t.save(
        {
            "mu": gaussians.mu.detach().cpu(),
            "q": gaussians.q.detach().cpu(),
            "s": gaussians.s.detach().cpu(),
            "alpha": gaussians.alpha.detach().cpu(),
            "A": gaussians.A.detach().cpu(),
            "k_j": gaussians.k_j.detach().cpu(),
            "N": gaussians.N,
        },
        path,
    )
