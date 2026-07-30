import torch as t
from gaussian_splatting.scene.gaussian import gaussian

def setup_optimizer(gaussian: gaussian, lrs: dict, lr : float = 0.01): 
    """
    Adds gaussian weights to be optimized and returns the optimizer Adam
    """
    params = [
        {
            "params": gaussian.mu, 
            "lr": lrs['mu'],  
            "name": "mu"
        },
        {
            "params": gaussian.q, 
            "lr": lrs['q'],  
            "name": "q"
        },
        {
            "params": gaussian.s, 
            "lr": lrs['s'],  
            "name": "Scale"
        },
        {
            "params": gaussian.alpha, 
            "lr": lrs['alpha'],  
            "name": "Opacity"
        },
        {
            "params": gaussian.A, 
            "lr": lrs['A'],  
            "name": "Albedo Coefficient"
        },
        {
            "params": gaussian.k_j, 
            "lr": lrs['k_j'],  
            "name": "Illumination Coefficients"
        }
    ]

    optimizer = t.optim.Adam(params, lr = lr)
    return optimizer