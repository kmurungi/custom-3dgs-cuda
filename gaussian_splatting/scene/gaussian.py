import torch as t

class gaussian: 
    # for N gaussians
    def __init__(self, N: int) -> None:
        self.N = N
        self.mu = t.zeros(N, 3, dtype=t.float32)  # means
        self.q = t.zeros(N, 4, dtype=t.float32)  # quaternions (w, x, y, z)
        self.s = t.zeros(N, 3, dtype=t.float32)  # log-scales
        self.alpha = t.zeros(N, 1, dtype=t.float32)  # opacity logits
        self.A = t.zeros(N, 3, dtype=t.float32)  # albedo SH DC
        self.k_j = t.zeros(N, 15, 3, dtype=t.float32)  # SH illumination coefficients
