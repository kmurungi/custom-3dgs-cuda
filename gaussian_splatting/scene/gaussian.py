import torch as t

class gaussian: 
    # for N gaussians
    def __init__(self, N: int) -> None:
        self.N = N
        self.mu = t.zeros(N, 3) # center points
        self.q = t.zeros(N, 4)  # quarternions
        self.s = t.zeros(N, 3) # scaling factors


        self.alpha = t.zeros(N, 1) # opacity
        self.A = t.zeros(N, 3) # Albedo coefficients
        self.k_j = t.zeros(N, 15, 3) # Illumination coefficients
