import torch as t

class gaussian: 
    def __init__(self, N: int) -> None:
        self.mu = t.zeros(N, 3) # center points
        self.q = t.zeros(N, 4)  # quarternions
        self.s = t.zeros(N, 3) # scaling factors

        self.alpha = t.zeros(N, 1) # opacity
        self.color = t.zeros(N, 16, 3) # color parameters across RGB channels
