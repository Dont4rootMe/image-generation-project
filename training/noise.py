import torch
import math

class NoiseScheduler:
    def __init__(self, 
            start_std: float, 
            num_steps: int, 
            s: float = 0.2, 
            e: float = 1.0, 
            tau: float = 2.0, 
            clip_min=1e-3
    ):
        """
        On the Importance of Noise Scheduling for Diﬀusion Models 
        (https://arxiv.org/pdf/2301.10972)
        """
        self.start_std = start_std
        self.num_steps = num_steps
        self.s = s
        self.e = e
        self.tau = tau
        self.clip_min = clip_min

    def __call__(self, images, step):
        t = step / self.num_steps
        v_start = math.cos(self.s * math.pi / 2) ** (2 * self.tau)
        v_end = math.cos(self.e * math.pi / 2) ** (2 * self.tau)
        gamma = math.cos((t * (self.e - self.s) + self.s) * math.pi / 2) ** (2 * self.tau)
        gamma = (v_end - gamma) / (v_end - v_start)
        
        return torch.normal(mean=images, std=gamma * self.start_std)
