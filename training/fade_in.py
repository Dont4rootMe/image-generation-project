from torch.nn import functional as F
from functools import partial
import torch

class FadeInScheduler:
    @staticmethod
    def compute_level_thresholds(total_steps, levels):
        cumsum = torch.cumsum(torch.tensor(levels), dim=0)[:-1]
        return total_steps * cumsum
    
    def __init__(self, total_steps, fade_in_config):
        assert 1 <= fade_in_config.start_from <= 5
        assert 5 - fade_in_config.start_from + 1 == len(fade_in_config.lvl_fraction)
        
        self.tresholds = self.compute_level_thresholds(total_steps, fade_in_config.lvl_fraction)
        self.start_from = fade_in_config.start_from
        self.adapters = {
            0: partial(F.interpolate, size=( 4,  4), mode='bilinear', align_corners=True),
            1: partial(F.interpolate, size=( 8,  8), mode='bilinear', align_corners=True),
            2: partial(F.interpolate, size=(16, 16), mode='bilinear', align_corners=True),
            3: partial(F.interpolate, size=(32, 32), mode='bilinear', align_corners=True),
            4: torch.nn.Identity()
        }

    def __call__(self, real_images, step):
        curr_level = torch.sum(self.tresholds < step).item()
        scaled_images = self.adapters[curr_level + self.start_from - 1](real_images)
        
        next_level = torch.sum(self.tresholds < step + 1) + self.start_from
        
        return scaled_images, next_level
