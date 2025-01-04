import torch
import torch.nn.functional as F
from utils.class_registry import ClassRegistry
import math

scheduler_registry = ClassRegistry()

class DiffusionNoiseScheduler:
    @staticmethod
    def cosine_schedule(numsteps, start=0.2, end=1, tau=2, clip_min=1e-9):
        # A gamma function based on cosine function.
        v_start = math.cos(start * math.pi / 2) ** (2 * tau)
        v_end = math.cos(end * math.pi / 2) ** (2 * tau)
        output = torch.cos(((torch.arange(numsteps) / numsteps) * (end - start) + start) * math.pi / 2) ** (2 * tau)
        output = (v_end - output) / (v_end - v_start)
        return torch.clip(output, clip_min, 1.).flip(0)

    @staticmethod
    def sigmoid_schedule(numsteps, start=-3, end=3, tau=0.9, clip_min=1e-9):
        # A gamma function based on sigmoid function.
        v_start = torch.sigmoid(torch.tensor([start / tau]))
        v_end = torch.sigmoid(torch.tensor([end / tau]))
        output = torch.sigmoid(((torch.arange(numsteps) / numsteps) * (end - start) + start) / tau)
        output = (v_end - output) / (v_end - v_start)
        return torch.clip(output, clip_min, 1.).flip(0)

    def __init__(self, scheduler_type, beta1, beta2, num_timesteps, device):
        if beta1 is None or beta2 is None:
            scale = 1000 / num_timesteps
            beta1 = scale * 1e-4
            beta2 = scale * 0.02

        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"

        self.scheduler_type = scheduler_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.steps = num_timesteps

        assert scheduler_type in ['linear', 'cosine', 'sigmoid']

        # compute betas
        if scheduler_type == 'linear':
            self.betas = torch.linspace(beta1, beta2, num_timesteps, device=device)
        elif scheduler_type == 'cosine':
            self.betas = self.cosine_schedule(num_timesteps).to(device)
            self.betas = self.betas * (beta2 - beta1) + beta1
        elif scheduler_type == 'sigmoid':
            self.betas = self.sigmoid_schedule(num_timesteps).to(device)
            self.betas = self.betas * (beta2 - beta1) + beta1

        # compute stats
        alphas = 1 - self.betas
        cumprod = torch.cumsum(torch.log(alphas), dim=0).exp()
        prev_cumprod = F.pad(cumprod[:-1], (1, 0), value=1.0)
        self.alhpa_1 = torch.sqrt(cumprod)
        self.alhpa_2 = torch.sqrt(1 - cumprod)
        self.recip_alphas = torch.sqrt(1 / alphas)
        self.posterior_variance = torch.sqrt(self.betas * (1 - prev_cumprod) / (1 - cumprod))

        # on last step mean of distribution is generated image
        self.posterior_variance[0] = 0.0

    def diffusion_forward(self, images, steps):
        """ 
        Takes an image and a timestep as input and 
        returns the noisy version of it
        """
        noise = torch.randn_like(images)
        alpha_1 = torch.gather(self.alhpa_1, 0, steps)
        alpha_2 = torch.gather(self.alhpa_2, 0, steps)

        return alpha_1[:, None, None, None] * images + alpha_2[:, None, None, None] * noise, noise

    def __call__(self, image, steps):
        return self.diffusion_forward(image, steps)

    def backward_proccess(self, images, noise, steps):
        # get required constants of noise
        betas = torch.gather(self.betas, 0, steps)
        alphas_2 = torch.gather(self.alhpa_2, 0, steps)
        recip_alphas = torch.gather(self.recip_alphas, 0, steps)
        posterior_variance = torch.gather(self.posterior_variance, 0, steps)

        # compute predicted mean
        predicted_mean = recip_alphas[:, None, None, None] * (
            images - betas[:, None, None, None] * noise / alphas_2[:, None, None, None]
        )

        # return denoised images
        addon_noise = torch.randn_like(images)
        return predicted_mean + posterior_variance[:, None, None, None] * addon_noise


class NCSNScheduler:
    def __init__(self, num_timesteps, sigma_max=1, sigma_min=0.01, device="cpu"):
        """
        Инициализация шедулера для NCSN.
        """
        self.num_timesteps = num_timesteps
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.device = device

        # logarithmic scale for noise
        self.sigmas = torch.logspace(
            math.log10(self.sigma_max),
            math.log10(self.sigma_min), 
            num_timesteps, 
            device=device
        )
        
        # normalizing constants
        self.inv_sigmas = 1 / self.sigmas

    def diffusion_forward(self, images, steps):
        # generate normal noise
        noise = torch.randn_like(images)
        
        # get successing variances
        sigma = torch.gather(self.sigmas, 0, steps)
        
        # noise images
        noisy_images = images + sigma[:, None, None, None] * noise
        return noisy_images, torch.clamp(noise / sigma[:, None, None, None], -1000, 1000)

    def __call__(self, images, steps):
        return self.diffusion_forward(images, steps)