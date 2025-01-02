import torch 
import torch.nn.functional as F
from utils.class_registry import ClassRegistry

scheduler_registry = ClassRegistry()

@scheduler_registry.add_to_registry(name="linear_diff")
class DiffusionNoiseScheduler:
    def __init__(self, scheduler_type, beta1, beta2, num_timesteps, device):
        assert beta1 < beta2 < 1.0, "beta1 and beta2 must be in (0, 1)"
        
        self.scheduler_type = scheduler_type
        self.beta1 = beta1
        self.beta2 = beta2
        self.steps = num_timesteps
        
        assert scheduler_type in ['linear']
        
        # compute betas
        if scheduler_type == 'linear':
            self.betas = torch.linspace(beta1, beta2, num_timesteps, device=device)
        
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
    