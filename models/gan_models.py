import torch
from torch import nn

from torch.nn import functional as F
from utils.class_registry import ClassRegistry

import os


gens_registry = ClassRegistry()
discs_registry = ClassRegistry()


@gens_registry.add_to_registry(name="base_gen")
class VerySimpleGenarator(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.z_dim = model_config["z_dim"]
        self.hidden_dim = model_config["hidden_dim"]
        self.blocks_num = model_config["blocks_num"]

        self.z_to_x = nn.Linear(self.z_dim, self.hidden_dim * 4 * 4)

        self.blocks = nn.ModuleList([])
        for i in range(self.blocks_num):
            self.blocks.append(VerySimpleBlock(self.hidden_dim, self.hidden_dim))

        self.to_rgb_block = VerySimpleBlock(self.hidden_dim, 3)

    def forward(self, z):
        x = self.z_to_x(z).reshape(-1, self.hidden_dim, 4, 4)

        for block in self.blocks:
            x = block(x)
            x = F.interpolate(x, x.size(-1) * 2)
        x = self.to_rgb_block(x)
        return x
    
    def load_model(self, path):
        assert os.path.exists(path), f'Base Generator model weights were not found on path {path}'
        self.load_state_dict(torch.load(path))
        return self


@discs_registry.add_to_registry(name="base_disc")
class VerySimpleDiscriminator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.hidden_dim = model_config["hidden_dim"]
        self.blocks_num = model_config["blocks_num"]
        
        self.from_rgb_block = VerySimpleBlock(3, self.hidden_dim)

        self.blocks = nn.ModuleList([])
        for i in range(self.blocks_num):
            self.blocks.append(VerySimpleBlock(self.hidden_dim, self.hidden_dim))

        self.to_label = nn.Conv2d(self.hidden_dim, 1, kernel_size=4, stride=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.from_rgb_block(x)

        for block in self.blocks:
            x = block(x)
            x = F.interpolate(x, x.size(-1) // 2)
        x = self.to_label(x)
        return x
    
    def load_model(self, path):
        assert os.path.exists(path), f'Base Discriminator model weights were not found on path {path}'
        self.load_state_dict(torch.load(path))
        return self


class VerySimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.LeakyReLU(-0.2) 

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        return x

@gens_registry.add_to_registry(name="wasserstain_gen")
class WasserstainGenerator(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        
        self.latent_dim = model_config.generator_args.z_dim
        self.feature_map_size = model_config.generator_args.hidden_dim
        self.img_channels = 3
        
        self.model = nn.Sequential(
            # (latent_dim) -> (feature_map_size * 16) x 4 x 4
            nn.ConvTranspose2d(self.latent_dim, self.feature_map_size * 16, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 16),
            nn.LeakyReLU(-0.2),

            # (feature_map_size * 16) x 4 x 4 -> (feature_map_size * 8) x 8 x 8
            nn.ConvTranspose2d(self.feature_map_size * 16, self.feature_map_size * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 8),
            nn.LeakyReLU(-0.2),

            # (feature_map_size * 8) x 8 x 8 -> (feature_map_size * 4) x 16 x 16
            nn.ConvTranspose2d(self.feature_map_size * 8, self.feature_map_size * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 4),
            nn.LeakyReLU(-0.2),

            # (feature_map_size * 4) x 16 x 16 -> (feature_map_size * 2) x 32 x 32
            nn.ConvTranspose2d(self.feature_map_size * 4, self.feature_map_size * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.feature_map_size * 2),
            nn.LeakyReLU(-0.2),

            # (feature_map_size * 2) x 32 x 32 -> img_channels x 64 x 64
            nn.ConvTranspose2d(self.feature_map_size * 2, self.img_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = x.view(1, self.latent_dim, 1, 1)
        return self.model(x)
    
@discs_registry.add_to_registry(name="wasserstain_critic")
class WasserstainCritic(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.feature_map_size = model_config.generator_args.hidden_dim
        self.img_channels = 3
        
        self.model = nn.Sequential(
            # img_channels x 64 x 64 -> (feature_map_size) x 32 x 32
            nn.Conv2d(self.img_channels, self.feature_map_size, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_map_size) x 32 x 32 -> (feature_map_size * 2) x 16 x 16
            nn.Conv2d(self.feature_map_size, self.feature_map_size * 2, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_map_size * 2) x 16 x 16 -> (feature_map_size * 4) x 8 x 8
            nn.Conv2d(self.feature_map_size * 2, self.feature_map_size * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_map_size * 4) x 8 x 8 -> (feature_map_size * 8) x 4 x 4
            nn.Conv2d(self.feature_map_size * 4, self.feature_map_size * 8, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            # (feature_map_size * 8) x 4 x 4 -> 1 x 1 x 1 (скаляр)
            nn.Conv2d(self.feature_map_size * 8, 1, kernel_size=4, stride=1, padding=0)
        )
    
    def forward(self, x):
        return self.model(x).squeeze()