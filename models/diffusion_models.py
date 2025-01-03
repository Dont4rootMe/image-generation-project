import os
import torch
from torch import nn

from torch.nn import functional as F
from utils.class_registry import ClassRegistry


diffusion_models_registry = ClassRegistry()


class VerySimpleUnetBlock(nn.Module):
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


@diffusion_models_registry.add_to_registry(name="base_diffusion")
class VerySimpleUnet(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.num_steps = model_config["num_steps"]
        self.base_hidden_dim = model_config["base_hidden_dim"]
        self.blocks_num = model_config["blocks_num"]

        self.input_conv = nn.Conv2d(3, self.base_hidden_dim, kernel_size=3, stride=1, padding=1)
        self.time_embed = nn.Embedding(self.num_steps, self.base_hidden_dim)
        self.down_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])

        for i in range(self.blocks_num):
            lower_dim = self.base_hidden_dim * 2**i
            higher_dim = lower_dim * 2

            self.down_blocks.append(VerySimpleUnetBlock(lower_dim, higher_dim))
            self.up_blocks.append(VerySimpleUnetBlock(higher_dim, lower_dim))

    def forward(self, x, t):
        t = self.time_embed(t)
        t = t[..., None, None]

        x = self.input_conv(x)
        x = x + t

        for block in self.down_blocks:
            x = block(x)
            x = F.interpolate(x, x.size(-1) // 2)

        for block in reversed(self.up_blocks):
            x = block(x)
            x = F.interpolate(x, x.size(-1) * 2)

        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, time_embedding_dim: int, up: bool):
        super().__init__()

        self.time_mlp = nn.Sequential(
            nn.Linear(time_embedding_dim, out_channels),
            nn.GELU(),
            nn.Linear(out_channels, out_channels)
        )

        # projection on target amount of channels
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(out_channels)
        )

        # blending projected images with time embeddings
        self.time_blend_in = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(0.05),
            nn.BatchNorm2d(out_channels)
        )

        # image dimmensions changing
        if up:
            self.transform = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)
        else:
            self.transform = nn.Conv2d(out_channels, out_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, images: torch.Tensor, times: torch.Tensor):
        # inner representation of input tensors
        time_embed = self.time_mlp(times)
        images_prj = self.projection(images)

        # blending representations with time
        time_inception = images_prj + time_embed[..., None, None]
        blended_images = (images_prj + self.time_blend_in(time_inception)) / 1.414

        # returning next level
        return self.transform(blended_images)


@diffusion_models_registry.add_to_registry(name="blend_diffusion")
class BlendingUnet(nn.Module):
    @staticmethod
    def get_sinusoidal_embeddings(diffusion_steps, time_embedding_dim, n=1000.0):
        assert time_embedding_dim % 2 == 0, f'Sinusoidal positional embedding cannot apply to odd token embedding dim (got dim={time_embedding_dim})'

        T = diffusion_steps
        d = time_embedding_dim  # d_model=head_num*d_k, not d_q, d_k, d_v

        positions = torch.arange(0, T).unsqueeze_(1)
        embeddings = torch.zeros(T, d)

        denominators = torch.pow(n, 2*torch.arange(0, d//2)/d)  # 10000^(2i/d_model), i is the index of embedding
        embeddings[:, 0::2] = torch.sin(positions/denominators)  # sin(pos/10000^(2i/d_model))
        embeddings[:, 1::2] = torch.cos(positions/denominators)  # cos(pos/10000^(2i/d_model))

        return embeddings

    def __init__(self, model_config):
        super().__init__()

        self.diffusion_steps = model_config.diffusion_steps
        self.feature_map_size = model_config.feature_map_size
        self.time_embedding_dim = model_config.time_embedding_dim
        self.img_channels = 3

        # projections from ing_channels to feature_map and reverse
        self.input_projection = nn.Conv2d(self.img_channels, self.feature_map_size, kernel_size=3, padding=1)
        self.output_projection = nn.Conv2d(self.feature_map_size, self.img_channels, kernel_size=1)

        # definition of positional encodings in diffusional-proccess predictor
        self.time_embeddings = nn.Embedding(self.diffusion_steps, self.time_embedding_dim)
        self.time_embeddings.weight.data.copy_(self.get_sinusoidal_embeddings(self.diffusion_steps, self.time_embedding_dim))
        self.time_embeddings.weight.requires_grad = True

        # convolutional blocks in down sampling
        self.downs = nn.ModuleList([
            # (feature_map_size * 1) x 64 x 64 -> (feature_map_size * 2) x 32 x 32
            ConvBlock(self.feature_map_size * 1, self.feature_map_size * 2, self.time_embedding_dim, up=False),

            # (feature_map_size * 2) x 32 x 32 -> (feature_map_size * 4) x 16 x 16
            ConvBlock(self.feature_map_size * 2, self.feature_map_size * 4, self.time_embedding_dim, up=False),

            # (feature_map_size * 4) x 16 x 16 -> (feature_map_size * 8) x 8 x 8
            ConvBlock(self.feature_map_size * 4, self.feature_map_size * 8, self.time_embedding_dim, up=False),

            # (feature_map_size * 8) x 8 x 8 -> (feature_map_size * 16) x 4 x 4
            ConvBlock(self.feature_map_size * 8, self.feature_map_size * 16, self.time_embedding_dim, up=False)
        ])

        # convolutional blocks in up sampling
        self.ups = nn.ModuleList([
            # 2 x (feature_map_size * 16) x 4 x 4 -> (feature_map_size * 8) x 8 x 8
            ConvBlock(2 * self.feature_map_size * 16, self.feature_map_size * 8, self.time_embedding_dim, up=True),

            # 2 x (feature_map_size * 8) x 8 x 8 -> (feature_map_size * 4) x 16 x 16
            ConvBlock(2 * self.feature_map_size * 8, self.feature_map_size * 4, self.time_embedding_dim, up=True),

            # 2 x (feature_map_size * 4) x 16 x 16 -> (feature_map_size * 2) x 32 x 32
            ConvBlock(2 * self.feature_map_size * 4, self.feature_map_size * 2, self.time_embedding_dim, up=True),

            # 2 x (feature_map_size * 2) x 32 x 32 -> (feature_map_size * 1) x 64 x 64
            ConvBlock(2 * self.feature_map_size * 2, self.feature_map_size * 1, self.time_embedding_dim, up=True),
        ])

    def forward(self, images: torch.Tensor, steps: torch.Tensor):
        x = self.input_projection(images)
        time_embeds = self.time_embeddings(steps)

        residual_connections = []
        for down in self.downs:
            x = down(x, time_embeds)
            residual_connections.append(x)
        for up, left in zip(self.ups, reversed(residual_connections)):
            x = torch.cat((x, left), dim=1)
            x = up(x, time_embeds)

        return self.output_projection(x)
    
    def load_model(self, path):
        assert os.path.exists(path), f'BlendingUnet model weights were not found on path {path}'
        self.load_state_dict(torch.load(path))
        return self
