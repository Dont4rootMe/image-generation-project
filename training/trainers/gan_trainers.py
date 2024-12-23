from utils.class_registry import ClassRegistry
from utils.model_utils import requires_grad
from training.trainers.base_trainer import BaseTrainer

from models.gan_models import gens_registry, discs_registry
from training.optimizers import optimizers_registry
from training.losses.gan_losses import GANLossBuilder

from torchvision.utils import save_image
import torch
import os

gan_trainers_registry = ClassRegistry()


@gan_trainers_registry.add_to_registry(name="base_gan_trainer")
class BaseGANTrainer(BaseTrainer):
    def setup_models(self):
        self.generator = gens_registry['base_gen'](self.config.generator_args).to(self.device)
        self.descriminator = discs_registry['base_disc'](self.config.discriminator_args).to(self.device)

        if self.config.train.checkpoint_path is not None:
            self.generator.load_model(self.checkpoint_path / 'generator.pth')
            self.descriminator.load_model(self.checkpoint_path / 'discriminator.pth')

    def setup_optimizers(self):
        self.generator_optimizer = optimizers_registry[self.config['train']['gen_optimizer']](
            self.generator.parameters(), **self.config['gen_optimizer_args']
        )
        self.dicriminator_optimizer = optimizers_registry[self.config['train']['disc_optimizer']](
            self.descriminator.parameters(), **self.config['disc_optimizer_args']
        )

        if self.config.train.checkpoint_path is not None:
            self.generator_optimizer.load_model(self.checkpoint_path / 'gen_optimizer.pth')
            self.dicriminator_optimizer.load_model(self.checkpoint_path / 'disc_optimizer.pth')

    def setup_losses(self):
        self.loss_builder = GANLossBuilder(self.config)

    def to_train(self):
        self.generator.train()
        self.descriminator.train()

    def to_eval(self):
        self.generator.eval()
        self.descriminator.eval()

    def train_step(self):
        # TRAIN DISCRIMINATOR
        with torch.amp.autocast(enabled=self.config['train']['use_amp'], device_type=self.device):
            # get real images from dataset
            real_images = next(self.train_dataloader)['images'].to(self.device)

            # get synthetic images via generator
            z = torch.normal(0, 1, (self.config.data.train_batch_size, self.config.generator_args.z_dim), device=self.device)
            fake_images = self.generator(z)

            # calculate disc losses, make discriminator step
            preds_dict = {
                'real_preds': self.descriminator(real_images),
                'fake_preds': self.descriminator(fake_images.detach())
            }

            # Compute loss of the discriminator
            total_loss_disc, loss_dict_disc = self.loss_builder.calculate_loss(preds_dict, loss_type='disc')

        # compute gradient over discriminator loss and make dicriminator step
        self.dicriminator_optimizer.zero_grad()
        if self.config.train.use_amp:
            self.scaler.scale(total_loss_disc).backward()
            self.scaler.step(self.dicriminator_optimizer)
            self.scaler.update()
        else:
            total_loss_disc.backward()
            self.dicriminator_optimizer.step()

        # TRAIN GENERATOR
        with torch.amp.autocast(enabled=self.config.train.use_amp, device_type=self.device):
            # get synthetic images via generator
            z = torch.normal(0, 1, (self.config.data.train_batch_size, self.config.generator_args.z_dim), device=self.device)
            fake_images = self.generator(z)

            # calculate gen losses, make generator step
            preds_dict = {
                'fake_preds': self.descriminator(fake_images)
            }

            # Compute loss of the generator
            total_loss_gen, loss_dict_gen = self.loss_builder.calculate_loss(preds_dict, loss_type='gen')

        # compute gradient over generator loss and make generator step
        self.generator_optimizer.zero_grad()
        if self.config.train.use_amp:
            self.scaler.scale(total_loss_gen).backward()
            self.scaler.step(self.generator_optimizer)
            self.scaler.update()
        else:
            total_loss_gen.backward()
            self.generator_optimizer.step()

        return {
            'total_gen_loss': total_loss_gen,
            'total_disc_loss': total_loss_disc,
            'gen_loss': {**loss_dict_gen},
            'disc_loss': {**loss_dict_disc}
        }

    def save_checkpoint(self):
        torch.save(self.generator.state_dict(), self.save_path / 'generator.pth')
        torch.save(self.descriminator.state_dict(), self.save_path / 'discriminator.pth')
        torch.save(self.generator_optimizer.state_dict(), self.save_path / 'gen_optimizer.pth')
        torch.save(self.dicriminator_optimizer.state_dict(), self.save_path / 'disc_optimizer.pth')

    def synthesize_images(self, step=None):
        z = torch.normal(0, 1, (self.config.data.val_batch_size, self.config.generator_args.z_dim), device=self.device)

        with torch.no_grad():
            generated_images = self.generator(z)

        # prepare path to save images
        if step is None:
            path_to_saved_pics = self.inference_path
        else:
            path_to_saved_pics = self.image_path / f"train/step_{step}"
            path_to_saved_pics.mkdir(parents=True, exist_ok=True)
            
        mean = self.data_mean
        std = self.data_std

        # save images
        for i in range(self.config.data.val_batch_size):
            image_path = path_to_saved_pics / f"generated_image_{i + 1}.png"
            save_image(generated_images[i] * std[:, None, None] - mean[:, None, None] , image_path)

        return generated_images, path_to_saved_pics
