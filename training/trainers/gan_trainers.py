from utils.class_registry import ClassRegistry
from utils.model_utils import requires_grad
from training.trainers.base_trainer import BaseTrainer
from training.noise import NoiseScheduler
from training.fade_in import FadeInScheduler

from models.gan_models import gens_registry, discs_registry
from training.optimizers import optimizers_registry
from training.losses.gan_losses import GANLossBuilder

from torchvision.utils import save_image
import numpy as np
import shutil
import torch
import math

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
        self.generator_optimizer = optimizers_registry[self.config['train']['models']['gen_optimizer']](
            self.generator.parameters(), **self.config['gen_optimizer_args']
        )
        self.dicriminator_optimizer = optimizers_registry[self.config['train']['models']['disc_optimizer']](
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
        with torch.amp.autocast(enabled=self.config.train.use_amp, device_type=self.device):
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
            'loss/total_gen_loss': total_loss_gen.item(),
            'loss/total_disc_loss': total_loss_disc.item(),
        }

    def get_modules_dict(self):
        return {
            'gen': self.generator,
            'disc': self.descriminator,
            'optimizer_gan': self.generator_optimizer,
            'optimizer_disc': self.dicriminator_optimizer
        }

    def save_checkpoint(self):
        torch.save(self.generator.state_dict(), self.save_path / 'generator.pth')
        torch.save(self.descriminator.state_dict(), self.save_path / 'discriminator.pth')
        torch.save(self.generator_optimizer.state_dict(), self.save_path / 'gen_optimizer.pth')
        torch.save(self.dicriminator_optimizer.state_dict(), self.save_path / 'disc_optimizer.pth')

    def synthesize_images(self, step=None):
        generated_images = []

        for i in range(math.ceil(self.test_size / self.config.data.val_batch_size)):
            z = torch.normal(0, 1, (self.config.data.val_batch_size, self.config.generator_args.z_dim), device=self.device)

            with torch.no_grad():
                gen_ims = self.generator(z).detach().cpu()

            generated_images.append(gen_ims)

        generated_images = torch.cat(generated_images, dim=0)[:self.test_size]

        # revert normalization
        mean = self.data_mean
        std = self.data_std
        generated_images = generated_images * std[:, None, None] + mean[:, None, None]

        # prepare path to save images
        if step is None:
            # if it is inference – save to inference path
            path_to_saved_pics = self.inference_path
        else:
            # if we save intermediate results - create dir for that step
            if self.config.data.n_save_images is not None:
                path_to_saved_pics = self.image_path / f"train/step_{step}"
                path_to_saved_pics.mkdir(parents=True, exist_ok=True)

            # clear temporal dir with images for validation
            path_validation = self.image_path / "generative_temp"
            try:
                shutil.rmtree(path_validation)
            except:
                pass
            path_validation.mkdir(parents=True, exist_ok=True)

        # save images for validation
        for i in range(len(generated_images)):
            image_path = path_validation / f"generated_image_{i + 1}.png"
            save_image(generated_images[i], image_path, format='png')

        # save samples from training step
        if self.config.data.n_save_images is not None:
            # get random samples from generated images
            indexes = torch.as_tensor(np.random.choice(generated_images.size(0), self.config.data.n_save_images))
            sampled_images = generated_images[indexes.to(generated_images.device)]

            # save each
            for i in range(len(sampled_images)):
                image_path = path_to_saved_pics / f"generated_image_{i + 1}.png"
                save_image(sampled_images[i], image_path, format='png')
        else:
            indexes = torch.as_tensor(np.random.choice(generated_images.size(0), 16))
            sampled_images = generated_images[indexes.to(generated_images.device)]

        return sampled_images, path_validation


@gan_trainers_registry.add_to_registry(name="wasserstain_gan_trainer")
class WasserstainGANTrainer(BaseTrainer):
    def __init__(self, config):
        super().__init__(config)

        if 'noise' in self.config.train and self.config.train.noise.add_noise:
            self.noise_sceduler = NoiseScheduler(self.config.train.noise.start_noise_std, self.config.train.proccessing.steps)

        if 'fade_in' in self.config.train and self.config.train.fade_in.use_fade_in:
            self.fade_in_scheduler = FadeInScheduler(self.config.train.proccessing.steps, self.config.train.fade_in)

    def setup_models(self):
        self.generator = gens_registry['wasserstain_gen'](self.config.generator_args).to(self.device)
        self.critic = discs_registry['wasserstain_critic'](self.config.critic_args).to(self.device)

        # load models if checkpoint path is provided
        if self.config.train.checkpoint_path is not None:
            self.generator.load_model(self.checkpoint_path / 'generator.pth', self.device)
            self.critic.load_model(self.checkpoint_path / 'critic.pth', self.device)

        # if fade in is off then use whole model
        if not ('fade_in' in self.config.train and self.config.train.fade_in.use_fade_in and self.step is not None):
            self.generator.set_num_blocks(self.generator.max_blocks)
            self.critic.set_num_blocks(self.critic.max_blocks)
        else:
            self.generator.set_num_blocks(self.fade_in_scheduler.check_step(self.start_step))
            self.critic.set_num_blocks(self.fade_in_scheduler.check_step(self.start_step))

    def setup_optimizers(self):
        self.generator_optimizer = optimizers_registry[self.config.train.models.gen_optimizer](
            self.generator.parameters(), **self.config.gen_optimizer_args
        )
        self.critic_optimizer = optimizers_registry[self.config.train.models.critic_optimizer](
            self.critic.parameters(), **self.config.critic_optimizer_args
        )

        if self.config.train.checkpoint_path is not None:
            self.generator_optimizer.load_model(self.checkpoint_path / 'gen_optimizer.pth')
            self.critic_optimizer.load_model(self.checkpoint_path / 'critic_optimizer.pth')

    def setup_losses(self):
        self.loss_builder = GANLossBuilder(self.config)

    def to_train(self):
        self.generator.train()
        self.critic.train()

    def to_eval(self):
        self.generator.eval()
        self.critic.eval()

    def get_modules_dict(self):
        return {
            'gen': self.generator,
            'critic': self.critic,
            'optimizer_gan': self.generator_optimizer,
            'optimizer_critic': self.critic_optimizer
        }

    def train_step(self):
        # TRAIN DISCRIMINATOR
        if 'critic_ministeps' in self.config.train.models:
            discriminator_ministeps = self.config.train.models.critic_ministeps
        else:
            discriminator_ministeps = 1

        sum_loss_disc = 0
        sum_loss_gp = 0
        for _ in range(discriminator_ministeps):
            with torch.amp.autocast(enabled=self.config['train']['use_amp'], device_type=self.device):
                # get real images from dataset
                real_images = next(self.train_dataloader)['images'].to(self.device)

                # add noise to real samples
                if self.config.train.noise.add_noise:
                    real_images = self.noise_sceduler(real_images, self.step)
                    if self.config.generator_args.add_tanh:
                        real_images = torch.clamp(real_images, min=-1, max=1)

                # apply fade in scheduler
                if self.config.train.fade_in.use_fade_in:
                    real_images, new_lavel = self.fade_in_scheduler(real_images, self.step)

                # log some train images after apply of noise and fade in
                if self.step % self.config.train.proccessing.val_step == 0:
                    mean = self.data_mean
                    std = self.data_std
                    imgs_to_log = real_images.cpu() * std[:, None, None] + mean[:, None, None]
                    self.logger.log_batch_of_images(imgs_to_log, step=self.step, images_type="train_imgs")

                # get synthetic images via generator
                z = torch.normal(0, 1, (real_images.size(0), self.config.generator_args.z_dim), device=self.device)
                fake_images = self.generator(z)

                # create interpolated images
                batch_size = real_images.size(0)
                epsilon = torch.rand(real_images.size(0), 1, 1, 1, device=real_images.device)
                interpolated_data = epsilon * real_images + (1 - epsilon) * fake_images
                interpolated_data.requires_grad_(True)

                # calculate disc losses, make discriminator step
                preds_dict = {
                    'real_preds': self.critic(real_images),
                    'fake_preds': self.critic(fake_images.detach()),
                    'interpolated_data': interpolated_data,
                    'interpolated_preds': self.critic(interpolated_data),
                    'batch_size': batch_size
                }

                if self.config.train.fade_in.use_fade_in:
                    self.generator.block_number = new_lavel
                    self.critic.block_number = new_lavel

                # Compute loss of the discriminator
                total_loss_disc, loss_dict_disc = self.loss_builder.calculate_loss(preds_dict, loss_type='critic')

            # compute gradient over discriminator loss and make dicriminator step
            self.critic_optimizer.zero_grad()
            if self.config.train.use_amp:
                self.scaler.scale(total_loss_disc).backward()
                self.scaler.step(self.critic_optimizer)
                self.scaler.update()
            else:
                total_loss_disc.backward()
                self.critic_optimizer.step()

            sum_loss_disc += total_loss_disc.item() / discriminator_ministeps
            sum_loss_gp += loss_dict_disc['wasserstein_gp'] / discriminator_ministeps

        # TRAIN GENERATOR
        if 'generator_ministeps' in self.config.train.models:
            generator_ministeps = self.config.train.models.generator_ministeps
        else:
            generator_ministeps = 1

        sum_loss_gen = 0
        for _ in range(generator_ministeps):
            with torch.amp.autocast(enabled=self.config.train.use_amp, device_type=self.device):
                # get synthetic images via generator
                z = torch.normal(0, 1, (self.config.data.train_batch_size, self.config.generator_args.z_dim), device=self.device)
                fake_images = self.generator(z)

                # calculate gen losses, make generator step
                preds_dict = {
                    'fake_preds': self.critic(fake_images)
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

            sum_loss_gen += total_loss_gen.item() / generator_ministeps

        return {
            'loss/total_gen_loss': sum_loss_gen,
            'loss/total_disc_loss': sum_loss_disc,
            'loss/penalty_gradient': sum_loss_gp
        }

    def save_checkpoint(self):
        torch.save(self.generator.state_dict(), self.save_path / 'generator.pth')
        torch.save(self.critic.state_dict(), self.save_path / 'critic.pth')
        torch.save(self.generator_optimizer.state_dict(), self.save_path / 'gen_optimizer.pth')
        torch.save(self.critic_optimizer.state_dict(), self.save_path / 'critic_optimizer.pth')

    def synthesize_images(self, step=None):
        generated_images = []

        for i in range(math.ceil(self.test_size / self.config.data.val_batch_size)):
            z = torch.normal(0, 1, (self.config.data.val_batch_size, self.config.generator_args.z_dim), device=self.device)

            with torch.no_grad():
                gen_ims = self.generator(z).detach().cpu()

            generated_images.append(gen_ims)

        generated_images = torch.cat(generated_images, dim=0)[:self.test_size]

        # revert normalization
        mean = self.data_mean
        std = self.data_std
        generated_images = generated_images * std[:, None, None] + mean[:, None, None]
        generated_images = torch.clamp(generated_images, 0, 1)

        # prepare path to save images
        if self.config.data.n_save_images is not None:
            if step is None:
                # if it is inference – save to inference path
                path_to_saved_pics = self.inference_path
            else:
                # if we save intermediate results - create dir for that step
                path_to_saved_pics = self.image_path / f"train/step_{step}"
            
            path_to_saved_pics.mkdir(parents=True, exist_ok=True)

        # clear temporal dir with images for validation
        path_validation = self.image_path / "generative_temp"
        try:
            shutil.rmtree(path_validation)
        except:
            pass
        path_validation.mkdir(parents=True, exist_ok=True)

        # save images for validation
        for i in range(len(generated_images)):
            image_path = path_validation / f"generated_image_{i + 1}.png"
            save_image(generated_images[i], image_path, format='png')

        # save samples from training step
        if self.config.data.n_save_images is not None:
            # get random samples from generated images
            indexes = torch.as_tensor(np.random.choice(generated_images.size(0), self.config.data.n_save_images))
            sampled_images = generated_images[indexes.to(generated_images.device)]

            # save each
            for i in range(len(sampled_images)):
                image_path = path_to_saved_pics / f"generated_image_{i + 1}.png"
                save_image(sampled_images[i], image_path, format='png')
        else:
            indexes = torch.as_tensor(np.random.choice(generated_images.size(0), 16))
            sampled_images = generated_images[indexes.to(generated_images.device)]

        return sampled_images, path_validation
