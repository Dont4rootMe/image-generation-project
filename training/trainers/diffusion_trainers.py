from utils.class_registry import ClassRegistry
from training.trainers.base_trainer import BaseTrainer

from models.diffusion_models import diffusion_models_registry
from training.optimizers import optimizers_registry
from training.losses.diffusion_losses import DiffusionLossBuilder
from training.diffusion_scheduler import DiffusionNoiseScheduler, NCSNScheduler

from torchvision.utils import save_image
import numpy as np
import shutil
import torch
import math


diffusion_trainers_registry = ClassRegistry()


@diffusion_trainers_registry.add_to_registry(name="base_diffusion_trainer")
class BaseDiffusionTrainer(BaseTrainer):
    def setup_models(self):
        self.ddpm = diffusion_models_registry[self.config.train.models.model](self.config.model_args).to(self.device)

        # model if checkpoint path is provided
        if self.config.train.checkpoint_path is not None:
            self.ddpm.load_model(self.checkpoint_path / 'ddpm.pth')

        # define noise scheduler for Gaussian proccess
        self.noise_scheduler = DiffusionNoiseScheduler(
            scheduler_type=self.config.train.noise_scheduler.scheduler_type,
            beta1=self.config.train.noise_scheduler.beta1,
            beta2=self.config.train.noise_scheduler.beta2,
            num_timesteps=self.config.train.noise_scheduler.num_timesteps,
            device=self.device
        )

    def setup_optimizers(self):
        self.ddpm_optimizer = optimizers_registry[self.config.train.models.optimizer](
            self.ddpm.parameters(), **self.config.optimizer_args
        )

        if self.config.train.checkpoint_path is not None:
            self.generator_optimizer.load_model(self.checkpoint_path / 'ddpm_optimizer.pth')

    def setup_losses(self):
        self.loss_builder = DiffusionLossBuilder(self.config)

    def to_train(self):
        self.ddpm.train()

    def to_eval(self):
        self.ddpm.eval()

    def train_step(self):
        with torch.amp.autocast(enabled=self.config['train']['use_amp'], device_type=self.device):
            # get image samples from dataset
            sample_images = next(self.train_dataloader)['images'].to(self.device)

            # get samples of timesteps for diffusion
            sample_time_staps = torch.randint(
                0, self.config.train.noise_scheduler.num_timesteps, (len(sample_images),), device=self.device
            )

            # get predictions of ddpm
            noisy_imgs, noise = self.noise_scheduler(sample_images, sample_time_staps)
            noise_predictions = self.ddpm(noisy_imgs, sample_time_staps)

            # log some train images after apply of noise and fade in
            if self.step % self.config.train.proccessing.val_step == 0:
                mean = self.data_mean
                std = self.data_std
                imgs_to_log = noisy_imgs.cpu() * std[:, None, None] + mean[:, None, None]
                imgs_to_log = torch.clip(imgs_to_log, 0, 1)
                self.logger.log_batch_of_images(imgs_to_log, step=self.step, images_type="train_imgs")

            # calculate loss
            preds_dict = {
                'real_noise': noise,
                'pred_noise': noise_predictions
            }
            total_loss_disc, loss_dict_disc = self.loss_builder.calculate_loss(preds_dict)

        self.ddpm_optimizer.zero_grad()
        if self.config.train.use_amp:
            self.scaler.scale(total_loss_disc).backward()
            self.scaler.step(self.ddpm_optimizer)
            self.scaler.update()
        else:
            total_loss_disc.backward()
            self.ddpm_optimizer.step()

        return {
            'loss/total_ddpm_loss': total_loss_disc.item(),
        }

    def get_modules_dict(self):
        return {
            'ddpm': self.ddpm,
            'optimizer_ddpm': self.ddpm_optimizer
        }

    def save_checkpoint(self):
        torch.save(self.ddpm.state_dict(), self.save_path / 'ppdm.pth')
        torch.save(self.ddpm_optimizer.state_dict(), self.save_path / 'ddpm_optimizer.pth')

    def synthesize_images(self, step=None):
        generated_images = []

        with torch.no_grad():
            for i in range(math.ceil(self.test_size / self.config.data.val_batch_size)):
                gen_imgs = torch.normal(0, 1, (self.config.data.val_batch_size, 3, 64, 64), device=self.device)

                for i in reversed(range(self.config.train.noise_scheduler.num_timesteps)):
                    steps = torch.tensor([i] * self.config.data.val_batch_size, device=self.device)
                    gen_noise = self.ddpm(gen_imgs, steps)
                    gen_imgs = self.noise_scheduler.backward_proccess(gen_imgs, gen_noise, steps)

                generated_images.append(gen_imgs)

        generated_images = torch.cat(generated_images, dim=0)[:self.test_size].cpu()

        # revert normalization
        mean = self.data_mean
        std = self.data_std
        generated_images = generated_images * std[:, None, None] + mean[:, None, None]
        generated_images = torch.clip(generated_images, 0, 1)

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


@diffusion_trainers_registry.add_to_registry(name="base_ncsn_trainer")
class BaseNCSNTrainer(BaseTrainer):
    def setup_models(self):
        self.ncsn = diffusion_models_registry[self.config.train.models.model](self.config.model_args).to(self.device)

        # model if checkpoint path is provided
        if self.config.train.checkpoint_path is not None:
            self.ncsn.load_model(self.checkpoint_path / 'ncsn.pth')

        # define noise scheduler for Gaussian proccess
        self.noise_scheduler = NCSNScheduler(
            num_timesteps=self.config.train.noise_scheduler.num_timesteps,
            sigma_max=self.config.train.noise_scheduler.sigma_max,
            sigma_min=self.config.train.noise_scheduler.sigma_min,
            device=self.device
        )

    def setup_optimizers(self):
        self.ncsn_optimizer = optimizers_registry[self.config.train.models.optimizer](
            self.ncsn.parameters(), **self.config.optimizer_args
        )

        if self.config.train.checkpoint_path is not None:
            self.generator_optimizer.load_model(self.checkpoint_path / 'ncsn_optimizer.pth')

    def setup_losses(self):
        self.loss_builder = DiffusionLossBuilder(self.config)

    def to_train(self):
        self.ncsn.train()

    def to_eval(self):
        self.ncsn.eval()

    def train_step(self):
        with torch.amp.autocast(enabled=self.config['train']['use_amp'], device_type=self.device):
            # get image samples from dataset
            sample_images = next(self.train_dataloader)['images'].to(self.device)

            # get samples of timesteps for diffusion
            sample_time_staps = torch.randint(
                0, self.config.train.noise_scheduler.num_timesteps, (len(sample_images),), device=self.device
            )

            # get predictions of ddpm
            noisy_imgs, noise = self.noise_scheduler(sample_images, sample_time_staps)
            noise_predictions = self.ncsn(noisy_imgs, sample_time_staps)

            # log some train images after apply of noise and fade in
            if self.step % self.config.train.proccessing.val_step == 0:
                mean = self.data_mean
                std = self.data_std
                imgs_to_log = noisy_imgs.cpu() * std[:, None, None] + mean[:, None, None]
                imgs_to_log = torch.clip(imgs_to_log, 0, 1)
                self.logger.log_batch_of_images(imgs_to_log, step=self.step, images_type="train_imgs")

            # calculate loss
            preds_dict = {
                'real_noise': noise,
                'pred_noise': noise_predictions
            }
            total_loss_disc, loss_dict_disc = self.loss_builder.calculate_loss(preds_dict)

        self.ncsn_optimizer.zero_grad()
        if self.config.train.use_amp:
            self.scaler.scale(total_loss_disc).backward()
            self.scaler.step(self.ncsn_optimizer)
            self.scaler.update()
        else:
            total_loss_disc.backward()
            self.ncsn_optimizer.step()

        return {
            'loss/total_ncsn_loss': total_loss_disc.item(),
        }

    def get_modules_dict(self):
        return {
            'ncsn': self.ncsn,
            'optimizer_ncsn': self.ncsn_optimizer
        }

    def save_checkpoint(self):
        torch.save(self.ncsn.state_dict(), self.save_path / 'ncsn.pth')
        torch.save(self.ncsn_optimizer.state_dict(), self.save_path / 'ncsn_optimizer.pth')

    def synthesize_images(self, step=None):
        generated_images = []

        with torch.no_grad():
            for i in range(math.ceil(self.test_size / self.config.data.val_batch_size)):
                gen_imgs = torch.normal(0, 1, (self.config.data.val_batch_size, 3, 64, 64), device=self.device)

                for l in range(self.config.train.noise_scheduler.num_timesteps):
                    step_size = 0.01 * ((self.noise_scheduler.sigmas[l] / self.noise_scheduler.sigmas[0]) ** 2)
                    
                    for _ in range(self.config.train.noise_scheduler.langeven_inner_steps):
                        steps = torch.tensor([i] * self.config.data.val_batch_size, device=self.device)
                        addon_noise = torch.sqrt(step_size) * torch.randn_like(gen_imgs)
                        gen_imgs = gen_imgs + 0.5 * step_size * self.ncsn(gen_imgs, steps) + addon_noise

                generated_images.append(gen_imgs)
                
        generated_images = torch.cat(generated_images, dim=0)[:self.test_size].cpu()

        # revert normalization
        mean = self.data_mean
        std = self.data_std
        generated_images = generated_images * std[:, None, None] + mean[:, None, None]
        generated_images = torch.clip(generated_images, 0, 1)

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
