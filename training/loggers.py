import wandb
import torch
from PIL import Image
from collections import defaultdict
from omegaconf import OmegaConf
import os


class WandbLogger:
    def __init__(self, config):
        if config.wandb.WANDB_KEY is not None:
            wandb.login(key=config.wandb.WANDB_KEY.strip())

        if config.train.checkpoint_path != "" and config.train.checkpoint_path is not None:
            # resume training run from checkpoint
            if config.wandb.run_id is not None:
                wandb_run_id = config.wandb.run_id.strip()
            else:
                with open(f"{config.train.checkpoint_path}/wandb_run_id.txt", "r") as file:
                    wandb_run_id = file.read().strip()
            self.wandb_args = {
                "id": wandb_run_id,
                "entity": config.wandb.entity,
                "project": config.wandb.project,
                "resume": "must",
            }
        else:
            # create new wandb run and save args, config and etc.
            if 'run_name' not in config.wandb or config.wandb.run_name is None:
                wandb_run_id = wandb.util.generate_id()
                print(f"Wandb run id: {wandb_run_id}")
                with open(f"{config.exp_dir}/wandb_run_id.txt", "w") as file:
                    file.write(wandb_run_id)
                    file.flush()

            self.wandb_args = {
                "id": wandb_run_id,
                "entity": config.wandb.entity,
                "project": config.wandb.project,
                "name": config.wandb.run_name,
                "tags": config.wandb.tags,
                "config": OmegaConf.to_container(config),
                "resume": "never",
            }

        wandb.init(**self.wandb_args)

    @staticmethod
    def log_values(values_dict: dict, step: int):
        # log values to wandb
        wandb.log(values_dict, step=step)

    @staticmethod
    def log_images(images: dict, step: int):
        # log images
        wandb_images = {key: wandb.Image(img) for key, img in images.items()}
        wandb.log(wandb_images, step=step)


class TrainingLogger:
    def __init__(self, config):
        self.logger = WandbLogger(config)
        self.losses_memory = defaultdict(list)

    def log_train_losses(self, step: int):
        # avarage losses in losses_memory
        avg_losses = {
            loss_name: sum(loss_vals) / len(loss_vals)
            for loss_name, loss_vals in self.losses_memory.items()
        }

        # log them and clear losses_memory
        self.logger.log_values(avg_losses, step)
        self.losses_memory.clear()

    def log_val_metrics(self, val_metrics: dict, step: int):
        self.logger.log_values(val_metrics, step)

    def log_batch_of_images(self, batch: torch.Tensor, step: int, images_type: str = ""):
        # batch of tensors -> images
        images = {}
        for i, img_tensor in enumerate(batch):
            img = Image.fromarray((img_tensor.cpu().numpy().transpose(1, 2, 0) * 255).astype("uint8"))
            images[f"{images_type}/image_{i}"] = img

        # log images
        self.logger.log_images(images, step)

    def update_losses(self, losses_dict):
        # it is useful to average losses over a number of steps rather than track them at each step
        # this makes training curves smoother
        for loss_name, loss_val in losses_dict.items():
            self.losses_memory[loss_name].append(loss_val)
            
    def log_model_parameters(self, modules, step):
        """
        Logging parameters of models and optimizers
        """
        # logging generator weights and gradients
        for name, param in modules['gen'].named_parameters():
            if 'adapters' in name: continue
            if param.requires_grad:
                wandb.log({
                    f"generator_weights/{name}": wandb.Histogram(param.data.cpu().numpy())
                }, step=step)
                # wandb.log({
                #     f"generator_gradients/{name}": wandb.Histogram(param.grad.data.cpu().numpy())
                # }, step=step)

        if 'disc' in modules:
            # logging discriminator weights
            for name, param in  modules['disc'].named_parameters():
                if 'adapters' in name: continue
                if param.requires_grad:
                    wandb.log({
                        f"discriminator_weights/{name}": wandb.Histogram(param.data.cpu().numpy())
                    }, step=step)
                    # wandb.log({
                    #     f"discriminator_gradients/{name}": wandb.Histogram(param.grad.data.cpu().numpy())
                    # }, step=step)
        else:
            # logging critic weights and gradients
            for name, param in  modules['critic'].named_parameters():
                if 'adapters' in name: continue
                if param.requires_grad:
                    wandb.log({
                        f"critic_weights/{name}": wandb.Histogram(param.data.cpu().numpy())
                    }, step=step)
                    # wandb.log({
                    #     f"critic_gradients/{name}": wandb.Histogram(param.grad.data.cpu().numpy())
                    # }, step=step)

        # logging hyperparameters and states of optimizers
        if 'disc' in modules:    
            wandb.log({
                "hyperparam/generator_lr": modules['optimizer_gan'].param_groups[0]['lr'],
                "hyperparam/discriminator_lr": modules['optimizer_disc'].param_groups[0]['lr'],
                "hyperparam/generator_weight_decay": modules['optimizer_gan'].param_groups[0].get('weight_decay', 0),
                "hyperparam/discriminator_weight_decay": modules['optimizer_disc'].param_groups[0].get('weight_decay', 0)
            }, step=step)
        else:
            wandb.log({
                "hyperparam/generator_lr": modules['optimizer_gan'].param_groups[0]['lr'],
                "hyperparam/critic_lr": modules['optimizer_critic'].param_groups[0]['lr'],
                "hyperparam/generator_weight_decay": modules['optimizer_gan'].param_groups[0].get('weight_decay', 0),
                "hyperparam/critic_weight_decay": modules['optimizer_critic'].param_groups[0].get('weight_decay', 0)
            }, step=step)
        
        
