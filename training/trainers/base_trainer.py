import torch
import tqdm
import os
import shutil
import numpy as np
from pathlib import Path
from itertools import repeat
from glob import glob

from abc import abstractmethod
from datasets.dataloaders import InfiniteLoader
from training.loggers import TrainingLogger
from datasets.datasets import datasets_registry
from metrics.metrics import metrics_registry
from torchvision import transforms


class BaseTrainer:
    @staticmethod
    def repeater(data_loader):
        for loader in repeat(data_loader):
            for data in loader:
                yield data

    def __init__(self, config):
        self.config = config

        self.device = config.exp.device
        self.start_step = config.train.proccessing.start_step
        self.step = 0

        if config.train.use_amp:
            self.scaler = torch.amp.GradScaler()

        self.checkpoint_path = Path(config.train.checkpoint_path) if config.train.checkpoint_path else None

    def setup(self):
        self.setup_experiment_dir()

        self.setup_models()
        self.setup_optimizers()
        self.setup_losses()

        self.setup_metrics()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()

    def setup_inference(self):
        self.setup_experiment_dir()

        self.setup_models()

        self.setup_metrics()
        self.setup_logger()

        self.setup_datasets()
        self.setup_dataloaders()

    @abstractmethod
    def setup_models(self):
        pass

    @abstractmethod
    def setup_optimizers(self):
        pass

    @abstractmethod
    def setup_losses(self):
        pass

    @abstractmethod
    def to_train(self):
        pass

    @abstractmethod
    def to_eval(self):
        pass

    def setup_experiment_dir(self):
        # create dir for experiment
        self.experiment_dir = Path(self.config.exp.exp_dir)
        self.experiment_dir.mkdir(parents=True, exist_ok=True)

        # create dir for checkpoint dumps in experiment dir
        self.save_path = self.experiment_dir / 'checkpoints'
        self.save_path.mkdir(parents=True, exist_ok=True)

        # create dir for image logging in experiment dir
        self.image_path = self.experiment_dir / 'images'
        self.image_path.mkdir(parents=True, exist_ok=True)

        # create dir for inference image generation
        self.inference_path = self.image_path / 'inference'
        self.inference_path.mkdir(parents=True, exist_ok=True)
        
        # create dir for validation image generation
        self.all_validation_images = np.asarray(glob(self.config.data.input_val_dir + '/*/*.jpg'))
        self.validation_temp_dir = self.experiment_dir / 'validation_temp'

    def setup_metrics(self):
        self.metrics = []
        for metric_name in self.config.train.val_metrics:
            self.metrics.append(metrics_registry[metric_name]())

    def setup_logger(self):
        self.logger = TrainingLogger(self.config)

    def setup_datasets(self):
        transformers = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((64, 64))
        ])
        
        if 'norm_mean' in self.config.data and self.config.data.norm_mean is not None:
            assert len(self.config.data.norm_mean) == 3
            mean = self.config.data.norm_mean
        else:
            mean = None
            
        if 'norm_std' in self.config.data and self.config.data.norm_std is not None:
            assert len(self.config.data.norm_std) == 3
            std = self.config.data.norm_std
        else:
            std = None
            

        # prepare train dataset
        self.train_dataset = datasets_registry[self.config.data.dataset_name](
            self.config.data.input_train_dir, transformers,
            samples_to_normalize=self.config.data.samples_to_normalize,
            normalize_data=self.config.data.normalize_data,
            mean=None, std=None
        )
        
        # get size of test
        self.test_size = len(self.all_validation_images)

        # get normalization parameters from train dataset
        data_mean, data_std = self.train_dataset.mean, self.train_dataset.std

        self.data_mean = data_mean.detach().clone()
        self.data_std = data_std.detach().clone()

    def setup_dataloaders(self):
        self.train_dataloader = InfiniteLoader(
            self.train_dataset,
            batch_size=self.config.data.train_batch_size,
            shuffle=True,
            num_workers=self.config.data.workers
        )
        
    @abstractmethod
    def get_modules_dict(self):
        pass

    def training_loop(self):
        self.to_train()

        for self.step in tqdm.trange(self.start_step, self.config.train.proccessing.steps + 1, desc="Training"):
            losses_dict = self.train_step()
            self.logger.update_losses(losses_dict)

            if self.step % self.config.train.proccessing.val_step == 0:
                self.logger.log_model_parameters(self.get_modules_dict(), step=self.step)

                val_metrics_dict, images = self.validate()
                self.logger.log_batch_of_images(images, step=self.step, images_type="validation")
                self.logger.log_val_metrics(val_metrics_dict, step=self.step)
                
            if self.step % self.config.train.proccessing.log_step == 0:
                self.logger.log_train_losses(self.step)

            if self.step % self.config.train.proccessing.checkpoint_step == 0:
                self.save_checkpoint()
                
        # delete temporal dir with images for validation
        path_validation = self.image_path / "temp"
        try: shutil.rmtree(path_validation)
        except: pass

    @abstractmethod
    def train_step(self):
        pass

    @abstractmethod
    def save_checkpoint(self):
        pass

    @torch.no_grad()
    def validate(self):
        self.to_eval()
        images_sample, images_pth = self.synthesize_images(self.step)
        
        # creation temporate dir for validation images
        if self.validation_temp_dir.exists():
            try: shutil.rmtree(self.validation_temp_dir)
            except: pass
        self.validation_temp_dir.mkdir(parents=True, exist_ok=True)
        for i, path in enumerate(self.all_validation_images):
            shutil.copy(path, self.validation_temp_dir / f"{i}.jpg")

        metrics_dict = {}
        for metric in self.metrics:
            stats = metric(
                orig_pth=str(self.validation_temp_dir),
                synt_pth=str(images_pth),
                device=self.device
            )
            metrics_dict.update(stats) 
        
        # delition of temporate dir 
        try: shutil.rmtree(self.validation_temp_dir)
        except: pass
        
        return metrics_dict, images_sample

    @abstractmethod
    def synthesize_images(self):
        pass

    @torch.no_grad()
    def inference(self):
        self.to_eval()
        images_sample, images_pth = self.synthesize_images(None)

        # # Validate your model, save images
        # # Calculate metrics
        # # Log if needed
        # raise NotImplementedError()
