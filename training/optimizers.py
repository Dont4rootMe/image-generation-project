from utils.class_registry import ClassRegistry
from torch.optim import Adam, AdamW, RMSprop
import torch
import os

optimizers_registry = ClassRegistry()


@optimizers_registry.add_to_registry(name="adam")
class Adam_(Adam):
    def load_model(self, path):
        assert os.path.exists(path), f'Adam optimizer weights were not found on path {path}'
        self.load_state_dict(torch.load(path))
        return self


@optimizers_registry.add_to_registry(name="adamW")
class AdamW_(AdamW):
    def load_model(self, path):
        assert os.path.exists(path), f'AdamW optimizer weights were not found on path {path}'
        self.load_state_dict(torch.load(path))
        return self


@optimizers_registry.add_to_registry(name="rmsprop")
class RMSprop_(RMSprop):
    def load_model(self, path):
        assert os.path.exists(path), f'RMSprop optimizer weights were not found on path {path}'
        self.load_state_dict(torch.load(path))
        return self
