import torch
from torch import nn
import torch.autograd as autograd

from torch.nn import functional as F
from utils.class_registry import ClassRegistry


gen_losses_registry = ClassRegistry()
disc_losses_registry = ClassRegistry()


class GANLossBuilder:
    def __init__(self, config):
        self.gen_losses = {}
        self.disc_losses = {}
        self.coefs = {}

        for loss_name, loss_coef in config.gen_losses.items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if 'losses_args' in config and loss_name in config.losses_args:
                loss_args = config.losses_args
            self.gen_losses[loss_name] = gen_losses_registry[loss_name](**loss_args)

        for loss_name, loss_coef in config.disc_losses.items():
            self.coefs[loss_name] = loss_coef
            loss_args = {}
            if 'losses_args' in config and loss_name in config.losses_args:
                loss_args = config.losses_args
            self.disc_losses[loss_name] = disc_losses_registry[loss_name](**loss_args)

    def calculate_loss(self, batch_data, loss_type):
        # batch_data is a dict with all necessary data for loss calculation
        loss_dict = {}
        total_loss = 0.0

        if loss_type == "gen":
            losses = self.gen_losses
        elif loss_type == "disc" or loss_type == "critic":
            losses = self.disc_losses

        for loss_name, loss in losses.items():
            loss_val = loss(batch_data)
            total_loss += self.coefs[loss_name] * loss_val
            loss_dict[loss_name] = float(loss_val)

        return total_loss, loss_dict

# SOFTPLUS FAMILIES
@gen_losses_registry.add_to_registry(name="softplus_gen")
class SoftPlusGenLoss(nn.Module):
    def forward(self, batch):
        return F.softplus(-batch['fake_preds']).mean()
    
@disc_losses_registry.add_to_registry(name="softplus_disc")
class SoftPlusDiscLoss(nn.Module):
    def forward(self, batch):
        return torch.log(-batch['real_preds']).mean() + torch.log(batch['fake_preds']).mean()

# BCE FAMILIES
@gen_losses_registry.add_to_registry(name="bce_gen")
class BCELossGen(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.loss_fn(batch['fake_preds'], torch.ones_like(batch['fake_preds']))

@disc_losses_registry.add_to_registry(name="bce_disc")
class BCELossDisc(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, batch):
        return self.loss_fn(batch['real_preds'], torch.ones_like(batch['real_preds'])) + self.loss_fn(batch['fake_preds'], torch.zeros_like(batch['fake_preds']))

# WASSERSTAIN FAMILIES
@gen_losses_registry.add_to_registry(name="wasserstain_gen")
class WassersteinGenLoss(nn.Module):
    def forward(self, batch):
        return -batch['fake_preds'].mean()
    
@disc_losses_registry.add_to_registry(name="wasserstain_critic")
class WassersteinCriticLoss(nn.Module):
    def forward(self, batch):
        return batch['fake_preds'].mean() - batch['real_preds'].mean()
    
@disc_losses_registry.add_to_registry(name="wasserstein_gp")
class WassersteinGradientPenalty(nn.Module):
    def forward(self, batch):
        """
        Get penalty over WGAN-GP backpropagation.
        """
        # get gradient w.r.t. interpolated data
        gradients = autograd.grad(outputs=batch['interpolated_preds'], 
                                  inputs=batch['interpolated_data'], 
                                  grad_outputs=torch.ones_like(batch['interpolated_preds']), 
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        # get norm of gradients
        grad_norm = gradients.view(batch['batch_size'], -1).norm(2, dim=1)
        
        # compute penalty MSE
        gradient_penalty = ((grad_norm - 1) ** 2).mean()
        
        return gradient_penalty