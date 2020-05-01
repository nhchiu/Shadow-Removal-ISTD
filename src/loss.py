#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom lost functions
"""

from collections import OrderedDict

import torch
import torch.nn as nn
from torch.nn import functional as F


class DataLoss(nn.Module):
    """
    Loss between shadow parameters
    """
    __slots__ = ["reduction", "norm"]

    def __init__(self, norm=F.smooth_l1_loss, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm

    def forward(self, y_pred, y_target):
        return self.norm(y_pred, y_target, reduction=self.reduction)


class VisualLoss(nn.Module):
    """
    Loss between predicted image and target image
    """
    __slots__ = ["reduction", "norm"]

    def __init__(self, norm=F.smooth_l1_loss, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm

    def forward(self, x, y_pred, img_target):
        # torch.reshape(
        # outputs, (outputs.shape[0], 3, 2, outputs.shape[2], outputs.shape[3]))
        # img_pred = torch.add(torch.mul(sp_pred[:,:, 0,...], 100/255), torch.mul(sp_pred[:,:, 1,...], img_in))
        img_pred = y_pred.mul(x).clamp_(0, 1)
        return self.norm(img_pred, img_target, reduction=self.reduction)


class AdversarialLoss(nn.Module):
    """
    Objective of a conditional GAN:
    E_(x,y){[log(D(x, y)]} + E_(x,z){[log(1-D(x, G(x, z))}

    The BCEWithLogitsLoss combines a Sigmoid layer and the BCELoss in one single class
    """

    def __init__(self):
        super().__init__()
        self.register_buffer('real_label', torch.tensor(1.0))
        self.register_buffer('fake_label', torch.tensor(0.0))

    def forward(self, D_out, is_real):
        target = self.real_label if is_real else self.fake_label
        target = target.expand_as(D_out)
        return F.binary_cross_entropy_with_logits(D_out, target)


class SoftAdapt(nn.Module):
    __slots__ = ["losses", "loss_tensor", "prev_losses", "differences",
                 "weights", "alpha", "beta", "epsilon",
                 "weighted", "normalized"]

    def __init__(self, keys: list,
                 beta=0.1,
                 epsilon=1e-8,
                 weighted=True,
                 normalized=True):
        self.losses = OrderedDict.fromkeys(keys, torch.tensor([0.0]))
        self.loss_tensor = torch.zeros(len(keys))
        self.prev_losses = torch.zeros(len(keys))
        self.differences = torch.zeros(len(keys))
        self.weights = torch.ones(len(keys)) / len(keys)
        self.beta: float = beta
        self.epsilon: float = epsilon
        self.weighted: bool = weighted
        self.normalized: bool = normalized
        self.alpha: float = 0.7  # smoothing factor

    def _loss_tensor(self):
        return torch.stack(tuple(self.losses.values()), dim=0)

    def update(self, losses: dict):
        self.losses.update(losses)

    def update_loss(self, loss: str, data: torch.Tensor):
        self.losses[loss] = data

    @torch.no_grad()
    def update_weights(self,):
        # Update gradient and average of previous losses
        self.loss_tensor = self._loss_tensor()
        with torch.no_grad():
            loss_detached = self.loss_tensor.detach()
            self.differences = self.alpha * self.differences + \
                (1-self.alpha) * (loss_detached-self.prev_losses)
            self.prev_losses = self.alpha * self.prev_losses + \
                (1-self.alpha) * loss_detached
            # Updated weights
            grad = self.differences
            if self.normalized:
                grad /= torch.clamp(grad.abs().sum(), min=self.epsilon)
            self.weights = F.softmax(grad, dim=0)
            if self.weighted:
                self.weights *= self.prev_losses
                self.weights /= torch.clamp(self.weights.sum(),
                                            min=self.epsilon)

    def forward(self, losses, update_weights=False):
        self.update(losses)
        if update_weights:
            self.update_weights()
        return torch.sum(self.loss_tensor * self.weights)

    def get_loss(self, key):
        return self.losses[key].item()

    def get_weights(self,):
        return dict(self.losses.keys(), self.weights.tolist())

    # def to(self, device):
    #     self.prev_losses = self.prev_losses.to(device)
    #     self.differences = self.differences.to(device)
    #     self.weights = self.weights.to(device)
