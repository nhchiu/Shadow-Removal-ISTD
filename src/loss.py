#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom lost functions
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
from dataset import ISTDDataset


class DataLoss(nn.Module):
    """
    Loss between shadow parameters
    """
    __slots__ = ["reduction", "norm"]

    def __init__(self, norm=F.l1_loss, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm

    def forward(self, y_pred, y_target):
        return self.norm(y_pred, y_target, reduction=self.reduction)


class VisualLoss(nn.Module):
    """
    Feature reconstruction perceptual loss with VGG-19.
    Measured by the norms between the features after passing
    through the pool4 layer.
    """

    def __init__(self, norm=F.mse_loss, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.norm = norm
        VGG19 = models.vgg19_bn(pretrained=True, progress=False)
        self.VGG = VGG19.features[:40].requires_grad_(False)
        self.register_buffer(
            'mean',
            torch.tensor(ISTDDataset.mean).reshape((3, 1, 1)))
        self.register_buffer(
            'std',
            torch.tensor(ISTDDataset.std).reshape((3, 1, 1)))

    def forward(self, x, y_pred, img_target):
        img_in = x.mul(self.std).add(self.mean)
        img_pred = y_pred.mul(img_in).clamp_(0, 1)
        feature_pred = self.VGG(img_pred)
        with torch.no_grad():
            feature_target = self.VGG(img_target)
        return self.norm(feature_pred, feature_target,
                         reduction=self.reduction)


class AdversarialLoss(nn.Module):
    """
    Objective of a conditional GAN:
    E_(x,y){[log(D(x, y)]} + E_(x,z){[log(1-D(x, G(x, z))}

    The BCEWithLogitsLoss combines a Sigmoid layer
    and the BCELoss in one single class.
    """

    def __init__(self, ls=False, rel=False, avg=False):
        super().__init__()
        if not ls:
            self.register_buffer('real_label', torch.tensor(1.0))
            self.register_buffer('fake_label', torch.tensor(0.0))
        else:
            self.register_buffer('real_label', torch.tensor(1.0))
            self.register_buffer('fake_label', torch.tensor(-1.0))
        self.ls = ls
        self.rel = rel
        self.avg = avg

    def forward(self, D_out, is_real):
        target = self.real_label if is_real else self.fake_label
        target = target.expand_as(D_out)
        if not self.ls:
            return F.mse_loss(D_out, target)
        else:
            return F.binary_cross_entropy_with_logits(D_out, target)


class SoftAdapt(nn.Module):

    def __init__(self, losses: list,
                 init_weights=None,
                 beta=0.1,
                 epsilon=1e-8,
                 min_=1e-4,
                 weighted=True,
                 normalized=True):
        super().__init__()
        self.loss = losses
        self.size = len(losses)
        self.register_buffer('current_loss', torch.ones(self.size))
        self.register_buffer('prev_loss', torch.ones(self.size))
        self.register_buffer('gradient', torch.zeros(self.size))
        if init_weights is None:
            self.register_buffer('weights',
                                 torch.ones(self.size) / self.size)
        else:
            assert len(init_weights) == self.size
            self.register_buffer('weights',
                                 torch.tensor(init_weights,
                                              dtype=torch.float32))
            self.weights /= self.weights.sum()
        self.beta: float = beta
        self.epsilon: float = epsilon
        self.weighted: bool = weighted
        self.normalized: bool = normalized
        self.alpha: float = 0.9  # smoothing factor
        self.min_ = min_

    # def _loss_tensor(self):
    #     return torch.stack(tuple(self.losses.values()), dim=0)

    def update(self, losses: dict):
        # for i, l in enumerate(self.loss):
        #     self.current_loss[i] = losses[l]
        loss_list = [losses[k] for k in self.loss]
        self.current_loss = torch.stack(tuple(loss_list), dim=0)

    # def update_loss(self, loss: str, data: torch.Tensor):
    #     self.losses[loss] = data

    @torch.no_grad()
    def update_weights(self,):
        # Update gradient and average of previous losses
        loss_detached = self.current_loss.detach()
        self.gradient = (loss_detached-self.prev_loss)
        # Updated weights
        grad = self.gradient
        if self.normalized:  # use relative ratios intead of absolute values
            grad /= self.prev_loss.clamp(min=self.epsilon)
            # grad /= torch.clamp(grad.abs().sum(), min=self.epsilon)
        grad -= grad.max()
        new_weight = F.softmax(self.beta * grad, dim=0)
        if self.weighted:  # account for losses of different ranges
            new_weight *= (self.prev_loss.sum() - self.prev_loss)
            new_weight /= new_weight.sum()
            # self.prev_loss.max()/self.prev_loss.clamp(min=self.epsilon)
        self.weights = self.alpha * self.weights + (1-self.alpha) * new_weight
        assert self.weights.requires_grad is False
        self.prev_loss = (loss_detached)
        # self.weights *= self.prev_loss
        # self.weights /= torch.clamp(self.weights.sum(),
        #                             min=self.epsilon)

    def forward(self, losses, update_weights=False):
        self.update(losses)
        if update_weights:
            self.update_weights()
        return torch.sum(self.current_loss * self.weights)

    def get_loss(self,):
        return dict(zip(self.loss, self.current_loss.tolist()))

    def get_weights(self,):
        return dict(zip(self.loss, self.weights.tolist()))
