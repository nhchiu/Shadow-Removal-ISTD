#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom lost functions
"""

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

import utils


class DataLoss(nn.Module):
    """
    Loss between shadow parameters
    """

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
    E_(x,y){[log(D(x, y)]} + E_(x,z){[log(1 − D(x, G(x, z))}

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




