#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Custom lost function
"""

import torch
import torch.nn as nn
from torch.nn import functional as F


class CustomLoss(nn.Module):
    def __init__(self, weight, reduction='mean'):
        super().__init__()
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, outputs, target_imgs, target_sps):
        # loss between shadow parameters
        sp_loss = F.l1_loss(
            outputs, target_sps, reduction=self.reduction)

        # loss between predicted image and target image
        sp_pred = torch.reshape(
            outputs, (outputs.shape[0], 3, 2, outputs.shape[2], outputs.shape[3]))
        img_in = inputs
        img_pred = torch.add(torch.mul(sp_pred[:,:, 0,...], 100/255), torch.mul(sp_pred[:,:, 1,...], img_in))
        img_pred = torch.clamp(img_pred, 0, 1)
        img_loss = F.mse_loss(img_pred, target_imgs, reduction=self.reduction)

        return torch.add(sp_loss, torch.mul(img_loss, self.weight))
