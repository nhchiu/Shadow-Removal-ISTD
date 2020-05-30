#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
M-Net Architectures from
Le, H., & Samaras, D. (2019).
Shadow Removal via Shadow Image Decomposition. ICCV.
http://arxiv.org/abs/1908.08628

@InProceedings{Le_2019_ICCV,
    author = {Le, Hieu and Samaras, Dimitris},
    title = {Shadow Removal via Shadow Image Decomposition},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    month = {October},
    year = {2019}
}
"""

import torch
import torch.nn as nn

from src.models import opt_layers
from src.models.skip_connection_layer import SkipConnectionLayer


class MNet(nn.Module):

    def __init__(self, in_channels, out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=True,
                 use_selu=False,
                 activation=None, **kwargs):
        super(MNet, self).__init__()
        depth = 4

        self.conv = nn.Conv2d(in_channels=in_channels,
                              out_channels=ngf,
                              kernel_size=4,
                              stride=2,
                              padding=1,
                              padding_mode='reflect',
                              bias=False)

        block = SkipConnectionLayer(
            _conv_block((2 ** min(depth-1, 3))*ngf,
                        (2 ** min(depth, 3))*ngf),
            _up_block((2 ** min(depth, 3))*ngf,
                      (2 ** min(depth-1, 3))*ngf, no_conv_t),
            drop_rate=drop_rate)

        for i in reversed(range(1, depth-1)):
            features_in = (2 ** min(i, 3)) * ngf
            features_out = (2 ** min(i+1, 3)) * ngf
            block = SkipConnectionLayer(
                _conv_block(features_in, features_out),
                _up_block(2*features_out, features_in, no_conv_t),
                submodule=block,
                drop_rate=drop_rate)

        self.block = SkipConnectionLayer(
            _conv_block(ngf, ngf*2),
            _up_block(ngf*4, ngf, no_conv_t),
            submodule=block,
            drop_rate=0)

        upsample = opt_layers.get_upsample(no_conv_t, ngf*2, out_channels)
        if activation != "none":
            activation_layer = opt_layers.get_activation(activation)
        self.up_conv = nn.Sequential(upsample, activation_layer)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)
        return self.up_conv(x)


class _conv_block(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.model = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=features,
                      kernel_size=4,
                      stride=2,
                      padding=1,
                      padding_mode='reflect',
                      bias=False),
            nn.BatchNorm2d(num_features=features)
        )

    def forward(self, x):
        return self.model(x), x


class _up_block(nn.Module):
    def __init__(self, in_channels, features, no_conv_t=True):
        super().__init__()
        upconv = opt_layers.get_upsample(no_conv_t, in_channels, features)
        self.model = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            upconv,
            nn.BatchNorm2d(num_features=features)
        )

    def forward(self, x, link):
        return torch.cat((self.model(x), link), dim=1)
