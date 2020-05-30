#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn


def get_activation(key):
    # "none", "sigmoid", "tanh", "htanh"
    if key == "sigmoid":
        return nn.Sigmoid()
    elif key == "tanh":
        return nn.Tanh()
    elif key == "htanh":
        return nn.Hardtanh(min_val=-1.0, max_val=1.0, inplace=True)
    elif key == "none":
        return None
    else:
        raise ValueError


def get_norm(use_selu: bool, num_features: int):
    if use_selu:
        return nn.SELU(inplace=True)
    else:
        return nn.Sequential(nn.LeakyReLU(negative_slope=0.2, inplace=True),
                             nn.BatchNorm2d(num_features=num_features))


def get_dropout(use_selu: bool, drop_rate):
    if drop_rate == 0:
        return None
    else:
        if use_selu:
            return nn.AlphaDropout(p=drop_rate)
        else:
            return nn.Dropout2d(p=drop_rate)


def get_upsample(use_upsample: bool, in_channels, out_channels):
    if use_upsample:
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode="reflect",
                      bias=False))
    else:
        return nn.ConvTranspose2d(in_channels=in_channels,
                                  out_channels=out_channels,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1,
                                  bias=False)
