#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn


class DummyNet(nn.Module):

    def __init__(self, in_channels, out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=True,
                 use_selu=False,
                 activation=None, **kwargs):
        super(DummyNet, self).__init__()
        self.out_channels = out_channels
        self.dummy_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.dummy_conv(x)
