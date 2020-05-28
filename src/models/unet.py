#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://github.com/mateuszbuda/brain-segmentation-pytorch
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas
    with shape features automatically extracted by a deep learning algorithm},
  author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
  journal={Computers in Biology and Medicine},
  volume={109},
  year={2019},
  publisher={Elsevier},
  doi={10.1016/j.compbiomed.2019.05.002}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.skip_connection_layer import SkipConnectionLayer


class UNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=False,
                 activation=None, **kwargs):
        super(UNet, self).__init__()
        depth = 4

        block = nn.Sequential(
            nn.Conv2d(in_channels=ngf*(2**(depth-1)),
                      out_channels=ngf*(2**depth),
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect',
                      bias=False,),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(num_features=ngf*(2**depth)),
            nn.Conv2d(in_channels=ngf*(2**depth),
                      out_channels=ngf*(2**depth),
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect',
                      bias=False,),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(num_features=ngf*(2**depth)))

        for i in reversed(range(1, depth)):
            block = SkipConnectionLayer(_conv_block(ngf*(2**(i-1)),
                                                    ngf*2**i),
                                        _up_block(ngf*2**(i+1), ngf*2**i),
                                        submodule=block, drop_rate=drop_rate)

        block = SkipConnectionLayer(_conv_block(in_channels, ngf),
                                    _up_block(ngf*2, ngf),
                                    submodule=block, drop_rate=0)

        sequence = [block,
                    nn.Conv2d(in_channels=ngf,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False)]
        if activation is not None:
            assert isinstance(activation, nn.Module)
            sequence.append(activation)

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class _conv_block(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        sequence = [nn.Conv2d(in_channels=in_channels,
                              out_channels=features,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              padding_mode='reflect',
                              bias=False,),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(num_features=features),
                    nn.Conv2d(in_channels=features,
                              out_channels=features,
                              kernel_size=3,
                              stride=1,
                              padding=1,
                              padding_mode='reflect',
                              bias=False,),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.BatchNorm2d(num_features=features)]
        self.block = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.block(x)
        return F.max_pool2d(out, 2), out


class _up_block(nn.Module):
    def __init__(self, in_channels, features):
        super().__init__()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=2,
            stride=2,
            bias=False)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels=2*features,
                      out_channels=features,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect',
                      bias=False,),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(num_features=features),
            nn.Conv2d(in_channels=features,
                      out_channels=features,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode='reflect',
                      bias=False,),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.BatchNorm2d(num_features=features))

    def forward(self, x, link):
        x = self.up_conv(x)
        return self.conv_block(torch.cat((x, link), dim=1))
