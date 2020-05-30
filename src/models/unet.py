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

from src.models import opt_layers
from src.models.skip_connection_layer import SkipConnectionLayer


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=False,
                 use_selu=False,
                 activation=None, **kwargs):
        super(UNet, self).__init__()
        depth = 4

        block = conv(ngf*(2**(depth-1)), ngf*(2**depth), use_selu)

        for i in reversed(range(1, depth)):
            block = SkipConnectionLayer(_conv_block(ngf*(2**(i-1)),
                                                    ngf*2**i, use_selu),
                                        _up_block(ngf*2**(i+1), ngf*2**i,
                                                  use_selu, no_conv_t),
                                        submodule=block, drop_rate=drop_rate)

        block = SkipConnectionLayer(_conv_block(in_channels, ngf, use_selu),
                                    _up_block(ngf*2, ngf,
                                              use_selu, no_conv_t),
                                    submodule=block, drop_rate=0)

        sequence = [block,
                    nn.Conv2d(in_channels=ngf,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              bias=False)]
        if activation != "none":
            sequence.append(opt_layers.get_activation(activation))

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def conv(in_channels, features, use_selu: bool):
    return nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                   out_channels=features,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   padding_mode='reflect',
                                   bias=False,),
                         opt_layers.get_norm(use_selu, features),
                         nn.Conv2d(in_channels=features,
                                   out_channels=features,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1,
                                   padding_mode='reflect',
                                   bias=False,),
                         opt_layers.get_norm(use_selu, features))


class _conv_block(nn.Module):
    def __init__(self, in_channels, features, selu):
        super().__init__()
        self.block = conv(in_channels, features, selu)

    def forward(self, x):
        out = self.block(x)
        return F.max_pool2d(out, 2), out


class _up_block(nn.Module):
    def __init__(self, in_channels, features, selu, no_conv_t):
        super().__init__()
        self.up_conv = opt_layers.get_upsample(no_conv_t,
                                               in_channels, features)
        self.conv_block = conv(2*features, features, selu)

    def forward(self, x, link):
        x = self.up_conv(x)
        return self.conv_block(torch.cat((x, link), dim=1))
