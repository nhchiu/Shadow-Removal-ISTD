#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://github.com/mateuszbuda/brain-segmentation-pytorch
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape
         features automatically extracted by a deep learning algorithm},
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

from src.models.skip_connection_layer import SkipConnectionLayer


class DenseUNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ngf=48,
                 drop_rate=0,
                 no_conv_t=False,
                 activation=None, **kwargs):
        super(DenseUNet, self).__init__()
        depth = 5
        n_composite_layers = 2
        growth_rate = ngf // n_composite_layers

        in_conv = nn.Conv2d(in_channels=in_channels,
                            out_channels=ngf,
                            kernel_size=1,
                            stride=1,
                            padding=0,
                            bias=False)

        block = DenseUNet._bottleneck(
            ngf, layers=3*n_composite_layers, growth_rate=growth_rate)

        for i in reversed(range(depth)):
            block = SkipConnectionLayer(
                _conv_block(ngf, n_composite_layers, growth_rate),
                _up_block(ngf*4, ngf*2, n_composite_layers,
                          growth_rate, no_conv_t),
                submodule=block,
                drop_rate=drop_rate if i > 0 else 0)

        out_conv = nn.Conv2d(in_channels=4*ngf,
                             out_channels=out_channels,
                             kernel_size=1,
                             stride=1,
                             bias=False)

        sequence = [in_conv, block, out_conv]
        if (activation is not None) and (activation != "none"):
            assert isinstance(activation, nn.Module)
            sequence.append(activation)

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)

    @staticmethod
    def _trans_down(in_channels, out_channels=None, drop_rate=0.01):
        if out_channels is None:
            out_channels = in_channels // 2
        block = []
        block.append(nn.BatchNorm2d(num_features=in_channels))
        block.append(nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False))
        if drop_rate > 0:
            block.append(nn.Dropout2d(p=drop_rate, inplace=True))
        block.append(nn.AvgPool2d(2))
        return nn.Sequential(*block)

    @staticmethod
    def _trans_up(in_channels, out_channels=None, no_conv_t=False):
        if out_channels is None:
            out_channels = in_channels // 4

        if no_conv_t:
            return nn.Sequential(
                nn.Upsample(scale_factor=2),
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
                                      kernel_size=2,
                                      stride=2,
                                      bias=False)

    @staticmethod
    def _bottleneck(in_channels, layers=8, growth_rate=8):
        return DenseUNet._dense_block(
            in_channels, layers=layers, growth_rate=growth_rate, drop_rate=0)

    class _dense_block(nn.Module):
        def __init__(self, in_channels, layers=4,
                     growth_rate=8, drop_rate=0.01):
            super().__init__()
            self.composite_layers = nn.ModuleList([
                DenseUNet._dense_block._composite(
                    in_channels+i*growth_rate, growth_rate, drop_rate)
                for i in range(layers)
            ])

        def forward(self, x):
            for composite_layer in self.composite_layers:
                y = x
                x = composite_layer(x)
                x = torch.cat((x, y), dim=1)
            return x

        @staticmethod
        def _composite(in_channels, growth_rate, drop_rate):
            """
            Create a composite layer in Dense Block.
            """
            layer = [
                nn.BatchNorm2d(num_features=in_channels),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(in_channels=in_channels,
                          out_channels=growth_rate,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode='reflect',
                          bias=False)]
            if drop_rate > 0:
                layer.append(nn.Dropout2d(p=drop_rate, inplace=True))
            return nn.Sequential(*layer)


class _conv_block(nn.Module):
    def __init__(self, in_channels, layers, growth_rate):
        super().__init__()
        self.dense_block = DenseUNet._dense_block(in_channels,
                                                  layers=layers,
                                                  growth_rate=growth_rate,
                                                  drop_rate=0)
        self.trans_down = DenseUNet._trans_down(
            in_channels+layers*growth_rate, in_channels, drop_rate=0)

    def forward(self, x):
        link = self.dense_block(x)
        return self.trans_down(link), link


class _up_block(nn.Module):
    def __init__(self, in_channels, link_channels, layers, growth_rate,
                 no_conv_t=False):
        super().__init__()
        tu_out_channels = link_channels - layers * growth_rate
        self.trans_up = DenseUNet._trans_up(in_channels,
                                            tu_out_channels,
                                            no_conv_t)
        self.dense_block = DenseUNet._dense_block(
            tu_out_channels+link_channels,
            layers=layers,
            growth_rate=growth_rate,
            drop_rate=0)

    def forward(self, x, link):
        return self.dense_block(torch.cat((self.trans_up(x), link), dim=1))
