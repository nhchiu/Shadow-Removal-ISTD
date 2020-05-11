#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://github.com/mateuszbuda/brain-segmentation-pytorch
@article{buda2019association,
  title={Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm},
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


class DenseUNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ngf=48, no_conv_t=False,
                 drop_rate=0, **kwargs):
        super(DenseUNet, self).__init__()
        depth = 5
        default_layers = 2
        growth_rate = ngf // default_layers

        self.in_conv = nn.Conv2d(in_channels=in_channels,
                                 out_channels=ngf,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0,
                                 bias=False)

        self.DenseBlockEncoders = nn.ModuleDict([
            (f"DBEncoder{i}", DenseUNet._dense_block(ngf,
                                                     layers=default_layers,
                                                     growth_rate=growth_rate,
                                                     drop_rate=drop_rate))
            for i in range(depth)
        ])

        self.TransDowns = nn.ModuleDict([
            (f"TD{i}", DenseUNet._trans_down(
                2*ngf, ngf, drop_rate=drop_rate))
            for i in range(depth)
        ])

        self.bottleneck = DenseUNet._bottleneck(
            ngf, layers=default_layers, growth_rate=growth_rate)

        self.TransUps = nn.ModuleDict(
            [
                (f"TU{depth-1}", DenseUNet._trans_up(2*ngf, ngf, no_conv_t))
            ]+[
                (f"TU{i}", DenseUNet._trans_up(4*ngf, ngf, no_conv_t))
                for i in reversed(range(depth-1))
            ])

        self.DenseBlockDecoders = nn.ModuleDict([
            (f"DBDecoder{i}", DenseUNet._dense_block(
                3 * ngf, layers=default_layers, growth_rate=growth_rate))
            for i in reversed(range(depth))
        ])

        self.out_conv = nn.Conv2d(in_channels=4*ngf,
                                  out_channels=out_channels,
                                  kernel_size=1,
                                  stride=1,
                                  bias=False)

    def forward(self, x):
        x = self.in_conv(x)
        # encode
        links = []
        for denseblock, trans_down in zip(self.DenseBlockEncoders.values(),
                                          self.TransDowns.values()):
            x = denseblock(x)
            links.append(x)
            x = trans_down(x)

        x = self.bottleneck(x)
        # decode
        for denseblock, trans_up in zip(self.DenseBlockDecoders.values(),
                                        self.TransUps.values()):
            x = trans_up(x)
            x = torch.cat((x, links.pop()), dim=1)
            x = denseblock(x)

        return self.out_conv(x)

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
            in_channels, layers=layers, growth_rate=growth_rate)

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
