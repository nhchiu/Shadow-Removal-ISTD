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

from collections import OrderedDict

import torch
import torch.nn as nn
from mydataset import MyDataset


class UNet(nn.Module):

    def __init__(self,
                 in_channels=MyDataset.in_channels,
                 out_channels=MyDataset.out_channels,
                 in_features=64):
        super(UNet, self).__init__()

        features = in_features
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=features,
            kernel_size=4,
            stride=2,
            padding=1,
            bias=True
        )
        self.encoder1 = UNet._block(features, features * 2, name="enc1")
        self.encoder2 = UNet._block(features * 2, features * 4, name="enc2")
        self.encoder3 = UNet._block(features * 4, features * 8, name="enc3")
        self.encoder4 = UNet._block(features * 8, features * 8, name="enc4")

        self.decoder4 = UNet._up_block(features * 8, features * 8, name="dec4")
        self.decoder3 = UNet._up_block(
            (features * 8) * 2, features * 4, name="dec3")
        self.decoder2 = UNet._up_block(
            (features * 4) * 2, features * 2, name="dec2")
        self.decoder1 = UNet._up_block(
            (features * 2) * 2, features, name="dec1")

        # self.activate = nn.Tanh()
        self.up_conv = nn.ConvTranspose2d(
            in_channels=features * 2,
            out_channels=out_channels,
            kernel_size=4,
            stride=2,
            padding=1)

    def forward(self, x):
        conv = self.conv(x)
        enc1 = self.encoder1(conv)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)

        dec4 = self.decoder4(enc4)
        dec3 = self.decoder3(torch.cat((dec4, enc3), dim=1))
        dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
        dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))
        upconv = self.up_conv(torch.cat((dec1, conv), dim=1))
        return upconv

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([
                (name + "lrelu", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
                (name + "conv",
                        nn.Conv2d(in_channels=in_channels,
                                  out_channels=features,
                                  kernel_size=4,
                                  stride=2,
                                  padding=1,
                                  bias=False,)),
                (name + "norm", nn.BatchNorm2d(num_features=features))
            ])
        )

    @staticmethod
    def _up_block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict([
                (name + "relu", nn.ReLU(inplace=True)),
                (name + "conv", nn.ConvTranspose2d(
                    in_channels=in_channels,
                    out_channels=features,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,)),
                (name + "norm", nn.BatchNorm2d(num_features=features))
            ])
        )
