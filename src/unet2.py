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

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from mydataset import MyDataset


class UNet2(nn.Module):

    def __init__(self,
                 in_channels=MyDataset.in_channels,
                 out_channels=MyDataset.out_channels,
                 in_features=64):
        super(UNet2, self).__init__()
        features = in_features
        self.depth = 5

        down_features = [in_channels]
        for i in range(self.depth):
            down_features.append(features * (2**i))

        self.encoders = nn.ModuleList()
        for i in range(self.depth):
            self.encoders.append(
                UNet2._block(down_features[i], down_features[i+1]))

        up_features = [features * (2**(self.depth-1))]
        for i in reversed(range(self.depth-1)):
            up_features.append(features * (2**i))

        self.decoders = nn.ModuleList()
        for i in range(self.depth-1):
            self.decoders.append(
                UNet2._up_block(up_features[i], up_features[i+1]))

        self.out_conv = nn.Conv2d(
            in_channels=features,
            out_channels=out_channels,
            kernel_size=1,
            stride=1)

    def forward(self, x):
        # encode
        links = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i != self.depth - 1:  # apply downsample except the last block
                links.append(x)
                x = F.max_pool2d(x, 2)

        # decode
        for i, decoder in enumerate(self.decoders):
            linkage = links.pop()
            x = decoder(x, linkage)

        return self.out_conv(x)

    class _block(nn.Module):
        def __init__(self, in_channels, features):
            super().__init__()
            self.conv_block = nn.Sequential(
                OrderedDict([
                    ("conv0", nn.Conv2d(in_channels=in_channels,
                                        out_channels=features,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False,)),
                    ("lrelu0", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
                    ("norm0", nn.BatchNorm2d(num_features=features)),
                    ("conv1", nn.Conv2d(in_channels=features,
                                        out_channels=features,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                        bias=False,)),
                    ("lrelu1", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
                    ("norm1", nn.BatchNorm2d(num_features=features))
                ])
            )

        def forward(self, x):
            return self.conv_block(x)

    class _up_block(nn.Module):
        def __init__(self, in_channels, features):
            super().__init__()
            self.features = features
            self.up_conv = nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=2,
                stride=2,
                bias=False
            )
            self.conv_block = UNet2._block(2*features, features)

        def forward(self, x, link):
            assert(link.size(1) == self.features)
            x = self.up_conv(x)
            x = torch.cat((x, link), dim=1)
            return self.conv_block(x)
