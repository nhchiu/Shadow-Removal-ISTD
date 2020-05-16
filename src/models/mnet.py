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

from skip_connection_layer import SkipConnectionLayer


class MNet(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 ngf=64,
                 drop_rate=0,
                 no_conv_t=True,
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
        # self.encoders = nn.ModuleList()
        # for i in range(depth):
        #     features_in = (2 ** min(i, 3)) * ngf
        #     features_out = (2 ** min(i+1, 3)) * ngf
        #     self.encoders.append(MNet._block(features_in, features_out))
        # self.encoder1 = MNet._block(ngf, ngf * 2)
        # self.encoder2 = MNet._block(ngf * 2, ngf * 4)
        # self.encoder3 = MNet._block(ngf * 4, ngf * 8)
        # self.encoder4 = MNet._block(ngf * 8, ngf * 8)

        # self.decoders = nn.ModuleList()
        # for i in reversed(range(depth)):
        #     features_in = (2 ** min(i+1, 3)) * ngf
        #     features_out = (2 ** min(i, 3)) * ngf
        #     self.decoders.append(MNet._up_block(features_in,
        #                                         features_out, no_conv_t)
        #                          if i == depth-1 else
        #                          MNet._up_block(features_in*2,
        #                                         features_out, no_conv_t))

        # self.decoder4 = MNet._up_block(ngf * 8, ngf * 8)
        # self.decoder3 = MNet._up_block(
        #     (ngf * 8) * 2, ngf * 4)
        # self.decoder2 = MNet._up_block(
        #     (ngf * 4) * 2, ngf * 2)
        # self.decoder1 = MNet._up_block(
        #     (ngf * 2) * 2, ngf)

        # self.activate = nn.Tanh()
        self.up_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(in_channels=ngf * 2,
                      out_channels=out_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      padding_mode="reflect",
                      bias=False)) if no_conv_t else \
            nn.ConvTranspose2d(in_channels=ngf * 2,
                               out_channels=out_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False)

    def forward(self, x):
        # conv = self.conv(x)
        # enc1 = self.encoder1(conv)
        # enc2 = self.encoder2(enc1)
        # enc3 = self.encoder3(enc2)
        # enc4 = self.encoder4(enc3)

        # dec4 = self.decoder4(enc4)
        # dec3 = self.decoder3(torch.cat((dec4, enc3), dim=1))
        # dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
        # dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))
        # upconv = self.up_conv(torch.cat((dec1, conv), dim=1))
        x = self.conv(x)
        # links = []
        # for encoder in self.encoders:
        #     links.append(x)
        #     x = encoder(x)

        # for decoder in self.decoders:
        #     x = decoder(x)
        #     x = torch.cat((x, links.pop()), dim=1)

        # assert len(links) == 0
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
        if no_conv_t:
            upconv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(in_channels=in_channels,
                          out_channels=features,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          padding_mode="reflect",
                          bias=False))
        else:
            upconv = nn.ConvTranspose2d(in_channels=in_channels,
                                        out_channels=features,
                                        kernel_size=4,
                                        stride=2,
                                        padding=1,
                                        bias=False)
        self.model = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            upconv,
            nn.BatchNorm2d(num_features=features)
        )

    def forward(self, x, link):
        return torch.cat((self.model(x), link), dim=1)


# def _block(in_channels, features):
#     return nn.Sequential(
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         nn.Conv2d(in_channels=in_channels,
#                   out_channels=features,
#                   kernel_size=4,
#                   stride=2,
#                   padding=1,
#                   padding_mode='reflect',
#                   bias=False),
#         nn.BatchNorm2d(num_features=features)
#     )


# def _up_block(in_channels, features, no_conv_t):
#     if no_conv_t:
#         upconv = nn.Sequential(
#             nn.Upsample(scale_factor=2),
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=features,
#                       kernel_size=3,
#                       stride=1,
#                       padding=1,
#                       padding_mode="reflect",
#                       bias=False))
#     else:
#         upconv = nn.ConvTranspose2d(in_channels=in_channels,
#                                     out_channels=features,
#                                     kernel_size=4,
#                                     stride=2,
#                                     padding=1,
#                                     bias=False)
#     return nn.Sequential(
#         nn.LeakyReLU(negative_slope=0.2, inplace=True),
#         upconv,
#         nn.BatchNorm2d(num_features=features)
#     )
