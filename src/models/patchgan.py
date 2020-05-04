#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
  year={2017}
}
"""

import torch
import torch.nn as nn


class PatchGAN(nn.Module):

    def __init__(self,
                 in_channels,
                 in_channels2,
                 ndf=64,
                 n_layers=3, **kwargs):
        super(PatchGAN, self).__init__()
        ksize = 4
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels=in_channels+in_channels2,
                          out_channels=ndf,
                          kernel_size=ksize,
                          stride=2,
                          padding=1),
                nn.LeakyReLU(0.2, inplace=True)
            )
        ])
        # (N, ndf, H/2, W/2)
        prev_channels = ndf
        for n in range(1, n_layers):  # gradually increase the number of filters
            if n < 4:
                self.blocks.append(self._block(prev_channels, prev_channels*2))
                prev_channels *= 2
                # (N, ndf*(2**n), H/(2**(n+1)), W/(2**(n+1)))
            else:
                self.blocks.append(self._block(prev_channels, prev_channels))
                # (N, ndf*(2**3), H/(2**(n+1)), W/(2**(n+1)))

        out_channels = prev_channels*2 if n_layers < 4 else prev_channels
        self.blocks.append(
            nn.Sequential(
                nn.Conv2d(in_channels=prev_channels,
                          out_channels=out_channels,
                          kernel_size=ksize,
                          stride=1,
                          padding=1,
                          padding_mode='reflect',
                          bias=False),
                nn.BatchNorm2d(num_features=out_channels),
                nn.LeakyReLU(0.2, inplace=True)
            )
        )
        # (N, ndf*(2**min(n_layers, 3)), H/(2**(n_layers)) - 1, W/(2**(n_layers)) - 1)

        # squeeze to 1 channel prediction map
        self.blocks.append(
            nn.Conv2d(out_channels, 1,
                      kernel_size=ksize, stride=1, padding=1,
                      padding_mode='reflect', bias=False)
        )
        # (N, ndf*(2**min(n_layers, 3)), H/(2**(n_layers)) - 2, W/(2**(n_layers)) - 2)
        # self.activation = nn.Sigmoid()
        # Use BCEWithLogitsLoss instead of BCELoss!

    def forward(self, img_in, sp):
        x = torch.cat((img_in, sp), dim=1)
        for block in self.blocks:
            x = block(x)
        return x

    def _block(self, in_channels, out_channels=None):
        """ Conv -> BatchNorm -> LeakyReLU"""
        if out_channels is None:
            out_channels = in_channels * 2
        return nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=4,
                stride=2,
                padding=1,
                padding_mode='reflect',
                bias=False),
            nn.BatchNorm2d(num_features=out_channels),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
