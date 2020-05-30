#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
@inproceedings{isola2017image,
  title={Image-to-Image Translation with Conditional Adversarial Networks},
  author={Isola, Phillip and Zhu, Jun-Yan and Zhou,
          Tinghui and Efros, Alexei A},
  booktitle={Computer Vision and Pattern Recognition (CVPR),
             2017 IEEE Conference on},
  year={2017}
}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models import opt_layers


def conv_block(in_dim, out_dim, use_selu=False):
    return nn.Sequential(
        # nn.Conv2d(in_dim, in_dim, kernel_size=3, stride=1, padding=1),
        # opt_layers.get_norm(use_selu, in_dim),
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        opt_layers.get_norm(use_selu, out_dim),
        nn.MaxPool2d(kernel_size=2, stride=2))


def deconv_block(in_dim, out_dim, use_selu=False):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1),
        opt_layers.get_norm(use_selu, out_dim),
        # nn.Conv2d(out_dim, out_dim, kernel_size=3, stride=1, padding=1),
        # opt_layers.get_norm(use_selu, out_dim),
        nn.Upsample(scale_factor=2, mode='nearest'))


class BEGAN(nn.Module):

    def __init__(self, in_channels, out_channels=None,
                 ndf=64,
                 n_layers=3,
                 use_selu=False,
                 use_sigmoid=False, **kwargs):
        super(BEGAN, self).__init__()
        self.in_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                               out_channels=ndf,
                                               kernel_size=3,
                                               stride=1,
                                               padding=1),
                                     opt_layers.get_norm(use_selu, ndf))
        # (N, ndf, H, W)
        self.downsamples = nn.ModuleList()
        prev_channels = ndf
        for n in range(1, n_layers):  # increasing the number of filters
            self.downsamples.append(conv_block(prev_channels, ndf*n, use_selu))
            prev_channels = ndf*n
            # (N, ndf*n, H/(2**n), W/(2**n))
        # (N, ndf*(n_layers-1), H/(2**(n_layers-1), W/(2**(n_layers-1)))
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ndf*(n_layers-1), ndf,
                      kernel_size=3, stride=1, padding=1),
            nn.Conv2d(ndf, ndf,
                      kernel_size=3, stride=1, padding=1))
        # (N, ndf, H/(2**(n_layers-1), W/(2**(n_layers-1)))
        self.decoders = nn.ModuleList([deconv_block(ndf, ndf, use_selu)])
        # (N, ndf, H/(2**(n_layers-2), W/(2**(n_layers-2)))
        for n in reversed(range(1, n_layers-1)):
            self.decoders.append(deconv_block(2*ndf, ndf, use_selu))
            # (N, ndf, H/(2**(n-1), W/(2**(n-1)))
        # (N, ndf, H, W)
        if out_channels is None:
            out_channels = in_channels
        out_conv = nn.Conv2d(in_channels=ndf,
                             out_channels=out_channels,
                             kernel_size=3,
                             stride=1,
                             padding=1)
        act = nn.Sigmoid() if use_sigmoid else nn.Tanh()
        self.out_conv = nn.Sequential(out_conv, act)

    def forward(self, x):
        x = self.in_conv(x)
        for encoder in self.downsamples:
            x = encoder(x)
        x = self.bottleneck(x)
        y = x
        for i, decoder in enumerate(self.decoders):
            if i < len(self.decoders) - 1:
                y = torch.cat(
                    (F.interpolate(x, scale_factor=2**(i+1), mode='nearest'),
                     decoder(y)), dim=1)
            else:
                y = decoder(y)
        return self.out_conv(y)
