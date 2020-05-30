#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch.nn as nn

from src.models import opt_layers


class SkipConnectionLayer(nn.Module):
    """
    Defines the Unet submodule with skip connection.
    +--------------------identity-------------------+
    |__ downsampling __ [submodule] __ upsampling __|
    """

    def __init__(self, down_block, up_block,
                 submodule=None,
                 use_selu=False,
                 drop_rate=0,):
        """Construct a Unet submodule with skip connections.
        Parameters:
            inner_nc (int) -- the number of filters in this layer
            input_nc (int) -- the number of channels in input
            submodule (_layer) -- previously defined submodules
            drop_rate (float)  -- dropout rate.
        """
        super().__init__()
        self.downsample = down_block
        self.submodule = submodule
        self.upsample = up_block
        self.dropout = opt_layers.get_dropout(use_selu=use_selu,
                                              drop_rate=drop_rate)

    def forward(self, x):
        y, link = self.downsample(x)
        if self.submodule is not None:
            y = self.submodule(y)
        z = self.upsample(y, link)
        if self.dropout is not None:
            return self.dropout(z)
        else:
            return z
