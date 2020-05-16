#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch.nn as nn


class SkipConnectionLayer(nn.Module):
    """
    Defines the Unet submodule with skip connection.
    +--------------------identity-------------------+
    |__ downsampling __ [submodule] __ upsampling __|
    """

    def __init__(self, down_block, up_block,
                 submodule=None,
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
        self.upsample = up_block
        self.submodule = submodule
        self.dropout = nn.Dropout2d(p=drop_rate) if drop_rate > 0 else None

    def forward(self, x):
        y, link = self.downsample(x)
        if self.submodule is not None:
            y = self.submodule(y)
        z = self.upsample(y, link)
        if self.dropout is not None:
            return self.dropout(z)
        else:
            return z
