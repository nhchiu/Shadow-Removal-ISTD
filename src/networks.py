#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, unique

import torch
import torch.nn as nn

from models.denseunet import DenseUNet
from models.mnet import MNet
from models.patchgan import PatchGAN
from models.unet import UNet


@torch.no_grad()
def weights_init(m):
    """custom weights initialization called on network model"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


@unique
class Generators(Enum):
    UNET = UNet
    MNET = MNet
    DENSEUNET = DenseUNet


@unique
class Discriminators(Enum):
    PATCHGAN = PatchGAN


def get_generator(key: str, *args, **kwargs):
    return Generators[key.upper()].value(*args, **kwargs)


def get_discriminator(key: str, *args, **kwargs):
    return Discriminators[key.upper()].value(*args, **kwargs)
