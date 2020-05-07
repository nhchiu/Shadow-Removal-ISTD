#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum, unique

import torch
from models.denseunet import DenseUNet
from models.mnet import MNet
from models.patchgan import PatchGAN
from models.unet import UNet


@torch.no_grad()
def weights_init(m):
    """custom weights initialization called on network model"""
    # classname = m.__class__.__name__
    # if classname.find('Conv') != -1 or \
    #         classname.find('BatchNorm') != -1 or \
    #         classname.find('Linear') != -1:
    #     pass
        # nn.init.normal_(m.weight.data, 0.0, 0.02)
        # if m.bias is not None:
        #     nn.init.constant_(m.bias.data, 0)


@unique()
class Generators(Enum):
    UNET = UNet
    MNET = MNet
    DENSEUNET = DenseUNet


@unique()
class Discriminators(Enum):
    PATCHGAN = PatchGAN


def get_generator(key: str, *args, **kwargs):
    return Generators[key.upper()].value(*args, **kwargs)


def get_discriminator(key: str, *args, **kwargs):
    return Discriminators[key.upper()].value(*args, **kwargs)


# """
# https://github.com/mateuszbuda/brain-segmentation-pytorch
# @article{buda2019association,
#   title={Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm},
#   author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
#   journal={Computers in Biology and Medicine},
#   volume={109},
#   year={2019},
#   publisher={Elsevier},
#   doi={10.1016/j.compbiomed.2019.05.002}
# }
# """


# class DenseUNet(nn.Module):
#     def __init__(self, in_channels=MyDataset.in_channels,
#                  out_channels=MyDataset.out_channels,
#                  ngf=48,
#                  drop_rate=0.01, **kwargs):
#         super(DenseUNet, self).__init__()
#         depth = 5
#         default_layers = 2
#         growth_rate = ngf // default_layers

#         self.in_conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=ngf,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#             bias=False
#         )

#         self.DenseBlockEncoders = nn.ModuleDict([
#             (f"DBEncoder{i}", DenseUNet._dense_block(
#                 ngf, layers=default_layers, growth_rate=growth_rate, drop_rate=drop_rate))
#             for i in range(depth)
#         ])

#         self.TransDowns = nn.ModuleDict([
#             (f"TD{i}", DenseUNet._trans_down(
#                 2*ngf, ngf, drop_rate=drop_rate))
#             for i in range(depth)
#         ])

#         self.bottleneck = DenseUNet._bottleneck(
#             ngf, layers=default_layers, growth_rate=growth_rate)

#         self.TransUps = nn.ModuleDict(
#             [
#                 (f"TU{depth-1}", DenseUNet._trans_up(2*ngf, ngf))
#             ]+[
#                 (f"TU{i}", DenseUNet._trans_up(4*ngf, ngf))
#                 for i in reversed(range(depth-1))
#             ])

#         self.DenseBlockDecoders = nn.ModuleDict(
#             [
#                 (f"DBDecoder{i}", DenseUNet._dense_block(
#                     3 * ngf, layers=default_layers, growth_rate=growth_rate))
#                 for i in reversed(range(depth))
#             ])

#         self.out_conv = nn.Conv2d(
#             in_channels=4*ngf,
#             out_channels=out_channels,
#             kernel_size=1,
#             stride=1,
#             bias=False
#         )

#     def forward(self, x):
#         x = self.in_conv(x)
#         # encode
#         links = []
#         for denseblock, trans_down in \
#             zip(self.DenseBlockEncoders.values(),
#                 self.TransDowns.values()):
#             x = denseblock(x)
#             links.append(x)
#             x = trans_down(x)

#         x = self.bottleneck(x)
#         # decode
#         for denseblock, trans_up in \
#             zip(self.DenseBlockDecoders.values(),
#                 self.TransUps.values()):
#             x = trans_up(x)
#             x = torch.cat((x, links.pop()), dim=1)
#             x = denseblock(x)

#         return self.out_conv(x)

#     @staticmethod
#     def _trans_down(in_channels, out_channels=None, drop_rate=None):
#         if out_channels is None:
#             out_channels = in_channels // 2
#         block = [nn.BatchNorm2d(num_features=in_channels),
#                  nn.Conv2d(in_channels=in_channels,
#                            out_channels=out_channels,
#                            kernel_size=1,
#                            stride=1,
#                            padding=0,
#                            bias=False)]
#         if drop_rate is not None and drop_rate > 0:
#             block.append(nn.Dropout2d(p=drop_rate, inplace=True))
#         block.append(nn.AvgPool2d(2))
#         return nn.Sequential(*block)

#     @staticmethod
#     def _trans_up(in_channels, out_channels=None):
#         if out_channels is None:
#             out_channels = in_channels // 4
#         return nn.ConvTranspose2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=2,
#             stride=2,
#             bias=False)

#     @staticmethod
#     def _bottleneck(in_channels, layers=8, growth_rate=8):
#         return DenseUNet._dense_block(
#             in_channels, layers=layers, growth_rate=growth_rate)

#     class _dense_block(nn.Module):
#         def __init__(self, in_channels,
#                      layers=4, growth_rate=8, drop_rate=None):
#             super().__init__()
#             self.composite_layers = nn.ModuleList([
#                 DenseUNet._dense_block._composite(
#                     in_channels+i*growth_rate,
#                     growth_rate,
#                     drop_rate) for i in range(layers)
#             ])

#         def forward(self, x):
#             for composite_layer in self.composite_layers:
#                 y = x
#                 x = composite_layer(x)
#                 x = torch.cat((x, y), dim=1)
#             return x

#         @staticmethod
#         def _composite(in_channels, growth_rate, drop_rate):
#             """
#             Create a composite layer in Dense Block.
#             """
#             layer = [
#                 nn.BatchNorm2d(num_features=in_channels),
#                 nn.LeakyReLU(negative_slope=0.1, inplace=True),
#                 nn.Conv2d(
#                     in_channels=in_channels,
#                     out_channels=growth_rate,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     padding_mode='reflect',
#                     bias=False)]
#             if drop_rate is not None and drop_rate > 0:
#                 layer.append(nn.Dropout2d(p=drop_rate, inplace=True))
#             return nn.Sequential(*layer)


# """
# https://github.com/mateuszbuda/brain-segmentation-pytorch
# @article{buda2019association,
#   title={Association of genomic subtypes of lower-grade gliomas
#   with shape features automatically extracted by a deep learning algorithm},
#   author={Buda, Mateusz and Saha, Ashirbani and Mazurowski, Maciej A},
#   journal={Computers in Biology and Medicine},
#   volume={109},
#   year={2019},
#   publisher={Elsevier},
#   doi={10.1016/j.compbiomed.2019.05.002}
# }
# """


# class UNet(nn.Module):

#     def __init__(self, in_channels=MyDataset.in_channels,
#                  out_channels=MyDataset.out_channels,
#                  ngf=64, **kwargs):
#         super(UNet, self).__init__()

#         features = ngf
#         self.conv = nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=features,
#             kernel_size=4,
#             stride=2,
#             padding=1,
#             bias=True
#         )
#         self.encoder1 = UNet._block(features, features * 2, name="enc1")
#         self.encoder2 = UNet._block(features * 2, features * 4, name="enc2")
#         self.encoder3 = UNet._block(features * 4, features * 8, name="enc3")
#         self.encoder4 = UNet._block(features * 8, features * 8, name="enc4")

#         self.decoder4 = UNet._up_block(features * 8, features * 8, name="dec4")
#         self.decoder3 = UNet._up_block(
#             (features * 8) * 2, features * 4, name="dec3")
#         self.decoder2 = UNet._up_block(
#             (features * 4) * 2, features * 2, name="dec2")
#         self.decoder1 = UNet._up_block(
#             (features * 2) * 2, features, name="dec1")

#         # self.activate = nn.Tanh()
#         self.up_conv = nn.ConvTranspose2d(in_channels=features * 2,
#                                           out_channels=out_channels,
#                                           kernel_size=4,
#                                           stride=2,
#                                           padding=1)

#     def forward(self, x):
#         conv = self.conv(x)
#         enc1 = self.encoder1(conv)
#         enc2 = self.encoder2(enc1)
#         enc3 = self.encoder3(enc2)
#         enc4 = self.encoder4(enc3)

#         dec4 = self.decoder4(enc4)
#         dec3 = self.decoder3(torch.cat((dec4, enc3), dim=1))
#         dec2 = self.decoder2(torch.cat((dec3, enc2), dim=1))
#         dec1 = self.decoder1(torch.cat((dec2, enc1), dim=1))
#         upconv = self.up_conv(torch.cat((dec1, conv), dim=1))
#         return upconv

#     @staticmethod
#     def _block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict([
#                 (name + "lrelu", nn.LeakyReLU(negative_slope=0.1, True)),
#                 (name + "conv", nn.Conv2d(in_channels=in_channels,
#                                           out_channels=features,
#                                           kernel_size=4,
#                                           stride=2,
#                                           padding=1,
#                                           bias=False)),
#                 (name + "norm", nn.BatchNorm2d(num_features=features))
#             ])
#         )

#     @staticmethod
#     def _up_block(in_channels, features, name):
#         return nn.Sequential(
#             OrderedDict([
#                 (name + "relu", nn.ReLU(inplace=True)),
#                 (name + "conv", nn.ConvTranspose2d(
#                     in_channels=in_channels,
#                     out_channels=features,
#                     kernel_size=4,
#                     stride=2,
#                     padding=1,
#                     bias=False)),
#                 (name + "norm", nn.BatchNorm2d(num_features=features))
#             ])
#         )


# class UNet2(nn.Module):

#     def __init__(self, in_channels=MyDataset.in_channels,
#                  out_channels=MyDataset.out_channels,
#                  in_features=64, **kwargs):
#         super(UNet2, self).__init__()
#         features = in_features
#         self.depth = 5

#         down_features = [in_channels]
#         for i in range(self.depth):
#             down_features.append(features * (2**i))

#         self.encoders = nn.ModuleList()
#         for i in range(self.depth):
#             self.encoders.append(
#                 UNet2._block(down_features[i], down_features[i+1]))

#         up_features = [features * (2**(self.depth-1))]
#         for i in reversed(range(self.depth-1)):
#             up_features.append(features * (2**i))

#         self.decoders = nn.ModuleList()
#         for i in range(self.depth-1):
#             self.decoders.append(
#                 UNet2._up_block(up_features[i], up_features[i+1]))

#         self.out_conv = nn.Conv2d(in_channels=features,
#                                   out_channels=out_channels,
#                                   kernel_size=1,
#                                   stride=1)

#     def forward(self, x):
#         # encode
#         links = []
#         for i, encoder in enumerate(self.encoders):
#             x = encoder(x)
#             if i != self.depth - 1:  # downsample except the last block
#                 links.append(x)
#                 x = F.max_pool2d(x, 2)
#         # decode
#         for i, decoder in enumerate(self.decoders):
#             linkage = links.pop()
#             x = decoder(x, linkage)
#         return self.out_conv(x)

#     class _block(nn.Module):
#         def __init__(self, in_channels, features):
#             super().__init__()
#             self.conv_block = nn.Sequential(
#                 OrderedDict([
#                     ("conv0", nn.Conv2d(in_channels=in_channels,
#                                         out_channels=features,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1,
#                                         bias=False)),
#                     ("lrelu0", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
#                     ("norm0", nn.BatchNorm2d(num_features=features)),
#                     ("conv1", nn.Conv2d(in_channels=features,
#                                         out_channels=features,
#                                         kernel_size=3,
#                                         stride=1,
#                                         padding=1,
#                                         bias=False)),
#                     ("lrelu1", nn.LeakyReLU(negative_slope=0.1, inplace=True)),
#                     ("norm1", nn.BatchNorm2d(num_features=features))
#                 ])
#             )

#         def forward(self, x):
#             return self.conv_block(x)

#     class _up_block(nn.Module):
#         def __init__(self, in_channels, features):
#             super().__init__()
#             self.features = features
#             self.up_conv = nn.ConvTranspose2d(
#                 in_channels=in_channels,
#                 out_channels=features,
#                 kernel_size=2,
#                 stride=2,
#                 bias=False)
#             self.conv_block = UNet2._block(2*features, features)

#         def forward(self, x, link):
#             assert(link.size(1) == self.features)
#             x = self.up_conv(x)
#             x = torch.cat((x, link), dim=1)
#             return self.conv_block(x)


# """
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
# @inproceedings{isola2017image,
#   title={Image-to-Image Translation with Conditional Adversarial Networks},
#   author={Isola, Phillip and Zhu, Jun-Yan and Zhou, Tinghui and Efros, Alexei A},
#   booktitle={Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference on},
#   year={2017}
# }
# """


# class Discriminator(nn.Module):

#     def __init__(self, in_channels=MyDataset.in_channels,
#                  in_channels2=MyDataset.out_channels,
#                  ndf=64,
#                  n_layers=3, **kwargs):
#         super(Discriminator, self).__init__()
#         ksize = 4
#         self.blocks = nn.ModuleList([
#             nn.Sequential(
#                 nn.Conv2d(in_channels=in_channels+in_channels2,
#                           out_channels=ndf,
#                           kernel_size=ksize,
#                           stride=2,
#                           padding=1),
#                 nn.LeakyReLU(0.2, inplace=True)
#             )
#         ])
#         # (N, features, H/2, W/2)
#         prev_channels = ndf
#         for n in range(1, n_layers):  # gradually increase the number of filters
#             if n < 4:
#                 self.blocks.append(self._block(prev_channels, prev_channels*2))
#                 prev_channels *= 2
#                 # (N, features*(2**n), H/(2**(n+1)), W/(2**(n+1)))
#             else:
#                 self.blocks.append(self._block(prev_channels, prev_channels))
#                 # (N, features*(2**3), H/(2**(n+1)), W/(2**(n+1)))

#         out_channels = prev_channels*2 if n_layers < 4 else prev_channels
#         self.blocks.append(
#             nn.Sequential(nn.Conv2d(in_channels=prev_channels,
#                                     out_channels=out_channels,
#                                     kernel_size=ksize,
#                                     stride=1,
#                                     padding=1,
#                                     padding_mode='reflect',
#                                     bias=False),
#                           nn.BatchNorm2d(num_features=out_channels),
#                           nn.LeakyReLU(0.2, inplace=True)))
#         # (N, features*(2**min(n_layers, 3)), H/(2**(n_layers)) - 1, W/(2**(n_layers)) - 1)

#         # squeeze to 1 channel prediction map
#         self.blocks.append(
#             nn.Conv2d(out_channels, 1,
#                       kernel_size=ksize, stride=1, padding=1,
#                       padding_mode='reflect', bias=False))
#         # (N, features*(2**min(n_layers, 3)), H/(2**(n_layers)) - 2, W/(2**(n_layers)) - 2)
#         # self.activation = nn.Sigmoid()
#         # Use BCEWithLogitsLoss instead of BCELoss!

#     def forward(self, img_in, sp):
#         x = torch.cat((img_in, sp), dim=1)
#         for block in self.blocks:
#             x = block(x)
#         return x

#     def _block(self, in_channels, out_channels=None):
#         """ Conv -> BatchNorm -> LeakyReLU"""
#         if out_channels is None:
#             out_channels = in_channels * 2
#         return nn.Sequential(
#             nn.Conv2d(in_channels=in_channels,
#                       out_channels=out_channels,
#                       kernel_size=4,
#                       stride=2,
#                       padding=1,
#                       padding_mode='reflect',
#                       bias=False),
#             nn.BatchNorm2d(num_features=out_channels),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True))
