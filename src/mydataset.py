#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
from matplotlib import pyplot as plt
from torchvision import transforms, utils
from tqdm.auto import tqdm

import utils


class MyDataset(torch.utils.data.Dataset):
    """Shadow removal dataset based on ISTD dataset."""
    in_channels: int = 3
    out_channels: int = 3

    def __init__(self,
                 root_dir,
                 subset="train",
                 transforms=None,
                 image_size=256):
        """
        Args:
            root_dir (string): Directory with all the images.
            subset ({"train", "test"}, optional): Select the set of data to load,
                default is "train".
            transforms (callable, optional): Optional transform to be applied
                on a sample.
            image_size (int, optional): The size that the images will be resized to,
                default is 256.
        """
        assert subset in ["train", "test"]
        self.root = root_dir
        self.transforms = transforms
        self.input_dir = os.path.join(root_dir, subset, subset + "_A")
        self.sp_dir = os.path.join(root_dir, subset, "sp")
        self.img_dir = os.path.join(
            root_dir, subset, subset + "_C_fixed_official")
        # list all images and targets,
        # sorting them without the suffix to ensure that they are aligned
        self.inputs = sorted(os.listdir(self.input_dir),
                             key=lambda f: os.path.splitext(f)[0])
        self.sps = sorted(os.listdir(self.sp_dir),
                          key=lambda f: os.path.splitext(f)[0])
        self.imgs = sorted(os.listdir(self.img_dir),
                           key=lambda f: os.path.splitext(f)[0])
        assert(len(self.inputs) == len((self.sps)))
        assert(len(self.inputs) == len((self.imgs)))
        # print("[Mydataset] Done creating {} dataset".format(subset))

    def __getitem__(self, idx):
        # leave the reading of images to __getitem__ for memory efficiency
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # read images and npy
        img_in = cv.imread(os.path.join(
            self.input_dir, self.inputs[idx]), cv.IMREAD_COLOR)
        # img_in = utils.uint2float(img_in)
        img_target = cv.imread(os.path.join(
            self.img_dir, self.imgs[idx]), cv.IMREAD_COLOR)
        # img_target = utils.uint2float(img_target)
        sp = np.load(os.path.join(
            self.sp_dir, self.sps[idx])).astype(np.float32)

        if self.transforms is not None:
            img_in, sp, img_target = self.transforms((img_in, sp, img_target))

        img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        # ndarray(H, W, C) to tensor(C, H, W)
        img_in_tensor = img_transform(img_in)
        img_target_tensor = img_transform(img_target)
        sp_tensor = torch.as_tensor(sp.transpose(2, 0, 1), dtype=torch.float32)

        filename = os.path.splitext(self.inputs[idx])[0]

        return (img_in_tensor, {"image": img_target_tensor, "sp": sp_tensor}, filename)

    def __len__(self):
        return len(self.inputs)
