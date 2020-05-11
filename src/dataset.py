#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

import cv2 as cv
import numpy as np
import torch
import torch.utils.data
import transform
import utils
# from torchvision import transforms, utils


class ISTDDataset(torch.utils.data.Dataset):
    """Shadow removal dataset based on ISTD dataset."""
    in_channels: int = 3
    out_channels: int = 3

    # B, G, R
    mean = [0.54, 0.57, 0.57]
    std = [0.14, 0.14, 0.14]

    def __init__(self, root_dir,
                 subset="train", transforms=None, preload=True):
        """
        Args:
            root_dir (string): Directory with all the images.
            subset ({"train", "test"}, optional): Select the set of data,
                default is "train".
            transforms (callable, optional): Optional transform to be applied
                on a sample.
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
        self.preload = preload
        if self.preload:
            self.input_imgs = [utils.uint2float(
                cv.imread(os.path.join(self.input_dir, f), cv.IMREAD_COLOR))
                for f in self.inputs]
            self.target_imgs = [utils.uint2float(
                cv.imread(os.path.join(self.img_dir, f), cv.IMREAD_COLOR))
                for f in self.imgs]
            self.sp = [np.load(
                os.path.join(self.sp_dir, f)).astype(np.float32)
                for f in self.sps]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # read images and npy
        if not self.preload:  # Load from disk
            input_image = cv.imread(os.path.join(
                self.input_dir, self.inputs[idx]), cv.IMREAD_COLOR)
            input_image = utils.uint2float(input_image)

            target_img = cv.imread(os.path.join(
                self.img_dir, self.imgs[idx]), cv.IMREAD_COLOR)
            target_img = utils.uint2float(target_img)

            sp = np.load(os.path.join(
                self.sp_dir, self.sps[idx])).astype(np.float32)
        else:
            assert self.input_imgs is not None
            assert self.target_imgs is not None
            assert self.sp is not None
            input_image = self.input_imgs[idx]
            target_img = self.target_imgs[idx]
            sp = self.sp[idx]

        normalize = transform.Normalize(ISTDDataset.mean, ISTDDataset.std)
        input_image, target_img = normalize(input_image, target_img)

        if self.transforms is not None:
            input_image, sp, target_img = self.transforms(
                input_image, sp, target_img)

        # ndarray(H, W, C) to tensor(C, H, W)
        input_img_tensor = torch.as_tensor(input_image.transpose(2, 0, 1),
                                           dtype=torch.float32)
        target_img_tensor = torch.as_tensor(target_img.transpose(2, 0, 1),
                                            dtype=torch.float32)
        sp_tensor = torch.as_tensor(sp.transpose(2, 0, 1),
                                    dtype=torch.float32)

        filename = os.path.splitext(self.inputs[idx])[0]

        return (filename,
                input_img_tensor,
                target_img_tensor,
                sp_tensor,)

    def __len__(self):
        return len(self.inputs)
