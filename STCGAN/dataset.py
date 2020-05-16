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
    # mean = [0.54, 0.57, 0.57]
    # std = [0.14, 0.14, 0.14]

    def __init__(self, root_dir,
                 subset,
                 datas: list = ["img", "mask", "target"],
                 transforms=None, preload=False):
        """
        Args:
            root_dir (string): Directory with all the images.
            subset {"train", "test"}: Select the set of data.
            transforms (callable, optional): Optional transform to be applied
                on a sample.
            datas: what data to load.
                a list of {img, mask, target, matte}.
        """
        assert subset in ["train", "test"]
        self.root = root_dir
        self.transforms = transforms
        self.img_dir = os.path.join(root_dir, subset, subset + "_A")
        self.mask_dir = os.path.join(root_dir, subset, subset + "_B")
        self.matte_dir = os.path.join(root_dir, subset, subset + "_matte")
        self.target_dir = os.path.join(root_dir, subset, subset + "_C_fixed")
        # self.sp_dir = os.path.join(root_dir, subset, "sp")

        # list all images and targets,
        # sorting them without the suffix to ensure that they are aligned
        self.img_files = sorted(os.listdir(self.img_dir),
                                key=lambda f: os.path.splitext(f)[0])
        self.mask_files = sorted(os.listdir(self.mask_dir),
                                 key=lambda f: os.path.splitext(f)[0])
        self.matte_files = sorted(os.listdir(self.matte_dir),
                                  key=lambda f: os.path.splitext(f)[0])
        self.target_files = sorted(os.listdir(self.target_dir),
                                   key=lambda f: os.path.splitext(f)[0])
        # self.sps = sorted(os.listdir(self.sp_dir),
        #                   key=lambda f: os.path.splitext(f)[0])
        assert(len(self.img_files) == len((self.mask_files)))
        assert(len(self.img_files) == len((self.matte_files)))
        assert(len(self.img_files) == len((self.target_files)))
        # assert(len(self.img_files) == len((self.sps)))
        self.preload = preload
        if self.preload:
            self.datas = {}
            if "img" in datas:
                self.datas["img"] = [cv.imread(os.path.join(
                    self.img_dir, f), cv.IMREAD_COLOR)
                    for f in self.img_files]
            if "mask" in datas:
                self.datas["mask"] = [cv.imread(os.path.join(
                    self.mask_dir, f), cv.IMREAD_GRAYSCALE)
                    for f in self.mask_files]
            if "matte" in datas:
                self.datas["matte"] = [cv.imread(os.path.join(
                    self.matte_dir, f), cv.IMREAD_GRAYSCALE)
                    for f in self.matte_files]
            if "target" in datas:
                self.datas["target"] = [cv.imread(os.path.join(
                    self.target_dir, f), cv.IMREAD_COLOR)
                    for f in self.target_files]
            # self.sp = [np.load(
            #     os.path.join(self.sp_dir, f)).astype(np.float32)
            #     for f in self.sps]
        else:
            self.datas = datas

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # read images and npy
        sample = {}
        if not self.preload:  # Load from disk
            if "img" in self.datas:
                image = cv.imread(os.path.join(
                    self.img_dir, self.img_files[idx]), cv.IMREAD_COLOR)
                sample["img"] = utils.uint2float(image)

            if "mask" in self.datas:
                mask = cv.imread(os.path.join(
                    self.mask_dir, self.mask_files[idx]), cv.IMREAD_GRAYSCALE)
                sample["mask"] = utils.uint2float(mask)

            if "matte" in self.datas:
                matte = cv.imread(os.path.join(
                    self.matte_dir, self.matte_files[idx]), cv.IMREAD_GRAYSCALE)
                sample["matte"] = utils.uint2float(matte)

            if "target" in self.datas:
                target = cv.imread(os.path.join(
                    self.target_dir, self.target_files[idx]), cv.IMREAD_COLOR)
                sample["target"] = utils.uint2float(target)
            # sp = np.load(os.path.join(
            #     self.sp_dir, self.sps[idx])).astype(np.float32)
        else:
            # assert self.input_imgs is not None
            # assert self.target_imgs is not None
            # assert self.sp is not None
            for k in self.datas:
                sample[k] = utils.uint2float(self.datas[k][idx])

        # normalize = transform.Normalize(ISTDDataset.mean, ISTDDataset.std)
        input_image = (input_image-0.5) * 2
        mask_img = (mask_img-0.5) * 2
        target_img = (target_img-0.5) * 2
        sample_list = []
        for k in sorted(sample.keys()):
            sample_list.append(sample[k])

        if self.transforms is not None:
            sample_list = self.transforms(*sample_list)
            # input_image, mask_img, target_img = self.transforms(
            #     input_image, mask_img, target_img)
        for i in range(len(sample_list)):
            # add a new axis to images with only one channel
            if sample_list[i].ndim == 2:
                sample_list[i] = sample_list[i][:, :, np.newaxis]

        filename = os.path.splitext(self.img_files[idx])[0]
        return_list = [filename]
        # ndarray(H, W, C) to tensor(C, H, W)
        for s in sample_list:
            return_list.append(torch.as_tensor(s.transpose(2, 0, 1),
                                               dtype=torch.float32))

        return tuple(return_list)
        # ["filename", "img", "mask", "matte", "target"]

    def __len__(self):
        return len(self.img_files)
