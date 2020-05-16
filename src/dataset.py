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
        assert(len(self.img_files) == len((self.mask_files)))
        assert(len(self.img_files) == len((self.matte_files)))
        assert(len(self.img_files) == len((self.target_files)))
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
        else:
            self.datas = datas
            # self.imgs = [cv.imread(os.path.join(self.img_dir, f), cv.IMREAD_COLOR))
            #     for f in self.img_files]
            # self.targets = [cv.imread(os.path.join(self.target_dir, f), cv.IMREAD_COLOR))
            #     for f in self.target_files]
            # self.masks = [cv.imread(os.path.join(self.mask_dir, f), cv.IMREAD_GRAYSCALE))
            #     for f in self.mask_files]
            # self.mattes = [cv.imread(os.path.join(self.matte_dir, f), cv.IMREAD_GRAYSCALE))
            #     for f in self.matte_files]

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
            #     self.matte_dir, self.matte_files[idx])).astype(np.float32)
        else:
            for k in self.datas:
                sample[k] = utils.uint2float(self.datas[k][idx])
            # image = self.imgs[idx]
            # target = self.targets[idx]
            # mask = self.masks[idx]
            # matte = self.mattes[idx]
            # sp = self.sp[idx]

        # normalize = transform.Normalize(ISTDDataset.mean, ISTDDataset.std)
        # for k in sample:
        #     sample[k] = (sample[k] - 0.5) * 2
        # image = (image-0.5) * 2
        # mask = (mask-0.5) * 2
        # matte = (matte-0.5) * 2
        # target = (target-0.5) * 2
        sample_list = []
        for k in sorted(sample.keys()):
            sample_list.append(sample[k])

        if self.transforms is not None:
            sample_list = self.transforms(*sample_list)
            # image, mask, matte, target = self.transforms(
            #     image, mask, matte, target)

        for i in range(len(sample_list)):
            if sample_list[i].ndim == 2:
                sample_list[i] = sample_list[i][:, :, np.newaxis]
        # if "mask" in sample:
        #     sample["mask"] = sample["mask"][:, :, np.newaxis]
        # if "matte" in sample:
        #     sample["matte"] = sample["matte"][:, :, np.newaxis]

        filename = os.path.splitext(self.img_files[idx])[0]
        return_list = [filename]
        # ndarray(H, W, C) to tensor(C, H, W)
        for s in sample_list:
            return_list.append(torch.as_tensor(s.transpose(2, 0, 1),
                                               dtype=torch.float32))
        # image_tensor = torch.as_tensor(image.transpose(2, 0, 1),
        #                                dtype=torch.float32)
        # mask_tensor = torch.as_tensor(mask.transpose(2, 0, 1),
        #                               dtype=torch.float32)
        # matte_tensor = torch.as_tensor(matte.transpose(2, 0, 1),
        #                                dtype=torch.float32)
        # target_tensor = torch.as_tensor(target.transpose(2, 0, 1),
        #                                 dtype=torch.float32)
        # sp_tensor = torch.as_tensor(sp.transpose(2, 0, 1),
        #                             dtype=torch.float32)

        return tuple(return_list)
        # ["filename", "img", "mask", "matte", "target"]

    def __len__(self):
        return len(self.img_files)
