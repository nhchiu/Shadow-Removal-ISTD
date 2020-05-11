#!/usr/bin/env python
# -*- coding: utf-8 -*-

import h5py
import torch
import torch.utils.data

import transform


class ISTDDataset(torch.utils.data.Dataset):
    """Shadow removal dataset based on ISTD dataset."""
    in_channels: int = 3
    out_channels: int = 3

    # B, G, R
    mean = [0.54, 0.57, 0.57]
    std = [0.14, 0.14, 0.14]

    def __init__(self,
                 file,
                 subset="train",
                 transforms=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            subset ({"train", "test"}, optional): Select the set of data,
                default is "train".
            transforms (callable, optional): Optional transform to be applied
                on a sample.
        """
        super().__init__()
        assert subset in ["train", "test"]
        self.data_set = h5py.File(file, 'r')[subset]
        self.transforms = transforms

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # read images and npy
        input_img = self.data_set["input_img"][idx]
        target_img = self.data_set["target_img"][idx]
        sp = self.data_set["sp"][idx]
        filename = self.data_set["filename"][idx]

        normalize = transform.Normalize(ISTDDataset.mean, ISTDDataset.std)
        input_img, target_img = normalize(input_img, target_img)

        if self.transforms is not None:
            input_img, sp, target_img = \
                self.transforms(input_img, sp, target_img)

        # ndarray(H, W, C) to tensor(C, H, W)
        input_img_tensor = torch.as_tensor(input_img.transpose(2, 0, 1),
                                           dtype=torch.float32)
        target_img_tensor = torch.as_tensor(target_img.transpose(2, 0, 1),
                                            dtype=torch.float32)
        sp_tensor = torch.as_tensor(sp.transpose(2, 0, 1),
                                    dtype=torch.float32)

        return (filename,
                input_img_tensor,
                target_img_tensor,
                sp_tensor,)

    def __len__(self):
        return self.data_set["filename"].shape[0]
