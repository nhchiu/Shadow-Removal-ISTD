#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import os
import time

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from torch.utils.data import DataLoader
from tqdm.auto import tqdm, trange

# from logger import Logger
from mydataset import MyDataset
from unet import UNet


def main(args):
    makedirs(args)
    device = torch.device(
        "cpu" if not torch.cuda.is_available() else args.device)

    loader = data_loader(args)

    with torch.set_grad_enabled(False):
        unet = UNet(in_channels=MyDataset.in_channels,
                    out_channels=MyDataset.out_channels)
        state_dict = torch.load(args.weights, map_location=device)
        unet.load_state_dict(state_dict)
        unet.eval()
        unet.to(device)

        # input_list = []
        # pred_list = []
        # true_list = []

        for i, data in tqdm(enumerate(loader), total=len(loader), desc="Loading data"):
            input_list = []
            pred_list = []

            x, _ = data
            x = x.to(device)

            y_pred = unet(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s].transpose(1, 2, 0)
                              for s in range(y_pred_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            input_list.extend([x_np[s].transpose(1, 2, 0)
                               for s in range(x_np.shape[0])])

            for p in range(len(input_list)):
                filename = loader.dataset.imgs[i*args.batch_size+p]
                sp_pred = pred_list[p]
                sp_pred = np.reshape(
                    sp_pred, (sp_pred.shape[0], sp_pred.shape[1], 3, 2))
                img_in = input_list[p]*255
                img_pred = np.clip(
                    (sp_pred[..., 0]*100 + sp_pred[..., 1]*img_in), 0, 255).astype(np.uint8)
                cv.imwrite(os.path.join(args.predictions, filename), img_pred)


def data_loader(args):
    dataset_valid = MyDataset(args.images, subset="test")
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=1
    )
    return loader_valid


def makedirs(args):
    os.makedirs(args.predictions, exist_ok=True)
    # os.makedirs("./sp", exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for shadow removal."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: %(default)s)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: %(default)d)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="path to weights file (required)"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="../ISTD_DATASET",
        help="root folder with images (default: %(default)s)"
    )
    # parser.add_argument(
    #     "--image-size",
    #     type=int,
    #     default=256,
    #     help="target input image size (default: 256)",
    # )
    parser.add_argument(
        "--predictions",
        type=str,
        default="../predictions",
        help="folder for saving images with prediction outlines (default: %(default)s)",
    )
    # parser.add_argument(
    #     "--figure",
    #     type=str,
    #     default="./dsc.png",
    #     help="filename for DSC distribution figure",
    # )

    args = parser.parse_args()
    main(args)
