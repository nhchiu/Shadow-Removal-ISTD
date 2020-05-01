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

import utils
from denseunet import DenseUNet
# from logger import Logger
from mydataset import MyDataset
from unet import UNet
from unet2 import UNet2


def main(args):
    makedirs(args)
    # use CuDNN backend
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    device = torch.device(
        "cpu" if not torch.cuda.is_available() else args.device)

    loader = data_loader(args)

    network_choices = {
        'unet': UNet,
        'unet2': UNet2,
        'denseunet': DenseUNet
    }
    network = network_choices.get(args.net, DenseUNet)

    with torch.set_grad_enabled(False):
        generator = network(in_channels=MyDataset.in_channels,
                            out_channels=MyDataset.out_channels)
        state_dict = torch.load(args.weights, map_location=device)
        generator.load_state_dict(state_dict)
        generator.eval()
        generator.to(device)

        for (x, _, filenames) in tqdm(loader, desc="Processing data"):
            input_list = []
            pred_list = []

            x = x.to(device)

            y_pred = generator(x)
            y_pred_np = y_pred.detach().cpu().numpy()
            pred_list.extend([y_pred_np[s].transpose(1, 2, 0)
                              for s in range(y_pred_np.shape[0])])

            x_np = x.detach().cpu().numpy()
            x_np = (x_np * 0.5)+0.5
            input_list.extend([x_np[s].transpose(1, 2, 0)
                               for s in range(x_np.shape[0])])

            assert len(input_list) == len(pred_list)
            for (img_in, sp_pred, name) in zip(input_list, pred_list, filenames):
                # filename = data.imgs[i*args.batch_size+p]
                # sp_pred = np.reshape(
                #     sp_pred, (sp_pred.shape[0], sp_pred.shape[1], 3, 2))
                img_pred = utils.float2uint(utils.apply_sp(img_in, sp_pred))
                cv.imwrite(os.path.join(
                    args.predictions, name+".png"), img_pred)
                if args.save_sp:
                    np.save(os.path.join(
                        f"{args.predictions}-sp", name), sp_pred)


def data_loader(args):
    dataset_valid = MyDataset(args.data_dir, subset="test")
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=1
    )
    return loader_valid


def makedirs(args):
    os.makedirs(args.predictions, exist_ok=True)
    if args.save_sp:
        os.makedirs(args.predictions+"-sp", exist_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inference for shadow removal."
    )
    parser.add_argument(
        "--device", help="device for inference (default: %(default)s)",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--batch-size", help="input batch size for inference (default: %(default)d)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--weights", help="path to generator weights file (required)",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--data-dir", help="root folder with images (default: %(default)s)",
        type=str,
        default="../ISTD_DATASET",
    )
    parser.add_argument(
        "--predictions", help="folder for saving images with prediction outlines (default: %(default)s)",
        type=str,
        default="../predictions",
    )
    parser.add_argument(
        "--save-sp", help="whether to save the shadow parameters (default: %(default)s)",
        type=bool,
        nargs='?',
        const=True,
        default=True,
    )
    parser.add_argument(
        "--net", help="the generator network module (default: %(default)s)",
        type=str,
        default="denseunet",
        choices=['unet', 'unet2', 'denseunet']
    )

    args = parser.parse_args()
    main(args)
