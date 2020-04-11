#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import logging
import os
import sys
import time

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm, trange

from logger import Logger
from mydataset import MyDataset
from transform import transforms
from unet import UNet
from loss import CustomLoss


def main(args):
    makedirs(args)
    snapshotargs(args)

    log_file = os.path.join(args.logs, __file__.split('.')[0]+".log")

    logger = Logger(log_file, level=logging.DEBUG)

    device = torch.device(
        "cpu" if not torch.cuda.is_available() else args.device)
    logger.logger.info("Using device: {:s}".format(device.__str__()))

    logger.logger.debug("Creating data loaders")
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    logger.logger.debug("Creating model")
    unet = UNet(in_channels=MyDataset.in_channels,
                out_channels=MyDataset.out_channels)
    unet.to(device)

    # smooth_l1_loss = nn.L1Loss()
    smooth_l1_loss = CustomLoss(args.loss_weight)
    best_validation_loss = 1000.0

    optimizer = optim.Adam(unet.parameters(), lr=args.lr)

#     logger = Logger(args.logs)
    loss_train = []
    loss_valid = []

    step = 0

    logger.logger.debug("Start training")
    start_time = time.time()
    progress = logger.trange(args.epochs, desc="train")
    for epoch in progress:
        for phase in ["train", "valid"]:
            progress.set_description(phase)
            if phase == "train":
                unet.train()
            else:
                unet.eval()

            # validation_pred = []
            # validation_true = []

            for i, data in enumerate(loaders[phase]):
                progress.set_description(phase+" batch {:<4d}".format(i))
                if phase == "train":
                    step += 1

                x, y_true = data
                x = x.to(device)
                y_true_img = y_true["image"].to(device)
                y_true_sp = y_true["sp"].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    y_pred = unet(x)

                    loss = smooth_l1_loss(x, y_pred, y_true_img, y_true_sp)

                    if phase == "valid":
                        loss_valid.append(loss.item())
                        # y_pred_np = y_pred.detach().cpu().numpy()
                        # validation_pred.extend(
                        #     [y_pred_np[s]
                        #         for s in range(y_pred_np.shape[0])]
                        # )
                        # y_true_np = y_true.detach().cpu().numpy()
                        # validation_true.extend(
                        #     [y_true_np[s]
                        #         for s in range(y_true_np.shape[0])]
                        # )
                        # if (epoch % args.vis_freq == 0) or (epoch == args.epochs - 1):
                        #     if i * args.batch_size < args.vis_images:
                        #         tag = "image/{}".format(i)
                        #         num_images = args.vis_images - i * args.batch_size
                        # logger.image_list_summary(
                        #     tag,
                        #     log_images(x, y_true, y_pred)[:num_images],
                        #     step,
                        # )

                    if phase == "train":
                        loss_train.append(loss.item())
                        loss.backward()
                        optimizer.step()

                if phase == "train" and (step + 1) % 10 == 0:
                    # log_loss_summary(logger, loss_train, step)
                    loss_train = []

            if phase == "valid":
                # log_loss_summary(logger, loss_valid, step, prefix="val_")
                mean_loss = np.mean(np.array(loss_valid))
                # logger.scalar_summary("val_dsc", mean_dsc, step)
                if mean_loss < best_validation_loss:
                    best_validation_loss = mean_loss
                    torch.save(unet.state_dict(), os.path.join(
                        args.weights, "unet_best.pt"))
                    logger.logger.info("Network saved after epoch {}, error = {}".format(
                        epoch, best_validation_loss))
                loss_valid = []

            torch.save(unet.state_dict(), os.path.join(
                args.weights, "unet_latest.pt"))

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.logger.info('Training time {:s}'.format(total_time_str))
    logger.logger.info(
        "Best validation loss: {:4f}".format(best_validation_loss))


def data_loaders(args):
    dataset_train, dataset_valid = datasets(args)

    def worker_init(worker_id):
        np.random.seed(42 + worker_id)

    loader_train = DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
    )

    return loader_train, loader_valid


def datasets(args):
    train = MyDataset(args.images, subset="train",
                      transforms=transforms(
                          scale=args.aug_scale,
                          angle=args.aug_angle,
                          flip_prob=0.5,
                          crop_size=args.image_size
                      ))
    valid = MyDataset(args.images, subset="test")
    return train, valid


# def dsc_per_volume(validation_pred, validation_true, patient_slice_index):
#     dsc_list = []
#     num_slices = np.bincount([p[0] for p in patient_slice_index])
#     index = 0
#     for p in range(len(num_slices)):
#         y_pred = np.array(validation_pred[index : index + num_slices[p]])
#         y_true = np.array(validation_true[index : index + num_slices[p]])
#         dsc_list.append(dsc(y_pred, y_true))
#         index += num_slices[p]
#     return dsc_list

# def log_loss_summary(logger, loss, step, prefix=""):
#     logger.scalar_summary(prefix + "loss", np.mean(loss), step)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args):
    args_file = os.path.join(args.logs, "args.json")
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp)


# def setlogger(args):
#     log_file = os.path.join(args.logs, __file__.split('.')[0]+".log")
#     logger = logging.getLogger('__file__')
#     logger.setLevel(logging.DEBUG)
#     # create file handler which logs even debug messages
#     fh = logging.FileHandler(log_file, mode='w')
#     fh.setLevel(logging.DEBUG)
#     # create console handler with a higher log level
#     ch = logging.StreamHandler()
#     ch.setLevel(logging.ERROR)
#     # create formatter and add it to the handlers
#     file_formatter = logging.Formatter(
#         '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s',
#         datefmt="",
#         style='{')
#     fh.setFormatter(file_formatter)
#     console_formatter = logging.Formatter(
#         '%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s: %(message)s')
#     ch.setFormatter(console_formatter)
#     # add the handlers to logger
#     logger.addHandler(ch)
#     logger.addHandler(fh)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for shadow removal"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="input batch size for training (default: %(default)d)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=250,
        help="number of epochs to train (default: %(default)d)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="initial learning rate (default: %(default).4f)",
    )
    parser.add_argument(
        "--loss-weight",
        type=float,
        default=5.0,
        help="weight of img loss (default: %(default).2f)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="device for training (default: %(default)s)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="number of workers for data loading (default: %(default)d)",
    )
    # parser.add_argument(
    #     "--vis-images",
    #     type=int,
    #     default=200,
    #     help="number of visualization images to save in log file (default: 200)",
    # )
    # parser.add_argument(
    #     "--vis-freq",
    #     type=int,
    #     default=10,
    #     help="frequency of saving images to log file (default: 10)",
    # )
    parser.add_argument(
        "--weights",
        type=str,
        default="../weights",
        help="folder to save weights (default: %(default)s)"
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="../logs",
        help="folder to save logs (default: %(default)s)"
    )
    parser.add_argument(
        "--images",
        type=str,
        default="../ISTD_DATASET",
        help="root folder with images (default: %(default)s)"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        help="target input image size (default: %(default)d)",
    )
    parser.add_argument(
        "--aug-scale",
        type=float,
        default=0.05,
        help="scale factor range for augmentation (default: %(default).2f)",
    )
    parser.add_argument(
        "--aug-angle",
        type=int,
        default=15,
        help="rotation angle range in degrees for augmentation (default: %(default)d)",
    )
    args = parser.parse_args()
    main(args)
