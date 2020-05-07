#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.utils.data

from cgan import CGAN


def main(args):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    makedirs(args)
    snapshotargs(args, filename=f"args-{time_str}.json")

    # use CuDNN backend
    torch.backends.cudnn.enabled = True
    if args.manual_seed != -1:
        set_manual_seed(args.manual_seed)

    log_file = os.path.join(
        args.logs, os.path.splitext(__file__)[0]+"-"+time_str+".log")
    # logger = Logger(log_file, level=logging.DEBUG)
    set_logger(log_file)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    logger.info(args)

    net = CGAN(args)

    if "train" in args.tasks:
        net.train(args.epochs)
    if "infer" in args.tasks:
        net.infer()


def set_logger(log_file):
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logFormatter = logging.Formatter(
        '%(asctime)s [%(module)s::%(funcName)s] %(levelname)s: %(message)s',
        datefmt="%H:%M:%S",
        style='%')
    rootLogger = logging.getLogger()
    rootLogger.setLevel(logging.INFO)

    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)


def set_manual_seed(manual_seed):
    """manual random seed for reproducible results"""
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(manual_seed)


def makedirs(args):
    os.makedirs(args.weights, exist_ok=True)
    os.makedirs(args.logs, exist_ok=True)


def snapshotargs(args, filename="args.json"):
    args_file = os.path.join(args.logs, filename)
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for shadow removal"
    )
    parser.add_argument(
        "tasks",
        help="the task to run (default: %(default)s)",
        type=str,
        nargs='+',
        default="train",
        choices=["train", "infer"])
    parser.add_argument(
        "--devices",
        help="device for training (default: %(default)s)",
        default=["cuda"], type=str, nargs='+',)
    parser.add_argument(
        "--batch-size",
        help="input batch size for training (default: %(default)d)",
        type=int,
        default=8,)
    parser.add_argument(
        "--epochs",
        help="number of epochs to train (default: %(default)d)",
        type=int,
        default=5000,)
    parser.add_argument(
        "--lr-D",
        help="initial learning rate of discriminator (default: %(default).5f)",
        default=0.0001, type=float,)
    parser.add_argument(
        "--lr-G",
        help="initial learning rate of generator (default: %(default).5f)",
        default=0.0001, type=float,)
    parser.add_argument(
        "--workers",
        help="number of workers for data loading (default: %(default)d)",
        type=int,
        default=4,)
    parser.add_argument(
        "--weights",
        help="folder to save weights (default: %(default)s)",
        type=str,
        default="../weights",)
    parser.add_argument(
        "--logs",
        help="folder to save logs (default: %(default)s)",
        type=str,
        default="../logs",)
    parser.add_argument(
        "--data-dir",
        help="root folder with images (default: %(default)s)",
        type=str,
        default="../ISTD_DATASET",)
    parser.add_argument(
        "--image-size",
        help="target input image size (default: %(default)d)",
        type=int,
        default=256,)
    parser.add_argument(
        "--aug-scale",
        help=("scale factor range for augmentation "
              "(default: %(default).2f)"),
        type=float,
        default=0.05,)
    parser.add_argument(
        "--aug-angle",
        help=("rotation range in degrees for augmentation "
              "(default: %(default)d)"),
        type=int,
        default=15,)
    parser.add_argument(
        "--net-g",
        help="the generator model (default: %(default)s)",
        type=str,
        default="mnet",
        choices=['unet', 'mnet', 'denseunet'])
    parser.add_argument(
        "--net-d",
        help="the discriminator model (default: %(default)s)",
        type=str,
        default="patchgan",
        choices=['patchgan'])
    parser.add_argument(
        "--load-weights-g",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-weights-d",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        '--manual_seed',
        help='manual random seed (default: %(default)s)',
        default=38107943,
        type=int)
    parser.add_argument(
        "--beta1",
        help=("Adam betas[0], "
              "DCGAN paper recommends .5 instead of the usual .9"),
        default=0.5, type=float)
    parser.add_argument(
        "--beta2",
        help="Adam betas[1]",
        default=0.999, type=float)
    args = parser.parse_args()
    main(args)
