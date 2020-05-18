#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Predict a grayscale shadow matte from input image
"""

import argparse
import json
import logging
import os
import random
import time

import numpy as np
import torch
import torch.utils.data

from src.cgan import CGAN


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
        "%(asctime)s [%(module)s::%(funcName)s] %(levelname)s: %(message)s",
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
    arg_str = f"_lr{args.lr_G:.5f}_"
    if args.D_loss_type == "normal":
        arg_str += ""
    elif args.D_loss_type == "rel":
        arg_str += "Rp"
    else:
        arg_str += "Ra"
    if args.D_loss_fn == "standard":
        arg_str += "SGAN"
    else:
        arg_str += "LSGAN"
    args.weights += arg_str
    args.logs += arg_str
    os.makedirs(args.logs, exist_ok=True)
    if "train" in args.tasks:
        os.makedirs(args.weights, exist_ok=True)
    if "infer" in args.tasks:
        os.makedirs(args.infered, exist_ok=True)
        os.makedirs(os.path.join(args.infered, "shadowless"), exist_ok=True)
        os.makedirs(os.path.join(args.infered, "matte"), exist_ok=True)


def snapshotargs(args, filename="args.json"):
    args_file = os.path.join(args.logs, filename)
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Training U-Net model for shadow removal"
    )
    parser.add_argument(
        "--tasks",
        help="the task to run (default: %(default)s)",
        required=True, choices=["train", "infer"], type=str, nargs='+',)
    parser.add_argument(
        "--devices",
        help="device for training (default: %(default)s)",
        default=["cuda"], type=str, nargs='+',)
    parser.add_argument(
        "--batch-size",
        help="input batch size for training (default: %(default)d)",
        default=16, type=int,)
    parser.add_argument(
        "--epochs",
        help="number of epochs to train (default: %(default)d)",
        default=100000, type=int,)
    parser.add_argument(
        "--lr-D",
        help="initial learning rate of discriminator (default: %(default).5f)",
        default=0.0002, type=float,)
    parser.add_argument(
        "--lr-G",
        help="initial learning rate of generator (default: %(default).5f)",
        default=0.0005, type=float,)
    parser.add_argument(
        "--decay",
        help=("Decay to apply to lr each cycle. (default: %(default).6f)"
              "(1-decay)^n_iter * lr gives the final lr. "
              "e.g. 0.00002 will lead to .13 of lr after 100k cycles"),
        default=0.00005, type=float)
    parser.add_argument(
        "--workers",
        help="number of workers for data loading (default: %(default)d)",
        default=4, type=int,)
    parser.add_argument(
        "--weights",
        help="folder to save weights (default: %(default)s)",
        default="../weights", type=str,)
    parser.add_argument(
        "--infered",
        help="folder to save infered images (default: %(default)s)",
        default="../infered", type=str,)
    parser.add_argument(
        "--logs",
        help="folder to save logs (default: %(default)s)",
        default="../logs", type=str,)
    parser.add_argument(
        "--data-dir",
        help="root folder with images (default: %(default)s)",
        default="../ISTD_DATASET", type=str,)
    parser.add_argument(
        "--image-size",
        help="target input image size (default: %(default)d)",
        default=256, type=int,)
    parser.add_argument(
        "--aug-scale",
        help=("scale factor range for augmentation "
              "(default: %(default).2f)"),
        default=0.05, type=float,)
    parser.add_argument(
        "--aug-angle",
        help=("rotation range in degrees for augmentation "
              "(default: %(default)d)"),
        default=15, type=int,)
    parser.add_argument(
        "--net-G",
        help="the generator model (default: %(default)s)",
        default="mnet", choices=["unet", "mnet", "denseunet"], type=str,)
    parser.add_argument(
        "--net-D",
        help="the discriminator model (default: %(default)s)",
        default="patchgan", choices=["patchgan"], type=str,)
    parser.add_argument(
        "--load-weights-g1",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-weights-g2",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-weights-d1",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-weights-d2",
        help="load weights to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--D-loss-fn",
        help="loss funtion of discriminator (default: %(default)s)",
        default="standard", choices=["standard", "leastsquare"], type=str)
    parser.add_argument(
        "--D-loss-type",
        help="Use relative discriminator loss (default: %(default)s)",
        default="normal", choices=["normal", "rel", "rel_avg"], type=str)
    parser.add_argument(
        "--softadapt", help="Adapt the weight of losses dynamically",
        type=bool, default=False, const=True, nargs='?')
    parser.add_argument(
        "--manual_seed",
        help="manual random seed (default: %(default)s)",
        default=38107943, type=int)
    parser.add_argument(
        "--SELU",
        help=("Using scaled exponential linear units (SELU) "
              "which are self-normalizing instead of ReLU with BatchNorm. "
              "Used only in arch=0. This improves stability."),
        default=False, type=bool,)
    parser.add_argument(
        "--beta1",
        help=("Adam betas[0], (default: %(default).2f) "
              "DCGAN paper recommends .5 instead of the usual .9"),
        default=0.5, type=float)
    parser.add_argument(
        "--beta2",
        help="Adam betas[1] (default: %(default).4f) ",
        default=0.999, type=float)
    parser.add_argument(
        "--NN-upconv", help=("This approach minimize checkerboard artifacts "
                             "during training. Used only by arch=0. Uses "
                             "nearest-neighbor resized convolutions instead "
                             "of strided convolutions "
                             "(https://distill.pub/2016/deconv-checkerboard/ "
                             "and github.com/abhiskk/fast-neural-style)."),
        type=bool, default=False, const=True, nargs='?')
    parser.add_argument(
        "--no-batch-norm-G", help="If True, no batch norm in G.",
        type=bool, default=False, const=True, nargs='?')
    parser.add_argument(
        "--no-batch-norm-D", help="If True, no batch norm in D.",
        type=bool, default=False, const=True, nargs='?')
    parser.add_argument(
        "--activation", help="Activation functin of G",
        default="none", choices=["none", "sigmoid", "tanh", "htanh"],
        type=str)
    parser.add_argument(
        "--jointly", help="Train NN jointly or seperately.",
        type=bool, default=True, const=True, nargs='?')
    parser.add_argument(
        "--log-every",
        help=("log to tensorboard"
              "(default: %(default)d)"),
        default=3, type=int,)
    parser.add_argument(
        "--valid-every",
        help=("validation"
              "(default: %(default)d)"),
        default=10, type=int,)
    args = parser.parse_args()
    main(args)
