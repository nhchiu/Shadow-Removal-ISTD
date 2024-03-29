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
import re
import time

import numpy as np
import torch
import torch.utils.data

from src.cgan import CGAN


def main(args):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    makedirs(args)
    snapshotargs(args, filename="args.json")
    if args.load_args is not None:
        with open(args.load_args, "r") as f:
            arg_dict = json.load(f)
        preserved_args = [
            "load_args"
            "load_checkpoint",
            "load_weights_g1",
            "load_weights_g2",
            "load_weights_d1",
            "load_weights_d2",
            "weights", "logs"]
        for k in preserved_args:
            if k in arg_dict:
                arg_dict.pop(k)
        args.__dict__.update(arg_dict)

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
    if args.load_checkpoint is not None:
        if not os.path.isfile(args.load_checkpoint):
            print(f"{args.load_checkpoint} is not a file")
        else:
            net.load(path=args.load_checkpoint)

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
    if args.D_type == "normal":
        arg_str += ""
    elif args.D_type == "rel":
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


def snapshotargs(args, filename="args.json"):
    args_file = os.path.join(args.logs, filename)
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "y", "1")


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
        default=["cuda"], type=lambda s: re.split(', *| +', s),)
    parser.add_argument(
        "--batch-size",
        help="input batch size for training (default: %(default)d)",
        default=16, type=int,)
    parser.add_argument(
        "--epochs",
        help="number of epochs to train (default: %(default)d)",
        default=100000, type=int,)
    parser.add_argument(
        "--data-dir",
        help="root folder with images (default: %(default)s)",
        default=[], type=lambda s: re.split(', *| +', s),)
    parser.add_argument(
        "--workers",
        help="number of workers for data loading (default: %(default)d)",
        default=4, type=int,)
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
        default="mnet", choices=["unet", "mnet", "denseunet", "stcgan"])
    parser.add_argument(
        "--net-D",
        help="the discriminator model (default: %(default)s)",
        default="patchgan", choices=["patchgan", "began", "stcgan", "dummy"])
    parser.add_argument(
        "--ngf",
        help=("initial number of features in G (default: %(default)d)"),
        default=64, type=int,)
    parser.add_argument(
        "--ndf",
        help=("initial number of features in D (default: %(default)d)"),
        default=64, type=int,)
    parser.add_argument(
        "--droprate",
        help="Dropout rate (default: %(default).3f)",
        default=0.05, type=float,)
    parser.add_argument(
        "--lr-D",
        help="initial learning rate of discriminator (default: %(default).5f)",
        default=0.0001, type=float,)
    parser.add_argument(
        "--lr-G",
        help="initial learning rate of generator (default: %(default).5f)",
        default=0.0005, type=float,)
    parser.add_argument(
        "--decay",
        help=("Decay to apply to lr each cycle. (default: %(default).5f)"
              "(1-decay)^n_iter * lr gives the final lr. "
              "e.g. 0.003 will lead to .05 of lr after 1k cycles"),
        default=0.003, type=float)
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
        "--lambda1",
        help="data2 loss coeficient (default: %(default).4f) ",
        default=5, type=float)
    parser.add_argument(
        "--lambda2",
        help="GAN1 loss coeficient (default: %(default).4f) ",
        default=0.5, type=float)
    parser.add_argument(
        "--lambda3",
        help="GAN2 loss coeficient (default: %(default).4f) ",
        default=0.5, type=float)
    parser.add_argument(
        "--lambda4",
        help="visual1 loss coeficient (default: %(default).4f) ",
        default=5, type=float)
    parser.add_argument(
        "--lambda5",
        help="visual2 coeficient (default: %(default).4f) ",
        default=50, type=float)
    parser.add_argument(
        "--manual_seed",
        help="manual random seed (default: %(default)s)",
        default=38107943, type=int)
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
        "--load-args",
        help="load args from previous runs (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--load-checkpoint",
        help="load checkpoint to continue training (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--D-loss-fn",
        help="loss funtion of discriminator (default: %(default)s)",
        default="standard", choices=["standard", "leastsquare"])
    parser.add_argument(
        "--D-type",
        help="Use relative discriminator loss (default: %(default)s)",
        default="normal", choices=["normal", "rel", "rel_avg"])
    parser.add_argument(
        "--softadapt", help="Adapt the weight of losses dynamically",
        default=False, const=True, nargs='?', type=str2bool)
    parser.add_argument(
        "--SELU",
        help=("Using scaled exponential linear units (SELU) "
              "which are self-normalizing instead of ReLU with BatchNorm. "
              "Used only in arch=0. This improves stability."),
        default=False, const=True, nargs='?', type=str2bool)
    parser.add_argument(
        "--NN-upconv", help=("This approach minimize checkerboard artifacts "
                             "during training. Used only by arch=0. Uses "
                             "nearest-neighbor resized convolutions instead "
                             "of strided convolutions "
                             "(https://distill.pub/2016/deconv-checkerboard/ "
                             "and github.com/abhiskk/fast-neural-style)."),
        type=str2bool, default=False, const=True, nargs='?')
    # parser.add_argument(
    #     "--no-batch-norm-G", help="If True, no batch norm in G.",
    #     type=str2bool, default=False, const=True, nargs='?')
    # parser.add_argument(
    #     "--no-batch-norm-D", help="If True, no batch norm in D.",
    #     type=str2bool, default=False, const=True, nargs='?')
    parser.add_argument(
        "--activation", help="Activation functin of G",
        default="tanh", choices=["none", "sigmoid", "tanh", "htanh"])
    # parser.add_argument(
    #     "--jointly", help="Train NN jointly or seperately.",
    #     type=str2bool, default=True, const=True, nargs='?')
    parser.add_argument(
        "--log-every",
        help="log interval to tensorboard (default: %(default)d)",
        default=3, type=int,)
    parser.add_argument(
        "--valid-every",
        help="validation interval (default: %(default)d)",
        default=10, type=int,)
    parser.add_argument(
        "--vis-every",
        help="visualize images to tensorboard (default: %(default)d)",
        default=50, type=int,)
    parser.add_argument(
        "--save-every",
        help="save checkpoints (default: %(default)d)",
        default=50, type=int,)
    parser.add_argument(
        "--weights",
        help="folder to save weights (default: %(default)s)",
        default="./weights",)
    parser.add_argument(
        "--infered",
        help="folder to save infered images (default: %(default)s)",
        default="./infered",)
    parser.add_argument(
        "--logs",
        help="folder to save logs (default: %(default)s)",
        default="./logs",)
    args = parser.parse_args()
    main(args)
