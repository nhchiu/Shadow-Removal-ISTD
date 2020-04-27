#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import logging
import os
import sys
import time
import random

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from denseunet import DenseUNet
from logger import Logger
from loss import DataLoss, VisualLoss, AdversarialLoss
from mydataset import MyDataset
from transform import transforms
from unet import UNet
from unet2 import UNet2
from discriminator import Discriminator


def main(args):
    makedirs(args)
    snapshotargs(args)

    # use CuDNN backend
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    manual_seed = 36
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    log_file = os.path.join(args.logs, os.path.splitext(__file__)[0]+".log")
    logger = Logger(log_file, level=logging.DEBUG)

    run_dir = os.path.join(args.logs, os.path.splitext(__file__)[0]+"-run")
    if (os.path.exists(run_dir) and os.path.isdir(run_dir)):
        for file in os.listdir(run_dir):
            os.remove(os.path.join(run_dir, file))
    writer = SummaryWriter(run_dir)

    # tensor(N, C, H, W)
    dummy_input = (torch.zeros((1, 3, args.image_size, args.image_size)), )

    device = torch.device(
        "cpu" if not torch.cuda.is_available() else args.device)
    logger.logger.info(f"Using device: {device.__str__()}")
    if torch.cuda.is_available() and torch.cuda.device_count() > 1 and args.device2 != "":
        device2 = torch.device(args.device2)
        logger.logger.info(f"Using device2: {device2.__str__()}")
    else:
        device2 = None

    logger.logger.debug("Creating data loaders")
    loader_train, loader_valid = data_loaders(args)
    loaders = {"train": loader_train, "valid": loader_valid}

    logger.logger.debug("Creating model")
    network_choices = {
        'unet': UNet,
        'unet2': UNet2,
        'denseunet': DenseUNet
    }
    network = network_choices.get(args.net, DenseUNet)
    generator = network(in_channels=MyDataset.in_channels,
                        out_channels=MyDataset.out_channels,
                        drop_rate=0.001)
    writer.add_graph(generator, dummy_input)
    if args.load_weights_g != "":
        state_dict = torch.load(args.load_weights_g, map_location=device)
        generator.load_state_dict(state_dict)
        logger.logger.info(f"Loaded G weights: {args.load_weights_g}")
    else:
        generator.apply(weights_init)
    generator.to(device)
    if device2:
        generator = nn.DataParallel(generator, [device, device2])

    discriminator = Discriminator(features=64, n_layers=3)
    if args.load_weights_d != "":
        state_dict = torch.load(args.load_weights_d, map_location=device)
        discriminator.load_state_dict(state_dict)
        logger.logger.info(f"Loaded D weights: {args.load_weights_d}")
    else:
        discriminator.apply(weights_init)
    discriminator.to(device)

    loss_weights = {
        'vis': args.visual_weight,
        'data': args.data_weight,
        'GAN': args.gan_weight
    }
    loss_fns = {
        'data': DataLoss().to(device),
        'vis': VisualLoss().to(device),
        'GAN': AdversarialLoss().to(device)
    }
    best_validation_loss = 1000.0

    optimizer_G = optim.Adam(generator.parameters(),
                             lr=args.lr, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(),
                             lr=args.lr, betas=(0.5, 0.999))
    module_g = generator.module if device2 else generator
    module_d = discriminator.module if device2 else discriminator

    logger.logger.debug("Start training")
    # writer.add_custom_scalars_multilinechart(['loss_train', 'loss_valid'])
    start_time = time.time()
    progress = logger.trange(args.epochs, desc="train")
    for epoch in progress:
        writer.add_text('progress', f"{epoch:5d} / {args.epochs}")
        for phase in ["train", "valid"]:
            if phase == "train":
                generator.train()
                discriminator.train()
            else:
                # validate every 3 epoch
                if epoch % 3 != 0:
                    continue
                generator.eval()
                discriminator.eval()
            progress.set_description(phase)

            loss_sum = {
                "total": 0,
                "D": 0,
                "G": 0,
                "G_gan": 0,
                "G_data": 0,
                "G_vis": 0
            }
            D_real = []
            D_fake = []
            for batch, (x, y_true, _) in enumerate(loaders[phase]):
                progress.set_description(phase + f" batch {batch:<2d}")

                chunks = len(x) // args.mini_batch
                x_chunks = x.chunk(chunks)
                y_true_img_chunks = y_true["image"].chunk(chunks)
                y_true_sp_shunks = y_true["sp"].chunk(chunks)

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    discriminator.zero_grad()
                    for (x, _, y_true_sp) in zip(x_chunks, y_true_img_chunks, y_true_sp_shunks):
                        x = x.to(device)
                        y_true_sp = y_true_sp.to(device)

                        discriminator.requires_grad_(True)
                        # Train discriminator with real y_true_sp
                        D_out_real = discriminator(x, y_true_sp)
                        D_real.append(
                            np.mean(D_out_real.detach().cpu().numpy()))
                        D_loss_real = loss_fns['GAN'](
                            D_out_real, is_real=True)*0.5
                        # Train discriminator with generated y_pred
                        y_pred = generator(x).detach()
                        D_out_fake = discriminator(x, y_pred)
                        D_fake.append(
                            np.mean(D_out_fake.detach().cpu().numpy()))
                        D_loss_fake = loss_fns['GAN'](
                            D_out_fake, is_real=False)*0.5

                        D_loss = ((D_loss_fake + D_loss_real)/2) * \
                            loss_weights['GAN'] / chunks
                        if phase == "train":
                            D_loss.backward()
                        loss_sum["D"] += D_loss.detach().item()
                    if phase == "train":
                        optimizer_D.step()

                    generator.zero_grad()
                    for (x, y_true_img, y_true_sp) in zip(x_chunks, y_true_img_chunks, y_true_sp_shunks):
                        x = x.to(device)
                        y_true_img = y_true_img.to(device)
                        y_true_sp = y_true_sp.to(device)

                        discriminator.requires_grad_(False)
                        y_pred = generator(x)
                        D_out = discriminator(x, y_pred)
                        G_loss_gan = loss_fns['GAN'](
                            D_out, is_real=True)/chunks
                        G_loss_data = loss_fns['data'](
                            y_pred, y_true_sp)/chunks
                        G_loss_vis = loss_fns['vis'](
                            x, y_pred, y_true_img)/chunks
                        G_loss = (
                            G_loss_gan * loss_weights['GAN'] +
                            G_loss_data * loss_weights['data'] +
                            G_loss_vis * loss_weights['vis']
                        )
                        if phase == "train":
                            G_loss.backward()
                        loss_sum["G_gan"] += G_loss_gan.detach().item()
                        loss_sum["G_data"] += G_loss_data.detach().item()
                        loss_sum["G_vis"] += G_loss_vis.detach().item()
                        loss_sum["G"] += G_loss.detach().item()
                    if phase == "train":
                        optimizer_G.step()
            loss_sum["total"] = loss_sum["G"] + loss_sum["D"]
            # Log metrics to tensorboard
            for k in loss_sum:
                loss_sum[k] /=len(loaders[phase])
                writer.add_scalar(
                    f"{phase}/loss/{k}", loss_sum[k], epoch)
            writer.add_scalar(
                f"{phase}/D_out_real", np.mean(D_real), epoch)
            writer.add_scalar(
                f"{phase}/D_out_fake", np.mean(D_fake), epoch)

            if phase == "train":
                torch.save(module_g.state_dict(), os.path.join(
                    args.weights, f"{args.net}_latest.pt"))
                torch.save(module_d.state_dict(), os.path.join(
                    args.weights, "discriminator_latest.pt"))

            if phase == "valid":
                if loss_sum["total"] < best_validation_loss:
                    best_validation_loss = loss_sum["total"]
                    torch.save(module_g.state_dict(), os.path.join(
                        args.weights, f"{args.net}_best.pt"))
                    torch.save(module_d.state_dict(), os.path.join(
                        args.weights, "discriminator_best.pt"))
                    logger.logger.info(
                        f"Best module saved after epoch {epoch}, error = {best_validation_loss}")
            loss_sum.clear()
            D_real.clear()
            D_fake.clear()

            torch.cuda.empty_cache()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.logger.info(f'Training time {total_time_str}')
    logger.logger.info(f"Best validation loss: {best_validation_loss:4f}")
    writer.close()


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
        pin_memory=(args.device != 'cpu' and torch.cuda.is_available())
    )
    loader_valid = DataLoader(
        dataset_valid,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        worker_init_fn=worker_init,
        pin_memory=False
    )

    return loader_train, loader_valid


def datasets(args):
    train = MyDataset(args.data_dir, subset="train",
                      transforms=transforms(
                          scale=args.aug_scale,
                          angle=args.aug_angle,
                          flip_prob=0.5,
                          crop_size=args.image_size
                      ))
    valid = MyDataset(args.data_dir, subset="test")
    return train, valid

# custom weights initialization called on netG and netD


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
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
        json.dump(vars(args), fp, indent=4, sort_keys=True)


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
        "--batch-size", help="input batch size for training (default: %(default)d)",
        type=int,
        default=16,
    )
    parser.add_argument(
        "--mini-batch", help="input batch size for training (default: %(default)d)",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--epochs", help="number of epochs to train (default: %(default)d)",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--lr", help="initial learning rate (default: %(default).4f)",
        type=float,
        default=0.0001,
    )
    parser.add_argument(
        "--visual-weight", help="weight of img visual loss (default: %(default).2f)",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--data-weight", help="weight of shadow parameter loss (default: %(default).2f)",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--gan-weight", help="weight of GAN loss (default: %(default).2f)",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--device", help="device for training (default: %(default)s)",
        type=str,
        default="cuda:0",
    )
    parser.add_argument(
        "--device2", help="second device for Data Parallelism (default: %(default)s)",
        type=str,
        default="",
    )
    parser.add_argument(
        "--workers", help="number of workers for data loading (default: %(default)d)",
        type=int,
        default=4,
    )
    # parser.add_argument(
    #     "--vis-images", help="number of visualization images to save in log file (default: 200)",
    #     type=int,
    #     default=200,
    # )
    # parser.add_argument(
    #     "--vis-freq", help="frequency of saving images to log file (default: 10)",
    #     type=int,
    #     default=10,
    # )
    parser.add_argument(
        "--weights", help="folder to save weights (default: %(default)s)",
        type=str,
        default="../weights",
    )
    parser.add_argument(
        "--logs", help="folder to save logs (default: %(default)s)",
        type=str,
        default="../logs",
    )
    parser.add_argument(
        "--data-dir", help="root folder with images (default: %(default)s)",
        type=str,
        default="../ISTD_DATASET",
    )
    parser.add_argument(
        "--image-size", help="target input image size (default: %(default)d)",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--aug-scale", help="scale factor range for augmentation (default: %(default).2f)",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--aug-angle", help="rotation angle range in degrees for augmentation (default: %(default)d)",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--net", help="the network module (default: %(default)s)",
        type=str,
        default="denseunet",
        choices=['unet', 'unet2', 'denseunet']
    )
    parser.add_argument(
        "--load-weights-g", help="load weights to continue training (default: %(default)s)",
        type=str,
        default=""
    )
    parser.add_argument(
        "--load-weights-d", help="load weights to continue training (default: %(default)s)",
        type=str,
        default=""
    )
    args = parser.parse_args()
    main(args)
