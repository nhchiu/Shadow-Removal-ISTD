#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import datetime
import json
import logging
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from denseunet import DenseUNet
from logger import Logger
from loss import DataLoss, VisualLoss, AdversarialLoss, SoftAdapt
from mydataset import MyDataset
from transform import transforms
from unet import UNet
from unet2 import UNet2
from discriminator import Discriminator


def main(args):
    time_str = time.strftime("%Y%m%d-%H%M%S")
    makedirs(args)
    snapshotargs(args, filename=f"args-{time_str}.json")

    # use CuDNN backend
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    manual_seed = 36
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    torch.cuda.manual_seed_all(manual_seed)

    log_file = os.path.join(args.logs,
                            os.path.splitext(__file__)[0]+"-"+time_str+".log")
    logger = Logger(log_file, level=logging.DEBUG)

    run_dir = os.path.join(
        args.logs, os.path.splitext(__file__)[0]+"-"+time_str)
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
                        drop_rate=0)
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

    data_loss = DataLoss().to(device)
    # visual_loss = VisualLoss().to(device)
    gan_loss = AdversarialLoss().to(device)
    soft_adapt = SoftAdapt(["data", "GAN"],
                           beta=0.1,
                           weighted=True,
                           normalized=False)
    soft_adapt.to(device)
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
                "D": 0,
                "G": 0,
                "G_gan": 0,
                "G_data": 0,
                # "G_vis": 0
            }
            D_real = []
            D_fake = []
            for batch, (x, y_true, _) in enumerate(loaders[phase]):
                progress.set_description(phase + f" batch {batch:<2d}")

                x = x.to(device)
                y_true_img = y_true["image"].to(device)
                y_true_sp = y_true["sp"].to(device)

                optimizer_D.zero_grad()
                optimizer_G.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    discriminator.zero_grad()
                    discriminator.requires_grad_(True)
                    # Train discriminator with real y_true_sp
                    D_out_real = discriminator(x, y_true_sp)
                    D_loss_real = gan_loss(D_out_real, is_real=True)
                    D_real.append(
                        np.mean(D_out_real.detach().cpu().numpy()))
                    # Train discriminator with generated y_pred
                    y_pred = generator(x)
                    D_out_fake = discriminator(x, y_pred.detach())
                    D_loss_fake = gan_loss(D_out_fake, is_real=False)
                    D_fake.append(
                        np.mean(D_out_fake.detach().cpu().numpy()))

                    D_loss = (D_loss_fake*0.5 + D_loss_real*0.5)
                    if phase == "train":
                        D_loss.backward()
                        optimizer_D.step()
                    loss_sum["D"] += D_loss.item()

                    generator.zero_grad()
                    discriminator.requires_grad_(False)
                    # Train generator with updated discriminator
                    D_out = discriminator(x, y_pred)
                    G_losses = {
                        "GAN": gan_loss(D_out, is_real=True),
                        "data": data_loss(y_pred, y_true_sp),
                        # "vis": visual_loss(x, y_pred, y_true_img)
                    }
                    soft_adapt.update(G_losses)
                    G_loss = soft_adapt.loss(update_weights=(phase == "train"))
                    if phase == "train":
                        G_loss.backward()
                        optimizer_G.step()

                    loss_sum["G_gan"] += soft_adapt.get("GAN")
                    loss_sum["G_data"] += soft_adapt.get("data")
                    # loss_sum["G_vis"] += soft_adapt.get("vis")
                    loss_sum["G"] += G_loss.item()

            loss_sum["total"] = loss_sum["G"]*0.8 + loss_sum["D"]*0.2
            # Log metrics to tensorboard
            n_batches = len(loaders[phase])
            for k in loss_sum:
                loss_sum[k] /= n_batches
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
                          resize=(300, 400),
                          scale=args.aug_scale,
                          angle=args.aug_angle,
                          flip_prob=0.5,
                          crop_size=args.image_size
                      ))
    valid = MyDataset(args.data_dir, subset="test")
    return train, valid


@torch.no_grad()
def weights_init(m):
    """custom weights initialization called on netG and netD"""
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or \
            classname.find('BatchNorm') != -1 or \
            classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)


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
        "--batch-size", help="input batch size for training (default: %(default)d)",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--epochs", help="number of epochs to train (default: %(default)d)",
        type=int,
        default=5000,
    )
    parser.add_argument(
        "--lr", help="initial learning rate (default: %(default).5f)",
        type=float,
        default=0.00002,
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
