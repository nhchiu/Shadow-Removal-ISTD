#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import os
import time
from collections import OrderedDict

import cv2 as cv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

import networks
import transform
import utils
from dataset import ISTDDataset
from loss import AdversarialLoss, DataLoss, SoftAdapt, VisualLoss


class CGAN(object):
    # __slots__ = ["logger", "device",
    #              "G", "D", "optim_G", "optim_D",
    #              "data_loss", "d_loss", "g_loss",
    #              "train_loader", "valid_loader",
    #              "train_logdir", "valid_logdir", "weights_dir"]

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(
            args.device[0] if torch.cuda.is_available() else "cpu")
        # network models
        self.logger.info("Creating network model")
        self.G = networks.get_generator(
            args.net_g,
            in_channels=ISTDDataset.in_channels,
            out_channels=ISTDDataset.out_channels,
            ngf=64,
            drop_rate=0)
        self.D = networks.get_discriminator(
            args.net_d,
            in_channels=ISTDDataset.in_channels,
            in_channels2=ISTDDataset.out_channels,
            ndf=64, n_layers=3)
        if "infer" in args.tasks and "train" not in args.tasks:
            assert args.load_weights_g is not None
        self.init_weight(g_weights=args.load_weights_g,
                         d_weights=args.load_weights_d)
        self.G.to(self.device)
        self.D.to(self.device)
        if len(args.device) > 1 and torch.cuda.is_available():
            self.G = nn.DataParallel(self.G, args.device)
            self.D = nn.DataParallel(self.D, args.device)
        self.optim_G = optim.Adam(self.G.parameters(),
                                  lr=args.lr, betas=(0.5, 0.999))
        self.optim_D = optim.Adam(self.D.parameters(),
                                  lr=args.lr, betas=(0.5, 0.999))
        # data loaders
        self.logger.info("Creating data loaders")
        train_dataset = ISTDDataset(args.data_dir, subset="train",
                                    transforms=transform.transforms(
                                        resize=(300, 400),
                                        scale=args.aug_scale,
                                        angle=args.aug_angle,
                                        flip_prob=0.5,
                                        crop_size=args.image_size))
        valid_dataset = ISTDDataset(args.data_dir, subset="test",
                                    transforms=transform.transforms(
                                        resize=(256, 256)))
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=args.workers,
            worker_init_fn=lambda id: np.random.seed(42+id),
            pin_memory=(self.device.type == 'cuda'))
        self.valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=args.workers,
            worker_init_fn=lambda id: np.random.seed(42+id),
            pin_memory=False)
        # loss
        if "train" in args.tasks:
            self.logger.info("Creating loss functions")
            self.d_loss = AdversarialLoss().to(self.device, non_blocking=True)
            self.g_loss = SoftAdapt(OrderedDict({"adv": AdversarialLoss(),
                                                 "data": DataLoss(),
                                                 "vis": VisualLoss(), }),
                                    init_weights=[1, 100, 50],
                                    beta=0.1,
                                    weighted=True,
                                    normalized=True).to(self.device, non_blocking=True)

            self.train_logdir = os.path.join(args.logs, 'train')
            self.valid_logdir = os.path.join(args.logs, 'valid')
            if (os.path.isdir(self.train_logdir)):
                for file in os.listdir(self.train_logdir):
                    os.remove(os.path.join(self.train_logdir, file))
            if (os.path.isdir(self.valid_logdir)):
                for file in os.listdir(self.valid_logdir):
                    os.remove(os.path.join(self.valid_logdir, file))
            self.weights_dir = args.weights

    def train(self, epochs=5000):
        train_writer = SummaryWriter(self.train_logdir)
        valid_writer = SummaryWriter(self.valid_logdir)
        train_writer.add_custom_scalars_multilinechart(
            ["SoftAdapt_weights/data",
             "SoftAdapt_weights/vis",
             "SoftAdapt_weights/adv"])

        self.logger.info("Start training")
        best_loss = 100000.0
        start_time = time.time()
        progress = trange(epochs, desc="epochs", position=0,
                          ncols=80, ascii=True,)
        for epoch in progress:
            measures = self.run_epoch()
            self.log_scalars(train_writer, measures, epoch)
            self.save(self.weights_dir, "latest")
            if epoch % 5 == 0:
                measures = self.run_epoch(training=False)
                self.log_scalars(valid_writer, measures, epoch)
                if measures["loss"]["total"] < best_loss:
                    best_loss = measures["loss"]["total"]
                    self.save(self.weights_dir, "best")
                    valid_writer.add_text(
                        'best', f"{epoch:5d}: loss={best_loss:.3f}")
                    self.logger.info(
                        f"Best module saved after epoch {epoch}, "
                        f"error = {best_loss}")

        total_time = datetime.timedelta(seconds=(time.time()-start_time))
        self.logger.info(f'Training time {total_time:s}')
        self.logger.info(f"Best validation loss: {best_loss:.3f}")
        train_writer.close()
        valid_writer.close()

    def run_epoch(self, training=True):
        if training:
            self.G.train()
            self.D.train()
        else:
            self.G.eval()
            self.D.eval()

        # "G_vis"
        loss = dict.fromkeys(["D", "G", "G_adv", "G_data", "G_vis"], 0.0)
        D_out = dict.fromkeys(["real", "fake"], 0.0)
        data_loader = self.train_loader if training else self.valid_loader
        for(x, y_true, _) in tqdm(data_loader,
                                  total=len(data_loader),
                                  desc="train" if training else "valid",
                                  ncols=80, ascii=True,
                                  leave=False, position=1):
            y_true_img = y_true["image"]
            y_true_sp = y_true["sp"]

            x_gpu = x.to(self.device, non_blocking=True)
            y_true_sp_gpu = y_true_sp.to(self.device, non_blocking=True)

            self.optim_D.zero_grad()
            self.optim_G.zero_grad()
            with torch.set_grad_enabled(training):
                self.optim_D.zero_grad()
                # Train discriminator with real y_true_sp
                D_out_real = self.D(x_gpu, y_true_sp_gpu)
                D_loss_real = self.d_loss(D_out_real, is_real=True)
                # Train discriminator with generated y_pred
                y_pred_gpu = self.G(x_gpu)
                D_out_fake = self.D(x_gpu, y_pred_gpu.detach())
                D_loss_fake = self.d_loss(D_out_fake, is_real=False)
                D_loss = (D_loss_fake + D_loss_real) * 0.5 * 0.1
                if training:
                    D_loss.backward()
                    self.optim_D.step()
                D_out["real"] += D_out_real.detach().cpu().numpy().mean()
                D_out["fake"] += D_out_fake.detach().cpu().numpy().mean()
                loss["D"] += D_loss.item()

                self.optim_G.zero_grad()
                # Train self.G with updated discriminator
                D_out_ = self.D(x_gpu, y_pred_gpu)
                # y_pred = y_pred_gpu.to("cpu")

                G_losses = {"adv": (D_out_, True),
                            "data": (y_pred_gpu,
                                     y_true_sp.to(self.device, non_blocking=True)),
                            "vis": (x_gpu,
                                    y_pred_gpu,
                                    y_true_img.to(self.device, non_blocking=True))}
                G_loss = self.g_loss(G_losses, update_weights=False)
                if training:
                    G_loss.backward()
                    self.optim_G.step()

                g_loss = self.g_loss.get_loss()
                for k, v in g_loss.items():
                    loss["G_"+k] += v
                loss["G"] += G_loss.item()
        loss["total"] = loss["G"]*0.8 + loss["D"]*0.2

        # return metrics
        n_batches = len(data_loader)
        for key in loss:
            loss[key] /= n_batches
        for key in D_out:
            D_out[key] /= n_batches
        softadapt_weights = self.g_loss.get_weights()
        torch.cuda.empty_cache()
        return {"loss": loss,
                "D_out": D_out,
                "SoftAdapt_weights": softadapt_weights}

    def infer(self, save_sp=False):
        os.makedirs(os.path.join(os.path.pardir, "infered"), exist_ok=True)
        with torch.no_grad():
            self.G.eval()
            data_loader = self.valid_loader
            normalization = transform.Normalize(
                ISTDDataset.mean, ISTDDataset.std)
            for (x, _, filenames) in tqdm(data_loader,
                                          desc="Processing data",
                                          total=len(data_loader),
                                          ncols=80, ascii=True):
                input_list = []
                pred_list = []

                x = x.to(self.device, non_blocking=True)
                y_pred = self.G(x)
                y_pred_np = y_pred.detach().cpu().numpy()
                pred_list.extend([y_pred_np[s].transpose(1, 2, 0)
                                  for s in range(y_pred_np.shape[0])])

                x_np = x.detach().cpu().numpy()
                input_list.extend([x_np[s].transpose(1, 2, 0)
                                   for s in range(x_np.shape[0])])
                input_list = normalization(*input_list, inverse=True)

                assert len(input_list) == len(pred_list)
                for (img_in, sp_pred, name) \
                        in zip(input_list, pred_list, filenames):
                    img_pred = utils.float2uint(
                        utils.apply_sp(img_in, sp_pred))
                    img_pred = cv.resize(img_pred, (640, 480))
                    cv.imwrite(os.path.join(
                        os.path.pardir, "infered", name+".png"), img_pred)
                    if save_sp:
                        np.save(os.path.join(
                            os.path.pardir, "infered-sp", name), sp_pred)
        return

    def log_scalars(self, writer, measures: dict, epoch: int):
        for m in measures:
            for key in measures[m]:
                writer.add_scalar(
                    f"{m}/{key}", measures[m][key], epoch)

    def save(self, weights="../weights", suffix="latest"):
        module_G = self.G.module if isinstance(self.G, nn.DataParallel) \
            else self.G
        module_D = self.D.module if isinstance(self.D, nn.DataParallel) \
            else self.D
        G_name = module_G.__class__.__name__
        D_name = module_D.__class__.__name__
        torch.save(
            module_G.state_dict(),
            os.path.join(weights, f"{G_name}_{suffix}.pt"))
        torch.save(
            module_D.state_dict(),
            os.path.join(weights, f"{D_name}_{suffix}.pt"))

    def init_weight(self, g_weights=None, d_weights=None):
        if g_weights:
            state_dict = torch.load(g_weights, map_location=self.device)
            self.G.load_state_dict(state_dict)
            self.logger.info(f"Loaded G weights: {g_weights}")
        # else:
        #     self.G.apply(networks.weights_init)
        if d_weights:
            state_dict = torch.load(d_weights, map_location=self.device)
            self.D.load_state_dict(state_dict)
            self.logger.info(f"Loaded D weights: {d_weights}")
        # else:
        #     self.D.apply(networks.weights_init)
