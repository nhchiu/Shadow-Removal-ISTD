#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import logging
import os
import time

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

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(
            args.devices[0] if torch.cuda.is_available() else "cpu")

        # network models
        self.logger.info("Creating network model")
        self.G = networks.get_generator(
            args.net_G,
            in_channels=ISTDDataset.in_channels,
            out_channels=ISTDDataset.out_channels,
            ngf=32, no_conv_t=args.NN_upconv,
            drop_rate=0)
        self.D = networks.get_discriminator(
            args.net_D,
            in_channels=ISTDDataset.in_channels,
            in_channels2=ISTDDataset.out_channels,
            ndf=32, n_layers=3)
        if "infer" in args.tasks and "train" not in args.tasks:
            assert args.load_weights_g is not None
        self.init_weight(g_weights=args.load_weights_g,
                         d_weights=args.load_weights_d)
        self.G.to(self.device)
        self.D.to(self.device)
        if len(args.devices) > 1 and torch.cuda.is_available():
            self.logger.info(f"DataParallel on devices: {args.devices}")
            devices = [torch.device(d) for d in args.devices]
            self.G = nn.DataParallel(self.G, devices)
            self.D = nn.DataParallel(self.D, devices)
        self.optim_G = optim.Adam(self.G.parameters(),
                                  lr=args.lr_G,
                                  betas=(args.beta1, args.beta2))
        self.optim_D = optim.Adam(self.D.parameters(),
                                  lr=args.lr_D,
                                  betas=(args.beta1, args.beta2))
        self.decay_G = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim_G,
            cooldown=10, min_lr=1e-6, factor=0.8, verbose=True)
        self.decay_D = optim.lr_scheduler.ReduceLROnPlateau(
            self.optim_D,
            cooldown=10, min_lr=1e-6, factor=0.8, verbose=True)

        # data loaders
        self.logger.info("Creating data loaders")
        train_dataset = ISTDDataset(args.data_dir, subset="train",
                                    transforms=transform.transforms(
                                        resize=(300, 400),
                                        scale=args.aug_scale,
                                        angle=args.aug_angle,
                                        flip_prob=0.5,
                                        crop_size=args.image_size))
        valid_dataset = ISTDDataset(args.data_dir, subset="test")

        def worker_init(id):
            return np.random.seed(42+id)

        self.train_loader = DataLoader(train_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=True,
                                       drop_last=True,
                                       num_workers=args.workers,
                                       worker_init_fn=worker_init,
                                       pin_memory=(self.device.type == "cuda"))
        self.valid_loader = DataLoader(valid_dataset,
                                       batch_size=args.batch_size,
                                       shuffle=False,
                                       drop_last=False,
                                       num_workers=args.workers,
                                       worker_init_fn=worker_init,
                                       pin_memory=False)

        # loss
        if "train" in args.tasks:
            self.logger.info("Creating loss functions")
            self.d_loss_fn = args.D_loss_fn
            self.d_loss_type = args.D_loss_type
            self.adv_loss = AdversarialLoss(
                ls=(args.D_loss_fn == "leastsqure"))
            self.adv_loss.to(self.device, non_blocking=True)

            self.data_loss = DataLoss().to(self.device)
            self.visual_loss = VisualLoss().to(self.device)
            self.lambda1 = 10  # data loss
            self.lambda2 = 100   # visual loss
            self.adapt = args.softadapt
            if self.adapt:
                self.soft_adapt = SoftAdapt(
                    ["adv", "data", "visual"],
                    init_weights=[1, self.lambda1, self.lambda2],
                    beta=0.1, weighted=True, normalized=True)
                self.soft_adapt.to(self.device, non_blocking=True)

            self.train_logdir = os.path.join(args.logs, "train")
            self.valid_logdir = os.path.join(args.logs, "valid")

            if (os.path.isdir(self.train_logdir)):
                for file in os.listdir(self.train_logdir):
                    os.remove(os.path.join(self.train_logdir, file))
            if (os.path.isdir(self.valid_logdir)):
                for file in os.listdir(self.valid_logdir):
                    os.remove(os.path.join(self.valid_logdir, file))
            self.weights_dir = args.weights
            self.log_interval = args.log_every
            self.valid_interval = args.valid_every
        if "infer" in args.tasks:
            self.inferd_dir = args.infered

    def train(self, epochs=5000):
        train_writer = SummaryWriter(self.train_logdir)
        valid_writer = SummaryWriter(self.valid_logdir)
        if self.adapt:
            train_writer.add_custom_scalars_multilinechart(
                ["SoftAdapt/adv",
                 "SoftAdapt/data",
                 "SoftAdapt/vis"])
            train_writer.add_custom_scalars_multilinechart(
                ["Loss/G_adv",
                 "Loss/G_data",
                 "Loss/G_vis"])

        self.logger.info("Start training")
        best_loss = 100000.0
        start_time = time.time()
        progress = trange(epochs, desc="epochs", position=0,
                          ncols=80, ascii=True,)

        for epoch in progress:
            measures = self.run_epoch()
            if (epoch % self.log_interval == 0):
                self.log_scalars(train_writer, measures, epoch)
                self.save(self.weights_dir, "latest")

            if (epoch % self.valid_interval == 0):
                measures = self.run_epoch(training=False)
                self.log_scalars(valid_writer, measures, epoch)
                if measures["Loss"]["total"] < best_loss:
                    best_loss = measures["Loss"]["total"]
                    self.save(self.weights_dir, "best")
                    valid_writer.add_text("best",
                                          f"{epoch}: loss={best_loss}", epoch)
                    self.logger.info(
                        f"Improvement after epoch {epoch}, "
                        f"error = {best_loss:4f}")

        total_time = datetime.timedelta(seconds=(time.time()-start_time))
        self.logger.info(f"Training time {total_time}")
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

        loss = dict.fromkeys(["D", "G", "G_adv", "G_data", "G_vis"], 0.0)
        D_out = dict.fromkeys(["real", "fake"], 0.0)
        data_loader = self.train_loader if training else self.valid_loader
        for(_, x, y_img, y_sp) in tqdm(data_loader,
                                       total=len(data_loader),
                                       desc="train" if training else "valid",
                                       ncols=80, ascii=True,
                                       leave=False, position=1):
            x = x.to(self.device, non_blocking=True)
            y_sp = y_sp.to(self.device, non_blocking=True)
            y_img = y_img.to(self.device, non_blocking=True)

            self.optim_D.zero_grad()
            self.optim_G.zero_grad()
            with torch.set_grad_enabled(training):
                self.D.zero_grad()
                self.D.requires_grad_(True)
                # Train D
                C_real = self.D(x, y_sp)
                y_sp_pred = self.G(x)
                C_fake = self.D(x, y_sp_pred.detach())
                if self.d_loss_type == "normal":
                    D_loss_real = self.adv_loss(C_real, is_real=True)
                    D_loss_fake = self.adv_loss(C_fake, is_real=False)
                    D_loss = (D_loss_fake + D_loss_real) * 0.5
                elif self.d_loss_type == "rel":
                    D_loss = self.adv_loss(C_real-C_fake, is_real=True)
                else:  # "rel_avg"
                    D_loss_real = self.adv_loss(
                        C_real - C_fake.mean(dim=0, keepdim=True),
                        is_real=True)
                    D_loss_fake = self.adv_loss(
                        C_fake - C_real.mean(dim=0, keepdim=True),
                        is_real=False)
                    D_loss = (D_loss_fake + D_loss_real) * 0.5

                if training:
                    D_loss.backward()
                    self.optim_D.step()
                    self.optim_D.zero_grad()
                D_out["real"] += C_real.detach().cpu().numpy().mean()
                D_out["fake"] += C_fake.detach().cpu().numpy().mean()
                loss["D"] += D_loss.item()

                self.G.zero_grad()
                self.D.requires_grad_(False)
                # Train G with updated discriminator
                if training:
                    C_real = self.D(x, y_sp)
                    C_fake = self.D(x, y_sp_pred)
                if self.d_loss_type == "normal":
                    g_loss_adv = self.adv_loss(C_fake, is_real=True)
                elif self.d_loss_type == "rel":
                    g_loss_adv = self.adv_loss(C_fake - C_real, is_real=True)
                else:  # "rel_avg"
                    g_loss_adv_r = self.adv_loss(
                        C_fake - C_real.mean(dim=0, keepdim=True),
                        is_real=True)
                    g_loss_adv_f = self.adv_loss(
                        C_real - C_fake.mean(dim=0, keepdim=True),
                        is_real=False)
                    g_loss_adv = (g_loss_adv_r + g_loss_adv_f) * 0.5

                data_loss = self.data_loss(y_sp_pred, y_sp)
                visual_loss = self.visual_loss(x, y_sp_pred, y_img)
                if self.adapt:
                    G_loss = self.soft_adapt({"adv": g_loss_adv,
                                              "data": data_loss,
                                              "visual": visual_loss},
                                             update_weights=training)
                else:
                    G_loss = (g_loss_adv +
                              self.lambda1 * data_loss +
                              self.lambda2 * visual_loss)
                if training:
                    G_loss.backward()
                    self.optim_G.step()
                    self.optim_G.zero_grad()

                loss["G_adv"] += g_loss_adv.item()
                loss["G_data"] += data_loss.item()
                loss["G_vis"] += visual_loss.item()
                loss["G"] += G_loss.item()
        if training:
            self.decay_G.step(loss["G"])
            self.decay_D.step(loss["D"])
        torch.cuda.empty_cache()
        # return metrics
        loss["total"] = loss["G"]*0.8 + loss["D"]*0.2
        n_batches = len(data_loader)
        for key in loss:
            loss[key] /= n_batches
        for key in D_out:
            D_out[key] /= n_batches
        softadapt_weights = self.soft_adapt.get_weights() if self.adapt else {}
        return {"Loss": loss,
                "D_out": D_out,
                "SoftAdapt": softadapt_weights}

    def infer(self,):
        # os.makedirs(self.inferd_dir, exist_ok=True)
        # os.makedirs(self.inferd_dir+"sp", exist_ok=True)
        with torch.no_grad():
            self.G.eval()
            data_loader = self.valid_loader
            normalization = transform.Normalize(
                ISTDDataset.mean, ISTDDataset.std)
            for (filenames, x, _, _) in tqdm(data_loader,
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
                assert len(input_list) == len(filenames)
                for img_in, sp_pred, name \
                        in zip(input_list, pred_list, filenames):
                    img_pred = utils.float2uint(
                        utils.apply_sp(img_in, sp_pred))
                    cv.imwrite(os.path.join(
                        self.inferd_dir, name+".png"), img_pred)
                    # if save_sp:
                    #     np.save(os.path.join(
                    #         self.inferd_dir+"sp", name), sp_pred)
        return

    def log_scalars(self, writer, measures: dict, epoch: int):
        for m in measures:
            for k, v in measures[m].items():
                writer.add_scalar(f"{m}/{k}", v, epoch)

    def save(self, weights=None, suffix="latest"):
        module_G = self.G.module if isinstance(self.G, nn.DataParallel) \
            else self.G
        module_D = self.D.module if isinstance(self.D, nn.DataParallel) \
            else self.D
        G_name = module_G.__class__.__name__
        D_name = module_D.__class__.__name__
        if weights is None:
            weights = self.weights_dir
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
        else:
            self.G.apply(networks.weights_init)
        if d_weights:
            state_dict = torch.load(d_weights, map_location=self.device)
            self.D.load_state_dict(state_dict)
            self.logger.info(f"Loaded D weights: {d_weights}")
        else:
            self.D.apply(networks.weights_init)
