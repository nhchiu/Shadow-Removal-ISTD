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


class STCGAN(object):

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(
            args.devices[0] if torch.cuda.is_available() else "cpu")

        # network models
        self.logger.info("Creating network model")
        self.G1 = networks.get_generator(
            in_channels=3, out_channels=1, ngf=64,)
        self.G2 = networks.get_generator(
            in_channels=3+1, out_channels=3, ngf=64,)
        self.D1 = networks.get_discriminator(
            in_channels=3+1, ndf=64, n_layers=3, use_sigmoid=False)
        self.D2 = networks.get_discriminator(
            in_channels=3+3+1, ndf=64, n_layers=3, use_sigmoid=False)
        if "infer" in args.tasks and "train" not in args.tasks:
            assert args.load_weights_g1 is not None
            assert args.load_weights_g2 is not None
        self.init_weight(g1_weights=args.load_weights_g1,
                         g2_weights=args.load_weights_g2,
                         d1_weights=args.load_weights_d1,
                         d2_weights=args.load_weights_d2)
        self.G1.to(self.device)
        self.G2.to(self.device)
        self.D1.to(self.device)
        self.D2.to(self.device)
        if len(args.devices) > 1 and torch.cuda.is_available():
            self.logger.info(f"DataParallel on devices: {args.devices}")
            devices = [torch.device(d) for d in args.devices]
            self.G1 = nn.DataParallel(self.G1, devices)
            self.G2 = nn.DataParallel(self.G2, devices)
            self.D1 = nn.DataParallel(self.D1, devices)
            self.D2 = nn.DataParallel(self.D2, devices)
        self.optim_G = optim.Adam(
            list(self.G1.parameters())+list(self.G2.parameters()),
            lr=args.lr_G, betas=(args.beta1, args.beta2))
        self.optim_D = optim.Adam(
            list(self.D1.parameters())+list(self.D2.parameters()),
            lr=args.lr_D, betas=(args.beta1, args.beta2))
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
        valid_dataset = ISTDDataset(args.data_dir, subset="test",
                                    transforms=transform.transforms(
                                        resize=(256, 256)))

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
                                       pin_memory=(self.device.type == "cuda"))

        # loss
        if "train" in args.tasks:
            self.logger.info("Creating loss functions")
            self.d_loss_fn = args.D_loss_fn
            self.d_loss_type = args.D_loss_type
            self.adv_loss = AdversarialLoss(
                ls=(args.D_loss_fn == "leastsqure"))
            self.adv_loss.to(self.device, non_blocking=True)

            self.data_loss = DataLoss().to(self.device)
            # self.visual_loss = VisualLoss().to(self.device)
            self.lambda1 = 5  # data2 loss
            self.lambda2 = 0.1   # CGAN1 loss
            self.lambda3 = 0.1   # CGAN2 loss
            self.adapt = args.softadapt
            # if self.adapt:
            #     self.soft_adapt = SoftAdapt(
            #         ["adv", "data", "visual"],
            #         init_weights=[1, self.lambda1, self.lambda2],
            #         beta=0.1, weighted=True, normalized=True)
            #     self.soft_adapt.to(self.device, non_blocking=True)

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
        # if self.adapt:
        #     train_writer.add_custom_scalars_multilinechart(
        #         ["SoftAdapt/adv",
        #          "SoftAdapt/data",
        #          "SoftAdapt/vis"])
        #     train_writer.add_custom_scalars_multilinechart(
        #         ["Loss/G_adv",
        #          "Loss/G_data",
        #          "Loss/G_vis"])

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
            self.G1.train()
            self.G2.train()
            self.D1.train()
            self.D2.train()
        else:
            self.G1.eval()
            self.G2.eval()
            self.D1.eval()
            self.D2.eval()

        loss = dict.fromkeys(
            ["G", "D", "D1", "D2", "G1", "G2", "data1", "data2"], 0.0)
        D1_out = dict.fromkeys(["real", "fake"], 0.0)
        D2_out = dict.fromkeys(["real", "fake"], 0.0)
        data_loader = self.train_loader if training else self.valid_loader
        for(_, x, m, y) in tqdm(data_loader,
                                total=len(data_loader),
                                desc="train" if training else "valid",
                                ncols=80, ascii=True,
                                leave=False, position=1):
            x = x.to(self.device, non_blocking=True)
            m = m.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optim_D.zero_grad()
            self.optim_G.zero_grad()
            with torch.set_grad_enabled(training):
                self.optim_D.zero_grad()
                self.D1.requires_grad_(True)
                self.D2.requires_grad_(True)
                # Train D1, D2
                C1_real = self.D1(torch.cat((x, m), dim=1))
                m_pred = self.G1(x)
                C1_fake = self.D1(torch.cat((x, m_pred.detach()), dim=1))

                C2_real = self.D2(torch.cat((x, m, y), dim=1))
                y_pred = self.G2(torch.cat((x, m), dim=1))
                C2_fake = self.D2(torch.cat((x,
                                             m_pred.detach(),
                                             y_pred.detach()), dim=1))
                if self.d_loss_type == "normal":
                    D1_loss_real = self.adv_loss(C1_real, is_real=True)
                    D1_loss_fake = self.adv_loss(C1_fake, is_real=False)
                    D1_loss = (D1_loss_fake + D1_loss_real) * 0.5

                    D2_loss_real = self.adv_loss(C2_real, is_real=True)
                    D2_loss_fake = self.adv_loss(C2_fake, is_real=False)
                    D2_loss = (D2_loss_fake + D2_loss_real) * 0.5
                elif self.d_loss_type == "rel":
                    D1_loss = self.adv_loss(C1_real-C1_fake, is_real=True)
                    D2_loss = self.adv_loss(C2_real-C2_fake, is_real=True)
                else:  # "rel_avg"
                    D1_loss_real = self.adv_loss(
                        C1_real - C1_fake.mean(dim=0), is_real=True)
                    D1_loss_fake = self.adv_loss(
                        C1_fake - C1_real.mean(dim=0), is_real=False)
                    D1_loss = (D1_loss_fake + D1_loss_real) * 0.5

                    D2_loss_real = self.adv_loss(
                        C2_real - C2_fake.mean(dim=0), is_real=True)
                    D2_loss_fake = self.adv_loss(
                        C2_fake - C2_real.mean(dim=0), is_real=False)
                    D2_loss = (D2_loss_fake + D2_loss_real) * 0.5

                D_loss = self.lambda2 * D1_loss + self.lambda3 * D2_loss
                if training:
                    D_loss.backward()
                    self.optim_D.step()
                D1_out["real"] += C1_real.detach().cpu().numpy().mean()
                D1_out["fake"] += C1_fake.detach().cpu().numpy().mean()
                D2_out["real"] += C2_real.detach().cpu().numpy().mean()
                D2_out["fake"] += C2_fake.detach().cpu().numpy().mean()
                loss["D1"] += D1_loss.item()
                loss["D2"] += D2_loss.item()
                loss["D"] += D_loss.item()

                self.optim_G.zero_grad()
                self.D1.requires_grad_(False)
                self.D2.requires_grad_(False)
                # Train G with updated discriminator
                if training:  # D is not updates when validating
                    C1_real = self.D1(torch.cat((x, m), dim=1))
                    C1_fake = self.D1(torch.cat((x, m_pred), dim=1))
                    C2_real = self.D2(torch.cat((x, m, y), dim=1))
                    C2_fake = self.D2(torch.cat((x, m_pred, y_pred), dim=1))
                if self.d_loss_type == "normal":
                    G1_loss = self.adv_loss(C1_fake, is_real=True)
                    G2_loss = self.adv_loss(C2_fake, is_real=True)
                elif self.d_loss_type == "rel":
                    G1_loss = self.adv_loss(C1_fake - C1_real, is_real=True)
                    G2_loss = self.adv_loss(C2_fake - C2_real, is_real=True)
                else:  # "rel_avg"
                    G1_loss_r = self.adv_loss(
                        C1_fake - C1_real.mean(dim=0), is_real=True)
                    G1_loss_f = self.adv_loss(
                        C1_real - C1_fake.mean(dim=0), is_real=False)
                    G1_loss = (G1_loss_r + G1_loss_f) * 0.5

                    G2_loss_r = self.adv_loss(
                        C1_fake - C1_real.mean(dim=0), is_real=True)
                    G2_loss_f = self.adv_loss(
                        C1_real - C1_fake.mean(dim=0), is_real=False)
                    G2_loss = (G2_loss_r + G2_loss_f) * 0.5

                data1_loss = self.data_loss(m_pred, m)
                data2_loss = self.data_loss(y_pred, y)
                # if self.adapt:
                #     G_loss = self.soft_adapt({"adv": g_loss_adv,
                #                               "data": data_loss,
                #                               "visual": visual_loss},
                #                              update_weights=training)
                # else:
                G_loss = (data1_loss +
                          self.lambda1 * data2_loss +
                          self.lambda2 * G1_loss +
                          self.lambda3 * G2_loss)
                if training:
                    G_loss.backward()
                    self.optim_G.step()

                loss["G1"] += G1_loss.item()
                loss["G2"] += G2_loss.item()
                loss["data1"] += data1_loss.item()
                loss["data2"] += data2_loss.item()
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
        for key in D1_out:
            D1_out[key] /= n_batches
        for key in D2_out:
            D2_out[key] /= n_batches
        # softadapt_weights = self.soft_adapt.get_weights()
        # if self.adapt else {}
        return {"Loss": loss,
                "D1_out": D1_out,
                "D2_out": D2_out}

    def infer(self,):
        with torch.no_grad():
            self.G1.eval()
            self.G2.eval()
            data_loader = self.valid_loader
            # normalization = transform.Normalize(
            #     ISTDDataset.mean, ISTDDataset.std)
            for (filenames, x, _, _) in tqdm(data_loader,
                                             desc="Processing data",
                                             total=len(data_loader),
                                             ncols=80, ascii=True):
                input_list = []
                m_pred_list = []
                y_pred_list = []

                x = x.to(self.device, non_blocking=True)
                m_pred = self.G1(x)
                y_pred = self.G2(torch.cat((x, m_pred), dim=1))

                x_np = x.detach().cpu().numpy()*0.5 + 0.5
                m_pred_np = m_pred.detach().cpu().numpy()*0.5 + 0.5
                y_pred_np = y_pred.detach().cpu().numpy()*0.5 + 0.5

                input_list.extend([x_np[s].transpose(1, 2, 0)
                                   for s in range(x_np.shape[0])])
                m_pred_list.extend([m_pred_np[s].transpose(1, 2, 0)
                                    for s in range(m_pred_np.shape[0])])
                y_pred_list.extend([y_pred_np[s].transpose(1, 2, 0)
                                    for s in range(y_pred_np.shape[0])])

                assert len(input_list) == len(y_pred_list)
                assert len(input_list) == len(filenames)
                for img_in, m_pred, y_pred, name in \
                        zip(input_list, m_pred_list, y_pred_list, filenames):
                    img_pred = cv.resize(
                        y_pred, (256, 192), interpolation=cv.INTER_LINEAR)
                    img_pred = utils.float2uint(img_pred)
                    cv.imwrite(os.path.join(self.inferd_dir,
                                            "shadowless",
                                            name+".png"), img_pred)
                    mask_pred = cv.resize(
                        m_pred, (256, 192), interpolation=cv.INTER_LINEAR)
                    mask_pred = utils.float2uint(mask_pred)
                    cv.imwrite(os.path.join(self.inferd_dir,
                                            "mask",
                                            name+".png"), mask_pred)
                    # if save_sp:
                    #     np.save(os.path.join(
                    #         self.inferd_dir+"sp", name), m_pred)
        return

    def log_scalars(self, writer, measures: dict, epoch: int):
        for m in measures:
            for k, v in measures[m].items():
                writer.add_scalar(f"{m}/{k}", v, epoch)

    def save(self, weights=None, suffix="latest"):
        module_G1 = self.G1.module if isinstance(self.G1, nn.DataParallel) \
            else self.G1
        module_G2 = self.G2.module if isinstance(self.G2, nn.DataParallel) \
            else self.G2
        module_D1 = self.D1.module if isinstance(self.D1, nn.DataParallel) \
            else self.D1
        module_D2 = self.D2.module if isinstance(self.D2, nn.DataParallel) \
            else self.D2
        if weights is None:
            weights = self.weights_dir
        torch.save(module_G1.state_dict(),
                   os.path.join(weights, f"G1-{suffix}.pt"))
        torch.save(module_G2.state_dict(),
                   os.path.join(weights, f"G2-{suffix}.pt"))
        torch.save(module_D1.state_dict(),
                   os.path.join(weights, f"D1-{suffix}.pt"))
        torch.save(module_D2.state_dict(),
                   os.path.join(weights, f"D2-{suffix}.pt"))

    def init_weight(self, g1_weights=None, g2_weights=None,
                    d1_weights=None, d2_weights=None):
        if g1_weights:
            state_dict = torch.load(g1_weights, map_location=self.device)
            self.G1.load_state_dict(state_dict)
            self.logger.info(f"Loaded G1 weights: {g1_weights}")
        else:
            self.G1.apply(networks.weights_init)
        if g2_weights:
            state_dict = torch.load(g2_weights, map_location=self.device)
            self.G2.load_state_dict(state_dict)
            self.logger.info(f"Loaded G2 weights: {g2_weights}")
        else:
            self.G2.apply(networks.weights_init)
        if d1_weights:
            state_dict = torch.load(d1_weights, map_location=self.device)
            self.D1.load_state_dict(state_dict)
            self.logger.info(f"Loaded D1 weights: {d1_weights}")
        else:
            self.D1.apply(networks.weights_init)
        if d2_weights:
            state_dict = torch.load(d2_weights, map_location=self.device)
            self.D2.load_state_dict(state_dict)
            self.logger.info(f"Loaded D2 weights: {d2_weights}")
        else:
            self.D2.apply(networks.weights_init)
