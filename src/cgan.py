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
from torch.utils.data import ConcatDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm, trange

import src.networks as networks
import src.transform as transform
import src.utils as utils
from src.dataset import ISTDDataset
from src.loss import AdversarialLoss, DataLoss, VisualLoss  # , SoftAdapt


class CGAN(object):

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)
        self.device = torch.device(
            args.devices[0] if torch.cuda.is_available() else "cpu")

        # network models
        self.logger.info("Creating network model")
        self.G1 = networks.get_generator(
            args.net_G,
            in_channels=3,
            out_channels=1,
            ngf=args.ngf,
            drop_rate=args.droprate,
            no_conv_t=args.NN_upconv,
            use_selu=args.SELU,
            activation=args.activation,)
        self.G2 = networks.get_generator(
            args.net_G,
            in_channels=3+1,
            out_channels=3,
            ngf=args.ngf,
            drop_rate=args.droprate,
            no_conv_t=args.NN_upconv,
            use_selu=args.SELU,
            activation=args.activation,)
        self.D1 = networks.get_discriminator(
            args.net_D,
            in_channels=3+1,
            out_channels=1,
            ndf=args.ndf,  # n_layers=3,
            use_selu=args.SELU,
            use_sigmoid=False)
        self.D2 = networks.get_discriminator(
            args.net_D,
            in_channels=3+3+1,
            out_channels=3,
            ndf=args.ndf,  # n_layers=3,
            use_selu=args.SELU,
            use_sigmoid=False)
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
        self.decay_G = optim.lr_scheduler.ExponentialLR(
            self.optim_G, gamma=(1-args.decay), last_epoch=-1)
        self.decay_D = optim.lr_scheduler.ExponentialLR(
            self.optim_D, gamma=(1-args.decay), last_epoch=-1)

        # data loaders
        self.logger.info("Creating data loaders")
        train_sets = []
        valid_sets = []
        for directory in args.data_dir:
            assert os.path.isdir(directory)
            train_sets.append(ISTDDataset(directory,
                                          subset="train",
                                          datas=["img", "target", "matte"],
                                          transforms=transform.transforms(
                                              # resize=(300, 400),
                                              scale=args.aug_scale,
                                              angle=args.aug_angle,
                                              flip_prob=0.5,
                                              crop_size=args.image_size),
                                          preload=False,
                                          name=os.path.basename(directory)))
            valid_sets.append(ISTDDataset(directory,
                                          subset="test",
                                          datas=["img", "target", "matte"],
                                          # transforms=transform.transforms(
                                          #     resize=(256, 256)),
                                          preload=False,
                                          name=os.path.basename(directory)))
        train_dataset = ConcatDataset(train_sets)
        valid_dataset = ConcatDataset(valid_sets)

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
            # self.d_loss_fn = args.D_loss_fn
            # self.d_loss_type = args.D_loss_type
            self.adv_loss = AdversarialLoss(
                ls=(args.D_loss_fn == "leastsqure"),
                rel=("rel" in args.D_type),
                avg=("avg" in args.D_type))
            self.adv_loss.to(self.device, non_blocking=True)

            self.data_loss = DataLoss().to(self.device)
            self.visual_loss = VisualLoss().to(self.device)
            # data1 loss = 1
            self.lambda1 = 5     # data2 loss
            self.lambda2 = 0.5   # CGAN1 loss
            self.lambda3 = 0.5   # CGAN2 loss
            self.lambda4 = 5     # Vis1 loss
            self.lambda5 = 50    # Vis2 loss
            self.adapt = args.softadapt
            # if self.adapt:
            #     self.soft_adapt = SoftAdapt(
            #         ["adv", "data", "visual"],
            #         init_weights=[1, self.lambda1, self.lambda2],
            #         beta=0.1, weighted=True, normalized=True)
            #     self.soft_adapt.to(self.device, non_blocking=True)
            self.began = (args.net_D == "began")
            self.gamma = 0.7
            self.lambda_k = 0.001

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
            self.vis_interval = args.vis_every
        if "infer" in args.tasks:
            self.inferd_dir = args.infered

    def train(self, epochs=5000):
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

        if self.began:
            self.k1 = 0
            self.k2 = 0
        for epoch in progress:
            visualize = (epoch % self.vis_interval == 0)
            log_scalars = (epoch % self.log_interval == 0)
            self.run_epoch(visualization=visualize,
                           log_scalars=log_scalars, epoch=epoch)

            if (epoch % self.valid_interval == 0):
                loss = self.run_epoch(training=False, epoch=epoch)
                if loss < best_loss:
                    best_loss = loss
                    self.save(self.weights_dir, "best")
                    self.logger.info(f"Improvement after epoch {epoch}, "
                                     f"error = {best_loss:4f}")
                    with SummaryWriter(self.valid_logdir) as writer:
                        writer.add_text("best",
                                        f"{epoch}: loss={best_loss}", epoch)

        total_time = datetime.timedelta(seconds=(time.time()-start_time))
        self.logger.info(f"Training time {total_time}")
        self.logger.info(f"Best validation loss: {best_loss:.3f}")

    def run_epoch(self, training=True,
                  visualization=False, log_scalars=False, epoch=0):
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
            # Always log scalars and images when validating
            log_scalars = True
            visualization = True

        data_loader = self.train_loader if training else self.valid_loader
        logdir = self.train_logdir if training else self.valid_logdir
        if visualization:
            n_images_to_show = 8
            images_x = []
            images_m = []
            images_y = []
        if log_scalars:
            loss = dict.fromkeys(["G", "G1", "G2", "D", "D1", "D2",
                                  "data1", "data2", "vis1", "vis2"], 0.0)
            D1_out = dict.fromkeys(["real", "fake", "diff"], 0.0)
            D2_out = dict.fromkeys(["real", "fake", "diff"], 0.0)
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
                if self.began:
                    D1_loss_real = self.data_loss(C1_real, m.detach())
                    D1_loss_fake = self.data_loss(C1_fake, m_pred.detach())
                    D1_loss = D1_loss_real - self.k1 * D1_loss_fake

                    D2_loss_real = self.data_loss(C2_real, y.detach())
                    D2_loss_fake = self.data_loss(C2_fake, y_pred.detach())
                    D2_loss = D2_loss_real - self.k2 * D2_loss_fake
                else:
                    D1_loss = self.adv_loss(C1_real, C1_fake, D_loss=True)
                    D2_loss = self.adv_loss(C2_real, C2_fake, D_loss=True)

                D_loss = self.lambda2 * D1_loss + self.lambda3 * D2_loss
                if training:
                    D_loss.backward()
                    self.optim_D.step()

                if log_scalars:
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
                if training:  # D is not updated when validating
                    C1_real = self.D1(torch.cat((x, m), dim=1))
                    C1_fake = self.D1(torch.cat((x, m_pred), dim=1))
                    C2_real = self.D2(torch.cat((x, m, y), dim=1))
                    C2_fake = self.D2(torch.cat((x, m_pred, y_pred), dim=1))
                if self.began:
                    G1_loss = self.data_loss(C1_fake, m_pred.detach())
                    G2_loss = self.data_loss(C2_fake, y_pred.detach())
                else:
                    G1_loss = self.adv_loss(C1_real, C1_fake, D_loss=False)
                    G2_loss = self.adv_loss(C2_real, C2_fake, D_loss=False)

                data1_loss = self.data_loss(m_pred, m)
                data2_loss = self.data_loss(y_pred, y)
                vis1_loss = self.visual_loss(m_pred.expand(-1, 3, -1, -1),
                                             m.expand(-1, 3, -1, -1))
                vis2_loss = self.visual_loss(y_pred, y)
                # if self.adapt:
                #     G_loss = self.soft_adapt({"adv": g_loss_adv,
                #                               "data": data_loss,
                #                               "visual": visual_loss},
                #                              update_weights=training)
                # else:
                G_loss = (data1_loss +
                          self.lambda1 * data2_loss +
                          self.lambda2 * G1_loss +
                          self.lambda3 * G2_loss +
                          self.lambda4 * vis1_loss +
                          self.lambda5 * vis2_loss)
                if training:
                    G_loss.backward()
                    self.optim_G.step()
                    if self.began:
                        balance1 = (self.gamma * D1_loss_real.item() -
                                    D1_loss_fake.item())
                        self.k1 = np.clip(
                            self.k1 + self.lambda_k * balance1, 0, 1)
                        balance2 = (self.gamma * D2_loss_real.item() -
                                    D2_loss_fake.item())
                        self.k2 = np.clip(
                            self.k2 + self.lambda_k * balance2, 0, 1)

                if log_scalars:
                    loss["G1"] += G1_loss.item()
                    loss["G2"] += G2_loss.item()
                    loss["data1"] += data1_loss.item()
                    loss["data2"] += data2_loss.item()
                    loss["vis1"] += vis1_loss.item()
                    loss["vis2"] += vis2_loss.item()
                    loss["G"] += G_loss.item()

                if visualization and len(images_x) < n_images_to_show:
                    with torch.no_grad():
                        for x, m, y in zip(x.cpu().unbind(),
                                           m_pred.cpu().unbind(),
                                           y_pred.cpu().unbind()):
                            images_x.append(x[(2, 1, 0), ...])
                            images_m.append(m)
                            images_y.append(y[(2, 1, 0), ...])
                            if len(images_x) >= n_images_to_show:
                                break

        if training:
            self.decay_G.step()
            self.decay_D.step()

        if visualization:
            with SummaryWriter(log_dir=logdir) as writer:
                grid_x = make_grid(images_x, nrow=4,
                                   normalize=True, range=(-1, 1))
                grid_m = make_grid(images_m, nrow=4,
                                   normalize=True, range=(-1, 1))
                grid_y = make_grid(images_y, nrow=4,
                                   normalize=True, range=(-1, 1))
                writer.add_image("input", grid_x, global_step=epoch)
                writer.add_image("matte", grid_m, global_step=epoch)
                writer.add_image("output", grid_y, global_step=epoch)

        if log_scalars:
            loss["total"] = loss["G"]*0.8 + loss["D"]*0.2
            D1_out["diff"] = D1_out["real"] - D1_out["fake"]
            D2_out["diff"] = D2_out["real"] - D2_out["fake"]
            n_batches = len(data_loader)
            with SummaryWriter(log_dir=logdir) as writer:
                for key in loss:
                    writer.add_scalar(
                        f"Loss/{key}", loss[key]/n_batches, epoch)
                for key in D1_out:
                    writer.add_scalar(
                        f"D1_output/{key}", D1_out[key]/n_batches, epoch)
                for key in D2_out:
                    writer.add_scalar(
                        f"D2_output/{key}", D2_out[key]/n_batches, epoch)
            self.save(self.weights_dir, "latest")
            # softadapt_weights = self.soft_adapt.get_weights()
            # if self.adapt else {}

        torch.cuda.empty_cache()
        return loss["total"]/len(data_loader) if not training else None

    def infer(self,):
        with torch.no_grad():
            self.G1.eval()
            self.G2.eval()
            data_loader = self.valid_loader
            for r in ["shadowless", "matte"]:
                for s in self.valid_loader.dataset.datasets:
                    os.makedirs(os.path.join(self.inferd_dir, r, s.name))
            for (filenames, x, _, _) in tqdm(data_loader,
                                             desc="Processing data",
                                             total=len(data_loader),
                                             ncols=80, ascii=True):
                m_pred_list = []
                y_pred_list = []

                x = x.to(self.device, non_blocking=True)
                m_pred = self.G1(x)
                y_pred = self.G2(torch.cat((x, m_pred), dim=1))

                # x_np = x.detach().cpu().numpy()
                m_pred_np = m_pred.detach().cpu().numpy()
                y_pred_np = y_pred.detach().cpu().numpy()
                m_pred_list.extend([m_pred_np[s].transpose(1, 2, 0)
                                    for s in range(m_pred_np.shape[0])])
                y_pred_list.extend([y_pred_np[s].transpose(1, 2, 0)
                                    for s in range(y_pred_np.shape[0])])

                for m_pred, y_pred, name in \
                        zip(m_pred_list, y_pred_list, filenames):
                    # img_pred = cv.resize(
                    #     y_pred, (256, 192), interpolation=cv.INTER_LINEAR)
                    img_pred = utils.float2uint(y_pred*0.5+0.5)
                    cv.imwrite(os.path.join(
                        self.inferd_dir, "shadowless", name+".png"), img_pred)

                    # matte_pred = cv.resize(
                    #     m_pred, (256, 192), interpolation=cv.INTER_LINEAR)
                    matte_pred = utils.float2uint(m_pred*0.5+0.5)
                    cv.imwrite(os.path.join(
                        self.inferd_dir, "matte", name+".png"), matte_pred)
                    # if save_sp:
                    #     np.save(os.path.join(
                    #         self.inferd_dir+"sp", name), sp_pred)
        return

    def save(self, weights=None, suffix="latest"):
        module_G1 = self.G1.module if isinstance(self.G1, nn.DataParallel) \
            else self.G1
        module_G2 = self.G2.module if isinstance(self.G2, nn.DataParallel) \
            else self.G2
        module_D1 = self.D1.module if isinstance(self.D1, nn.DataParallel) \
            else self.D1
        module_D2 = self.D2.module if isinstance(self.D2, nn.DataParallel) \
            else self.D2
        G1_name = module_G1.__class__.__name__
        G2_name = module_G2.__class__.__name__
        D1_name = module_D1.__class__.__name__
        D2_name = module_D2.__class__.__name__
        if weights is None:
            weights = self.weights_dir
        torch.save(module_G1.state_dict(),
                   os.path.join(weights, f"G1_{G1_name}_{suffix}.pt"))
        torch.save(module_G2.state_dict(),
                   os.path.join(weights, f"G2_{G2_name}_{suffix}.pt"))
        torch.save(module_D1.state_dict(),
                   os.path.join(weights, f"D1_{D1_name}_{suffix}.pt"))
        torch.save(module_D2.state_dict(),
                   os.path.join(weights, f"D2_{D2_name}_{suffix}.pt"))

    def init_weight(self, g1_weights=None, g2_weights=None,
                    d1_weights=None, d2_weights=None):
        if g1_weights:
            state_dict = torch.load(g1_weights, map_location=self.device)
            self.G1.load_state_dict(state_dict)
            self.logger.info(f"Loaded G1 weights: {g1_weights}")
        if g2_weights:
            state_dict = torch.load(g2_weights, map_location=self.device)
            self.G2.load_state_dict(state_dict)
            self.logger.info(f"Loaded G2 weights: {g2_weights}")
        if d1_weights:
            state_dict = torch.load(d1_weights, map_location=self.device)
            self.D1.load_state_dict(state_dict)
            self.logger.info(f"Loaded D1 weights: {d1_weights}")
        if d2_weights:
            state_dict = torch.load(d2_weights, map_location=self.device)
            self.D2.load_state_dict(state_dict)
            self.logger.info(f"Loaded D2 weights: {d2_weights}")
