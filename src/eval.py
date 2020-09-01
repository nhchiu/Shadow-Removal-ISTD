#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate errors
"""

import argparse
import json
import logging
import os

import cv2 as cv
import numpy as np
from tqdm.auto import tqdm
from skimage import io, metrics, color, transform, util
# from skimage import io


def main(args):
    snapshotargs(args, filename="args.json")

    # log_file = os.path.join(
    #     args.logs, os.path.splitext(__file__)[0]+"-"+time_str+".log")
    # logger = Logger(log_file, level=logging.DEBUG)
    set_logger(args.logfile)
    logger = logging.getLogger(__name__)
    logger.info("Arguments:")
    logger.info(args)

    # rmmse = RMMSE(args.dir1, args.dir2, args.image_size)
    # mrmse = MRMSE(args.dir1, args.dir2, args.image_size)
    errors = all_metrics(args.dir1, args.dir2,
                         size=args.image_size,
                         maskdir=args.maskdir)
    # logger.info(f"RMMSE: {rmmse}")
    # logger.info(f"MRMSE: {mrmse}")
    for k in errors:
        logger.info(f"{k}: {errors[k]}")


def all_metrics(dir1, dir2, size=None, maskdir=None):
    files = os.listdir(dir1)
    rmses = []
    maes = []
    rmses_nonshadow = []
    maes_nonshadow = []
    pixels = []
    pixels_nonshadow = []
    psnrs = []
    ssims = []
    for f in tqdm(files):
        #     img1 = (cv.imread(os.path.join(dir1, f))/255.).astype(np.float32)
        #     img1 = cv.resize(img1, (size, size), interpolation=cv.INTER_LINEAR)
        #     img1 = cv.cvtColor(img1, cv.COLOR_BGR2LAB)

        #     img2 = (cv.imread(os.path.join(dir2, f))/255.).astype(np.float32)
        #     img2 = cv.resize(img2, (size, size), interpolation=cv.INTER_LINEAR)
        #     img2 = cv.cvtColor(img2, cv.COLOR_BGR2LAB)
        #     # rmses.append(np.sqrt(np.mean((img1-img2)**2)))
        #     rmses += (MAE(img1, img2))
        # return {"rmse": rmses/len(files)}

        img1 = util.img_as_float32(io.imread(os.path.join(dir1, f)))
        img2 = transform.resize(
            util.img_as_float32(io.imread(os.path.join(dir2, f))),
            img1.shape,  mode="edge", anti_aliasing=False)
        if maskdir is not None:
            mask = transform.resize(
                io.imread(os.path.join(maskdir, f), as_gray=True),
                (img1.shape[:2]),  mode="edge")
        else:
            mask = (np.ones((img1.shape[0], img1.shape[1])) == 1)
        if size is not None:
            img1_resized = transform.resize(
                img1, (size, size), mode="edge", anti_aliasing=False)
            img2_resized = transform.resize(
                img2, (size, size), mode="edge", anti_aliasing=False)
            # img1_resized = cv.resize(img1, (size, size))
            # img2_resized = cv.resize(img2, (size, size))
            mask_resized = util.img_as_bool(transform.resize(
                mask, (size, size), mode="edge"))
        else:
            img1_resized, img2_resized = img1, img2
            mask_resized = util.img_as_bool(mask)

        rmses.append(RMSE(color.rgb2lab(img1_resized),
                          color.rgb2lab(img2_resized),
                          mask_resized))
        maes.append(MAE(color.rgb2lab(img1_resized),
                        color.rgb2lab(img2_resized),
                        mask_resized))
        pixels.append(np.count_nonzero(mask_resized))
        mask_resized = np.logical_not(mask_resized)
        rmses_nonshadow.append(RMSE(color.rgb2lab(img1_resized),
                                    color.rgb2lab(img2_resized),
                                    mask_resized))
        maes_nonshadow.append(MAE(color.rgb2lab(img1_resized),
                                  color.rgb2lab(img2_resized),
                                  mask_resized))
        pixels_nonshadow.append(np.count_nonzero(mask_resized))
        if maskdir is None:
            psnrs.append(PSNR(img1, img2))
            ssims.append(SSIM(img1, img2))
    results = {"rmse": np.sum(rmses)/np.sum(pixels),
               "mae": np.sum(maes)/np.sum(pixels),
               "rmse_non": np.sum(rmses_nonshadow)/np.sum(pixels_nonshadow),
               "mae_non": np.sum(maes_nonshadow)/np.sum(pixels_nonshadow),
               "rmse_all": (np.sum(rmses_nonshadow)+np.sum(rmses))/(
                   np.sum(pixels_nonshadow)+np.sum(pixels)),
               "mae_all": (np.sum(maes_nonshadow)+np.sum(maes))/(
                   np.sum(pixels_nonshadow)+np.sum(pixels))}
    if maskdir is None:
        results["psnr"] = np.mean(psnrs)
        results["ssim"] = np.mean(ssims)
    return results


def MSE(img1, img2):
    # return np.mean(np.square(img1-img2))
    return metrics.mean_squared_error(img1, img2)


def MAE(img1, img2, mask):
    return np.sum(np.abs(img1-img2)[mask]).astype(np.float64)


def RMSE(img1, img2, mask):
    return np.sum(np.sqrt(
        np.sum((img1 - img2)**2, axis=-1))[mask]).astype(np.float64)


def PSNR(img1, img2):
    # return 10 * np.log10(1 / MSE(img1, img2))
    return metrics.peak_signal_noise_ratio(img1, img2)


def SSIM(img1, img2):
    return metrics.structural_similarity(img1, img2, multichannel=True)


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


def snapshotargs(args, filename="args.json"):
    args_file = os.path.join(os.path.curdir, filename)
    with open(args_file, "w") as fp:
        json.dump(vars(args), fp, indent=4, sort_keys=True)


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "y", "1")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate errors")
    parser.add_argument("dir1", type=str)
    parser.add_argument("dir2", type=str)
    parser.add_argument(
        "-m", "--maskdir", help="mask directory (default: %(default)s)",
        default=None)
    parser.add_argument(
        "--image-size", help="target image size (default: %(default)d)",
        default=256, type=int,)
    parser.add_argument(
        "--logfile", help=" (default: %(default)d)",
        default="./eval.log")
    args = parser.parse_args()
    main(args)
