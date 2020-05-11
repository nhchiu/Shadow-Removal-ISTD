#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from multiprocessing import Pool

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from numpy.polynomial import polynomial as P
from tqdm.auto import tqdm

import utils


def Polyfit(args: tuple) -> tuple:
    deg, ksize, r, c, channel, img, gt, weight, w2 = args
    x = img[r:r+ksize, c:c+ksize, channel].ravel()
    y = gt[r:r+ksize, c:c+ksize, channel].ravel()
    w1 = weight[r:r+ksize, c:c+ksize].ravel()
    coef, _ = P.polyfit(x, y, deg, full=True, w=w1*w2)
    return (r, c, channel, coef)


def process_images(args: tuple):
    image_dir, target_dir, filename, save_sp, save_img = args
    img = cv.imread(os.path.join(image_dir, filename), cv.IMREAD_COLOR)
    target = cv.imread(os.path.join(target_dir, filename), cv.IMREAD_COLOR)

    sp = utils.get_sp(img, target)
    if save_sp:
        sp_dir = os.path.join(target_dir, os.path.pardir, "sp")
        utils.mkdir(sp_dir)
        np.save(os.path.join(sp_dir, os.path.splitext(filename)[0]), sp)

    if save_img:
        img_dir = os.path.join(target_dir, os.path.pardir, "sp_restored_img")
        utils.mkdir(img_dir)
        cv.imwrite(os.path.join(img_dir, filename), utils.apply_sp(img, sp))
    return


def main(args):
    root = args.path
    subset = args.subset
    image_dir = os.path.join(root, subset, subset+"_A")
    target_dir = os.path.join(root, subset, subset+"_C_fixed_official")

    filenames = sorted(os.listdir(image_dir))
    print("{} files to process".format(len(filenames)), file=sys.stderr)
    results = map(process_images, tqdm(
        ((image_dir, target_dir, f, args.save_sp, args.save_img)
         for f in filenames),
        total=len(filenames)))

    errors = 0
    for it in results:
        if it is not None:
            errors += 1

    if errors > 0:
        print("there are {:d} errors".format(errors), file=sys.stderr)
    else:
        print("completed proprecessing.", file=sys.stderr)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess shadow removal dataset"
    )
    parser.add_argument(
        "--path", help="Path to ISTD dataset (default: %(default)s)",
        type=str,
        default="../ISTD_DATASET"
    )
    parser.add_argument(
        "--subset", help="the subset to process (default: %(default)s)",
        type=str,
        default="train",
        choices=['train', 'test']
    )
    parser.add_argument(
        "--save-sp", help="whether to save the shadow parameters (default: %(default)s)",
        type=bool,
        nargs='?',
        const=True,
        default=True
    )
    parser.add_argument(
        "--save-img", help="whether to save the iamge restored with SP (default: %(default)s)",
        type=bool,
        nargs='?',
        const=True,
        default=False
    )
    args = parser.parse_args()
    main(args)
