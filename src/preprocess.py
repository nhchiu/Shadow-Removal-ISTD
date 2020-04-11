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


def Polyfit(arg):
    deg, ksize, r, c, channel, img, gt, weight, w2 = arg
    x = img[r:r+ksize, c:c+ksize, channel].ravel()
    y = gt[r:r+ksize, c:c+ksize, channel].ravel()
    w1 = weight[r:r+ksize, c:c+ksize].ravel()
    coef, _ = P.polyfit(x, y, deg, full=True, w=w1*w2)
    return (r, c, channel, coef)


def GetShadowParameter(shadow, shadow_free, ksize=5, deg=1):
    """
    Calculate shadow parameters based on neighboring region.
    """
    hat_weight = np.concatenate(
        (np.arange(64), np.ones(128)*64, np.arange(64)[::-1]))

    border = (ksize-1)//2  # ksize should be an odd number
    bordered_shadow = cv.copyMakeBorder(
        shadow, border, border, border, border, cv.BORDER_REPLICATE)
    bordered_shadow_free = cv.copyMakeBorder(
        shadow_free, border, border, border, border, cv.BORDER_REPLICATE)
    gray_img = cv.cvtColor(bordered_shadow_free, cv.COLOR_BGR2GRAY)
    weight_gray = hat_weight[gray_img]

    sp = np.ones((shadow.shape[0], shadow.shape[1],
                  3, (deg+1)), dtype=np.float32)
    weight_distance = np.array([ksize-np.abs(i-border)-np.abs(j-border)
                                for i in range(ksize) for j in range(ksize)], dtype=np.int)

    a = [(deg, ksize, r, c, channel, bordered_shadow, bordered_shadow_free, weight_gray, weight_distance)
         for r in range(shadow.shape[0])
         for c in range(shadow.shape[1])
         for channel in range(3)]

    with Pool(min(10, os.cpu_count()//2)) as pool:
        coefs = pool.imap_unordered(Polyfit, a, chunksize=480)
    # coefs = map(Polyfit, a)
        for i in tqdm(coefs, total=shadow.shape[0]*shadow.shape[1]*3):
            r, c, channel, coef = i
            sp[r, c, channel, :] = coef
    # if show_plot:
    #     plt.scatter(x, y, marker='.', label='data')
    #     plt.plot(x, y_pred, 'b-', label='linear fit')
    #     plt.legend(loc='lower right')
    #     plt.title('Linear regression')
    #     if savepath is not None:
    #         plt.savefig(savepath)
    #     plt.show()

    return sp


# %%
def process_images(arg):
    image_dir, target_dir, filename, save_sp, save_img = arg
    img = cv.imread(os.path.join(image_dir, filename), cv.IMREAD_COLOR)
    target = cv.imread(os.path.join(target_dir, filename), cv.IMREAD_COLOR)
    sp = GetShadowParameter(img, target)
    if save_sp:
        sp_dir = os.path.join(target_dir, os.path.pardir, "sp")
        if not (os.path.exists(sp_dir) and os.path.isdir(sp_dir)):
            os.makedirs(sp_dir, exist_ok=True)
        np.save(os.path.join(sp_dir, os.path.splitext(filename)[0]), sp)
    result = np.clip((sp[..., 0] + sp[..., 1] * img), 0, 255).astype(np.uint8)
    if save_img:
        img_dir = os.path.join(target_dir, os.path.pardir, "sp_restored_img")
        if not (os.path.exists(img_dir) and os.path.isdir(img_dir)):
            os.makedirs(img_dir, exist_ok=True)
        cv.imwrite(os.path.join(img_dir, filename), result)
    # plt.figure()
    # plt.imshow(cv.cvtColor(result, cv.COLOR_BGR2RGB))
    # plt.close()
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
        print("ther are {:d} errors".format(errors), file=sys.stderr)
    else:
        print("completed proprecessing.", file=sys.stderr)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess shadow removal dataset"
    )
    parser.add_argument(
        "--path",
        type=str,
        default="../ISTD_DATASET",
        help="Path to ISTD dataset (default: %(default)s)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="train",
        help="the subset to process (default: %(default)s)",
        choices=['train', 'test']
    )
    parser.add_argument(
        "--save-sp",
        type=bool,
        nargs='?',
        const=True,
        default=True,
        help="whether to save the shadow parameters (default: %(default)s)"
    )
    parser.add_argument(
        "--save-img",
        type=bool,
        nargs='?',
        const=True,
        default=False,
        help="whether to save the iamge restored with SP (default: %(default)s)"
    )
    args = parser.parse_args()
    main(args)
