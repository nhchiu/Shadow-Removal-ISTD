import numpy as np
import os


def mkdir(path: str):
    if not (os.path.exists(path) and os.path.isdir(path)):
        os.makedirs(path, exist_ok=True)
    return


def get_sp(shadowed, shadowless, ksize: int = 5, deg: int = 1):
    """
    type: (np.ndarray, np.ndarray, int, int) -> np.ndarray[np.float32]
    Calculate shadow parameters based on neighboring region.
    """
    assert shadowed.dtype == shadowless.dtype
    # hat_weight = np.concatenate(
    #     (np.arange(64), np.ones(128)*64, np.arange(64)[::-1]))

    # border = (ksize-1)//2  # ksize should be an odd number
    # bordered_shadow = cv.copyMakeBorder(
    #     shadow, border, border, border, border, cv.BORDER_REPLICATE)
    # bordered_shadow_free = cv.copyMakeBorder(
    #     shadow_free, border, border, border, border, cv.BORDER_REPLICATE)
    # gray_img = cv.cvtColor(bordered_shadow_free, cv.COLOR_BGR2GRAY)
    # weight_gray = hat_weight[gray_img]

    # sp = np.ones((shadow.shape[0], shadow.shape[1],
    #               3, (deg+1)), dtype=np.float32)
    # weight_distance = np.array([ksize-np.abs(i-border)-np.abs(j-border)
    #                             for i in range(ksize) for j in range(ksize)], dtype=np.int)

    # a = [(deg, ksize, r, c, channel, bordered_shadow, bordered_shadow_free, weight_gray, weight_distance)
    #      for r in range(shadow.shape[0])
    #      for c in range(shadow.shape[1])
    #      for channel in range(3)]

    # with Pool(min(10, os.cpu_count()//2)) as pool:
    #     coefs = pool.imap_unordered(Polyfit, a, chunksize=480)
    #     for i in tqdm(coefs, total=shadow.shape[0]*shadow.shape[1]*3):
    #         r, c, channel, coef = i
    #         sp[r, c, channel, :] = coef
    shadowed[shadowed == 0] = 1
    sp = shadowless.astype(np.float32) / shadowed.astype(np.float32)
    return sp


def apply_sp(shadowed, sp):
    """
    type: (np.ndarray, np.ndarray[np.float32]) -> np.ndarray
    """
    if shadowed.dtype == np.uint8:
        return np.clip((sp * shadowed), 0, 255).astype(np.uint8)
    else:  # np.float32
        return np.clip((sp * shadowed), 0, 1).astype(np.float32)


def uint2float(array):
    return array.astype(np.float32)/255


def float2uint(array):
    return np.clip(array * 255, 0, 255).astype(np.uint8)


def normalize_ndarray(array):
    lower = np.percentile(array,  3)
    upper = np.percentile(array, 97)
    img = (array - lower) / (upper - lower)
    return float2uint(img)
