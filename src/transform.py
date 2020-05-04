import numbers

import cv2 as cv
import numpy as np


def transforms(resize=None,
               scale=None,
               angle=None,
               flip_prob=None,
               crop_size=None):
    transform_list = []
    if resize is not None:
        transform_list.append(Resize(resize))
    if scale is not None:
        transform_list.append(RandomScale(scale))
    if angle is not None:
        transform_list.append(RandomRotate(angle))
    if flip_prob is not None:
        transform_list.append(RandomHorizontalFlip(flip_prob))
    if crop_size is not None:
        transform_list.append(RandomCrop(crop_size))

    return Compose(transform_list)


class Compose(object):
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, *sample):
        for transform in self.transforms:
            sample = transform(*sample)
        return sample


class Normalize(object):
    def __init__(self, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)):
        self.mean = np.array(mean).reshape(-1)
        self.std = np.array(std).reshape(-1)

    def __call__(self, *datas, inverse=False):
        outputs = []
        for x in datas:
            assert x.shape[-1] == len(self.mean)
            assert x.shape[-1] == len(self.std)
            y = np.empty_like(x)
            if not inverse:
                y = (x-self.mean) / self.std
            else:
                y = (x*self.std) + self.mean
            outputs.append(y)
        if len(datas) > 1:
            return outputs
        else:
            return outputs[0]


class RandomScale(object):

    def __init__(self, scale):
        assert 0 <= scale and scale <= 0.5
        self.scale = scale

    def __call__(self, *datas):
        outputs = []
        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)
        interp = cv.INTER_LINEAR if scale > 1 else cv.INTER_AREA
        for x in datas:
            rows, cols = x.shape[:2]
            M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 0, scale)
            outputs.append(cv.warpAffine(x, M, (cols, rows),
                                         flags=interp,
                                         borderMode=cv.BORDER_CONSTANT))
        if len(datas) > 1:
            return outputs
        else:
            return outputs[0]


class RandomRotate(object):
    """
    Rotate the image by a random angle.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, *datas):
        outputs = []
        angle = np.random.uniform(low=-self.angle, high=self.angle)
        for x in datas:
            rows, cols = x.shape[:2]
            M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)
            outputs.append(cv.warpAffine(x, M, (cols, rows),
                                         borderMode=cv.BORDER_CONSTANT))
        if len(datas) > 1:
            return outputs
        else:
            return outputs[0]


class RandomHorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, *datas):
        if np.random.rand() > self.flip_prob:
            return datas
        else:
            outputs = [np.fliplr(x).copy() for x in datas]
            if len(datas) > 1:
                return outputs
            else:
                return outputs[0]


class RandomCrop(object):
    """
    Crop the np.ndarray at a random location.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            to_size = (int(size), int(size))
        else:
            to_size = size
        self.rows, self.cols = to_size

    def __call__(self, *datas):
        rows, cols = datas[0].shape[:2]  # datas should have the same size
        padding = self.rows > rows or self.cols > cols
        if padding:
            # padding is needed if the target size is larger than the image
            pad_height = max((self.rows - rows), 0)
            pad_width = max((self.cols - cols), 0)
            rows += 2*pad_height
            cols += 2*pad_width

        row_offset = np.random.randint(low=0, high=(rows-self.rows))
        col_offset = np.random.randint(low=0, high=(cols-self.cols))

        outputs = []
        for x in datas:
            if padding:
                x = cv.copyMakeBorder(x,
                                      pad_height, pad_height,
                                      pad_width, pad_width,
                                      cv.BORDER_CONSTANT, value=0)
            outputs.append(x[row_offset:row_offset+self.rows,
                             col_offset:col_offset+self.cols, ...].copy())
        if len(datas) > 1:
            return outputs
        else:
            return outputs[0]


class Resize(object):
    """Resize the image to the given size."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            to_size = (int(size), int(size))
        else:
            to_size = size
        self.rows, self.cols = to_size

    def __call__(self, *datas):
        outputs = []
        for x in datas:
            rows, cols = x.shape[:2]
            if self.rows < rows and self.cols < cols:
                interp = cv.INTER_AREA
            else:
                interp = cv.INTER_LINEAR
            outputs.append(
                cv.resize(x, (self.cols, self.rows), interpolation=interp))
        if len(datas) > 1:
            return outputs
        else:
            return outputs[0]
