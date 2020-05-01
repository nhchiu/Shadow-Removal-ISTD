import numbers

import cv2 as cv
import numpy as np
# from torchvision.transforms import Compose


def transforms(resize=None,
               scale=None,
               angle=None,
               flip_prob=None,
               crop_size=None
               ):
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

    def __call__(self, sample: tuple):
        for transform in self.transforms:
            sample = transform(sample)
        return sample


class RandomScale(object):

    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        rows, cols = sample[0].shape[:2]

        scale = np.random.uniform(low=1.0 - self.scale, high=1.0 + self.scale)

        M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 0, scale)
        interp = cv.INTER_LINEAR if scale > 1 else cv.INTER_AREA

        output = [
            cv.warpAffine(i, M, (cols, rows),
                          flags=interp,
                          borderMode=cv.BORDER_CONSTANT)
            for i in sample
        ]

        return tuple(output)


class RandomRotate(object):
    """
    Rotate the image by a random angle.
    """

    def __init__(self, angle):
        self.angle = angle

    def __call__(self, sample):
        rows, cols = sample[0].shape[:2]

        angle = np.random.uniform(low=-self.angle, high=self.angle)

        M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), angle, 1)

        output = [
            cv.warpAffine(i, M, (cols, rows), borderMode=cv.BORDER_CONSTANT)
            for i in sample
        ]
        return tuple(output)


class RandomHorizontalFlip(object):

    def __init__(self, flip_prob):
        self.flip_prob = flip_prob

    def __call__(self, sample):
        if np.random.rand() > self.flip_prob:
            return sample
        else:
            output = [
                np.fliplr(i).copy() for i in sample
            ]
            return tuple(output)


class RandomCrop(object):
    """
    Crop the np.ndarray at a random location.
    """

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        rows, cols = sample[0].shape[:2]
        to_rows, to_cols = self.size

        # padding is needed if the target size is larger than the image
        if to_rows > rows or to_cols > cols:
            pad_height = (to_rows - rows) if to_rows > rows else 0
            pad_width = (to_cols - cols) if to_cols > cols else 0
            sample = [
                cv.copyMakeBorder(i,
                                  pad_height,
                                  pad_height,
                                  pad_width,
                                  pad_width,
                                  cv.BORDER_CONSTANT, value=0)
                for i in sample
            ]
            # update the size size after padding
            rows, cols = sample[0].shape[:2]

        row_offset = np.random.randint(low=0, high=(rows-to_rows))
        col_offset = np.random.randint(low=0, high=(cols-to_cols))

        output = [
            i[row_offset:row_offset+to_rows,
                col_offset:col_offset+to_cols, ...].copy() for i in sample
        ]

        return tuple(output)


class Resize(object):
    """Resize the image to the given size."""

    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, sample):
        rows, cols = sample[0].shape[:2]
        if self.size[0] < rows and self.size[1] < cols:
            interp = cv.INTER_AREA
        else:
            interp = cv.INTER_LINEAR
        output = [
            cv.resize(i, (self.size[1], self.size[0]), interpolation=interp)
            for i in sample
        ]
        return tuple(output)
