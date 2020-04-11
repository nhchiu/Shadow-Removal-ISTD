#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Handling logs with logging while tracking the process with tqdm
"""

import io
import logging
import sys

from tqdm.auto import tqdm, trange

class TeeIo(object):
    """
    A io.TextIOWrapper that writes to a file and a stream at the same time.
    Similar to the funtion of "tee".
    """

    def __init__(self, file, stream=sys.stderr):
        self.file = open(file, 'w')
        self.stream = stream
        # self.stdout = sys.stdout
        # sys.stdout = self
        # super().__init__(*args, **kwargs)

    def close(self):
        # sys.stdout = self.stdout
        self.file.close()

    def write(self, data, to_stream=True):
        self.file.write(data)
        if to_stream:
            self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()


class TqdmStreamHandler(logging.StreamHandler):
    """
    A StreamHandler that writes to the stream with tqdm.write().
    """

    def __init__(self, stream):
        super().__init__(stream)

    def emit(self, record):
        try:
            msg = self.format(record)
            return tqdm.write(msg, file=self.stream)
        except (KeyboardInterrupt, SystemExit):
            raise
        except:
            self.handleError(record)
        return super().emit(record)

    def flush(self):
        return super().flush()


class Logger(object):
    """
        A Logger that logs to a file and the console at the same time.
        The file handsler is compatable with tqdm.
    """

    def __init__(self, log_file, level=logging.INFO, *args, **kwargs):
        self.stream = TeeIo(log_file, stream=sys.stderr)
        self.logger = self._creat_logger(log_file, level=level)

    def _creat_logger(self,
                      log_file,
                      level=logging.DEBUG,
                      file_level=None,
                      console_level=None
                      ):
        file_level = level if file_level is None else file_level
        console_level = level if console_level is None else console_level

        logger = logging.getLogger(log_file)
        logger.setLevel(level)
        # create file handler which logs even debug messages
        fh = TqdmStreamHandler(self.stream)
        fh.setLevel(file_level)
        # create console handler with a higher log level
        # ch = logging.StreamHandler(sys.stderr)
        # ch.setLevel(console_level)
        # create formatter and add it to the handlers
        file_formatter = logging.Formatter(
            '%(asctime)s - %(filename)-15s %(levelname)-6s %(message)s',
            datefmt="%H:%M:%S",
            style='%')
        fh.setFormatter(file_formatter)
        # console_formatter = logging.Formatter(
        #     '%(message)s')
        # ch.setFormatter(console_formatter)
        # add the handlers to logger
        logger.addHandler(fh)
        # logger.addHandler(ch)
        return logger

    def __del__(self):
        self.stream.close()
        return

    def tqdm(self, *args, **kwargs):
        kwargs["file"] = self.stream
        return tqdm(*args, **kwargs)

    def trange(self, *args, **kwargs):
        kwargs["file"] = self.stream
        return trange(*args, **kwargs)

# brain-segmentation-pytorch/logger.py
# from io import BytesIO

# import scipy.misc
# import tensorflow as tf


# class Logger(object):

#     def __init__(self, log_dir):
#         self.writer = tf.summary.FileWriter(log_dir)

#     def scalar_summary(self, tag, value, step):
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()

#     def image_summary(self, tag, image, step):
#         s = BytesIO()
#         scipy.misc.toimage(image).save(s, format="png")

#         # Create an Image object
#         img_sum = tf.Summary.Image(
#             encoded_image_string=s.getvalue(),
#             height=image.shape[0],
#             width=image.shape[1],
#         )

#         # Create and write Summary
#         summary = tf.Summary(value=[tf.Summary.Value(tag=tag, image=img_sum)])
#         self.writer.add_summary(summary, step)
#         self.writer.flush()

#     def image_list_summary(self, tag, images, step):
#         if len(images) == 0:
#             return
#         img_summaries = []
#         for i, img in enumerate(images):
#             s = BytesIO()
#             scipy.misc.toimage(img).save(s, format="png")

#             # Create an Image object
#             img_sum = tf.Summary.Image(
#                 encoded_image_string=s.getvalue(),
#                 height=img.shape[0],
#                 width=img.shape[1],
#             )

#             # Create a Summary value
#             img_summaries.append(
#                 tf.Summary.Value(tag="{}/{}".format(tag, i), image=img_sum)
#             )

#         # Create and write Summary
#         summary = tf.Summary(value=img_summaries)
#         self.writer.add_summary(summary, step)
#         self.writer.flush()
