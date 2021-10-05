#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file getLayerTests.py
# @date 13 Se 2020
# @brief Generate *.nnlayergolden file
# *.nnlayergolden file is expected to contain following information **in order**
# - Initial Weights
# - inputs
# - outputs
# - *gradients
# - weights
# - derivatives
#
# @author Jihoon Lee <jhoon.it.lee@samsung.com>

from multiprocessing.sharedctypes import Value
import warnings
import random
from functools import partial

from recorder import record_single

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import tensorflow as tf
    from tensorflow.python import keras as K

##
# @brief inpsect if file is created correctly
# @note this just checks if offset is corretly set, The result have to inspected
# manually
def inspect_file(file_name):
    with open(file_name, "rb") as f:
        while True:
            sz = int.from_bytes(f.read(4), byteorder="little")
            if not sz:
                break
            print("size: ", sz)
            print(np.fromfile(f, dtype="float32", count=sz))


if __name__ == "__main__":
    fc = K.layers.Dense(5)
    record_single(fc, (3, 1, 1, 10), "fc_plain")
    fc = K.layers.Dense(4)
    record_single(fc, (1, 1, 1, 10), "fc_single_batch")
    bn = K.layers.BatchNormalization()
    record_single(bn, (2, 4, 2, 3), "bn_channels_training", {"training": True})
    record_single(bn, (2, 4, 2, 3), "bn_channels_inference", {"training": False})
    bn = K.layers.BatchNormalization()
    record_single(bn, (2, 10), "bn_width_training", {"training": True})
    record_single(bn, (2, 10), "bn_width_inference", {"training": False})

    conv = K.layers.Conv2D(3, 2)
    record_single(conv, (1, 1, 4, 4), "conv_sb_minimum")
    record_single(conv, (3, 1, 4, 4), "conv_mb_minimum")

    conv = K.layers.Conv2D(2, 3, padding="same")
    record_single(conv, (1, 1, 4, 4), "conv_sb_same_remain")
    record_single(conv, (3, 1, 4, 4), "conv_mb_same_remain")

    conv = K.layers.Conv2D(2, 3, strides=2, padding="same")
    record_single(conv, (1, 3, 4, 4), "conv_sb_same_uneven_remain")
    record_single(conv, (3, 3, 4, 4), "conv_mb_same_uneven_remain")

    conv = K.layers.Conv2D(2, 3, strides=2, padding="valid")
    record_single(conv, (1, 3, 7, 7), "conv_sb_valid_drop_last")
    record_single(conv, (3, 3, 7, 7), "conv_mb_valid_drop_last")

    conv = K.layers.Conv2D(3, 2, strides=3)
    record_single(conv, (1, 2, 5, 5), "conv_sb_no_overlap")
    record_single(conv, (3, 2, 5, 5), "conv_mb_no_overlap")

    conv = K.layers.Conv2D(3, 1, strides=2)
    record_single(conv, (1, 2, 5, 5), "conv_sb_1x1_kernel")
    record_single(conv, (3, 2, 5, 5), "conv_mb_1x1_kernel")

    attention = K.layers.Attention()
    record_single(attention, [(1, 2, 2), (1, 2, 2)],
        "attention_golden_shared_kv", {"training": False})

inspect_file("conv_sb_no_overlap.nnlayergolden")

