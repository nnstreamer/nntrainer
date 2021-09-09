#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file getLayerTests.py
# @date 13 Se 2020
# @brief Generate *.nnlayergolden file
# *.nnlayergolden file is expected to contain following information **in order**
# ## TBA ##
#
#
# @author Jihoon lee <jhoon.it.lee@samsung.com>

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

from transLayer import attach_trans_layer as TL


##
# @brief inpsect if file is created correctly
# @note this just checks if offset is corretly set, The result have to inspected
# manually
def inspect_file(file_name):
    import struct
    with open(file_name, "rb") as f:
        while True:
            sz = int.from_bytes(f.read(4), byteorder='little')
            if not sz:
                break
            print("size: ", sz)
            print(np.fromfile(f, dtype='float32', count=sz))

if __name__ == "__main__":
    fc = K.layers.Dense(5)
    record_single(fc, (3, 1, 1, 10), "fc_golden_plain.nnlayergolden")
    fc = K.layers.Dense(4)
    record_single(fc, (1, 1, 1, 10), "fc_golden_single_batch.nnlayergolden")

# inspect_file("fc_golden.nnlayergolden")



