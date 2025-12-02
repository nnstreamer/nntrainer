"""
SPDX-License-Identifier: Apache-2.0

Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>

@file   dataset.py
@date   15 July 2020
@brief  This is for mnist input generation
@see    https://github.com/nnstreamer/nntrainer
@author Jijoong Moon <jijoong.moon@samsung.com>
@bug    No known bugs except for NYI items
"""


import struct
import numpy as np

TOTAL_TRAIN_DATA_SIZE = 100
TOTAL_LABEL_SIZE = 100
TOTAL_VAL_DATA_SIZE = 20
FEATURE_SIZE = 3072


def get_data_info():
    """Return Data size

    Returns:
        t_data_size, v_data_size, TOTAL_LABEL_SIZE, FEATURE_SIZE
    """
    t_data_size = TOTAL_TRAIN_DATA_SIZE
    v_data_size = TOTAL_VAL_DATA_SIZE
    return t_data_size, v_data_size, TOTAL_LABEL_SIZE, FEATURE_SIZE

##
# @brief load input data from file
# @return (InputVector, InputLabel, Validation Vector, ValidationLabel)


def load_data(target):
    """Load data && save as file

    Args:
        target (str): train || validation

    Returns:
        input_vector, input_label, val_vector, val_label
    """
    # data_size = TOTAL_TRAIN_DATA_SIZE;
    d_size = get_data_info()

    if target == "validation":
        t_buf_size = d_size[0]
        v_buf_size = d_size[1]
    else:
        t_buf_size = d_size[0]*TOTAL_LABEL_SIZE
        v_buf_size = d_size[1]*TOTAL_LABEL_SIZE

    input_vector = np.zeros((t_buf_size, FEATURE_SIZE), dtype=np.float32)
    input_label = np.zeros((t_buf_size, TOTAL_LABEL_SIZE), dtype=np.float32)

    val_vector = np.zeros((v_buf_size, FEATURE_SIZE), dtype=np.float32)
    val_label = np.zeros((v_buf_size, TOTAL_LABEL_SIZE), dtype=np.float32)

    # read Input & Label

    with open('vgg_valSet.dat', 'rb') as fin:
        for i in range(v_buf_size):
            for j in range(FEATURE_SIZE):
                data_str = fin.read(4)
                val_vector[i, j] = struct.unpack('f', data_str)[0]
            for j in range(TOTAL_LABEL_SIZE):
                data_str = fin.read(4)
                val_label[i, j] = struct.unpack('f', data_str)[0]
        fin.close()

    # we are using same training data for validation to check how internal implementation is working
    with open('vgg_trainingSet.dat', 'rb') as fin:
        for i in range(t_buf_size):
            for j in range(FEATURE_SIZE):
                data_str = fin.read(4)
                input_vector[i, j] = struct.unpack('f', data_str)[0]
            for j in range(TOTAL_LABEL_SIZE):
                data_str = fin.read(4)
                input_label[i, j] = struct.unpack('f', data_str)[0]
        fin.close()

    return input_vector, input_label, val_vector, val_label
