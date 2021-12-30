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
    import tensorflow.keras as K

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
    record_single(conv, (1, 1, 4, 4), "conv2d_sb_minimum")
    record_single(conv, (3, 1, 4, 4), "conv2d_mb_minimum")

    conv = K.layers.Conv2D(2, 3, padding="same")
    record_single(conv, (1, 1, 4, 4), "conv2d_sb_same_remain")
    record_single(conv, (3, 1, 4, 4), "conv2d_mb_same_remain", input_type='float')

    conv = K.layers.Conv2D(2, 3, strides=2, padding="same")
    record_single(conv, (1, 3, 4, 4), "conv2d_sb_same_uneven_remain")
    record_single(conv, (3, 3, 4, 4), "conv2d_mb_same_uneven_remain")

    conv = K.layers.Conv2D(2, 3, strides=2, padding="valid")
    record_single(conv, (1, 3, 7, 7), "conv2d_sb_valid_drop_last")
    record_single(conv, (3, 3, 7, 7), "conv2d_mb_valid_drop_last")

    conv = K.layers.Conv2D(3, 2, strides=3)
    record_single(conv, (1, 2, 5, 5), "conv2d_sb_no_overlap")
    record_single(conv, (3, 2, 5, 5), "conv2d_mb_no_overlap")

    conv = K.layers.Conv2D(3, 1, strides=2)
    record_single(conv, (1, 2, 5, 5), "conv2d_sb_1x1_kernel")
    record_single(conv, (3, 2, 5, 5), "conv2d_mb_1x1_kernel")

    # use float data to generate input here
    attention = K.layers.Attention()
    record_single(attention, [(1, 5, 7), (1, 3, 7)],
                 "attention_shared_kv", {}, input_type='float')
    attention = K.layers.Attention()
    record_single(attention, [(2, 5, 7), (2, 3, 7)],
                 "attention_shared_kv_batched", {}, input_type='float')
    attention = K.layers.Attention()
    record_single(attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "attention_batched", {}, input_type='float')

    rnn = K.layers.SimpleRNN(units=5,
                         activation="tanh",
                         return_sequences=False,
                         return_state=False)
    record_single(rnn, (3, 1, 7), "rnn_single_step")

    lstm = K.layers.LSTM(units=5,
                         recurrent_activation="sigmoid",
                         activation="tanh",
                         return_sequences=False,
                         return_state=False)
    record_single(lstm, (3, 1, 7), "lstm_single_step")
    record_single(lstm, (3, 4, 7), "lstm_multi_step")

    lstm = K.layers.LSTM(units=5,
                         recurrent_activation="sigmoid",
                         activation="tanh",
                         return_sequences=True,
                         return_state=False)
    record_single(lstm, (3, 1, 7), "lstm_single_step_seq")
    record_single(lstm, (3, 4, 7), "lstm_multi_step_seq", input_type='float')

    lstm = K.layers.LSTM(units=5,
                         recurrent_activation="tanh",
                         activation="sigmoid",
                         return_sequences=True,
                         return_state=False)
    record_single(lstm, (3, 4, 7), "lstm_multi_step_seq_act")

    unit, batch_size, unroll_for, feature_size, state_num = [5, 3, 1, 7, 2]
    lstmcell = K.layers.LSTMCell(units=unit,
                         activation="tanh",
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform')
    record_single(lstmcell, [(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num)], "lstmcell_single_step", input_type='float')

    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=False,
                         return_state=False,
                         reset_after=False)
    record_single(gru, (3, 1, 7), "gru_single_step")
    record_single(gru, (3, 4, 7), "gru_multi_step")

    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=False)
    record_single(gru, (3, 1, 7), "gru_single_step_seq")
    record_single(gru, (3, 4, 7), "gru_multi_step_seq", input_type='float')

    gru = K.layers.GRU(units=5, activation="sigmoid", 
                         recurrent_activation="tanh",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=False,)
    record_single(gru, (3, 4, 7), "gru_multi_step_seq_act", input_type='float')

    # check reset_after
    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=False,
                         return_state=False,
                         reset_after=True,)
    record_single(gru, (3, 1, 7), "gru_reset_after_single_step")
    record_single(gru, (3, 4, 7), "gru_reset_after_multi_step")

    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=True)
    record_single(gru, (3, 1, 7), "gru_reset_after_single_step_seq")
    record_single(gru, (3, 4, 7), "gru_reset_after_multi_step_seq", input_type='float')

    gru = K.layers.GRU(units=5, activation="sigmoid", 
                         recurrent_activation="tanh",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=True)
    record_single(gru, (3, 4, 7), "gru_reset_after_multi_step_seq_act", input_type='float')

    unit, batch_size, unroll_for, feature_size = [5, 3, 1, 7]
    grucell = K.layers.GRUCell(units=unit,
                         recurrent_activation='sigmoid',
                         bias_initializer='glorot_uniform')
    record_single(grucell, [(batch_size, feature_size), (batch_size, unit)], "grucell_single_step", input_type='float')

    unit, batch_size, unroll_for, feature_size = [5, 3, 1, 7]
    grucell = K.layers.GRUCell(units=unit,
                         recurrent_activation='sigmoid',
                         bias_initializer='glorot_uniform',
                         reset_after=True)
    record_single(grucell, [(batch_size, feature_size), (batch_size, unit)], "grucell_reset_after_single_step", input_type='float')

    unit, batch_size, unroll_for, feature_size = [5, 3, 1, 7]
    grucell = K.layers.GRUCell(units=unit,
                         activation="sigmoid",
                         recurrent_activation="tanh",
                         bias_initializer='glorot_uniform')
    record_single(grucell, [(batch_size, feature_size), (batch_size, unit)], "grucell_single_step_act", input_type='float')

    dropout = K.layers.Dropout(rate=0.2)
    record_single(dropout, (2, 3, 2, 3), "dropout_20_training", {"training": True})
    record_single(dropout, (2, 3, 2, 3), "dropout_20_inference", {"training": False})

    dropout = K.layers.Dropout(rate=0.0)
    record_single(dropout, (2, 3, 2, 3), "dropout_0_training", {"training": True})

    dropout = K.layers.Dropout(rate=0.9999)
    record_single(dropout, (2, 3, 2, 3), "dropout_100_training", {"training": True})

    conv = K.layers.Conv1D(3, 2)
    record_single(conv, (1, 1, 1, 4), "conv1d_sb_minimum")
    record_single(conv, (3, 1, 1, 4), "conv1d_mb_minimum")

    conv = K.layers.Conv1D(2, 3, padding="same")
    record_single(conv, (1, 1, 1, 4), "conv1d_sb_same_remain")
    record_single(conv, (3, 1, 1, 4), "conv1d_mb_same_remain")

    conv = K.layers.Conv1D(2, 3, strides=2, padding="same")
    record_single(conv, (1, 3, 1, 4), "conv1d_sb_same_uneven_remain")
    record_single(conv, (3, 3, 1, 4), "conv1d_mb_same_uneven_remain")

    conv = K.layers.Conv1D(2, 3, strides=2, padding="valid")
    record_single(conv, (1, 3, 1, 7), "conv1d_sb_valid_drop_last")
    record_single(conv, (3, 3, 1, 7), "conv1d_mb_valid_drop_last")

    conv = K.layers.Conv1D(3, 2, strides=3)
    record_single(conv, (1, 2, 1, 5), "conv1d_sb_no_overlap")
    record_single(conv, (3, 2, 1, 5), "conv1d_mb_no_overlap")

    conv = K.layers.Conv1D(3, 1, strides=2)
    record_single(conv, (1, 2, 1, 5), "conv1d_sb_1x1_kernel")
    record_single(conv, (3, 2, 1, 5), "conv1d_mb_1x1_kernel")

    concat = K.layers.Concatenate(axis=3)
    record_single(concat, [(2,3,3,2), (2, 3, 3, 3)], "concat_dim3")

    concat = K.layers.Concatenate(axis=2)
    record_single(concat, [(2,3,2,3), (2, 3, 3, 3)], "concat_dim2")

    concat = K.layers.Concatenate(axis=1)
    record_single(concat, [(2,2,3,3), (2, 3, 3, 3)], "concat_dim1")

inspect_file("dropout_20_training.nnlayergolden")

