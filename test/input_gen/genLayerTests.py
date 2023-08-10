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

from recorder import record_single, record_single_fp16

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras as K

##
# @brief inpsect if file is created correctly
# @note this just checks if offset is corretly set, The result have to inspected
# manually
def inspect_file(file_name, _dtype = "float32"):
    with open(file_name, "rb") as f:
        while True:
            if (_dtype == "float32"):
                sz = int.from_bytes(f.read(4), byteorder="little")
            elif (_dtype =="float16"):
                sz = int.from_bytes(f.read(2), byteorder="little")
            if not sz:
                break
            print("size: ", sz)
            print(np.fromfile(f, dtype=_dtype, count=sz))

class PositionalEncoding(tf.keras.layers.Layer):
    def __init__(self, position, d_model):
        super(PositionalEncoding, self).__init__()
        self.position = position
        self.d_model = d_model

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        return pos * angle_rates

    def build(self, input_shape):
        angle_rads = self.get_angles(np.arange(self.position)[:, np.newaxis],
                        np.arange(self.d_model)[np.newaxis, :], self.d_model)

        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        self.pos_encoding = angle_rads[np.newaxis, ...]

        self.pos_encoding = tf.cast(self.pos_encoding, dtype=tf.float32)

    def call(self, inputs):
        inputs += self.pos_encoding[:, :tf.shape(inputs[0])[-2], :]
        return inputs

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
   
    ln = K.layers.LayerNormalization()
    record_single(ln, (2, 4, 2, 3), "ln_axis_1")
    ln = K.layers.LayerNormalization([1])
    record_single(ln, (2, 4, 2, 3), "ln_axis_2")
    ln = K.layers.LayerNormalization([2])
    record_single(ln, (2, 4, 2, 3), "ln_axis_3")
    ln = K.layers.LayerNormalization([1, 3])
    record_single(ln, (2, 4, 2, 3), "ln_axis_1_2")
    ln = K.layers.LayerNormalization([1, 2])
    record_single(ln, (2, 4, 2, 3), "ln_axis_2_3")
    ln = K.layers.LayerNormalization([2, 3])
    record_single(ln, (2, 4, 2, 3), "ln_axis_1_3")
    ln = K.layers.LayerNormalization([1, 2, 3])
    record_single(ln, (2, 4, 2, 3), "ln_axis_1_2_3")

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

    conv = K.layers.Conv2D(2, 3, dilation_rate=(2, 2))
    record_single(conv, (1, 3, 11, 11), "conv2d_sb_dilation")
    record_single(conv, (3, 3, 11, 11), "conv2d_mb_dilation")

    conv = K.layers.Conv2D(2, 3, padding="same", dilation_rate=(2, 2))
    record_single(conv, (1, 3, 11, 11), "conv2d_sb_same_dilation")
    record_single(conv, (3, 3, 11, 11), "conv2d_mb_same_dilation")

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

    # use float data to generate input here
    multi_head_attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=3)
    record_single(multi_head_attention, [(1, 5, 7), (1, 3, 7), (1, 3, 7)],
                 "multi_head_attention_single_batch", {}, input_type='float')
    record_single(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention", {}, input_type='float')
    record_single(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention_return_attention_scores", {"return_attention_scores":True}, input_type='float')
    multi_head_attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=3, value_dim=5)
    record_single(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention_value_dim", {}, input_type='float')
    multi_head_attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=3, output_shape=5)
    record_single(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention_output_shape", {}, input_type='float')

    rnn = K.layers.SimpleRNN(units=5,
                         activation="tanh",
                         return_sequences=False,
                         return_state=False)
    record_single(rnn, (3, 1, 7), "rnn_single_step")

    unit, batch_size, unroll_for, feature_size= [1, 1, 1, 1]
    rnncell = K.layers.SimpleRNNCell(units=unit,
                         bias_initializer='glorot_uniform')
    record_single(rnncell, [(batch_size, feature_size)] + [(batch_size, unit)], "rnncell_single_step", input_type='float')

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

    conv = K.layers.Conv1D(2, 3, dilation_rate=2)
    record_single(conv, (1, 3, 1, 11), "conv1d_sb_dilation")
    record_single(conv, (3, 3, 1, 11), "conv1d_mb_dilation")

    conv = K.layers.Conv1D(2, 3, padding="same", dilation_rate=2)
    record_single(conv, (1, 3, 1, 11), "conv1d_sb_same_dilation")
    record_single(conv, (3, 3, 1, 11), "conv1d_mb_same_dilation")

    conv = K.layers.Conv1D(3, 2, padding="causal")
    record_single(conv, (1, 1, 1, 4), "conv1d_sb_causal")
    record_single(conv, (3, 1, 1, 4), "conv1d_mb_causal")

    conv = K.layers.Conv1D(3, 2, padding="causal", dilation_rate=2)
    record_single(conv, (1, 1, 1, 4), "conv1d_sb_causal_dilation")
    record_single(conv, (3, 1, 1, 4), "conv1d_mb_causal_dilation")

    concat = K.layers.Concatenate(axis=3)
    record_single(concat, [(2,3,3,2), (2, 3, 3, 3)], "concat_dim3")

    concat = K.layers.Concatenate(axis=2)
    record_single(concat, [(2,3,2,3), (2, 3, 3, 3)], "concat_dim2")

    concat = K.layers.Concatenate(axis=1)
    record_single(concat, [(2,2,3,3), (2, 3, 3, 3)], "concat_dim1")

    positional_encoding = PositionalEncoding(10, 6)
    record_single(positional_encoding, [(3, 1, 7, 6)], "positional_encoding_partial")
    record_single(positional_encoding, [(3, 1, 10, 6)], "positional_encoding")


    # fp16fp16

    fc1616 = K.layers.Dense(5)
    record_single_fp16(fc1616, (3, 1, 1, 10), "fc_plain_fp16fp16",input_type='float16')
    fc1616 = K.layers.Dense(4)
    record_single_fp16(fc1616, (1, 1, 1, 10), "fc_single_batch_fp16fp16",input_type='float16')
 
    bn = K.layers.BatchNormalization()
    record_single_fp16(bn, (2, 4, 2, 3), "bn_channels_training_fp16fp16", {"training": True},input_type='float16')
    record_single_fp16(bn, (2, 4, 2, 3), "bn_channels_inference_fp16fp16", {"training": False},input_type='float16')
    bn = K.layers.BatchNormalization()
    record_single_fp16(bn, (2, 10), "bn_width_training_fp16fp16", {"training": True},input_type='float16')
    record_single_fp16(bn, (2, 10), "bn_width_inference_fp16fp16", {"training": False},input_type='float16')

    ln = K.layers.LayerNormalization()
    record_single_fp16(ln, (2, 4, 2, 3), "ln_axis_1_fp16fp16",input_type='float16')
    ln = K.layers.LayerNormalization([1])
    record_single_fp16(ln, (2, 4, 2, 3), "ln_axis_2_fp16fp16",input_type='float16')
    ln = K.layers.LayerNormalization([2])
    record_single_fp16(ln, (2, 4, 2, 3), "ln_axis_3_fp16fp16",input_type='float16')
    ln = K.layers.LayerNormalization([1, 3])
    record_single_fp16(ln, (2, 4, 2, 3), "ln_axis_1_2_fp16fp16",input_type='float16')
    ln = K.layers.LayerNormalization([1, 2])
    record_single_fp16(ln, (2, 4, 2, 3), "ln_axis_2_3_fp16fp16",input_type='float16')
    ln = K.layers.LayerNormalization([2, 3])
    record_single_fp16(ln, (2, 4, 2, 3), "ln_axis_1_3_fp16fp16",input_type='float16')
    ln = K.layers.LayerNormalization([1, 2, 3])
    record_single_fp16(ln, (2, 4, 2, 3), "ln_axis_1_2_3_fp16fp16",input_type='float16')

    conv = K.layers.Conv2D(3, 2)
    record_single_fp16(conv, (1, 1, 4, 4), "conv2d_sb_minimum_fp16fp16",input_type='float16')
    record_single_fp16(conv, (3, 1, 4, 4), "conv2d_mb_minimum_fp16fp16",input_type='float16')

    conv = K.layers.Conv2D(2, 3, padding="same")
    record_single_fp16(conv, (1, 1, 4, 4), "conv2d_sb_same_remain_fp16fp16",input_type='float16')
    record_single_fp16(conv, (3, 1, 4, 4), "conv2d_mb_same_remain_fp16fp16",input_type='float16')


    conv = K.layers.Conv2D(2, 3, strides=2, padding="same")
    record_single_fp16(conv, (1, 3, 4, 4), "conv2d_sb_same_uneven_remain_fp16fp16")
    record_single_fp16(conv, (3, 3, 4, 4), "conv2d_mb_same_uneven_remain_fp16fp16")

    conv = K.layers.Conv2D(2, 3, strides=2, padding="valid")
    record_single_fp16(conv, (1, 3, 7, 7), "conv2d_sb_valid_drop_last_fp16fp16")
    record_single_fp16(conv, (3, 3, 7, 7), "conv2d_mb_valid_drop_last_fp16fp16")

    conv = K.layers.Conv2D(3, 2, strides=3)
    record_single_fp16(conv, (1, 2, 5, 5), "conv2d_sb_no_overlap_fp16fp16")
    record_single_fp16(conv, (3, 2, 5, 5), "conv2d_mb_no_overlap_fp16fp16")

    conv = K.layers.Conv2D(3, 1, strides=2)
    record_single_fp16(conv, (1, 2, 5, 5), "conv2d_sb_1x1_kernel_fp16fp16")
    record_single_fp16(conv, (3, 2, 5, 5), "conv2d_mb_1x1_kernel_fp16fp16")

    conv = K.layers.Conv2D(2, 3, dilation_rate=(2, 2))
    record_single_fp16(conv, (1, 3, 11, 11), "conv2d_sb_dilation_fp16fp16")
    record_single_fp16(conv, (3, 3, 11, 11), "conv2d_mb_dilation_fp16fp16")

    conv = K.layers.Conv2D(2, 3, padding="same", dilation_rate=(2, 2))
    record_single_fp16(conv, (1, 3, 11, 11), "conv2d_sb_same_dilation_fp16fp16")
    record_single_fp16(conv, (3, 3, 11, 11), "conv2d_mb_same_dilation_fp16fp16")

    # use float data to generate input here
    attention = K.layers.Attention()
    record_single_fp16(attention, [(1, 5, 7), (1, 3, 7)],
                 "attention_shared_kv_fp16fp16", {}, input_type='float16')
    attention = K.layers.Attention()
    record_single_fp16(attention, [(2, 5, 7), (2, 3, 7)],
                 "attention_shared_kv_batched_fp16fp16", {}, input_type='float16')
    attention = K.layers.Attention()
    record_single_fp16(attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "attention_batched_fp16fp16", {}, input_type='float16')

    # use float data to generate input here
    multi_head_attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=3)
    record_single_fp16(multi_head_attention, [(1, 5, 7), (1, 3, 7), (1, 3, 7)],
                 "multi_head_attention_single_batch_fp16fp16", {}, input_type='float16')
    record_single_fp16(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention_fp16fp16", {}, input_type='float')
    record_single_fp16(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention_return_attention_scores_fp16fp16", {"return_attention_scores":True}, input_type='float16')
    multi_head_attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=3, value_dim=5)
    record_single_fp16(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention_value_dim_fp16fp16", {}, input_type='float16')
    multi_head_attention = K.layers.MultiHeadAttention(num_heads=2, key_dim=3, output_shape=5)
    record_single_fp16(multi_head_attention, [(2, 5, 7), (2, 3, 7), (2, 3, 7)],
                 "multi_head_attention_output_shape_fp16fp16", {}, input_type='float16')

    rnn = K.layers.SimpleRNN(units=5,
                         activation="tanh",
                         return_sequences=False,
                         return_state=False)
    record_single_fp16(rnn, (3, 1, 7), "rnn_single_step_fp16fp16")

    unit, batch_size, unroll_for, feature_size= [1, 1, 1, 1]
    rnncell = K.layers.SimpleRNNCell(units=unit,
                         bias_initializer='glorot_uniform')
    record_single_fp16(rnncell, [(batch_size, feature_size)] + [(batch_size, unit)], "rnncell_single_step_fp16fp16", input_type='float')

    lstm = K.layers.LSTM(units=5,
                         recurrent_activation="sigmoid",
                         activation="tanh",
                         return_sequences=False,
                         return_state=False)
    record_single_fp16(lstm, (3, 1, 7), "lstm_single_step_fp16fp16")
    record_single_fp16(lstm, (3, 4, 7), "lstm_multi_step_fp16fp16")

    lstm = K.layers.LSTM(units=5,
                         recurrent_activation="sigmoid",
                         activation="tanh",
                         return_sequences=True,
                         return_state=False)
    record_single_fp16(lstm, (3, 1, 7), "lstm_single_step_seq_fp16fp16")
    record_single_fp16(lstm, (3, 4, 7), "lstm_multi_step_seq_fp16fp16", input_type='float')

    lstm = K.layers.LSTM(units=5,
                         recurrent_activation="tanh",
                         activation="sigmoid",
                         return_sequences=True,
                         return_state=False)
    record_single_fp16(lstm, (3, 4, 7), "lstm_multi_step_seq_act_fp16fp16")

    unit, batch_size, unroll_for, feature_size, state_num = [5, 3, 1, 7, 2]
    lstmcell = K.layers.LSTMCell(units=unit,
                         activation="tanh",
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform')
    record_single_fp16(lstmcell, [(batch_size, feature_size)] + [(batch_size, unit) for _ in range(state_num)], "lstmcell_single_step_fp16fp16", input_type='float')

    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=False,
                         return_state=False,
                         reset_after=False)
    record_single_fp16(gru, (3, 1, 7), "gru_single_step_fp16fp16")
    record_single_fp16(gru, (3, 4, 7), "gru_multi_step_fp16fp16")

    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=False)
    record_single_fp16(gru, (3, 1, 7), "gru_single_step_seq_fp16fp16")
    record_single_fp16(gru, (3, 4, 7), "gru_multi_step_seq_fp16fp16", input_type='float16')

    gru = K.layers.GRU(units=5, activation="sigmoid", 
                         recurrent_activation="tanh",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=False,)
    record_single_fp16(gru, (3, 4, 7), "gru_multi_step_seq_act_fp16fp16", input_type='float16')

    # check reset_after
    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=False,
                         return_state=False,
                         reset_after=True,)
    record_single_fp16(gru, (3, 1, 7), "gru_reset_after_single_step_fp16fp16")
    record_single_fp16(gru, (3, 4, 7), "gru_reset_after_multi_step_fp16fp16")

    gru = K.layers.GRU(units=5, activation="tanh", 
                         recurrent_activation="sigmoid",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=True)
    record_single_fp16(gru, (3, 1, 7), "gru_reset_after_single_step_seq_fp16fp16")
    record_single_fp16(gru, (3, 4, 7), "gru_reset_after_multi_step_seq_fp16fp16", input_type='float16')

    gru = K.layers.GRU(units=5, activation="sigmoid", 
                         recurrent_activation="tanh",
                         bias_initializer='glorot_uniform',
                         return_sequences=True,
                         return_state=False,
                         reset_after=True)
    record_single_fp16(gru, (3, 4, 7), "gru_reset_after_multi_step_seq_act_fp16fp16", input_type='float16')

    unit, batch_size, unroll_for, feature_size = [5, 3, 1, 7]
    grucell = K.layers.GRUCell(units=unit,
                         recurrent_activation='sigmoid',
                         bias_initializer='glorot_uniform')
    record_single_fp16(grucell, [(batch_size, feature_size), (batch_size, unit)], "grucell_single_step_fp16fp16", input_type='float16')

    unit, batch_size, unroll_for, feature_size = [5, 3, 1, 7]
    grucell = K.layers.GRUCell(units=unit,
                         recurrent_activation='sigmoid',
                         bias_initializer='glorot_uniform',
                         reset_after=True)
    record_single_fp16(grucell, [(batch_size, feature_size), (batch_size, unit)], "grucell_reset_after_single_step_fp16fp16", input_type='float16')

    unit, batch_size, unroll_for, feature_size = [5, 3, 1, 7]
    grucell = K.layers.GRUCell(units=unit,
                         activation="sigmoid",
                         recurrent_activation="tanh",
                         bias_initializer='glorot_uniform')
    record_single_fp16(grucell, [(batch_size, feature_size), (batch_size, unit)], "grucell_single_step_act_fp16fp16", input_type='float16')

    dropout = K.layers.Dropout(rate=0.2)
    record_single_fp16(dropout, (2, 3, 2, 3), "dropout_20_training_fp16fp16", {"training": True})
    record_single_fp16(dropout, (2, 3, 2, 3), "dropout_20_inference_fp16fp16", {"training": False})

    dropout = K.layers.Dropout(rate=0.0)
    record_single_fp16(dropout, (2, 3, 2, 3), "dropout_0_training_fp16fp16", {"training": True})

    dropout = K.layers.Dropout(rate=0.9999)
    record_single_fp16(dropout, (2, 3, 2, 3), "dropout_100_training_fp16fp16", {"training": True})

    conv = K.layers.Conv1D(3, 2)
    record_single_fp16(conv, (1, 1, 1, 4), "conv1d_sb_minimum_fp16fp16")
    record_single_fp16(conv, (3, 1, 1, 4), "conv1d_mb_minimum_fp16fp16")

    conv = K.layers.Conv1D(2, 3, padding="same")
    record_single_fp16(conv, (1, 1, 1, 4), "conv1d_sb_same_remain_fp16fp16")
    record_single_fp16(conv, (3, 1, 1, 4), "conv1d_mb_same_remain_fp16fp16")

    conv = K.layers.Conv1D(2, 3, strides=2, padding="same")
    record_single_fp16(conv, (1, 3, 1, 4), "conv1d_sb_same_uneven_remain_fp16fp16")
    record_single_fp16(conv, (3, 3, 1, 4), "conv1d_mb_same_uneven_remain_fp16fp16")

    conv = K.layers.Conv1D(2, 3, strides=2, padding="valid")
    record_single_fp16(conv, (1, 3, 1, 7), "conv1d_sb_valid_drop_last_fp16fp16")
    record_single_fp16(conv, (3, 3, 1, 7), "conv1d_mb_valid_drop_last_fp16fp16")

    conv = K.layers.Conv1D(3, 2, strides=3)
    record_single_fp16(conv, (1, 2, 1, 5), "conv1d_sb_no_overlap_fp16fp16")
    record_single_fp16(conv, (3, 2, 1, 5), "conv1d_mb_no_overlap_fp16fp16")

    conv = K.layers.Conv1D(3, 1, strides=2)
    record_single_fp16(conv, (1, 2, 1, 5), "conv1d_sb_1x1_kernel_fp16fp16")
    record_single_fp16(conv, (3, 2, 1, 5), "conv1d_mb_1x1_kernel_fp16fp16")

    conv = K.layers.Conv1D(2, 3, dilation_rate=2)
    record_single_fp16(conv, (1, 3, 1, 11), "conv1d_sb_dilation_fp16fp16")
    record_single_fp16(conv, (3, 3, 1, 11), "conv1d_mb_dilation_fp16fp16")

    conv = K.layers.Conv1D(2, 3, padding="same", dilation_rate=2)
    record_single_fp16(conv, (1, 3, 1, 11), "conv1d_sb_same_dilation_fp16fp16")
    record_single_fp16(conv, (3, 3, 1, 11), "conv1d_mb_same_dilation_fp16fp16")

    conv = K.layers.Conv1D(3, 2, padding="causal")
    record_single_fp16(conv, (1, 1, 1, 4), "conv1d_sb_causal_fp16fp16")
    record_single_fp16(conv, (3, 1, 1, 4), "conv1d_mb_causal_fp16fp16")

    conv = K.layers.Conv1D(3, 2, padding="causal", dilation_rate=2)
    record_single_fp16(conv, (1, 1, 1, 4), "conv1d_sb_causal_dilation_fp16fp16")
    record_single_fp16(conv, (3, 1, 1, 4), "conv1d_mb_causal_dilation_fp16fp16")

    concat = K.layers.Concatenate(axis=3)
    record_single_fp16(concat, [(2,3,3,2), (2, 3, 3, 3)], "concat_dim3_fp16fp16")

    concat = K.layers.Concatenate(axis=2)
    record_single_fp16(concat, [(2,3,2,3), (2, 3, 3, 3)], "concat_dim2_fp16fp16")

    concat = K.layers.Concatenate(axis=1)
    record_single_fp16(concat, [(2,2,3,3), (2, 3, 3, 3)], "concat_dim1_fp16fp16")

    positional_encoding = PositionalEncoding(10, 6)
    record_single_fp16(positional_encoding, [(3, 1, 7, 6)], "positional_encoding_partial_fp16fp16")
    record_single_fp16(positional_encoding, [(3, 1, 10, 6)], "positional_encoding_fp16fp16")

inspect_file("dropout_20_training.nnlayergolden")
inspect_file("fc_plain_fp16fp16.nnlayergolden", _dtype = "float16")
