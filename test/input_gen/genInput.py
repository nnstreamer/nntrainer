#!/usr/bin/env python3
##
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
#
# SPDX-License-Identifier: Apache-2.0-only
#
# @file genTestInput.py
# @brief Generate test input
# @author Jijoong Moon <jijoong.moon@samsung.com>

import sys
import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import struct

##
# @brief save data into file with filename
# @param[in] data The data to be saved
def save(filename,data):
    if os.path.isfile(filename):
        os.remove(filename)

    with open(filename, 'ab') as outfile:
        np.array(data, dtype=np.float32).tofile(outfile)
    print(data.shape, " data is generated")

##
# @brief generate random data and save
# @param[in] outfile_name outfile_name
# @param[in] batch batch size
# @param[in] channel channel size
# @param[in] height height size
# @param[in] width width size
# @return data generted data

def gen_input(outfile_name, batch, channel, height, width):
    if os.path.isfile(outfile_name):
        os.remove(outfile_name)

    x=np.random.rand(batch, channel, height, width)
    save(outfile_name, x)
    return x

##
# @brief conv2d layer forwarding with tensorflow
# @param[in] x input data
# @param[in] kerenl weight data
# @param[in] batch batch size of x
# @param[in] channel channel size of x
# @param[in] height height size of x
# @param[in] width width size of x
# @param[in] k_width width size of kernel
# @param[in] k_height height size of kernel
# @param[in] k_num batch size of kernel
# @param[in] stride stride size
# @param[in] pad padding : SAME VALID FULL
# @param[in] bias bias data
# @return tf_o calculated result
def conv2d_tensorflow(x, kernel, batch, width, height, channel, k_width, k_height, k_num, stride, pad, bias):
    x_trans = np.transpose(x,[0,3,2,1])
    kernel = np.transpose(kernel, [3,2,1,0])
    tf_x = tf.constant(x_trans, dtype=dtypes.float32)

    scope = "conv_in_numpy"
    act = tf.nn.sigmoid

    with tf.Session() as sess:
        with tf.variable_scope(scope):
            nin = tf_x.get_shape()[3].value
            tf_w = tf.get_variable("w", [k_width, k_height, nin, k_num], initializer=tf.constant_initializer(kernel))
            tf_b = tf.get_variable(
                "b", [k_num],
                initializer=tf.constant_initializer(bias, dtype=dtypes.float32))
            tf_z = tf.nn.conv2d(
                tf_x, kernel, strides=[1, stride, stride, 1], padding=pad) + bias
            tf_h = act(tf_z)
            tf_p = tf.nn.max_pool(tf_h, ksize = [1,2,2,1], strides=[1,1,1,1], padding='VALID');
            sess.run(tf.global_variables_initializer())
            tf_c = sess.run(tf_h)
            tf_o = sess.run(tf_p)
    return tf_c, tf_o

if __name__ == "__main__":
    target = int(sys.argv[1])

# Input File Generation with given info
    if target == 1:
        if len(sys.argv) != 7 :
            print('wrong argument : 1 filename, batch, channel, height, width')
            exit()
        gen_input(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]))

# Convolution Test Case : Generate input & kernel & golden data
#  : Input Dimension (1, 3, 7, 7)
#  : Kernel (2, 3, 3, 3)
#  : output (1,2,5,5)
#  : stride 1, 1
#  : padding 0, 0 (VALID)
    if target == 2:
        i_b = 1
        i_c = 3
        i_h = 7
        i_w = 7
        k_c = 2
        k_h = 3
        k_w = 3
        padding = 'VALID'
        stride = 1
        x=gen_input("conv2dLayer.in", i_b, i_c, i_h, i_w)
        kernel=gen_input("conv2dKernel.in", k_c, i_c, k_h, k_w)
        bias=[0.0,0.0]
        with open("conv2dKernel.in", 'ab') as outfile:
            np.array(bias, dtype=np.float32).tofile(outfile)
        golden_conv, golden_pool=conv2d_tensorflow(x, kernel, i_b, i_h, i_w, i_c, k_h, k_w, k_c, stride, padding, bias)
        save("goldenConv2DResult.out", np.transpose(golden_conv,(0,3,2,1)))
        save("goldenPooling2DResult.out", np.transpose(golden_pool,(0,3,2,1)))
