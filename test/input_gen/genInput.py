#!/usr/bin/env python3
##
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
# Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
#
# SPDX-License-Identifier: Apache-2.0-only
#
# @file genTestInput.py
# @brief Generate test input
# @author Jijoong Moon <jijoong.moon@samsung.com>
# @author Parichay Kapoor <pk.kapoor@samsung.com>

import sys
import os

os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import numpy as np
import tensorflow as tf
from tensorflow.python.framework import dtypes
import struct
import random

DEBUG = True

# Fix the seeds across frameworks
SEED = 1234
tf.compat.v1.reset_default_graph()
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)

##
# @brief save data into file with filename
# @param[in] data The data to be saved
def save(filename, data):
    if os.path.isfile(filename):
        os.remove(filename)

    with open(filename, 'ab') as outfile:
        np.array(data, dtype=np.float32).tofile(outfile)
    print(data.shape, " data is generated")

##
# @brief generate random tensor
def gen_tensor(shape, dtype=None):
  return np.random.random_sample(input_shape)

##
# @brief generate random data and save
# @param[in] outfile_name outfile_name
# @param[in] batch batch size
# @param[in] channel channel size
# @param[in] height height size
# @param[in] width width size
# @return data generted data

def gen_input(outfile_name, input_shape, savefile=True):
    x=gen_tensor(input_shape)
    if savefile:
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
def conv2d_tf(x, kernel, batch, width, height, channel, k_width, k_height, k_num, stride, pad, bias):
    x_trans = np.transpose(x,[0,2,3,1])
    kernel = np.transpose(kernel, [2,3,1,0])
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
            tf_p = tf.nn.max_pool(tf_z, ksize = [1,2,2,1], strides=[1,1,1,1], padding='VALID');
            sess.run(tf.global_variables_initializer())
            tf_c = sess.run(tf_z)
            tf_o = sess.run(tf_p)
    return tf_c, tf_o

##
# Tested with tensorflow 1.x (1.14.0 and above)
# @brief fc layer forwarding with tensorflow
# @param[in] x input data
# @param[in] kernel weight data
# @param[in] bias bias data
# @param[in] activation activation after the operation
# @param[in] train train a few steps
# @return tf_o calculated result
def fc_tf(x, kernel, label, bias, activation, train=False, loss='mse', opt='sgd'):
    tf.compat.v1.reset_default_graph()
    tf_input = tf.placeholder(dtype = dtypes.float32, shape=x.shape)

    if (train and loss == 'cross'):
        stored_act = activation
        activation = None

    with tf.compat.v1.Session() as sess:
        with tf.compat.v1.variable_scope('fc'):
            model = tf.keras.Sequential([tf.keras.layers.Dense(kernel.shape[-1],
                    activation=activation,
                    use_bias=True,
                    kernel_initializer=tf.constant_initializer(kernel),
                    bias_initializer=tf.constant_initializer(bias),
                    input_shape=tf_input.shape)])
            tf_logit = model(tf_input, training=train)

            if train:
                tf_label = tf.placeholder(dtype = dtypes.float32, shape=label.shape)
                if loss == 'mse':
                    tf_loss = tf.reduce_mean(tf.keras.losses.MSE(tf_label, tf_logit))
                elif loss == 'cross':
                    if stored_act == tf.nn.sigmoid:
                        tf_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                            labels=tf_label, logits=tf_logit))
                    elif stored_act == tf.nn.softmax:
                        tf_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                            labels=tf_label, logits=tf_logit))
                    else:
                        raise 'unknown activation with cross entropy'
                else:
                    raise 'unknown loss'

                if opt == 'sgd':
                    optimizer = tf.keras.optimizers.SGD()
                elif opt == 'adam':
                    optimizer = tf.keras.optimizers.Adam()
                else:
                    raise 'unknown optimizer'

                trainable_variables = tf.compat.v1.trainable_variables()
                if DEBUG:
                    print([x.name for x in trainable_variables])

                tf_grad = optimizer.get_gradients(tf_loss, params=trainable_variables)
                train_op = optimizer.apply_gradients(list(zip(tf_grad, trainable_variables)))

                var_to_run = [tf_logit, tf_grad, train_op]
                feed_dict = {tf_input: x, tf_label: label}
            else:
                var_to_run = [tf_logit]
                feed_dict = {tf_input: x}

            sess.run(tf.compat.v1.global_variables_initializer())
            if DEBUG:
                old_w = sess.run(tf.compat.v1.trainable_variables())

            tf_outs = sess.run(var_to_run, feed_dict = feed_dict)

            if DEBUG:
                new_w = sess.run(tf.compat.v1.trainable_variables())

            if (train):
                tf_outs = tf_outs[:-1]
                tf_outs.append(sess.run(tf.compat.v1.trainable_variables()))
                # tf outs contain :
                # 1. forward output numpy array
                # 2. gradient for weights in list form
                # 3. updated weights in list form
                if DEBUG:
                    print(tf_outs[0].shape)
                    print(tf_outs[1][0].shape)
                    print(tf_outs[1][1].shape)
                    print(tf_outs[2][0].shape)
                    print(tf_outs[2][1].shape)

                if DEBUG:
                    if opt == 'sgd':
                        assert(np.isclose(new_w[1].all(), (old_w[1] - (tf_outs[1][1] * 0.1)).all()))
                    print(old_w[1])
                    print(new_w[1])
                    print(tf_outs[1][1])

    return tf_outs
  
##
# tested with tf 1.14.0
# @param[in] x input
# @param[in] trainable
# @return bn output, [updated_gamma, updated_beta], grad_result (0. dx / 1. gamma / 2. beta / 3. mean / 4. variance)
# for updated_gamma, updated_beta, x <- x - grad is used for easier calculation
def bn_tf(x, trainable=False):
    tf.compat.v1.reset_default_graph()
    tf_input = tf.compat.v1.placeholder(
        dtype=dtypes.float32, shape=x.shape, name='input')

    bnlayer = tf.keras.layers.BatchNormalization(
        axis=0,
        trainable=trainable,
        gamma_initializer=gen_tensor,
        beta_initializer=gen_tensor)(tf_input)

    bn_variables = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope='batch_normalization')
    input_variables = [tf_input] + bn_variables

    grad = tf.gradients(bnlayer, input_variables)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        bn_result = sess.run(bnlayer, feed_dict={tf_input: x})
        grad_result = sess.run(grad, feed_dict={tf_input: x})
        updated_gamma = sess.run(input_variables[1] - grad_result[1])
        updated_beta = sess.run(input_variables[1] - grad_result[2])

    if DEBUG:
        print(x[0], bn_result[0])
        print("updated_gamma: %s" % updated_gamma)
        print("updated_beta: %s" % updated_beta)
        for item, input_variable in zip(grad_result, input_variables):
            print(input_variable.name)
            print(item[0])

    return bn_result, [updated_gamma, updated_beta], grad_result

def gen_test_case_conv(i_b, i_c, i_h, i_w, k_c, k_h, k_w, padding, stride, bias, base_name):
    x=gen_input(base_name+"conv2DLayer.in", [i_b, i_c, i_h, i_w])
    kernel=gen_input(base_name+"conv2DKernel.in", [k_c, i_c, k_h, k_w])
    with open(base_name+"conv2DKernel.in", 'ab') as outfile:
        np.array(bias, dtype=np.float32).tofile(outfile)

    golden_conv, golden_pool=conv2d_tf(x, kernel, i_b, i_h, i_w, i_c, k_h, k_w, k_c, stride, padding, bias)
    save(base_name+"goldenConv2DResult.out", np.transpose(golden_conv,(0,3,2,1)))
    save(base_name+"goldenPooling2DResult.out", np.transpose(golden_pool,(0,3,2,1)))

##
# @brief generate fc test case data for forward and backward pass
def gen_test_case_fc(input_shape, kernel_shape, base_name):
    input_data = gen_input(base_name + "FCLayer.in", input_shape)
    label = gen_input(base_name + "FCLabel.in", input_shape[:-1] + [kernel_shape[-1]])

    kernel = gen_input(base_name + "FCKernel.in", kernel_shape)
    bias = gen_input(base_name + "FCKernel.in", kernel_shape[-1:], savefile=False)
    with open(base_name+"FCKernel.in", 'ab') as outfile:
        np.array(bias, dtype=np.float32).tofile(outfile)


    golden_fc = fc_tf(input_data, kernel, None, bias, activation=None)
    save(base_name + "goldenFCResultActNone.out", golden_fc[0])

    golden_fc = fc_tf(input_data, kernel, None, bias, activation=tf.nn.sigmoid)
    save(base_name + "goldenFCResultSigmoid.out", golden_fc[0])

    golden_fc = fc_tf(input_data, kernel, None, bias, activation=tf.nn.softmax)
    save(base_name + "goldenFCResultSoftmax.out", golden_fc[0])

def get_test_case_bn(input_shape, training=False):
    pass

if __name__ == "__main__":
    target = int(sys.argv[1])

# Input File Generation with given info
    if target == 1:
        if len(sys.argv) != 7 :
            print('wrong argument : 1 filename, batch, channel, height, width')
            exit()
        gen_input(sys.argv[2], [int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])])

# Convolution Test Case : Generate input & kernel & golden data
# first unit test case : 1, 3, 7, 7, 2, 3, 3, VALID, 1 test_1_
#  : Input Dimension (1, 3, 7, 7)
#  : Kernel (2, 3, 3, 3)
#  : output (1,2,5,5)
#  : stride 1, 1
#  : padding 0, 0 (VALID)
    if target == 2:
        bias1 = [0.0, 0.0]
        gen_test_case_conv(1, 3, 7, 7, 2, 3, 3, "VALID", 1, bias1, "test_1_")

# second unit test case : 2, 3, 7, 7, 3, 3, 3, VALID, 1 test_2_
#  : Input Dimension (2, 3, 7, 7)
#  : Kernel (3, 3, 3, 3)
#  : output (1,3,5,5)
#  : stride 1, 1
#  : padding 0, 0 (VALID)
    if target == 3:
        bias2 = [0.0, 0.0, 0.0]
        gen_test_case_conv(2, 3, 7, 7, 3, 3, 3, "VALID", 1, bias2, "test_2_")

# FC layer unit test case:
    if target == 4:
        gen_test_case_fc(input_shape = [3, 1, 1, 12],
                kernel_shape = [12, 15],
                base_name = "test_1_")
