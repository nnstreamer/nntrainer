#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0-only
##
# Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
# Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
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
def save(filename, *data):
    if os.path.isfile(filename):
        os.remove(filename)

    with open(filename, 'ab') as outfile:
        for item in data:
          np.array(item, dtype=np.float32).tofile(outfile)
          try:
            print(item.shape, " data is generated")
            print(item)
          except:
            pass


##
# @brief generate random tensor
def gen_tensor(shape, dtype=dtypes.float32):
    return np.random.randint(1, 10, shape).astype(np.float32)

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
    x = np.transpose(x,[0,2,3,1])
    kernel = np.transpose(kernel, [2,3,1,0])
    tf.compat.v1.reset_default_graph()
    input_shape = (batch, height, width, channel)

    tf_input = tf.compat.v1.placeholder(
        dtype=dtypes.float32, shape=input_shape, name='input')
    kernel_w = tf.constant_initializer(kernel)
    bias_w = tf.constant_initializer(bias)
    conv2d_layer = tf.keras.layers.Conv2D(k_num, k_width, strides = stride, padding=pad, kernel_initializer=kernel_w, bias_initializer=bias_w)(tf_input)

    conv2d_variables = tf.compat.v1.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)

    input_variables = [tf_input] + conv2d_variables

    grad = tf.gradients(conv2d_layer, input_variables)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        conv2d_result = sess.run(conv2d_layer, feed_dict={tf_input: x})
        grad_result = sess.run(grad, feed_dict={tf_input:x})
    if DEBUG:
        for item, input_variable in zip(grad_result, input_variables):
            print(input_variable.name)
            print(item)

    return conv2d_result, grad_result[0], grad_result[1], grad_result[2]

def pooling2d_tf(x, pool_size, stride, padding, pooling):
    x = np.transpose(x, [0,2,3,1])
    tf.compat.v1.reset_default_graph()
    input_shape = x.shape
    tf_input=tf.compat.v1.placeholder(dtype=dtypes.float32, shape=input_shape, name='input')

    if (pooling == "max"):
        pooling2d_layer=tf.keras.layers.MaxPooling2D(pool_size=pool_size, strides =stride, padding = "valid")(tf_input)
    elif (pooling == "average"):
        pooling2d_layer=tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides =stride, padding = "valid")(tf_input)
    elif (pooling == "global_max"):
        pooling2d_layer=tf.keras.layers.GlobalMaxPooling2D()(tf_input)
    elif (pooling == "global_average"):
        pooling2d_layer=tf.keras.layers.GlobalAveragePooling2D()(tf_input)

    pooling2d_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES)
    input_variables = [tf_input] + pooling2d_variables
    grad = tf.gradients(pooling2d_layer, input_variables)

    with tf.compat.v1.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pooling2d_result = sess.run(pooling2d_layer, feed_dict={tf_input:x})
        grad_result = sess.run(grad, feed_dict={tf_input:x})
    if DEBUG:
        for item, input_variable in zip(grad_result, input_variables):
            print(input_variable.name)
            print(item)
    return pooling2d_result, grad_result[0]

##
# Tested with tensorflow 1.x (1.14.0 and above)
# @brief fc layer forwarding with tensorflow
# @param[in] x input data
# @param[in] kernel weight data
# @param[in] bias bias data
# @param[in] activation activation after the operation
# @return tf_o calculated result
def fc_tf_simplified_backward(x, kernel, label, bias, activation, opt):
    tf.compat.v1.reset_default_graph()
    tf_input = tf.placeholder(dtype = dtypes.float32, shape=x.shape)

    fc_out = tf.keras.layers.Dense(kernel.shape[-1],
                    activation=activation,
                    use_bias=True,
                    kernel_initializer=tf.constant_initializer(kernel),
                    bias_initializer=tf.constant_initializer(bias),
                    input_shape=tf_input.shape)(tf_input)

    trainable_variables = tf.compat.v1.trainable_variables()
    all_variables = [tf_input] + trainable_variables

    if opt == 'sgd':
        optimizer = tf.keras.optimizers.SGD(learning_rate = 1)
    elif opt == 'adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate = 1,beta_1=0.9, beta_2=0.999, epsilon=1.0e-7)
    else:
        raise 'unknown optimizer'
    tf_grad = tf.gradients(fc_out, all_variables)
    train_op = optimizer.apply_gradients(list(zip(tf_grad[1:], trainable_variables)))

    with tf.compat.v1.Session() as sess:
      with tf.compat.v1.variable_scope('fc'):
        sess.run(tf.compat.v1.global_variables_initializer())

        old_w = sess.run(trainable_variables)
        tf_outs = sess.run([fc_out, tf_grad, train_op], feed_dict={tf_input: x})
        new_w = sess.run(trainable_variables)

        tf_outs = tf_outs[:-1] + [new_w]

    if DEBUG:
        print("FC simplified backward with activation.")
        print(tf_outs[0].shape)
        print(tf_outs[1][0].shape)
        print(tf_outs[1][1].shape)
        print(tf_outs[1][2].shape)
        print(tf_outs[2][0].shape)
        print(tf_outs[2][1].shape)
        print("-------------------")
        print(tf_outs[0])
        print(tf_outs[1][0])
        print(tf_outs[1][1])
        print(tf_outs[1][2])
        print(tf_outs[2][0])
        print(tf_outs[2][1])
        print("-------------------")

    return tf_outs

##
# Tested with tensorflow 1.x (1.14.0 and above)
# @brief fc layer forwarding and training with tensorflow
# @param[in] x input data
# @param[in] kernel weight data
# @param[in] bias bias data
# @param[in] activation activation after the operation
# @param[in] train train a few steps
# @return tf_o calculated result
def fc_tf(x, kernel, label, bias, activation, train=False, loss='mse', opt='sgd'):
    lr = 1
    tf.compat.v1.reset_default_graph()
    tf_input = tf.placeholder(dtype = dtypes.float32, shape=x.shape)

    if (loss == 'cross'):
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
                tf_logit = stored_act(tf_logit)
            else:
                raise 'unknown loss'

            if train:
                if opt == 'sgd':
                    optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
                elif opt == 'adam':
                    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
                else:
                    raise 'unknown optimizer'

                trainable_variables = tf.compat.v1.trainable_variables()
                if DEBUG:
                    print([x.name for x in trainable_variables])

                tf_grad = optimizer.get_gradients(tf_loss, params=[tf_input] + trainable_variables)
                train_op = optimizer.apply_gradients(list(zip(tf_grad[1:], trainable_variables)))

                var_to_run = [tf_logit, tf_loss, tf_grad, train_op]
                feed_dict = {tf_input: x, tf_label: label}
            else:
                var_to_run = [tf_logit, tf_loss]
                feed_dict = {tf_input: x, tf_label: label}

            sess.run(tf.compat.v1.global_variables_initializer())
            if DEBUG:
                old_w = sess.run(tf.compat.v1.trainable_variables())

            tf_outs = sess.run(var_to_run, feed_dict = feed_dict)

            if DEBUG:
                new_w = sess.run(tf.compat.v1.trainable_variables())

            if (train):
                # Replace the train_op value with updated weights
                tf_outs = tf_outs[:-1]
                tf_outs.append(sess.run(tf.compat.v1.trainable_variables()))
                # tf outs contain :
                # 1. forward output numpy array
                # 2. final loss value
                # 3. gradient for weights in list form
                # 4. updated weights in list form
                if DEBUG:
                    print(tf_outs[0].shape)
                    print(tf_outs[1].shape)
                    print(tf_outs[2][0].shape)
                    print(tf_outs[2][1].shape)
                    print(tf_outs[3][0].shape)
                    print(tf_outs[3][1].shape)

                if DEBUG:
                    if opt == 'sgd':
                        assert(np.isclose(new_w[1].all(), (old_w[1] - (tf_outs[2][1] * 0.1)).all()))
                    print(old_w[1])
                    print(new_w[1])
                    print(tf_outs[2][1])

    return tf_outs

##
# tested with tf 1.14.0
# @param[in] x input
# @param[in] trainable
# @return input_variables, bn output, output_variables, grad_result (0. dx / 1. gamma / 2. beta / 3. mean / 4. variance)
# for updated_gamma, updated_beta, x <- x - grad is used for easier calculation
def bn_tf(x, *, trainable=True, init_beta=gen_tensor, init_gamma=gen_tensor, axis=[1, 2, 3]):
    tf.compat.v1.reset_default_graph()
    tf_input = tf.compat.v1.placeholder(
        dtype=dtypes.float32, shape=x.shape, name='input')

    bnlayer = tf.keras.layers.BatchNormalization(
        axis=axis,
        trainable=trainable,
        momentum=1.0,
        gamma_initializer=gen_tensor,
        beta_initializer=gen_tensor)(tf_input)

    bn_variables = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.GLOBAL_VARIABLES,
                                               scope='batch_normalization')

    input_variables = [tf_input] + bn_variables

    grad = tf.gradients(bnlayer, input_variables)

    f_dict = {tf_input: x, tf.keras.backend.learning_phase(): trainable}

    with tf.compat.v1.Session() as sess:
      with tf.compat.v1.variable_scope('bn'):
        sess.run(tf.compat.v1.global_variables_initializer())

        old_var = sess.run(input_variables, feed_dict=f_dict)
        bn_result = sess.run(bnlayer, feed_dict=f_dict)
        grad_result = sess.run(grad, feed_dict=f_dict)

        updated_gamma = sess.run(input_variables[1] - grad_result[1])
        updated_beta = sess.run(input_variables[2] - grad_result[2])

        output_variables = [bn_result, updated_gamma, updated_beta]

    if DEBUG:
        print("======================================")
        print("Input:\n %s\n Output:\n %s"  % (x[0], bn_result[0]))
        print("dx: %s" % grad_result[0][0][0])
        print("gradient of gamma: %s" % grad_result[1][0][0], grad_result[1].shape)
        print("gradient of beta: %s" % grad_result[2][0][0], grad_result[2].shape)
        print("======================================")

    return old_var, output_variables, grad_result


def gen_test_case_conv(i_b, i_c, i_h, i_w, k_c, k_h, k_w, padding, stride, bias, base_name):
    x=gen_input(base_name+"_conv2DLayer.in", [i_b, i_c, i_h, i_w])
    kernel=gen_input(base_name+"_conv2DKernel.in", [k_c, i_c, k_h, k_w])
    with open(base_name+"_conv2DKernel.in", 'ab') as outfile:
        np.array(bias, dtype=np.float32).tofile(outfile)

    golden_conv, golden_grad_input, golden_grad_kernel, golden_grad_bias=conv2d_tf(x, kernel, i_b, i_h, i_w, i_c, k_h, k_w, k_c, stride, padding, bias)
    save(base_name+"_goldenConv2DResult.out", np.transpose(golden_conv,(0,3,1,2)))
    save(base_name+"_goldenInputGrad.out", np.transpose(golden_grad_input,(0,3,1,2)))
    save(base_name+"_goldenKernelGrad.out", np.transpose(golden_grad_kernel,(3,2,0,1)))
    save(base_name+"_goldenBiasGrad.out", golden_grad_bias)

def gen_test_case_pooling(input_shape, pool_size, stride, padding, pooling, base_name, gen_in):
    if gen_in:
        input_data = gen_input(base_name + ".in", input_shape)
    else:
        with open(base_name+".in", 'rb') as f:
            input_data = np.fromfile(f, dtype=np.float32)
            input_data=np.reshape(input_data, input_shape)
    golden_pooling, golden_grad_input = pooling2d_tf(input_data, pool_size, stride, padding, pooling)
    if (pooling == "global_average" or pooling == "global_max"):
        save(base_name+"_goldenPooling2D"+pooling+".out", golden_pooling)
    else:
        save(base_name+"_goldenPooling2D"+pooling+".out", np.transpose(golden_pooling,(0,3,1,2)))
    save(base_name+"_goldenPooling2D"+pooling+"Grad.out", np.transpose(golden_grad_input,(0,3,1,2)))

##
# @brief generate fc test case data for forward and backward pass with loss
def gen_test_case_fc(input_shape, kernel_shape, base_name):
    input_data = gen_input(base_name + "_FCLayer.in", input_shape)
    label = gen_input(base_name + "_FCLabel.in", input_shape[:-1] + [kernel_shape[-1]])

    kernel = gen_input(base_name + "_FCKernel.in", kernel_shape)
    bias = gen_input(base_name + "_FCKernel.in", kernel_shape[-1:], savefile=False)
    with open(base_name+"_FCKernel.in", 'ab') as outfile:
        np.array(bias, dtype=np.float32).tofile(outfile)

    golden_fc_simplified = fc_tf_simplified_backward(input_data, kernel, label, bias, activation=None, opt='adam')
    save(base_name + "_goldenFCAdam.out", golden_fc_simplified[0])
    save(base_name + "_goldenFCGradientAdam.out", golden_fc_simplified[1][0])
    save(base_name + "_goldenFCUpdatedWeightAdam.out", golden_fc_simplified[2][0])
    save(base_name + "_goldenFCUpdatedBiasAdam.out", golden_fc_simplified[2][1])

    golden_fc = fc_tf(input_data, kernel, label, bias, activation=None, train=True, loss='mse', opt='sgd')
    save(base_name + "_goldenFCResultActNone.out", golden_fc[0])
    save(base_name + "_goldenFCLossActNoneMse.out", golden_fc[1])
    save(base_name + "_goldenFCGradientDxActNoneMse.out", golden_fc[2][0])
    save(base_name + "_goldenFCGradientsActNoneMse.out", golden_fc[2][1], golden_fc[2][2])
    save(base_name + "_goldenFCUpdatedWeightsActNoneMse.out", golden_fc[3][0], golden_fc[3][1])
    golden_fc_simplified = fc_tf_simplified_backward(input_data, kernel, label, bias, activation=None, opt='sgd' )
    assert(golden_fc_simplified[0].all() == golden_fc[0].all())
    save(base_name + "_goldenFCGradientDxActNone.out", golden_fc_simplified[1][0])
    save(base_name + "_goldenFCGradientsActNone.out", golden_fc_simplified[1][1], golden_fc_simplified[1][2])
    save(base_name + "_goldenFCUpdatedWeightsActNone.out", golden_fc_simplified[2][0], golden_fc_simplified[2][1])

    golden_fc = fc_tf(input_data, kernel, label, bias, activation=tf.nn.sigmoid, train=True, loss='mse', opt='sgd')
    save(base_name + "_goldenFCResultSigmoidMse.out", golden_fc[0])
    save(base_name + "_goldenFCLossSigmoidMse.out", golden_fc[1])
    save(base_name + "_goldenFCGradientDxSigmoidMse.out", golden_fc[2][0])
    save(base_name + "_goldenFCGradientsSigmoidMse.out", golden_fc[2][1], golden_fc[2][2])
    save(base_name + "_goldenFCUpdatedWeightsSigmoidMse.out", golden_fc[3][0], golden_fc[3][1])
    golden_fc_simplified = fc_tf_simplified_backward(input_data, kernel, label, bias, activation=tf.nn.sigmoid, opt='sgd')
    assert(golden_fc_simplified[0].all() == golden_fc[0].all())
    save(base_name + "_goldenFCGradientDxSigmoid.out", golden_fc_simplified[1][0])
    save(base_name + "_goldenFCGradientsSigmoid.out", golden_fc_simplified[1][1], golden_fc_simplified[1][2])
    save(base_name + "_goldenFCUpdatedWeightsSigmoid.out", golden_fc_simplified[2][0], golden_fc_simplified[2][1])

    golden_fc = fc_tf(input_data, kernel, label, bias, activation=tf.nn.softmax, train=True, loss='mse', opt='sgd')
    save(base_name + "_goldenFCResultSoftmaxMse.out", golden_fc[0])
    save(base_name + "_goldenFCLossSoftmaxMse.out", golden_fc[1])
    save(base_name + "_goldenFCGradientDxSoftmaxMse.out", golden_fc[2][0])
    save(base_name + "_goldenFCGradientsSoftmaxMse.out", golden_fc[2][1], golden_fc[2][2])
    save(base_name + "_goldenFCUpdatedWeightsSoftmaxMse.out", golden_fc[3][0], golden_fc[3][1])
    golden_fc_simplified = fc_tf_simplified_backward(input_data, kernel, label, bias, activation=tf.nn.softmax, opt='sgd')
    assert(golden_fc_simplified[0].all() == golden_fc[0].all())
    save(base_name + "_goldenFCGradientDxSoftmax.out", golden_fc_simplified[1][0])
    save(base_name + "_goldenFCGradientsSoftmax.out", golden_fc_simplified[1][1], golden_fc_simplified[1][2])
    save(base_name + "_goldenFCUpdatedWeightsSoftmax.out", golden_fc_simplified[2][0], golden_fc_simplified[2][1])

    golden_fc = fc_tf(input_data, kernel, label, bias, activation=tf.nn.sigmoid, train=True, loss='cross', opt='sgd')
    save(base_name + "_goldenFCResultSigmoidCross.out", golden_fc[0])
    save(base_name + "_goldenFCLossSigmoidCross.out", golden_fc[1])
    save(base_name + "_goldenFCGradientDxSigmoidCross.out", golden_fc[2][0])
    save(base_name + "_goldenFCGradientsSigmoidCross.out", golden_fc[2][1], golden_fc[2][2])
    save(base_name + "_goldenFCUpdatedWeightsSigmoidCross.out", golden_fc[3][0], golden_fc[3][1])

    golden_fc = fc_tf(input_data, kernel, label, bias, activation=tf.nn.softmax, train=True, loss='cross', opt='sgd')
    save(base_name + "_goldenFCResultSoftmaxCross.out", golden_fc[0])
    save(base_name + "_goldenFCLossSoftmaxCross.out", golden_fc[1])
    save(base_name + "_goldenFCGradientDxSoftmaxCross.out", golden_fc[2][0])
    save(base_name + "_goldenFCGradientsSoftmaxCross.out", golden_fc[2][1], golden_fc[2][2])
    save(base_name + "_goldenFCUpdatedWeightsSoftmaxCross.out", golden_fc[3][0], golden_fc[3][1])

    golden_fc = fc_tf(input_data, kernel, label, bias, activation=tf.nn.softmax, train=True, loss='cross', opt='adam')
    save(base_name + "_goldenFCResultSoftmaxCrossAdam.out", golden_fc[0])
    save(base_name + "_goldenFCLossSoftmaxCrossAdam.out", golden_fc[1])
    save(base_name + "_goldenFCGradientDxSoftmaxCrossAdam.out", golden_fc[2][0])
    save(base_name + "_goldenFCGradientsSoftmaxCrossAdam.out", golden_fc[2][1], golden_fc[2][2])
    save(base_name + "_goldenFCUpdatedWeightsSoftmaxCrossAdam.out", golden_fc[3][0], golden_fc[3][1])

def gen_test_case_bn(input_shape, base_name, training=True):
    input_data = gen_input(base_name + "_BNLayerInput.in", input_shape)

    input_variables, output_variables, grad = bn_tf(input_data)

    # mu / var / gamma / beta
    save(base_name + "_BNLayerWeights.in", input_variables[3], input_variables[4], input_variables[1], input_variables[2])
    save(base_name + "_goldenBNResultForward.out", output_variables[0])
    # todo: change 0 to initial moving avg / std in case of training
    save(base_name + "_goldenBNLayerAfterUpdate.out", 0, 0, output_variables[1], output_variables[2])
    save(base_name + "_goldenBNLayerBackwardDx.out", grad[0])

if __name__ == "__main__":
    target = sys.argv[1]

# Input File Generation with given info
    if target == "gen_tensor":
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
    if target == "conv2d_1":
        bias1 = [1.0, 1.0]
        gen_test_case_conv(1, 3, 7, 7, 2, 3, 3, "VALID", 1, bias1, "tc_conv2d_1")

# second unit test case : 2, 3, 7, 7, 3, 3, 3, VALID, 1 test_2_
#  : Input Dimension (2, 3, 7, 7)
#  : Kernel (3, 3, 3, 3)
#  : output (1,3,5,5)
#  : stride 1, 1
#  : padding 0, 0 (VALID)
    if target == "conv2d_2":
        bias2 = [1.0, 1.0, 1.0]
        gen_test_case_conv(2, 3, 7, 7, 3, 3, 3, "VALID", 1, bias2, "tc_conv2d_2")

# FC layer unit test case:
    if target == "fc_1":
        gen_test_case_fc(input_shape = [3, 1, 1, 12],
                kernel_shape = [12, 15],
                base_name = "tc_fc_1")

# Bn layer unit test case:
    if target == "bn_1":
        gen_test_case_bn(input_shape = [3, 1, 4, 5], base_name = "tc_bn_1")

    if target == "pooling2d_1":
        gen_test_case_pooling(input_shape = [1,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="max", base_name="tc_pooling2d_1", gen_in=True)
        gen_test_case_pooling(input_shape = [1,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="average", base_name="tc_pooling2d_1", gen_in=False)
        gen_test_case_pooling(input_shape = [1,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="global_max", base_name="tc_pooling2d_1", gen_in=False)
        gen_test_case_pooling(input_shape = [1,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="global_average", base_name="tc_pooling2d_1", gen_in=False)

    if target == "pooling2d_2":
        gen_test_case_pooling(input_shape = [2,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="max", base_name="tc_pooling2d_2", gen_in=True)
        gen_test_case_pooling(input_shape = [2,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="average", base_name="tc_pooling2d_2", gen_in=False)
        gen_test_case_pooling(input_shape = [2,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="global_max", base_name="tc_pooling2d_2", gen_in=False)
        gen_test_case_pooling(input_shape = [2,2,5,5], pool_size=[2,2], stride=[1,1], padding=[0,0], pooling="global_average", base_name="tc_pooling2d_2", gen_in=False)
