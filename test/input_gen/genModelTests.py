#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file genModelTests.py
# @date 13 October 2020
# @brief Generate *.info file using KerasRecorder
# *.info file is expected to contain following information **in order**
# you can pass ("value") to debug to print out
# *************************************************** #
# 1. reference input ["initial_input"]
# 2. reference label ["label"]
# 3. initial weight data for each layer in order ["initial_weights"]
# for each iteration...
# 4. layer forward output for each layer in order["output"]
# 5. layer backward output for each layer in order ["dx"]
# 6. weight gradient for each layer(for trainable weights) ["gradients"]
# 7. updated weights after optimization ["weights"]
# after iteration...
# 8. inference result (NYI)
# *************************************************** #
# @author Jihoon lee <jhoon.it.lee@samsung.com>

import warnings
from functools import partial

from recorder import record

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import tensorflow as tf
    from tensorflow.python import keras as K

opt = tf.keras.optimizers

if __name__ == "__main__":
    ## please generate all test cases since golden data format can change anytime
    fc_sigmoid = [
        K.Input(shape=(3, 3)),
        K.layers.Dense(5),
        K.layers.Activation("sigmoid"),
        K.layers.Dense(10),
        K.layers.Activation("softmax"),
    ]

    fc_sigmoid_tc = partial(
        record,
        model=fc_sigmoid,
        input_shape=(3, 3),
        label_shape=(3, 10),
        iteration=10,
        optimizer=opt.SGD(learning_rate=1.0),
    )

    fc_sigmoid_tc(file_name="fc_sigmoid_mse.info", loss_fn_str="mse")

    fc_sigmoid_tc(
        file_name="fc_sigmoid_cross.info", loss_fn_str="cross_softmax",
    )

    fc_relu = [
        K.Input(shape=(3)),
        K.layers.Dense(10),
        K.layers.Activation("relu"),
        K.layers.Dense(2),
        K.layers.Activation("sigmoid"),
    ]

    fc_relu_tc = partial(
        record, model=fc_relu, input_shape=(3, 3), label_shape=(3, 2), iteration=10
    )

    fc_relu_tc(
        file_name="fc_relu_mse.info",
        loss_fn_str="mse",
        optimizer=opt.SGD(learning_rate=0.1),
    )

    fc_bn_sigmoid = [
        K.Input(shape=(3)),
        K.layers.Dense(10),
        K.layers.BatchNormalization(),
        K.layers.Activation("sigmoid"),
        K.layers.Dense(10),
        K.layers.Activation("softmax"),
    ]

    fc_bn_sigmoid_tc = partial(
        record,
        model=fc_bn_sigmoid,
        input_shape=(3, 3),
        label_shape=(3, 10),
        optimizer=opt.SGD(learning_rate=1),
        iteration=10,
    )

    fc_bn_sigmoid_tc(
        file_name="fc_bn_sigmoid_cross.info",
        loss_fn_str="cross_softmax",
        # debug=["summary", "iteration", "weights"],
    )

    fc_bn_sigmoid_tc(
        file_name="fc_bn_sigmoid_mse.info", loss_fn_str="mse",
    )

    _mnist_block = lambda filter_size: [
        K.layers.Conv2D(filters=filter_size, kernel_size=(3, 4)),
        K.layers.Activation("sigmoid"),
        K.layers.AveragePooling2D(pool_size=(2, 2)),
    ]

    mnist_conv = [
        K.Input(shape=(2, 4, 5)),
        *_mnist_block(2),
        K.layers.Flatten(),
        K.layers.Dense(10),
        K.layers.Activation("softmax"),
    ]

    mnist_conv_tc = partial(
        record, model=mnist_conv, optimizer=opt.SGD(learning_rate=0.1), iteration=10,
    )

    mnist_conv_tc(
        input_shape=(3, 2, 4, 5),
        label_shape=(3, 10),
        file_name="mnist_conv_cross.info",
        loss_fn_str="cross_softmax",
        # debug=["summary", "loss", "layer_name", "initial_weights"],
    )

    mnist_conv_tc(
        input_shape=(1, 2, 4, 5),
        label_shape=(1, 10),
        file_name="mnist_conv_cross_one_input.info",
        loss_fn_str="cross_softmax",
        # debug=["summary", "loss", "layer_name", "initial_weights"],
    )

    conv_nxn_model = lambda kernel_size: [
        K.Input(shape=(2, 4, 5)),
        K.layers.Conv2D(filters=4, kernel_size=kernel_size),
        K.layers.Activation("sigmoid"),
        K.layers.Flatten(),
        K.layers.Dense(10),
        K.layers.Activation("softmax"),
    ]

    conv_nxn_tc = partial(
        record,
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(3, 2, 4, 5),
        label_shape=(3, 10),
        loss_fn_str="cross_softmax",
    )

    # 1x1 kernel size
    conv_nxn_tc(
        model=conv_nxn_model((1, 1)), file_name="conv_1x1.info",
    )

    # height width is same as input size
    conv_nxn_tc(
        model=conv_nxn_model((4, 5)), file_name="conv_input_matches_kernel.info"
    )

    conv_layer_tc = lambda **conv_args: partial(
        record,
        model=[
            K.Input(shape=(2, 5, 3)),
            K.layers.Conv2D(filters=4, kernel_size=(3, 3), **conv_args),
            K.layers.Activation("sigmoid"),
            K.layers.Flatten(),
            K.layers.Dense(10),
            K.layers.Activation("softmax"),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(3, 2, 5, 3),
        label_shape=(3, 10),
        loss_fn_str="cross_softmax",
    )

    conv_layer_tc()(file_name="conv_basic.info")
    conv_layer_tc(padding="same")(file_name="conv_same_padding.info")  # padding: 1, 1
    conv_layer_tc(strides=(2, 2))(file_name="conv_multi_stride.info")
    conv_layer_tc(padding="same", strides=(2, 2))(  # padding: 1, 1
        file_name="conv_same_padding_multi_stride.info"
    )

    conv_layer_tc(strides=(3, 3))(file_name="conv_uneven_strides.info", debug="summary")

    pool_layer_tc = lambda pool_layer: partial(
        record,
        model=[
            K.Input(shape=(2, 5, 3)),
            pool_layer,
            K.layers.Activation("sigmoid"),
            K.layers.Flatten(),
            K.layers.Dense(10),
            K.layers.Activation("softmax"),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(3, 2, 5, 3),
        label_shape=(3, 10),
        loss_fn_str="cross_softmax",
    )

    pool_layer_tc(K.layers.MaxPooling2D(pool_size=3, strides=1, padding="same"))(
        file_name="pooling_max_same_padding.info",  # debug="output"
    )  # padding: 1, 1

    pool_layer_tc(K.layers.MaxPooling2D(pool_size=3, strides=1, padding="valid"))(
        file_name="pooling_max_valid_padding.info",  # debug="output"
    )  # padding: 1, 1

    pool_layer_tc(K.layers.AveragePooling2D(pool_size=3, strides=1, padding="same"))(
        file_name="pooling_avg_same_padding.info",  # debug="dx"
    )  # padding: 1, 1

    pool_layer_tc(K.layers.AveragePooling2D(pool_size=3, strides=1, padding="valid"))(
        file_name="pooling_avg_valid_padding.info",  # debug="dx"
    )

    pool_layer_tc(K.layers.GlobalAvgPool2D(data_format="channels_first"))(
        file_name="pooling_global_avg.info",  # debug="summary"
    )

    pool_layer_tc(K.layers.GlobalMaxPool2D(data_format="channels_first"))(
        file_name="pooling_global_max.info",  # debug="dx"
    )

    pool_layer_tc2 = lambda pool_layer: partial(
        record,
        model=[
            K.Input(shape=(2, 3, 5)),
            pool_layer,
            K.layers.Activation("sigmoid"),
            K.layers.Flatten(),
            K.layers.Dense(10),
            K.layers.Activation("softmax"),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(3, 2, 3, 5),
        label_shape=(3, 10),
        loss_fn_str="cross_softmax",
    )

    pool_layer_tc2(K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same"))(
        file_name="pooling_max_same_padding_multi_stride.info",  # debug="dx"
    )

    pool_layer_tc2(K.layers.AveragePooling2D(pool_size=3, strides=2, padding="same"))(
        file_name="pooling_avg_same_padding_multi_stride.info", # debug="output"
    )

