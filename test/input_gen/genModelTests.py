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
        record, model=fc_sigmoid, input_shape=(3, 3), label_shape=(3, 10), iteration=10
    )

    fc_sigmoid_tc(
        file_name="fc_sigmoid_mse_sgd.info",
        loss_fn_str="mse",
        optimizer=opt.SGD(learning_rate=1.0),
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
        file_name="fc_relu_mse_sgd.info",
        loss_fn_str="mse",
        optimizer=opt.SGD(learning_rate=0.1),
        debug="initial_input"
    )
