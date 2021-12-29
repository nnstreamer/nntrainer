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
# 9. layer name ["name"]
# *************************************************** #
# @author Jihoon lee <jhoon.it.lee@samsung.com>

import warnings
from functools import partial

from recorder import record

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras as K

from transLayer import attach_trans_layer as TL
from transLayer import MultiOutLayer

opt = tf.keras.optimizers

if __name__ == "__main__":

    def multiout_test():
        # x -> [a, b] -> c
        x = K.Input(shape=(2, 3, 5), name="x")
        # because the sort order is x -> [b, a] -> c, b0 must out first.
        b0, a0 = MultiOutLayer(num_output=2)(x)
        a1 = TL(
            K.layers.Conv2D(
                filters=4, kernel_size=3, strides=2, padding="same", name="multiout_a1"
            )
        )(a0)
        a2 = K.layers.Activation("relu", name="multiout_a2")(a1)
        a3 = TL(
            K.layers.Conv2D(
                filters=4, kernel_size=3, padding="same", name="multiout_a3"
            )
        )(a2)
        a4 = K.layers.Flatten(name="multiout_a4")(a3)
        a5 = K.layers.Dense(10, name="multiout_a5")(a4)
        a6 = K.layers.Activation("softmax", name="multiout_a6")(a5)
        b1 = TL(
            K.layers.Conv2D(
                filters=4, kernel_size=1, strides=2, padding="same", name="multiout_b1"
            )
        )(b0)
        b2 = K.layers.Flatten(name="multiout_b2")(b1)
        b3 = K.layers.Dense(10, name="multiout_b3")(b2)
        b4 = K.layers.Activation("softmax", name="multiout_b4")(b3)

        return x, [x, b0, b1, b2, b3, b4, a0, a1, a2, a3, a4, a5, a6]

    x, y = multiout_test()
    record(
        loss_fn_str="mse",
        file_name="multiple_output_model.info",
        input_shape=(3, 2, 3, 5),
        label_shape=(3, 10),
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        inputs=x,
        outputs=y,
        multi_out=[5, 12],
        # debug=["name", "summary", "output", "initial_weights"],
    )

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

    conv_layer_tc(strides=(3, 3))(file_name="conv_uneven_strides.info")

    record(
        file_name="conv_uneven_strides2.info",
        model=[
            K.Input(shape=(2, 4, 4)),
            K.layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(1, 2)),
            K.layers.Activation("sigmoid"),
            K.layers.Flatten(),
            K.layers.Dense(10),
            K.layers.Activation("softmax"),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(3, 2, 4, 4),
        label_shape=(3, 10),
        loss_fn_str="cross_softmax",
        # debug="summary"
    )

    record(
        file_name="conv_uneven_strides3.info",
        model=[
            K.Input(shape=(2, 4, 4)),
            K.layers.Conv2D(filters=2, kernel_size=(2, 2), strides=(2, 1)),
            K.layers.Activation("sigmoid"),
            K.layers.Flatten(),
            K.layers.Dense(10),
            K.layers.Activation("softmax"),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(3, 2, 4, 4),
        label_shape=(3, 10),
        loss_fn_str="cross_softmax",
    )

    record(
        file_name="conv_bn.info",
        model=[
            K.Input(shape=(2, 3, 5)),
            K.layers.Conv2D(filters=2, kernel_size=(2, 2)),
            K.layers.BatchNormalization(),
            K.layers.Activation("relu"),
            K.layers.Flatten(),
            K.layers.Dense(10),
            K.layers.Activation("softmax"),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(3, 2, 3, 5),
        label_shape=(3, 10),
        loss_fn_str="cross_softmax",
        # debug=["summary", "initial_weights"]
    )

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
        file_name="pooling_avg_same_padding_multi_stride.info",  # debug="output"
    )

    def addition_test():
        # x -> [a, b] -> c
        x = K.Input(shape=(2, 3, 5), name="x")
        # because the sort order is x -> [b, a] -> c, b0 must out first.
        b0, a0 = MultiOutLayer(num_output=2)(x)
        a1 = TL(
            K.layers.Conv2D(
                filters=4, kernel_size=3, strides=2, padding="same", name="addition_a1"
            )
        )(a0)
        a2 = K.layers.Activation("relu", name="addition_a2")(a1)
        a3 = TL(
            K.layers.Conv2D(
                filters=4, kernel_size=3, padding="same", name="addition_a3"
            )
        )(a2)
        b1 = TL(
            K.layers.Conv2D(
                filters=4, kernel_size=1, strides=2, padding="same", name="addition_b1"
            )
        )(b0)
        c1 = K.layers.Add(name="addition_c1")([a3, b1])
        c2 = K.layers.Flatten(name="addition_c2")(c1)
        c3 = K.layers.Dense(10, name="addition_c3")(c2)
        c4 = K.layers.Activation("softmax", name="addition_c4")(c3)

        return x, [x, b0, b1, a0, a1, a2, a3, c1, c2, c3, c4]

    x, y = addition_test()
    record(
        loss_fn_str="mse",
        file_name="addition_resnet_like.info",
        input_shape=(3, 2, 3, 5),
        label_shape=(3, 10),
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        inputs=x,
        outputs=y,
        # debug=["name", "summary", "output", "initial_weights"],
    )


    def resnet18(num_class, input_shape):
        def block(x, filters, kernel_size, downsample = False):
            # because the sort order is x -> [b, a] -> c, b0 must out first.
            b0, a0 = MultiOutLayer(num_output=2)(x)
            a1 = TL(K.layers.Conv2D(kernel_size=kernel_size,
                    strides= (1 if not downsample else 2),
                    filters=filters,
                    padding="same"))(a0)
            a2 = TL(K.layers.BatchNormalization())(a1)
            a3 = TL(K.layers.ReLU())(a2)
            a4 = TL(K.layers.Conv2D(kernel_size=kernel_size,
                    strides=1,
                    filters=filters,
                    padding="same"))(a3)

            if downsample:
                b1 = TL(K.layers.Conv2D(kernel_size=1,
                        strides=2,
                        filters=filters,
                        padding="same"))(b0)
            else:
                b1 = b0
            o1 = K.layers.Add()([a4, b1])
            o2 = TL(K.layers.BatchNormalization())(o1)
            o3 = K.layers.Activation("relu")(o2)

            if (downsample):
                ret_array = [a0, a1, a2, a3, a4, b0, b1, o1, o2, o3]
            else:
                ret_array = [a0, a1, a2, a3, a4, b0, o1, o2, o3]
            return ret_array


        # x -> [a, b] -> c
        x = K.Input(shape=input_shape, name="x")
        out_nodes = [x]
        # initial section of resnet
        conv0 = TL(K.layers.Conv2D(
                filters=64, kernel_size=3, strides=1, padding="same"))
        bn0 = TL(K.layers.BatchNormalization())
        act0 = K.layers.Activation("relu")

        out_nodes.append(conv0(out_nodes[-1]))
        out_nodes.append(bn0(out_nodes[-1]))
        out_nodes.append(act0(out_nodes[-1]))

        # Add all the resnet blocks
        out_nodes.extend(block(out_nodes[-1], 64, 3, False))
        out_nodes.extend(block(out_nodes[-1], 64, 3, False))
        out_nodes.extend(block(out_nodes[-1], 128, 3, True))
        out_nodes.extend(block(out_nodes[-1], 128, 3, False))
        out_nodes.extend(block(out_nodes[-1], 256, 3, True))
        out_nodes.extend(block(out_nodes[-1], 256, 3, False))
        out_nodes.extend(block(out_nodes[-1], 512, 3, True))
        out_nodes.extend(block(out_nodes[-1], 512, 3, False))

        # add the suffix part
        pool0 = TL(K.layers.AveragePooling2D(pool_size=4))
        flat0 = K.layers.Flatten()
        dense0 = K.layers.Dense(num_class)
        sm0 = K.layers.Activation("softmax")

        out_nodes.append(pool0(out_nodes[-1]))
        out_nodes.append(flat0(out_nodes[-1]))
        out_nodes.append(dense0(out_nodes[-1]))
        out_nodes.append(sm0(out_nodes[-1]))

        return x, out_nodes

    x, y = resnet18(100, (3,32,32))
    record(
        loss_fn_str="cross_softmax",
        file_name="ResNet18.info",
        input_shape=(2, 3, 32, 32),
        label_shape=(2, 100),
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=2,
        inputs=x,
        outputs=y,
        record_only_outputs=True
        # debug=["file_shape_generation", "name"],
    )

    lstm_layer_tc = lambda batch, time, return_sequences: partial(
        record,
        model=[
            K.Input(shape=(time, 1)),
            K.layers.LSTM(
                time,
                recurrent_activation="sigmoid",
                activation="tanh",
                return_sequences=return_sequences,
            ),
            K.layers.Dense(1),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(batch, time, 1),
        label_shape=(batch, time, 1),
        is_onehot=False,
        loss_fn_str="mse",
    )

    lstm_layer_tc(1, 1, False)(file_name="lstm_basic.info")
    lstm_layer_tc(1, 2, True)(file_name="lstm_return_sequence.info")
    lstm_layer_tc(2, 2, True)(file_name="lstm_return_sequence_with_batch.info")

    multi_lstm_layer_tc = lambda batch, time: partial(
        record,
        model=[
            K.Input(batch_shape=(batch, time, 1)),
            K.layers.LSTM(
                time,
                recurrent_activation="sigmoid",
                activation="tanh",
                return_sequences=True,
            ),
            K.layers.LSTM(time, recurrent_activation="sigmoid", activation="tanh"),
            K.layers.Dense(1),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(batch, time, 1),
        label_shape=(batch, 1),
        is_onehot=False,
        loss_fn_str="mse",
    )
    multi_lstm_layer_tc(1,2)(file_name="multi_lstm_return_sequence.info")
    multi_lstm_layer_tc(2,2)(file_name="multi_lstm_return_sequence_with_batch.info")

    rnn_layer_tc = lambda batch, time, return_sequences: partial(
        record,
        model=[
            K.Input(shape=(time, 1)),
            K.layers.SimpleRNN(2, return_sequences=return_sequences),
            K.layers.Dense(1),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(batch, time, 1),
        label_shape=(batch, time, 1),
        is_onehot=False,
        loss_fn_str="mse",
    )
    rnn_layer_tc(1, 1, False)(file_name="rnn_basic.info")
    rnn_layer_tc(1, 2, True)(file_name="rnn_return_sequences.info")
    rnn_layer_tc(2, 2, True)(file_name="rnn_return_sequence_with_batch.info")

    multi_rnn_layer_tc = lambda batch, time: partial(
        record,
        model=[
            K.Input(batch_shape=(batch, time, 1)),
            K.layers.SimpleRNN(
                time,
                return_sequences=True,
            ),
            K.layers.SimpleRNN(time),
            K.layers.Dense(1),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(batch, time, 1),
        label_shape=(batch, 1),
        is_onehot=False,
        loss_fn_str="mse",
    )
    multi_rnn_layer_tc(1,2)(file_name="multi_rnn_return_sequence.info")
    multi_rnn_layer_tc(2,2)(file_name="multi_rnn_return_sequence_with_batch.info")

    gru_layer_tc = lambda batch, time, unit, feature_size, return_sequences, reset_after: partial(
        record,
        model=[
            K.Input(batch_shape=(batch, time, feature_size)),
            K.layers.GRU(
                unit,
                activation="tanh",
                recurrent_activation="sigmoid",
                bias_initializer='GlorotUniform',
                return_sequences=return_sequences,
                reset_after=reset_after,
            ),
            K.layers.Dense(1),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(batch, time, feature_size),
        label_shape=(batch, time, 1),
        is_onehot=False,
        loss_fn_str="mse"
    )

    gru_layer_tc(1, 1, 3, 4, False, False)(file_name="gru_basic.info")
    gru_layer_tc(1, 2, 3, 4, True, False)(file_name="gru_return_sequence.info")
    gru_layer_tc(2, 2, 3, 4, True, False)(file_name="gru_return_sequence_with_batch.info")
    # Check reset_after
    gru_layer_tc(1, 1, 3, 4, False, True)(file_name="gru_reset_after_basic.info")
    gru_layer_tc(1, 2, 3, 4, True, True)(file_name="gru_reset_after_return_sequence.info")
    gru_layer_tc(2, 2, 3, 4, True, True)(file_name="gru_reset_after_return_sequence_with_batch.info")

    multi_gru_layer_tc = lambda batch, time, unit, feature_size, reset_after: partial(
        record,
        model=[
            K.Input(batch_shape=(batch, time, feature_size)),
            K.layers.GRU(
                unit,
                activation="tanh",
                recurrent_activation="sigmoid",
                bias_initializer='GlorotUniform',
                return_sequences=True,
                reset_after=reset_after,
            ),
            K.layers.GRU(unit, activation="tanh", recurrent_activation="sigmoid", bias_initializer='GlorotUniform', reset_after=reset_after),
            K.layers.Dense(1),
        ],
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        input_shape=(batch, time, feature_size),
        label_shape=(batch, 1),
        is_onehot=False,
        loss_fn_str="mse",
    )
    multi_gru_layer_tc(1, 2, 3, 4, False)(file_name="multi_gru_return_sequence.info")
    multi_gru_layer_tc(2, 2, 3, 4, False)(file_name="multi_gru_return_sequence_with_batch.info")
    # Check reset_after
    multi_gru_layer_tc(1, 2, 3, 4, True)(file_name="multi_gru_reset_after_return_sequence.info")
    multi_gru_layer_tc(2, 2, 3, 4, True)(file_name="multi_gru_reset_after_return_sequence_with_batch.info")

    def multiout_test():
        # x -> [a, b] -> c
        x = K.Input(shape=(1, 10), name="x")
        fc = K.layers.Dense(2, name="fc")(x)
        b0, a0 = MultiOutLayer(num_output=2)(fc)
        fc1 = K.layers.Dense(2, name="fc1")(a0)
        fc2 = K.layers.Dense(2, name="fc2")(b0)
        add1 = K.layers.Add(name="add_1")([fc1, fc2]) # [a, b] -> c
        fc3 = K.layers.Dense(3, name="fc3")(add1)
        sm = K.layers.Activation("softmax", name="sm")(fc3)

        return x, [x, fc, b0, a0, fc1, fc2, add1, fc3, sm]

    x, y = multiout_test()
    record(
        loss_fn_str="mse",
        file_name="multiout_model.info",
        input_shape=(3, 10),
        label_shape=(3, 3),
        optimizer=opt.SGD(learning_rate=0.1),
        iteration=10,
        inputs=x,
        outputs=y,
        # debug=["name", "summary", "output", "initial_weights"],
    )
