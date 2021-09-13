#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file resnet18.py.py
# @date 02 November 2020
# @brief Resnet 18 model file
# @author Jihoon lee <jhoon.it.lee@samsung.com>
# @author Parichay Kapoor <pk.kapoor@samsung.com>
# @note tested with tensorflow 2.6


import random
import os
import sys

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, \
    Add, AveragePooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K

sys.path.insert(1, "../../../test/input_gen")
from transLayer import attach_trans_layer as TL

# Fix the seeds across frameworks
SEED = 412349
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

# enable to verify values of the model layer by layer
DEBUG = False

def resnet_block(x, filters, kernel_size, downsample = False):
    y = TL(Conv2D(kernel_size=kernel_size,
               strides= (1 if not downsample else 2),
               filters=filters,
               padding="same"))(x)
    y = TL(BatchNormalization())(y)
    y = TL(ReLU())(y)
    y = TL(Conv2D(kernel_size=kernel_size,
               strides=1,
               filters=filters,
               padding="same"))(y)
    # y = TL(BatchNormalization())(y)

    if downsample:
        x = TL(Conv2D(kernel_size=1,
                   strides=2,
                   filters=filters,
                   padding="same"))(x)
        # x = TL(BatchNormalization())(x)
    out = Add()([y, x])
    out = TL(BatchNormalization())(out)
    return ReLU()(out)


def Resnet18(input_shape):
    img = Input(shape=input_shape)
    x = TL(Conv2D(64, kernel_size=3, strides=1, padding='same',
                  bias_initializer='zeros',
                  kernel_initializer='glorot_uniform'))(img)
    x = TL(BatchNormalization())(x)
    x = ReLU()(x)
    x = resnet_block(x, 64, 3, False)
    x = resnet_block(x, 64, 3, False)
    x = resnet_block(x, 128, 3, True)
    x = resnet_block(x, 128, 3, False)
    x = resnet_block(x, 256, 3, True)
    x = resnet_block(x, 256, 3, False)
    x = resnet_block(x, 512, 3, True)
    x = resnet_block(x, 512, 3, False)
    x = TL(AveragePooling2D(4))(x)
    x = Flatten()(x)
    x = Dense(100)(x)
    x = tf.keras.layers.Softmax()(x)

    return tf.keras.models.Model(inputs=img, outputs=x)


if __name__ == '__main__':
    tf.config.run_functions_eagerly(True)

    model = Resnet18(input_shape=(3, 32, 32))

    # Fixed LR schedule for now
    opt = Adam(learning_rate=0.001)
    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy'],
        run_eagerly=True
    )
    model.summary()

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar100.load_data()

    x_train = np.transpose(x_train.astype('float32'), (0, 3, 1, 2))
    x_test = np.transpose(x_test.astype('float32'), (0, 3, 1, 2))
    x_train /= 255.0
    x_test /= 255.0

    y_train = tf.keras.utils.to_categorical(y_train, 100)
    y_test = tf.keras.utils.to_categorical(y_test, 100)

    def write_fn(*items, filename='./resnet.bin'):
        with open(filename, 'wb') as f:
            for item in items:
                try:
                    item.numpy().tofile(f)
                except AttributeError:
                    pass
            return items

    initial_weights = []
    for layer in model.layers:
        # print(layer.name)
        initial_weights += layer.weights.copy()

    write_fn(*initial_weights, './resnet_initial.bin')

    if DEBUG:
        input = tf.reshape(tf.convert_to_tensor(x_train[0]), [1,3,32,32])

        # run with training mode
        output = model.call(input)
        # run each layer with training mode
        K.set_learning_phase(1)
        for layer in model.layers:
            l_out = K.function(model.layers[0].input, layer.output)([input])
            if (len(l_out.shape) > 2):
                print(layer.name, l_out[0][0][0])
            else:
                print(layer.name, l_out[0])

    model.fit(
        x = x_train,
        y = y_train,
        epochs = 60,
        verbose = 1,
        validation_data=(x_test, y_test),
        batch_size=128,
    )

    final_weights = []
    for layer in model.layers:
        # print(layer.name)
        final_weights += layer.weights.copy()

    write_fn(*final_weights)
