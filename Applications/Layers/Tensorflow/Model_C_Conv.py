# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
#
# @file   main.cpp
# @date   01 Feb 2023
# @see    https://github.com/nnstreamer/nntrainer
# @author Donghak Park <donghak.park@samsung.com>
# @bug	  No known bugs except for NYI items
# @brief  This is Model_C_Conv Example for Tensorflow (with Dummy Dataset)
# @brief  -> Conv -> RELU -> Flatten ->

import tensorflow as tf
import numpy as np

print("Tensorflow Verison : ", tf.__version__)
print("Keras Version      : ", tf.keras.__version__)

NUM_EPOCHS = 100
BATCH_SIZE = 64
IMG_SIZE = 224
OUTPUT_SIZE = 10


class Model(tf.Module):

    def __init__(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
                tf.keras.layers.Conv2D(
                    3,
                    kernel_size=(3, 3),
                    padding="same",
                    name="conv1",
                    strides=(2, 2),
                    activation="relu",
                ),
                tf.keras.layers.Flatten(),
            ]
        )

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(), optimizer="sgd")


m = Model()
m.model.summary()
np.random.seed(1234)


for i in range(NUM_EPOCHS):
    x_train = np.random.randint(0, 255, (64, IMG_SIZE, IMG_SIZE, 3))
    y_train = np.random.randint(0, 1, (64, 37632)).astype(np.float32)
    x_train = (x_train / 255.0).astype(np.float32)
    result = m.model.fit(x_train, y_train, batch_size=BATCH_SIZE)
print("----------Training END--------------")
