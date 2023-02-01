# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
#
# @file   main.cpp
# @date   01 Feb 2023
# @see    https://github.com/nnstreamer/nntrainer
# @author Donghak Park <donghak.park@samsung.com>
# @bug	  No known bugs except for NYI items
# @brief  This is Model_A_Linear Example for Tensorflow (with Dummy Dataset)
# @brief  -> FC -> FC -> FC ->

import tensorflow as tf
import numpy as np

print("Tensorflow Verison : ", tf.__version__)
print("Keras Version      : ", tf.keras.__version__)

NUM_EPOCHS = 10
BATCH_SIZE = 2048
IMG_SIZE = 784
OUTPUT_SIZE = 100


class Model(tf.Module):

    def __init__(self):
        self.model = tf.keras.models.Sequential(
            [
                tf.keras.layers.Input(shape=(IMG_SIZE,)),
                tf.keras.layers.Dense(4096, name="dense1"),
                tf.keras.layers.Dense(2048, name="dense2"),
                tf.keras.layers.Dense(OUTPUT_SIZE, name="dense3"),
            ]
        )

        self.model.compile(
            loss=tf.keras.losses.MeanSquaredError(), optimizer="sgd")


m = Model()
m.model.summary()
np.random.seed(1234)

for i in range(NUM_EPOCHS):
    x_train = np.random.randint(0, 255, (BATCH_SIZE, IMG_SIZE))
    y_train = np.random.randint(
        0, 1, (BATCH_SIZE, OUTPUT_SIZE)).astype(np.float32)
    x_train = (x_train / 255.0).astype(np.float32)
    result = m.model.fit(x_train, y_train, batch_size=BATCH_SIZE)
print("----------Training END--------------")
