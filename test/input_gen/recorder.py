#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file recorder.py
# @date 13 October 2020
# @brief Generate tc from given keras model
# @author Jihoon lee <jhoon.it.lee@samsung.com>

from functools import wraps
import sys
import os
import warnings
import random

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import tensorflow as tf
    from tensorflow.python import keras as K

tf.compat.v1.enable_eager_execution()
# Fix the seeds across frameworks
SEED = 1234
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)


##
# Keras Recorder

##
# @brief Record Keras model with some watchers attached
# @note  The class might need to go through some rework for non-sequential model
# in case of the order of graph traversal is diffrent from NNTrainer
class KerasRecorder:
    def __init__(
        self,
        file_name,
        inputs,
        outputs,
        input_shape,
        label_shape,
        loss_fn=None,
        optimizer=tf.keras.optimizers.SGD(lr=1.0),
    ):
        self.inputs = inputs
        self.outputs = outputs
        self.model = K.Model(inputs=inputs, outputs=outputs)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if os.path.isfile(file_name):
          print("Warning: the file %s is being truncated and overwritten" % file_name)
        self.file = open(file_name, "wb")
        self.generate_data(input_shape, label_shape)

    def __del__(self):
        self.file.close()

    def _rand_like(self, tensorOrShape, scale=10):
        try:
            return tf.random.uniform(tensorOrShape.shape, dtype=tf.float32) * scale
        except AttributeError:
            return tf.random.uniform(tensorOrShape, dtype=tf.float32) * scale

    ##
    # @brief generate data using uniform data from a function and save to the file.
    # @note one-hot label is supported for now, this could be extended if needed.
    def generate_data(self, input_shape, label_shape):
        """This part loads data, should be changed if you are gonna load real data"""
        self.initial_input = self._rand_like(input_shape)
        self.label = tf.one_hot(indices=[1] * label_shape[0], depth=label_shape[1])

        self.initial_input.numpy().tofile(self.file)
        self.label.numpy().tofile(self.file)

    def _write_items(self, *items):
        for item in items:
            print(item)
            try:
                item.numpy().tofile(self.file)
            except AttributeError:
                pass
            try:
                print(item.shape, " data is generated")
            except:
                pass

    ##
    # @brief model iteration wrapper that listen to the gradient and outputs of the model
    # each results are recorded.
    def step(self):
        self.model.summary()

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(self.initial_input)
            outputs = self.model(self.initial_input)

            if self.loss_fn:
                loss = self.loss_fn(self.label, outputs[-1])
                outputs.append(loss)
                print("loss is %s" % loss)

        results = [self.initial_input] + outputs

        for idx, layer in enumerate(self.model.layers):
            print("generating for %s" % layer.name)

            weights = layer.trainable_weights.copy()
            gradients = tape.gradient(results[-1], layer.trainable_weights)
            dweights = tape.gradient(results[-1], results[idx])

            # input, weights, gradients, output, dx
            # you should take weight order to account (eg. I think conv2d has different order)
            self._write_items(
                *[results[idx], *weights, *gradients, results[idx + 1], dweights]
            )

            self.optimizer.apply_gradients(zip(gradients, layer.trainable_weights))

        self._write_items(results[-1])

    ##
    # @brief run function
    # @param iteration number of iteration to run
    def run(self, iteration = 1):
        for _ in range(iteration):
            self.step()
