#!/usr/bin/env python3
##
# Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# SPDX-License-Identifier: Apache-2.0-only
#
# @file recorder.py
# @brief Generate tc from given keras model
# @author Jihoon lee <jhoon.it.lee@samsung.com>

from functools import wraps
import sys
import os
import warnings
import random

os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

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
# @brief record given keras model
# model flows as
# 1. foward pass
# --------------------------------
#  in -> | model weights | -> out
# --------------------------------
# 2. calculate loss(omitable)
# --------------------------------
#  out ->    loss_fn       -> loss
# --------------------------------
# 3. backprop pass
# --------------------------------
#  dy(1) -> | model dweights | -> dx
# --------------------------------
# 4. weight update
# --------------------------------
# updated_weights <- opt(weights, dx)
# --------------------------------
#
# Recorder records feedable_list to *.in file (generated randomly)
#                  result_list to *.out file (calculated from *.in file)
class Recorder:
    feedable_list = ("in", "weights", "label")
    result_candidates = ("out", "dx", "dweights", "updated_weights")

    def __init__(
        self,
        model,
        loss_fn=None,
        optimizer=tf.keras.optimizers.SGD(lr=1.0),
        result_list=result_candidates,
    ):
        """
            (uncompiled) model with loss_fn, optimizer to be descripted
        """
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.result_list = result_list
        self.feed_list = self.feedable_list if loss_fn else self.feedable_list[:-1]

    def _rand_like(self, tensorOrShape, scale=10):
        try:
            return tf.random.uniform(tensorOrShape.shape, dtype=tf.float32) * scale
        except AttributeError:
            return tf.random.uniform(tensorOrShape, dtype=tf.float32) * scale

    def _generate_storage(self, input_shape, label_shape):
        x = self._rand_like(input_shape)
        label = self._rand_like(label_shape, scale=1) if label_shape else None

        storage = {}
        storage["in"] = x
        storage["label"] = label

        with tf.GradientTape(persistent=True) as tape:
            tape.watch(x)
            y = model(x)

            if self.loss_fn:
                y = self.loss_fn(label, x)

        print(model.summary())
        print("=========================WEIGHT INFO==============================")
        print(*[(item.name, item.shape) for item in model.trainable_weights], sep="\n")
        print("==================================================================")

        storage["weights"] = model.trainable_weights.copy()
        storage["out"] = y
        storage["dx"] = tape.gradient(y, x)
        storage["dweights"] = tape.gradient(y, model.trainable_weights)
        self.optimizer.apply_gradients(
            zip(storage["dweights"], model.trainable_weights)
        )

        storage["updated_weights"] = model.trainable_weights

        return storage

    def _write_items(self, fd, *items):
        for item in items:
            item.numpy().tofile(fd)
            try:
                print(item.shape, " data is generated")
            except:
                pass

    def _write(self, storage, saving_list, filename):
        f = open(filename, "wb")
        for item in saving_list:
            print(" generating [%s]..." % item)
            try:
                _item = storage[item]
            except KeyError:
                raise RuntimeError("result list seems wrong")
            if isinstance(_item, tf.Tensor):
                self._write_items(f, _item)
            else:
                self._write_items(f, *_item)
        f.close()

    ##
    # @brief generate_tc...
    # @param[in] input_shape shape object to determine input
    # @param[in] label_shape label object to deteremine label. Currently, label is generated as well
    # @param[in] case_name case name
    # @param[in] num_cases number of case to be generated
    # @param[in] save whether to save to real file or just print out
    def generate_tc(
        self, input_shape, label_shape=None, case_name="golden", num_cases=1, save=True
    ):
        for idx in range(num_cases):
            tc_name = "tc_%s_%d" % (case_name, idx)
            # storage is a dictionary that has all the tensors that need to make tc
            print("==========================================generating %s " % tc_name)
            storage = self._generate_storage(input_shape, label_shape)
            if save:
                print("'%s.in' file will contain %s" % (tc_name, str(self.feed_list)))
                self._write(storage, self.feed_list, "%s.in" % tc_name)
                print(
                    "'%s.out' file will contain %s" % (tc_name, str(self.result_list))
                )
                self._write(storage, self.result_list, "%s.out" % (tc_name))
            else:
                print(storage)


if __name__ == "__main__":
    inp = K.Input(shape=(3, 3))
    a = K.layers.Dense(32)(inp)
    # b = K.layers.Dense(32)(a)

    # or you could use Model.Sequential()
    model = K.Model(inputs=inp, outputs=a)
    Recorder(model).generate_tc((3, 3), case_name="tc_example", num_cases=2)
