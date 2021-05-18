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

from transLayer import attach_trans_layer

__all__ = ["record"]

tf.compat.v1.enable_eager_execution()
# Fix the seeds across frameworks
SEED = 1234
random.seed(SEED)
tf.compat.v1.set_random_seed(SEED)
np.random.seed(SEED)

LOSS_FN = {
    "mse": lambda: tf.keras.losses.MeanSquaredError(),
    "cross_sigmoid": lambda: tf.keras.losses.BinaryCrossentropy(from_logits=True),
    "cross_softmax": lambda: tf.keras.losses.CategoricalCrossentropy(from_logits=True),
}


def _get_loss_fn(loss_fn_representation):
    try:
        return LOSS_FN[loss_fn_representation]()
    except KeyError:
        raise ValueError("given loss fn representation is not available")


def _get_writer(file):
    def write_fn(*items):
        for item in items:
            try:
                item.numpy().tofile(file)
            except AttributeError:
                pass
        return items

    return write_fn


def _rand_like(tensorOrShape, scale=1):
    try:
        shape = tensorOrShape.shape
    except AttributeError:
        shape = tensorOrShape

    t = np.random.randint(-10, 10, shape).astype(dtype=np.float32)
    return tf.convert_to_tensor(t) * scale


_debug_default_formatter = lambda key, value: "\033[4;32mkey: {}\033[0m\n {}".format(
    key, value
)
##
# @brief Print debug information from the record
# @param debug list or string that filters debug information from @a data
# @param print_option print option for the print function
# @param print_format print formatter. a callable that takes key and value should be passed
# @param data data to passed to _debug_print
def _debug_print(
    debug=None,
    print_option={"end": "\n"},
    print_format=_debug_default_formatter,
    **data
):
    if not debug:
        return
    elif isinstance(debug, str):
        debug = [debug]

    for target in debug:
        try:
            print(print_format(target, data[target]), **print_option)
        except KeyError:
            pass


##
# @brief generate data using uniform data from a function and save to the file.
# @note one-hot label is supported for now, this could be extended if needed.
def prepare_data(model, input_shape, label_shape, writer_fn, **kwargs):
    initial_input = _rand_like(input_shape)
    label = tf.one_hot(
        indices=np.random.randint(0, label_shape[1] - 1, label_shape[0]),
        depth=label_shape[1],
    )

    initial_weights = []
    for layer in model.layers:
        initial_weights += layer.weights.copy()

    writer_fn(initial_input, label, *initial_weights)
    _debug_print(
        initial_input=initial_input,
        label=label,
        initial_weights=initial_weights,
        **kwargs
    )

    return initial_input, label


##
# @brief model iteration wrapper that listen to the gradient and outputs of the model
# each results are recorded.
def train_step(model, optimizer, loss_fn, initial_input, label, writer_fn, **kwargs):
    with tf.GradientTape(persistent=True) as tape:
        tape.watch(initial_input)

        inp = initial_input
        outp = model.call(inp, training=True)
        outputs = {}
        for layer in model.layers:
            output_indices = model.recorder__output_maps[layer.name]
            outputs[layer.name] = [outp[i] for i in output_indices]

        loss = loss_fn(label, outp[-1])

    layer_input = initial_input

    # if traversal order is different from nntrainer, this need to be restructured to rely on the order of output.
    for layer in model.layers:
        # we might need a bit of reordering if output is more than one
        layer_output = outputs[layer.name]

        # print("generating for %s" % layer.name)
        gradients = tape.gradient(loss, layer.trainable_weights)
        optimizer.apply_gradients(zip(gradients, layer.trainable_weights))

        if isinstance(optimizer, tf.keras.optimizers.Adam):
            wm = [optimizer.get_slot(var, "m") for var in layer.trainable_weights]
            wv = [optimizer.get_slot(var, "v") for var in layer.trainable_weights]
            _debug_print(wm=wm, wv=wv, **kwargs)

        _debug_print(lr=optimizer.lr, **kwargs)

        weights = layer.weights.copy()
        dx = tape.gradient(loss, layer_input)

        try:
            gradients = layer.to_nntr_trainable_weights(gradients)
        except AttributeError:
            pass

        writer_fn(
            *layer_output,  # output of forward
            *dx,  # output of backward
            *gradients,  # weight gradient output from backward
            *weights  # updated weight after optimization
        )

        _debug_print(name=layer.name, print_format=value_only_formatter, **kwargs)
        _debug_print(
            output=layer_output,
            dx=dx,
            weights=weights,
            gradients=gradients,
            dx_shape=[i.shape for i in dx],
            **kwargs
        )

        layer_input = layer_output

    writer_fn(loss)
    _debug_print(loss=loss, **kwargs)


##
# @brief inference_step of the result
def inference_step(loss_fn_str, initial_input, label, writer_fn):
    # Not yet implemented
    # because loss function with fromLogit is used, last layer fc layer should be added for the inference step
    if loss_fn_str == "cross_sigmoid" or loss_fn_str == "cross_entropy":
        # add last activation layer
        pass
    raise NotImplementedError("Not Implemented yet")


value_only_formatter = lambda key, value: value

##
# @brief generate recordable model
# @note if model, inputs, outputs is given, trans_layer will NOT be automatically attached
# @param loss_fn_str one of LOSS_FN string otherwise raise KeyError
# @param model base model to record, if model is present @a inputs and @a outputs is ignored
# @param inputs keras inputs to build a model
# @param outputs keras outputs to build a model
def generate_recordable_model(
    loss_fn_str, model=None, inputs=None, outputs=None, **kwargs
):
    if isinstance(model, list):
        model = [attach_trans_layer(layer) for layer in model]

        inputs = model[0]  # first layer must be input
        outputs = [inputs]
        for layer in model[1:]:
            current_output = layer(outputs[-1])
            outputs.append(current_output)

    if isinstance(model, K.models.Model) == False:
        # omit last activation layer if cross softmax or cross_sigmoid
        if loss_fn_str == "cross_softmax" or loss_fn_str == "cross_sigmoid":
            if isinstance(outputs[-1]._keras_history.layer, K.layers.Activation):
                outputs = outputs[:-1]

        model = K.Model(inputs=inputs, outputs=outputs)

    model.summary(
        print_fn=lambda x: _debug_print(
            summary=x, print_format=value_only_formatter, **kwargs
        )
    )

    output_maps = {}
    for idx, output in enumerate(model.outputs):
        layer_name = output._keras_history.layer.name
        try:
            output_maps[layer_name].append(idx)
        except KeyError:
            output_maps[layer_name] = [idx]

    model.recorder__output_maps = output_maps

    return model


##
# @brief record function that records weights, gradients, inputs and outputs for @a iteration
# @param loss_fn_str loss function representation
# @param optimizer keras optimizer
# @param file_name file name to save
# @param input_shape input shape to put
# @param label_shape label shape to put
# @param iteration number of iteration to run
# @param model base model to record, if model is present @a inputs and @a outputs is ignored
# @param inputs keras inputs to build a model
# @param outputs keras outputs to build a model
# @param debug a single string key or list of keys to print out particular information,
# checkout usage of _debug_print of which is printed. for example `_debug_print(loss=loss, **kwargs)`
# catches debug="loss" or debug=["loss"] to print out loss
def record(
    loss_fn_str,
    optimizer,
    file_name,
    input_shape,
    label_shape,
    iteration=1,
    model=None,
    inputs=None,
    outputs=None,
    **kwargs
):
    if os.path.isfile(file_name):
        print("Warning: the file %s is being truncated and overwritten" % file_name)

    loss_fn = _get_loss_fn(loss_fn_str)
    model = generate_recordable_model(loss_fn_str, model, inputs, outputs, **kwargs)

    with open(file_name, "wb") as f:
        write = _get_writer(f)

        initial_input, label = prepare_data(
            model, input_shape, label_shape, write, **kwargs
        )

        for _ in range(iteration):
            _debug_print(
                iteration="\033[1;33m[%d/%d]\033[0m" % (_ + 1, iteration),
                print_format=value_only_formatter,
                **kwargs
            )
            train_step(model, optimizer, loss_fn, initial_input, label, write, **kwargs)

        # self.inference_step(initial_input, label, write)
