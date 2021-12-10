#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file transLayer.py
# @date 02 November 2020
# @brief Proxy object to translate tensor to nntrainer layout from keras layer
# @author Jihoon lee <jhoon.it.lee@samsung.com>
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import numpy as np
    import tensorflow as tf
    import tensorflow.keras as K

import inspect
from inspect import signature

##
# @brief AbstractTransLayer is a proxy to translate a layer to nntrainer layout.
# This class delegates most of the method/member to self.tf_layer except that
# input, output, weights are reinterpreted to be in nntrainer_layer layout
class AbstractTransLayer(K.layers.Layer):
    def __init__(self, tf_layer, *args, **kwargs):
        if not isinstance(tf_layer, K.layers.Layer):
            raise ValueError("tf_layer must be type of keras layer")
        super().__init__(*args, **kwargs, name=tf_layer.name + "/translated")
        self.tf_layer = tf_layer
        self.call.__func__.__signature__ = signature(self.tf_layer.call)
        self.has_training = "training" in inspect.getfullargspec(self.call).args

    def build(self, input_shape):
        if not self.built:
            self.tf_layer.build(input_shape)
            super().build(input_shape)

    ##
    # @brief call function
    # @param nntr_input input with nntrainer layout
    def call(self, nntr_input, training=None):
        tf_input = self.to_tf_tensor(nntr_input)

        additional_args = {}
        if self.has_training:
            additional_args["training"] = training

        tf_output = self.tf_layer.call(tf_input, **additional_args)
        return self.to_nntr_tensor(tf_output)

    ##
    # @brief change a tensor to tf layout
    # @param tensor tensor to convert
    def to_tf_tensor(self, tensor):
        raise NotImplementedError("Abstract method should not be called!")

    ##
    # @brief change a tensor to tf layout
    def to_nntr_tensor(self, tensor):
        raise NotImplementedError("Abstract method should not be called!")

    ##
    # @brief convert tensor weights to nntrainer weights
    # @param tensorOrList tensor or list of tensor to convert
    def to_nntr_weights(self, tensorOrList):
        raise NotImplementedError("Abstract method should not be called!")

    ##
    # @brief convert tensor **trainable** weights to nntrainer weights
    # this can be used to convert gradient layout
    # @param tensorOrList tensor or list of tensors to convert
    def to_nntr_trainable_weights(self, tensorOrList):
        raise NotImplementedError("Abstract method should not be called!")

    ##
    # @brief hook to get weights
    @property
    def weights(self):
        return self.to_nntr_weights(self.tf_layer.weights)


##
# @brief Translayer that does nothing (only using identity function)
class IdentityTransLayer(AbstractTransLayer):
    def to_tf_tensor(self, tensor):
        return tensor

    def to_nntr_tensor(self, tensor):
        return tensor

    def to_nntr_weights(self, tensorOrList):
        return tensorOrList

    def to_nntr_trainable_weights(self, tensorOrList):
        return tensorOrList


##
# @brief Translayer to translate channel last <-> channel first
# @note This class relies on Permute layer. This should be skipped when
# iterating through keras layer
class ChannelLastTransLayer(AbstractTransLayer):
    # TO_CHANNELS_FIRST = (0, 3, 1, 2)
    TO_CHANNELS_FIRST = (3, 1, 2)
    # TO_CHANNELS_LAST = (0, 2, 3, 1)
    TO_CHANNELS_LAST = (2, 3, 1)
    # height, width, channel, filter_size -> filter_size, channel, height, width
    TO_NNTR_KERNEL_4D = (3, 2, 0, 1)
    # width, channel, filter_size -> filter_size, channel, width
    TO_NNTR_KERNEL_3D = (2, 1, 0)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.to_tf_layer_ = K.layers.Permute(ChannelLastTransLayer.TO_CHANNELS_LAST)
        self.to_nntr_layer_ = K.layers.Permute(ChannelLastTransLayer.TO_CHANNELS_FIRST)

    def build(self, input_shape):
        if self.built:
            return

        if isinstance(input_shape, tf.TensorShape):
            input_shape_list_ = input_shape.as_list()
        else:
            input_shape_list_ = input_shape
        transposed_list_ = [None] * 4

        for idx, i in enumerate((0,) + ChannelLastTransLayer.TO_CHANNELS_LAST):
            transposed_list_[idx] = input_shape_list_[i]

        transposed_input_shape = tf.TensorShape(transposed_list_)
        super().build(transposed_input_shape)

    def to_tf_tensor(self, tensor):
        return self.to_tf_layer_(tensor)

    def to_nntr_tensor(self, tensor):
        return self.to_nntr_layer_(tensor)

    @classmethod
    def _nntr_kernel(cls, tensor):
        if tf.rank(tensor).numpy() == 4:
            return tf.transpose(tensor, perm=cls.TO_NNTR_KERNEL_4D)
        elif tf.rank(tensor).numpy() == 3:
            return tf.transpose(tensor, perm=cls.TO_NNTR_KERNEL_3D)
        return tensor

    def to_nntr_trainable_weights(self, weights):
        return self.to_nntr_weights(weights)

    def to_nntr_weights(self, weights):
        return [self._nntr_kernel(t) for t in weights]


CHANNEL_LAST_LAYERS = (
    K.layers.Conv2D,
    K.layers.Conv1D,
    K.layers.AveragePooling2D,
    K.layers.AvgPool2D,
    K.layers.MaxPooling2D,
    K.layers.MaxPool2D,
)

##
# @brief Translayer for batch normalization layer
class BatchNormTransLayer(IdentityTransLayer):
    def build(self, input_shape):
        if self.built:
            return

        if len(input_shape) > 3:
            self.tf_layer = ChannelLastTransLayer(self.tf_layer)

        super().build(input_shape)

    def call(self, input, training=None):
        return self.tf_layer.call(input, training)

    def to_nntr_weights(self, tensorOrList):
        x = tensorOrList
        assert len(x) == 4
        return [x[2], x[3], x[0], x[1]]


##
# @brief Multiout wrapper layer, this class separate gradient
# to calculate derivative properly
# when calling, this returns [x] * @a num_output just like output Layer does in NNTrainer
class MultiOutLayer(IdentityTransLayer):

    ##
    # @brief init function
    # @param tf_layer tf_layer to get number of outputs
    # @param num_output explicit number to generate number of output
    def __init__(self, tf_layer=None, *args, num_output, **kwargs):
        if not tf_layer:
            tf_layer = K.layers.Layer()

        super().__init__(tf_layer, *args, **kwargs)

        # this enables seperating gradient one by one
        self.stub_layers = [K.layers.Lambda(lambda x: x + 0) for i in range(num_output)]

    ##
    # @brief call function
    # @param x input with nntrainer layout
    def call(self, x, training=None):
        additional_args = {}
        if self.has_training:
            additional_args["training"] = training

        tf_output = self.tf_layer.call(x, **additional_args)

        return [layer(tf_output) for layer in self.stub_layers]

##
# @brief Translayer for gru layer
class GRUTransLayer(IdentityTransLayer):
    def to_nntr_weights(self, tensorOrList):
        bias = tensorOrList[2]
        if bias.shape.rank == 2:
            bias_ih, bias_hh = bias[0], bias[1]
            return [tensorOrList[0], tensorOrList[1], bias_ih, bias_hh]
        else:
            return tensorOrList

    def to_nntr_trainable_weights(self, tensorOrList):
        return self.to_nntr_weights(tensorOrList)

##
# @brief A factory function to attach translayer to existing layer
# if nothing should be attached, it does not attach the layer
def attach_trans_layer(layer):
    if isinstance(layer,
        (K.layers.BatchNormalization),
    ):
        return BatchNormTransLayer(layer)

    if isinstance(layer, CHANNEL_LAST_LAYERS):
        return ChannelLastTransLayer(layer)

    if isinstance(layer, K.layers.GRU):
        return GRUTransLayer(layer)

    return layer
