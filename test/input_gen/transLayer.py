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
    from tensorflow.python import keras as K

##
# @brief AbstractTransLayer is a proxy to translate a layer to nntrainer layout.
# This class delegates most of the method/member to self.tf_layer except that
# input, output, weights are reinterpreted to be in nntrainer_layer layout
class AbstractTransLayer:
    def __init__(self, tf_layer):
        if not isinstance(tf_layer, K.engine.base_layer.Layer):
            raise ValueError("tf_layer must be type of keras layer")
        self.tf_layer = tf_layer

    def __getattr__(self, name):
        return self.tf_layer.__getattribute__(name)

    def __call__(self, *args, **kwargs):
        return self.call(*args, **kwargs)

    ##
    # @brief call function
    # @param nntr_input input with nntrainer layout
    def call(self, nntr_input, *args, **kwargs):
        tf_input = self.to_tf_tensor(nntr_input)
        tf_output = self.tf_layer.__call__(tf_input, *args, **kwargs)
        self.output = self.to_nntr_tensor(tf_output)
        return self.output

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
    # @brief overriding the identity of this class. This ensures that transLayer
    # is a type of a Layer.
    # Note that this more like a hack and it might break
    # any method using self.__class__
    @property
    def __class__(self):
        return K.engine.base_layer.Layer

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
# @brief Translayer for batch normalization layer
class BatchNormTransLayer(IdentityTransLayer):

    def to_nntr_weights(self, tensorOrList):
        x = tensorOrList
        assert len(x) == 4
        return [x[2], x[3], x[0], x[1]]


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
    TO_NNTR_KERNEL = (3, 2, 0, 1)

    def to_tf_tensor(self, tensor):
        return K.layers.Permute(ChannelLastTransLayer.TO_CHANNELS_LAST)(tensor)

    def to_nntr_tensor(self, tensor):
        return K.layers.Permute(ChannelLastTransLayer.TO_CHANNELS_FIRST)(tensor)

    @classmethod
    def _nntr_kernel(cls, tensor):
        if tf.rank(tensor).numpy() == 4:
            return tf.transpose(tensor, perm=cls.TO_NNTR_KERNEL)
        return tensor

    def to_nntr_trainable_weights(self, weights):
        return self.to_nntr_weights(weights)

    def to_nntr_weights(self, weights):
        return [self._nntr_kernel(t) for t in weights]


CHANNEL_LAST_LAYERS = (
    K.layers.Conv2D,
    K.layers.AveragePooling2D,
    K.layers.AvgPool2D,
    K.layers.MaxPooling2D,
    K.layers.MaxPool2D,
)


##
# @brief A factory function to attach translayer to existing layer
# if nothing should be attached, it does not attach the layer
def attach_trans_layer(layer):
    if isinstance(
        layer,
        (K.layers.BatchNormalization, K.layers.normalization_v2.BatchNormalization),
    ):
        return BatchNormTransLayer(layer)

    if isinstance(layer, CHANNEL_LAST_LAYERS):
        return ChannelLastTransLayer(layer)

    return layer
