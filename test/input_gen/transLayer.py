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
        return self.to_nntr_tensor(tf_output)

    ##
    # @brief change a tensor to tf layout
    # @param tensor tensor to convert
    def to_tf_tensor(self, tensor):
        return tensor

    ##
    # @brief change a tensor to tf lay
    def to_nntr_tensor(self, tensor):
        return tensor

    ##
    # @brief overriding the identity of this class. This ensures that transLayer
    # is a type of a Layer.
    # Note that this more like a hack and it might break
    # any method using self.__class__
    @property
    def __class__(self):
        return K.engine.base_layer.Layer

    ##
    # @brief hook to get trainable_weights
    @property
    def trainable_weights(self):
        return self.tf_layer.trainable_weights

    ##
    # @brief hook to get weights
    @property
    def weights(self):
        return self.tf_layer.weights


##
# @brief Translayer that does nothing (only using identity function)
class IdentityTransLayer(AbstractTransLayer):
    pass


##
# @brief Translayer for batch normalization layer
class BatchNormTransLayer(AbstractTransLayer):

    @property
    def weights(self):
        x = self.tf_layer.weights
        assert len(x) == 4
        return [x[2], x[3], x[0], x[1]]

##
# @brief A factory function to attach translayer to existing layer
# if nothing should be attached, it does not attach the layer
def attach_trans_layer(layer):
    if isinstance(layer, K.layers.BatchNormalization):
        return BatchNormTransLayer(layer)

    return layer
