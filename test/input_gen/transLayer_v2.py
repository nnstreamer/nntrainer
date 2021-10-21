#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file transLayer_v2.py
# @date 19 October 2021
# @brief Rewrite parameters in the order consistent with nntrainer for the torch model
# @author Jihoon lee <jhoon.it.lee@samsung.com>

import torch
from collections.abc import Iterable

__all__ = ["params_translated"]

# type to parameter mapper containing (classes, function)
handler_book = []

##
# Decorater to register class mapping to a function.
# This is to imitate function overloadding
def register_for_(classes):
    for already_registered_classes, _ in handler_book:
        if not isinstance(classes, Iterable):
            classes = (classes, )

        for cls_ in classes:
            if isinstance(cls_, already_registered_classes):
                raise ValueError("class is already registered %s" % cls_.__name__)

    def wrapper(func):
        handler_book.append((classes, func))
        return func

    return wrapper


def default_translate_(model):
    yield from model.named_parameters(recurse=False)

@register_for_(torch.nn.Linear)
def fc_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))
    new_params = [transpose_(params[0]), params[1]]
    yield from new_params

@register_for_(torch.nn.BatchNorm1d)
def bn1d_translate(model):
    gamma, beta = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    mu, var, _ = [(name, tensor.detach()) for name, tensor in model.named_buffers()]
    yield from [mu, var, gamma, beta]


@register_for_(torch.nn.LSTMCell)
def lstm_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    bias = ("bias", params[2][1] + params[3][1])
    # hidden, input -> input, hidden
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    new_params = [transpose_(params[0]), transpose_(params[1]), bias]
    yield from new_params

def translate(model):
    for child in model.children():
        for registered_classes, fn in handler_book:
            if isinstance(child, registered_classes):
                yield from fn(child)
                break
        else: # default case
            yield from translate(child)
    yield from default_translate_(model)

params_translated = translate

