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
from zoneout import Zoneout

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
    if len(params) == 2:
        new_params = [transpose_(params[0]), params[1]]
    else:
        new_params = [transpose_(params[0])]
    yield from new_params

@register_for_(torch.nn.BatchNorm1d)
def bn1d_translate(model):
    gamma, beta = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    mu, var, _ = [(name, tensor.detach()) for name, tensor in model.named_buffers()]
    yield from [mu, var, gamma, beta]


@register_for_((Zoneout))
def zoneout_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    hidden_state = ("hidden_state", torch.stack(model.hidden_state_zoneout_mask, dim=0))
    cell_state = ("cell_state", torch.stack(model.cell_state_zoneout_mask, dim=0))

    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    new_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3], hidden_state, cell_state]
    yield from new_params

@register_for_((torch.nn.LSTM))
def lstm_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    new_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3]]
    if model.bidirectional:
        reverse_params = [transpose_(params[4]), transpose_(params[5]), params[6], params[7]]
        new_params += reverse_params

    yield from new_params

@register_for_((torch.nn.RNNCell, torch.nn.LSTMCell))
def rnn_lstm_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]
    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    new_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3]]
    yield from new_params

@register_for_((torch.nn.GRUCell))
def gru_translate(model):
    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]

    # [hidden, input] -> [input, hidden]
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    # resetgate, inputgate, newgate -> inputgate, resetgate, newgate
    def reorder_weights(params):
        reordered_weights = []
        for (name, weight) in params: # param = ("name", weight)
            weight = weight.hsplit(3)
            reordered_weights.append((name, torch.hstack((weight[1], weight[0], weight[2])))) # reorder
        return reordered_weights

    transposed_params = [transpose_(params[0]), transpose_(params[1]), params[2], params[3]]
    new_params = reorder_weights(transposed_params)

    yield from new_params

@register_for_((torch.nn.MultiheadAttention))
def multi_head_attention_translate(model):
    def transpose_(weight):
        return (weight[0], weight[1].transpose(1, 0))

    params = [(name, tensor.detach()) for name, tensor in model.named_parameters()]

    getParamByName = lambda name: list(filter(lambda param: param[0] == name, params))[0]

    if model._qkv_same_embed_dim:
        in_proj_weight = getParamByName('in_proj_weight')
        w_q, w_k, w_v = in_proj_weight[1].chunk(3)
        q_proj_weight = ('q_proj_weight', w_q)
        k_proj_weight = ('k_proj_weight', w_k)
        v_proj_weight = ('v_proj_weight', w_v)
    else:
        q_proj_weight = getParamByName('q_proj_weight')
        k_proj_weight = getParamByName('k_proj_weight')
        v_proj_weight = getParamByName('v_proj_weight')

    if model.in_proj_bias is not None:
        in_proj_bias = getParamByName('in_proj_bias')
        w_q, w_k, w_v = in_proj_bias[1].chunk(3)
        q_proj_bias = ('q_proj_bias', w_q)
        k_proj_bias = ('k_proj_bias', w_k)
        v_proj_bias = ('v_proj_bias', w_v)

    out_proj_weight = getParamByName('out_proj.weight')

    if model.in_proj_bias is not None:
        out_proj_bias = getParamByName('out_proj.bias')

    if model.in_proj_bias is None:
        new_params = [transpose_(q_proj_weight), transpose_(k_proj_weight), transpose_(v_proj_weight), transpose_(out_proj_weight)]
    else:
        new_params = [transpose_(q_proj_weight), q_proj_bias, transpose_(k_proj_weight), k_proj_bias, transpose_(v_proj_weight), v_proj_bias, transpose_(out_proj_weight), out_proj_bias]

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

