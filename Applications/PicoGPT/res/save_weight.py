#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
#
# @file save_weight.py
# @date 19 May 2023
# @brief save weights from pico gpt
# @author Hyeonseok Lee <hs89.lee@samsung.com>


import numpy as np

file = open("pico_gpt.bin", "wb")
def save_params(params, n_head):
    def save_weight(weight):
        weight.tofile(file)

    def save_ln(weights):
        save_weight(weights['g'])
        save_weight(weights['b'])

    def save_fc(weights):
        save_weight(weights['w'])
        save_weight(weights['b'])

    def save_qkv(weights):
        qkv_ws = np.split(weights['w'], 3, axis=-1)
        qkv_bs = np.split(weights['b'], 3, axis=-1)
        qkv_heads_ws = list(map(lambda x: np.split(x, n_head, axis=-1), qkv_ws))
        qkv_heads_bs = list(map(lambda x: np.split(x, n_head, axis=-1), qkv_bs))
        for i, (qkv_w, qkv_b) in enumerate(zip(qkv_heads_ws, qkv_heads_bs)):
            for j, (w, b) in enumerate(zip(qkv_w, qkv_b)):
                save_weight(w)
                save_weight(b)

    def save_attn(weights):
        save_qkv(weights['c_attn'])
        save_fc(weights['c_proj'])

    def save_mlp(weights):
        save_fc(weights['c_fc'])
        save_fc(weights['c_proj'])

    save_weight(params['wpe'])
    save_weight(params['wte'])

    for i in range(len(params['blocks'])):
        block = params['blocks'][i]
        save_ln(block['ln_1'])
        save_attn(block['attn'])
        save_ln(block['ln_2'])
        save_mlp(block['mlp'])

    save_ln(params['ln_f'])
