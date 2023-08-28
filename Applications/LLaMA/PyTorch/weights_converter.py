# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
# 
# @file weights_converter.py
# @date 13 October 2023
#
# @author Seungbaek Hong <sb92.hong@samsung.com>

import torch
import numpy as np
from transformers import LlamaForCausalLM

##
# @brief convert and save weights as nntrainer format for naive model
def save_llama_for_nntrainer(params, n_layers, n_heads, dim, file, dtype):
    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)

    def save_embedding(weight):
        save_weight(weight)

    def save_attention(weights, layer_name, n_head):        
        save_weight(params[layer_name + 'input_layernorm' + '.weight'])
        split_size = (dim // n_head)
        for head_idx in range(1, n_head+1):            
            st_idx = (dim - split_size * head_idx)
            end_idx = st_idx + split_size
            save_weight(params[layer_name + 'self_attn.v_proj' + '.weight'][st_idx:end_idx, :].permute(1, 0))
            
        for head_idx in range(1, n_head+1):
            st_idx = (dim - split_size * head_idx)
            end_idx = st_idx + split_size
            save_weight(params[layer_name + 'self_attn.k_proj' + '.weight'][st_idx:end_idx, :].permute(1, 0))

        for head_idx in range(1, n_head+1):
            st_idx = (dim - split_size * head_idx)
            end_idx = st_idx + split_size
            save_weight(params[layer_name + 'self_attn.q_proj' + '.weight'][st_idx:end_idx, :].permute(1, 0))
        
        save_weight(params[layer_name + 'self_attn.o_proj' + '.weight'].permute(1, 0))

    def save_feed_forward(weights, layer_name):
        save_weight(params[layer_name + 'post_attention_layernorm' + '.weight'])        
        
        save_weight(params[layer_name + 'mlp.up_proj' + '.weight'].permute(1, 0))
        save_weight(params[layer_name + 'mlp.gate_proj' + '.weight'].permute(1, 0))        
        save_weight(params[layer_name + 'mlp.down_proj' + '.weight'].permute(1, 0))

    # save weights of embedding layer
    save_embedding(params['model.embed_tokens.weight'])
    
    # save weights of attention layers & feed forward layers
    for layer_idx in range(n_layers):
        save_attention(params, 'model.layers.{}.'.format(layer_idx), n_heads)
        save_feed_forward(params, 'model.layers.{}.'.format(layer_idx))

    # save weights of output batch-normalization layer
    save_weight(params['model.norm.weight'])

    # save weights of output fc layer
    save_weight(params['lm_head.weight'].permute(1, 0))

##
# @brief convert and save weights as nntrainer format for multi-head attention model
def save_llama_for_nntrainer_for_mha(params, n_layers, file, dtype):        
    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)

    def save_embedding(weight):
        save_weight(weight)

    def save_attention(weights, layer_name):        
        save_weight(params[layer_name + 'input_layernorm' + '.weight'])
        save_weight(params[layer_name + 'self_attn.q_proj' + '.weight'].permute(1, 0)) 
        save_weight(params[layer_name + 'self_attn.k_proj' + '.weight'].permute(1, 0))
        save_weight(params[layer_name + 'self_attn.v_proj' + '.weight'].permute(1, 0))
        save_weight(params[layer_name + 'self_attn.o_proj' + '.weight'].permute(1, 0))

    def save_feed_forward(weights, layer_name):
        save_weight(params[layer_name + 'post_attention_layernorm' + '.weight'])
        save_weight(params[layer_name + 'mlp.up_proj' + '.weight'].permute(1, 0))
        save_weight(params[layer_name + 'mlp.gate_proj' + '.weight'].permute(1, 0))        
        save_weight(params[layer_name + 'mlp.down_proj' + '.weight'].permute(1, 0))

    # save weights of embedding layer
    save_embedding(params['model.embed_tokens.weight'])
    
    # save weights of attention layers & feed forward layers
    for layer_idx in range(n_layers):
        save_attention(params, 'model.layers.{}.'.format(layer_idx))
        save_feed_forward(params, 'model.layers.{}.'.format(layer_idx))

    # save weights of output batch-normalization layer
    save_weight(params['model.norm.weight'])

    # save weights of output fc layer
    save_weight(params['lm_head.weight'].permute(1, 0))

model_path = '/USR_DIR/MODEL_DIR/'

model = LlamaForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map='cpu'
)

file = open("./llama_v2.bin", "wb")
save_llama_for_nntrainer(model.state_dict(), 28, 18, 1440, file, 'float16')

# file_mha = open("./llama_v2_mha.bin", "wb")
# save_llama_for_nntrainer_for_mha(model.state_dict(), 28, file_mha, 'float16')
