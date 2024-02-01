"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>

@file weights_converter.py
@date 13 October 2023

@author Seungbaek Hong <sb92.hong@samsung.com>
"""

import torch
import numpy as np
from transformers import LlamaForCausalLM


def save_llama_for_nntrainer(params, n_layers, file, dtype):
    """
    @brief convert and save weights as nntrainer format for multi-head attention model
    """

    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)

    def save_embedding(weight):
        save_weight(weight)

    def save_attention(layer_name):
        save_weight(params[layer_name + "input_layernorm" + ".weight"])
        save_weight(params[layer_name + "self_attn.q_proj" + ".weight"].permute(1, 0))
        save_weight(params[layer_name + "self_attn.k_proj" + ".weight"].permute(1, 0))
        save_weight(params[layer_name + "self_attn.v_proj" + ".weight"].permute(1, 0))
        save_weight(params[layer_name + "self_attn.o_proj" + ".weight"].permute(1, 0))

    def save_feed_forward(layer_name):
        save_weight(params[layer_name + "post_attention_layernorm" + ".weight"])
        save_weight(params[layer_name + "mlp.up_proj" + ".weight"].permute(1, 0))
        save_weight(params[layer_name + "mlp.gate_proj" + ".weight"].permute(1, 0))
        save_weight(params[layer_name + "mlp.down_proj" + ".weight"].permute(1, 0))

    # save weights of embedding layer
    save_embedding(params["model.embed_tokens.weight"])

    # save weights of attention layers & feed forward layers
    for layer_idx in range(n_layers):
        save_attention(f"model.layers.{layer_idx}.")
        save_feed_forward(f"model.layers.{layer_idx}.")

    # save weights of output batch-normalization layer
    save_weight(params["model.norm.weight"])

    # save weights of output fc layer
    save_weight(params["lm_head.weight"].permute(1, 0))


if __name__ == "__main__":
    MODEL_PATH = "/USR_DIR/MODEL_DIR/"

    model = LlamaForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float32, device_map="cpu"
    )

    with open("./llama_v2_mha.bin", "wb") as file_mha:
        save_llama_for_nntrainer(model.state_dict(), 28, file_mha, "float16")
