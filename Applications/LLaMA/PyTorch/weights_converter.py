"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>

@file weights_converter.py
@date 13 October 2023
@this script is tested on transformers 4.30.2

@author Seungbaek Hong <sb92.hong@samsung.com>
@author Eunju Yang <ej.yang@samsung.com>
"""

import torch
import numpy as np
from transformers import LlamaForCausalLM, AutoConfig


def save_llama_for_nntrainer(params, n_layers, file, dtype):  
    """
    @brief convert and save weights as nntrainer format for multi-head attention model
    """
      
    def save_weight(weight):
        np.array(weight, dtype=dtype).tofile(file)  

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(params[f"{layer_name}{proj_name}.base_layer.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_A.default.weight"].permute(1, 0))  
            save_weight(params[f"{layer_name}{proj_name}.lora_B.default.weight"].permute(1, 0))  
        else:  
            save_weight(params[f"{layer_name}{proj_name}.weight"].permute(1, 0))  

    def save_attention(layer_name):  
        """Save attention layer weights"""  
        save_weight(params[f"{layer_name}input_layernorm.weight"])  
          
        # Save Q/K/V/O projections using helper  
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:  
            save_projection(layer_name, f"self_attn.{proj}")  

    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        save_weight(params[f"{layer_name}post_attention_layernorm.weight"])  
          
        # Save MLP projections using helper  
        for proj in ["up_proj", "gate_proj", "down_proj"]:  
            save_projection(layer_name, f"mlp.{proj}")  

    # Save embedding layer  
    save_weight(params["model.embed_tokens.weight"])  

    # Process all layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        save_attention(layer_prefix)  
        save_feed_forward(layer_prefix)  

    # Save final layers  
    save_weight(params["model.norm.weight"])  
    save_weight(params["lm_head.weight"].permute(1, 0))  

if __name__ == "__main__":
    model_name_or_path = "./"
    data_dtype = "float16"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    config = AutoConfig.from_pretrained(model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        model_name_or_path, torch_dtype='auto', device_map=device
    )

    with open("./llama_fp16.bin", "wb") as f:
        save_llama_for_nntrainer(model.state_dict(), config.num_hidden_layers, f, data_dtype)
