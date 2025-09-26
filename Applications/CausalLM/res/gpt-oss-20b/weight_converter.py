"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Eunju Yang <ej.yang@samsung.com>

@file weights_converter.py
@date 08 May 2025
@this script is tested on transformers 4.53.2
@brief gpt-oss-20b
@author Eunju Yang <ej.yang@samsung.com>


@note

- weight(FP32/FP16) should be transposed
- vector should not be transposed

-expected save order
   * embed_tokens.weight
   * model.layers
       model.layers.0
               model.layers.0.input_layernorm.weight
               model.layers.0.self_attn
                   model.layers.0.self_attn.q_proj (bias = true)
                   model.layers.0.self_attn.k_proj (bias = true)
                   model.layers.0.self_attn.v_proj (bias = true)
                   model.layers.0.self_attn.sinks
                   model.layers.0.self_attn.o_proj (bias = true)
               model.layers.0.experts
                   model.layers.0.mlp.experts.router (bias = true)
                   model.layers.0.mlp.experts.0 (enforce to split)
                       model.layers.0.mlp.experts.0.up_proj (split from gate_up_proj) (bias = true)
                       model.layers.0.mlp.experts.0.gate_proj (split from gate_up_proj) (bias = true)
                       model.layers.0.mlp.experts.0.down_proj (bias = true)
"""

import torch
import numpy as np
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

total_size = 0
def save_gpt_oss_for_nntrainer(params, config, dtype, file):  
    """Convert and save weights as nntrainer format for multi-head attention model"""  

    n_layers = config.num_hidden_layers
    n_experts = config.num_local_experts

    print(dtype)
      
    def save_weight(weight_name, is_transpose=False):
        
        if is_transpose:
            print(weight_name, params[weight_name].permute(1,0).shape, params[weight_name].permute(1,0).flatten()[:3], params[weight_name].permute(1,0).flatten()[-3:])
            np.array(params[weight_name].permute(1,0).float(), dtype=dtype).tofile(file)  
        else:
            print(weight_name, params[weight_name].shape, params[weight_name].flatten()[:3], params[weight_name].flatten()[-3:])
            np.array(params[weight_name].float(), dtype=dtype).tofile(file)  

    def save_projection(layer_name, proj_name):  
        """Helper function to handle base/lora weight saving"""  
        lora_key = f"{layer_name}{proj_name}.lora_A.default.weight"  
        if lora_key in params:  
            save_weight(f"{layer_name}{proj_name}.base_layer.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_A.default.weight", True)
            save_weight(f"{layer_name}{proj_name}.lora_B.default.weight", True)  
        else:  
            save_weight(f"{layer_name}{proj_name}.weight", True)  

    def save_attention(layer_name):  
        """Save attention layer weights"""  
          
        # Save Q/K/V projections using helper  
        for proj in ["q_proj", "k_proj", "v_proj"]:  
            # save projection weight
            save_projection(layer_name, f"self_attn.{proj}")  
            # save projection bias
            save_weight(f"{layer_name}self_attn.{proj}.bias")
        
        # save attention sink
        save_weight(f"{layer_name}self_attn.sinks")
        
        # Save O_proj projections using helper  
        for proj in ["o_proj"]:
            # save projection weight
            save_projection(layer_name, f"self_attn.{proj}")  
            # save projection bias
            save_weight(f"{layer_name}self_attn.{proj}.bias")
        
    def save_feed_forward(layer_name):  
        """Save feed forward layer weights"""  
        
        save_weight(f"{layer_name}mlp.router.weight", True)  
        save_weight(f"{layer_name}mlp.router.bias")  
          
        # Save MoE projections using helper  
        for num_expert in range(n_experts):
            # save up_proj (weight should not be transposed)
            up_proj_weight = params[f"{layer_name}mlp.experts.gate_up_proj"][...,1::2][num_expert]
            up_proj_bias = params[f"{layer_name}mlp.experts.gate_up_proj_bias"][...,1::2][num_expert]
            print(f"{layer_name}mlp.experts.gate_up_proj.up", up_proj_weight.shape, up_proj_weight[0,:3], up_proj_weight[-1,-3:])
            print(f"{layer_name}mlp.experts.gate_up_proj.up_bias", up_proj_bias.shape, up_proj_bias[:3], up_proj_bias[-3:])
            np.array(up_proj_weight.float(), dtype=dtype).tofile(file)  
            np.array(up_proj_bias.float(), dtype=dtype).tofile(file)
            
            # save gate_proj
            gate_proj_weight = params[f"{layer_name}mlp.experts.gate_up_proj"][...,::2][num_expert]
            gate_proj_bias = params[f"{layer_name}mlp.experts.gate_up_proj_bias"][...,::2][num_expert]
            print(f"{layer_name}mlp.experts.gate_up.gate_proj", gate_proj_weight.shape, gate_proj_weight[0,:3], gate_proj_weight[-1,-3:])
            print(f"{layer_name}mlp.experts.gate_up_proj.gate_bias", gate_proj_bias.shape, gate_proj_bias[:3], gate_proj_bias[-3:])
            np.array(gate_proj_weight.float(), dtype=dtype).tofile(file)  
            np.array(gate_proj_bias.float(), dtype=dtype).tofile(file)
            
            # save down_proj
            down_proj_weight = params[f"{layer_name}mlp.experts.down_proj"][num_expert]
            down_proj_bias = params[f"{layer_name}mlp.experts.down_proj_bias"][num_expert]
            print(f"{layer_name}mlp.experts.down_proj", down_proj_weight.shape, down_proj_weight[0,:3], down_proj_weight[-1,-3:])
            print(f"{layer_name}mlp.experts.down_proj_bias", down_proj_bias.shape, down_proj_bias[:3], down_proj_bias[-3:])
            np.array(down_proj_weight.float(), dtype=dtype).tofile(file)  
            np.array(down_proj_bias.float(), dtype=dtype).tofile(file)
            

            

    ####################################################################
    # Save embedding layer  
    save_weight("model.embed_tokens.weight")  

    # Process all layers  
    for layer_idx in range(n_layers):  
        layer_prefix = f"model.layers.{layer_idx}."  
        save_weight(f"{layer_prefix}input_layernorm.weight")  
        save_attention(layer_prefix)  
        save_weight(f"{layer_prefix}post_attention_layernorm.weight")  
        save_feed_forward(layer_prefix)  

    # Save final layers  
    save_weight("model.norm.weight")  
    save_weight("lm_head.weight", True)


if __name__ == "__main__":
    data_dtype = "float32"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model_path = "."
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    config = AutoConfig.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=data_dtype, trust_remote_code=True)
    model.eval()


    with open("./nntr_gpt_oss_20b.bin", "wb") as f_model :
        save_gpt_oss_for_nntrainer(model.state_dict(), config, data_dtype, f_model)
