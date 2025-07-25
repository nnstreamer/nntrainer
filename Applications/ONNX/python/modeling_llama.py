"""
SPDX-License-Identifier: Apache-2.0
Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>

@file modeling_llama.py
@date 17 July 2025
@breif The simplest LLaMA modeling file that is compatible with Hugging Face and supports conversion
       from PyTorch to ONNX to NNTrainer. It does not include masking and caching for token generation.
@note This script has been tested with transformers version 4.51.3 and PyTorch version 2.3.1

@author Seungbaek Hong <sb92.hong@samsung.com>
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.modeling_utils import PreTrainedModel
from transformers import LlamaConfig

class LlamaRotaryEmbedding(torch.nn.Module):
    def __init__(self, config: LlamaConfig):
        super().__init__()
        inv_freq, _ = ROPE_INIT_FUNCTIONS["default"](config)        
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()      
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().detach(), emb.sin().detach()


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, hidden_states, variance_epsilon):
        input_dtype = hidden_states.dtype
        variance = hidden_states.pow(2).mean(3, keepdim=True)
        std_dev = torch.sqrt(variance + variance_epsilon)
        hidden_states = hidden_states * torch.pow(std_dev, -1)
        return self.weight * hidden_states.to(input_dtype)    


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :x.shape[-1]]
    return torch.cat((-x2, x1), dim=3)


def apply_rotary_pos_emb(q, k, cos, sin):    
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=config.mlp_bias)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj


class LlamaAttention(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx  # It will be used for managing caching mechanism later
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )

    def forward(self, hidden_states, cos, sin):
        q_hidden_shape = (1, 1, self.config.num_attention_heads, self.head_dim)
        kv_hidden_shape = (1, 1, self.config.num_key_value_heads, self.head_dim)       
        query_states = self.q_proj(hidden_states).view(q_hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(kv_hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(kv_hidden_shape).transpose(1, 2)
        
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(1, 1, 1, self.config.hidden_size).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, layer_idx):
        super().__init__()
        self.self_attn = LlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size)

    def forward(self, hidden_states, cos, sin, variance_epsilon):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states, variance_epsilon)
        hidden_states = self.self_attn.forward(hidden_states, cos, sin)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states, variance_epsilon)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(PreTrainedModel):
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, config.pad_token_id)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(config.hidden_size)        

    def forward(self, input_ids, cos, sin, variance_epsilon):
        hidden_states = self.embed_tokens(input_ids)
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states.reshape((1, 1, 1, self.config.hidden_size)),
                cos=cos,
                sin=sin,
                variance_epsilon=variance_epsilon
            )
        hidden_states = self.norm(hidden_states, variance_epsilon)
        return hidden_states


class NNTrainerLlamaForCausalLM(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, cos, sin, variance_epsilon): 
        last_hidden_state = self.model(
            input_ids = input_ids,
            cos = cos,
            sin = sin,
            variance_epsilon = variance_epsilon
        )
        logits = self.lm_head(last_hidden_state)
        return logits
