# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_qwen3_moe_reference.py
## @brief Generate golden fixtures for Qwen3MoE differential tests.
##
## Creates a tiny Qwen3MoE model (matching the C++ test adapter dimensions)
## with a fixed random seed, converts its weights via the existing
## weight_converter, runs a HuggingFace forward pass and greedy generation,
## then saves the results as JSON fixtures.
##
## Usage:
##   python3 generate_qwen3_moe_reference.py [--out <dir>] [--seed <int>] [--n <int>]
##
## The default output directory is:
##   test/unittest/models/causallm_reference/qwen3_moe_tiny/
## (relative to the nntrainer repo root, resolved from this file's location).
##
## Requirements: torch >= 2.0, transformers >= 4.51 (Qwen3MoE support)

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from transformers import Qwen3MoeForCausalLM, Qwen3MoeConfig

THIS_DIR = pathlib.Path(__file__).resolve().parent
# generators -> causallm_reference -> models -> unittest -> test -> repo root
REPO_ROOT = THIS_DIR.parents[4]
# Fixtures live one level up, next to this generators/ directory.
DEFAULT_OUT = THIS_DIR.parent / "qwen3_moe_tiny"

# Tiny model dimensions — must match makeTinyQwen3MoEConfig() in C++
TINY_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    vocab_size=32,
    max_position_embeddings=8,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    num_experts=4,
    num_experts_per_tok=2,
    # The nntrainer MoE layer always renormalizes the top-k router weights,
    # as the released Qwen3-MoE checkpoints do; the HF config default is False.
    norm_topk_prob=True,
    moe_intermediate_size=64,
    attention_dropout=0.0,
)

INPUT_IDS = [1, 4, 2, 3]
N_GEN = 4

TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {
            "id": 31,
            "content": "<eos>",
            "single_word": False,
            "lstrip": False,
            "rstrip": False,
            "normalized": False,
            "special": True,
        }
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0,
            "hello": 1,
            "world": 2,
            **{f"tok{i}": i for i in range(3, 31)},
            "<eos>": 31,
        },
        "unk_token": "<unk>",
    },
}


def build_tiny_config() -> Qwen3MoeConfig:
    return Qwen3MoeConfig(**TINY_CONFIG)


def build_model(config: Qwen3MoeConfig, seed: int) -> Qwen3MoeForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3MoeForCausalLM(config)
    model.eval()
    return model


def save_qwen3_moe_for_nntrainer(params: dict, config: Qwen3MoeConfig,
                                  dtype: str, f) -> None:
    """Save Qwen3MoE weights in NNTrainer binary format.

    Weight save order matches NNTrainer's graph traversal:
      embed_tokens
      for each layer:
        input_layernorm (rms_norm, no +1)
        q_proj^T, q_norm, k_proj^T, k_norm, v_proj^T, o_proj^T
        post_attention_layernorm (rms_norm)
        MoE gate^T, then for each expert: up^T, gate^T, down^T
      model.norm (rms_norm)
      lm_head^T (if not tied)
    """
    n_layers = config.num_hidden_layers
    n_experts = config.num_experts
    moe_int = config.moe_intermediate_size

    def save(tensor):
        np.array(tensor.detach().float().numpy(), dtype=dtype).tofile(f)

    # 1. Embedding
    save(params["model.embed_tokens.weight"])

    for i in range(n_layers):
        pfx = f"model.layers.{i}."

        # Attention norm
        save(params[f"{pfx}input_layernorm.weight"])

        # Q, q_norm, K, k_norm, V, O
        save(params[f"{pfx}self_attn.q_proj.weight"].T)
        save(params[f"{pfx}self_attn.q_norm.weight"])
        save(params[f"{pfx}self_attn.k_proj.weight"].T)
        save(params[f"{pfx}self_attn.k_norm.weight"])
        save(params[f"{pfx}self_attn.v_proj.weight"].T)
        save(params[f"{pfx}self_attn.o_proj.weight"].T)

        # FFN norm (post_attention_layernorm in Qwen3MoE)
        save(params[f"{pfx}post_attention_layernorm.weight"])

        # MoE layer: router gate, then per-expert weights
        # gate.weight: [num_experts, hidden] → save transposed → [hidden, num_experts]
        save(params[f"{pfx}mlp.gate.weight"].T)

        # gate_up_proj: [num_experts, 2*moe_int, hidden]
        # down_proj:    [num_experts, hidden, moe_int]
        gate_up = params[f"{pfx}mlp.experts.gate_up_proj"]
        down = params[f"{pfx}mlp.experts.down_proj"]

        for e in range(n_experts):
            # Qwen3MoE splits gate_up as [gate; up] along dim 0
            gate_w = gate_up[e, :moe_int, :]     # [moe_int, hidden]
            up_w = gate_up[e, moe_int:, :]        # [moe_int, hidden]
            down_w = down[e]                       # [hidden, moe_int]

            # NNTrainer expert order: up, gate, down (see qwen_moe_layer.cpp finalize)
            save(up_w.T)     # [hidden, moe_int]
            save(gate_w.T)   # [hidden, moe_int]
            save(down_w.T)   # [moe_int, hidden]

    # Final norm
    save(params["model.norm.weight"])

    # lm_head (if not tied)
    tie = getattr(config, "tie_word_embeddings", True)
    if not tie and "lm_head.weight" in params:
        save(params["lm_head.weight"].T)


def convert_weights(model: Qwen3MoeForCausalLM, config: Qwen3MoeConfig,
                    bin_path: pathlib.Path) -> None:
    params = model.state_dict()
    with open(bin_path, "wb") as f:
        save_qwen3_moe_for_nntrainer(params, config, "float32", f)
    print(f"[converter] saved {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")


def run_forward(model: Qwen3MoeForCausalLM, input_ids: list) -> list:
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    return out.logits[0, -1, :].float().tolist()


def run_greedy(model: Qwen3MoeForCausalLM, input_ids: list, n: int) -> list:
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(
            ids,
            max_new_tokens=n,
            do_sample=False,
            temperature=1.0,
            repetition_penalty=1.0,
            use_cache=True,
            eos_token_id=None,
        )
    return out[0, len(input_ids):].tolist()[:n]


def write_nntr_configs(out_dir: pathlib.Path, bin_name: str,
                       tokenizer_path: str) -> None:
    config_json = {
        "architectures": ["Qwen3MoeForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "head_dim": 8,
        "hidden_size": 64,
        "intermediate_size": 64,
        "is_causal": True,
        "max_position_embeddings": 8,
        "moe_intermediate_size": 64,
        "num_attention_heads": 8,
        "num_hidden_layers": 1,
        "num_key_value_heads": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "norm_topk_prob": True,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000,
        "tie_word_embeddings": True,
        "vocab_size": 32,
    }
    generation_config_json = {
        "bos_token_id": 0,
        "eos_token_id": 31,
        "do_sample": False,
        "top_k": 1,
        "top_p": 1.0,
        "temperature": 1.0,
    }
    nntr_config_json = {
        "bad_word_ids": [],
        "batch_size": 1,
        "embedding_dtype": "FP32",
        "fc_layer_dtype": "FP32",
        "init_seq_len": 4,
        "lmhead_dtype": "FP32",
        "max_seq_len": 8,
        "model_file_name": bin_name,
        "model_tensor_type": "FP32-FP32",
        "model_type": "CausalLM",
        "num_to_generate": 1,
        "tokenizer_file": tokenizer_path,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_json, f, indent=2)
    with open(out_dir / "nntr_config.json", "w") as f:
        json.dump(nntr_config_json, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3MoE tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=N_GEN)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] output dir: {out_dir}")
    print(f"[generate] seed={args.seed}, n_gen={args.n}, input_ids={INPUT_IDS}")

    config = build_tiny_config()
    model = build_model(config, args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_qwen3_moe_tiny_fp32.bin"
    bin_path = out_dir / bin_name
    convert_weights(model, config, bin_path)

    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)
    print(f"[generate] tokenizer: {tokenizer_path}")

    write_nntr_configs(out_dir, bin_name, str(tokenizer_path))
    print("[generate] config.json / generation_config.json / nntr_config.json written")

    ref_logits = run_forward(model, INPUT_IDS)
    with open(out_dir / "reference_logits.json", "w") as f:
        json.dump(ref_logits, f)
    argmax_tok = int(np.argmax(ref_logits))
    print(f"[generate] reference logits: {len(ref_logits)} values, argmax={argmax_tok}")

    ref_tokens = run_greedy(model, INPUT_IDS, args.n)
    with open(out_dir / "reference_tokens.json", "w") as f:
        json.dump(ref_tokens, f)
    print(f"[generate] reference tokens: {ref_tokens}")

    with open(out_dir / "input_ids.json", "w") as f:
        json.dump(INPUT_IDS, f)

    import transformers
    meta = {
        "seed": args.seed,
        "n_gen": args.n,
        "input_ids": INPUT_IDS,
        "logits_atol_fp32": 1e-2,
        "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[generate] meta.json written (transformers={transformers.__version__})")

    print("\n[generate] done. Commit the following directory:")
    print(f"  {out_dir}")


if __name__ == "__main__":
    main()
