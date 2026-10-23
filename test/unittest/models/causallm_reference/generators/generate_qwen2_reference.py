# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_qwen2_reference.py
## @brief Generate golden fixtures for Qwen2 differential tests.
##
## Creates a tiny Qwen2 model (matching the C++ test dimensions) with a fixed
## seed, converts its weights via the existing weight_converter, runs a
## HuggingFace forward pass and greedy generation, then saves the results as
## JSON fixtures.
##
## Usage:
##   python3 generate_qwen2_reference.py [--out <dir>] [--seed <int>] [--n <int>]
##
## The default output directory is:
##   test/unittest/models/causallm_reference/qwen2_tiny/
##
## Requirements: torch >= 2.0, transformers >= 4.40 (Qwen2 support)

import argparse
import json
import pathlib
import sys

import numpy as np
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

THIS_DIR = pathlib.Path(__file__).resolve().parent
# generators -> causallm_reference -> models -> unittest -> test -> repo root
REPO_ROOT = THIS_DIR.parents[4]
CONVERTER_DIR = (REPO_ROOT / "Applications" / "CausalLM" / "res" / "qwen2"
                 / "qwen2-0.5b")
# Fixtures live one level up, next to this generators/ directory.
DEFAULT_OUT = THIS_DIR.parent / "qwen2_tiny"

sys.path.insert(0, str(CONVERTER_DIR))
from weight_converter import save_qwen2_for_nntrainer  # noqa: E402

# Tiny model dimensions — must match makeTinyQwen2Config() in C++
TINY_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=1,
    num_attention_heads=8,
    num_key_value_heads=4,
    vocab_size=32,
    max_position_embeddings=8,
    rope_theta=10000.0,
    rms_norm_eps=1e-5,
    tie_word_embeddings=True,
    attention_dropout=0.0,
    hidden_act="silu",
)

# Fixed input token IDs — all < vocab_size (32), length == init_seq_len (4)
INPUT_IDS = [1, 4, 2, 3]
N_GEN = 4

# Tiny tokenizer JSON — matches writeTinyTokenizer() in C++
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


def build_model(seed: int) -> Qwen2ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen2ForCausalLM(Qwen2Config(**TINY_CONFIG))
    model.eval()
    return model


def convert_weights(model: Qwen2ForCausalLM, n_layers: int,
                    bin_path: pathlib.Path) -> None:
    params = model.state_dict()
    with open(bin_path, "wb") as f:
        save_qwen2_for_nntrainer(params, n_layers, "float32", f)
    print(f"[converter] saved {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")


def run_forward(model: Qwen2ForCausalLM, input_ids: list[int]) -> list[float]:
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model(ids, use_cache=False)
    return out.logits[0, -1, :].float().tolist()


def run_greedy(model: Qwen2ForCausalLM, input_ids: list[int], n: int) -> list[int]:
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
        "architectures": ["Qwen2ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "hidden_size": 64,
        "intermediate_size": 64,
        "is_causal": True,
        "max_position_embeddings": 8,
        "num_attention_heads": 8,
        "num_hidden_layers": 1,
        "num_key_value_heads": 4,
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
    parser = argparse.ArgumentParser(description="Generate Qwen2 tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n", type=int, default=N_GEN)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[generate] output dir: {out_dir}")

    model = build_model(args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_qwen2_tiny_fp32.bin"
    convert_weights(model, TINY_CONFIG["num_hidden_layers"], out_dir / bin_name)

    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)

    write_nntr_configs(out_dir, bin_name, str(tokenizer_path))

    ref_logits = run_forward(model, INPUT_IDS)
    with open(out_dir / "reference_logits.json", "w") as f:
        json.dump(ref_logits, f)
    print(f"[generate] reference logits: {len(ref_logits)} values, "
          f"argmax={int(np.argmax(ref_logits))}, "
          f"max|logit|={max(abs(v) for v in ref_logits):.4f}")

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
        # Q4_0 tolerance: tune by running the differential test once and reading
        # the reported max deviation; set to (observed max) x safety factor.
        "logits_atol_q40": 5.0,
        "prefix_match_min": 2,
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n[generate] done. Commit: {out_dir}")


if __name__ == "__main__":
    main()
