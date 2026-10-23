# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_qwen3_embedding_reference.py
## @brief Generate golden fixtures for the Qwen3Embedding differential test.
##
## Builds a tiny Qwen3 model (matching the C++ tiny config) with a fixed seed,
## saves its weights via the existing weight_converter (no lm_head — embedding
## models are headless), then runs a HuggingFace forward pass through the base
## model, applies last-token pooling + L2 normalization (matching nntrainer's
## SentenceTransformer module pipeline), and saves the resulting embedding as a
## JSON fixture.
##
## Usage:
##   python3 generate_qwen3_embedding_reference.py [--out <dir>] [--seed <int>]
##
## Default output directory:
##   test/unittest/models/causallm_reference/qwen3_embedding_tiny/
##
## Requirements: torch >= 2.0, transformers >= 4.51 (Qwen3 support)

import argparse
import json
import pathlib
import sys

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

THIS_DIR = pathlib.Path(__file__).resolve().parent
# generators -> causallm_reference -> models -> unittest -> test -> repo root
REPO_ROOT = THIS_DIR.parents[4]
CONVERTER_DIR = (REPO_ROOT / "Applications" / "CausalLM" / "res" / "qwen3"
                 / "qwen3-0.6b")
DEFAULT_OUT = THIS_DIR.parent / "qwen3_embedding_tiny"

sys.path.insert(0, str(CONVERTER_DIR))
from weight_converter import save_qwen3_for_nntrainer  # noqa: E402

# Tiny model dimensions — must match the C++ tiny config.
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
    attention_dropout=0.0,
    hidden_act="silu",
)

# Fixed input token IDs and the prompt that the tiny tokenizer maps to them.
#   hello=1, tok4=4, world=2, tok3=3
INPUT_IDS = [1, 4, 2, 3]
PROMPT = "hello tok4 world tok3"

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

# SentenceTransformer module pipeline: Transformer -> Pooling(last-token) ->
# Normalize.  Matches what the nntrainer Qwen3Embedding builds from modules.json.
MODULES_JSON = [
    {"idx": 0, "name": "0", "path": "",
     "type": "sentence_transformers.models.Transformer"},
    {"idx": 1, "name": "1", "path": "1_Pooling",
     "type": "sentence_transformers.models.Pooling"},
    {"idx": 2, "name": "2", "path": "",
     "type": "sentence_transformers.models.Normalize"},
]

POOLING_CONFIG = {
    "word_embedding_dimension": TINY_CONFIG["hidden_size"],
    "pooling_mode_cls_token": False,
    "pooling_mode_mean_tokens": False,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_weightedmean_tokens": False,
    "pooling_mode_lasttoken": True,
    "include_prompt": True,
}


def build_model(seed: int) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_CONFIG))
    model.eval()
    return model


def convert_weights(model: Qwen3ForCausalLM, bin_path: pathlib.Path) -> None:
    params = model.state_dict()
    with open(bin_path, "wb") as f:
        # tie=True → headless save (no lm_head), exactly what an embedding model
        # graph expects.
        save_qwen3_for_nntrainer(params, TINY_CONFIG["num_hidden_layers"],
                                 "float32", f, True)
    print(f"[converter] saved {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")


def run_embedding(model: Qwen3ForCausalLM, input_ids: list) -> list:
    """Last-token pooling + L2 normalization over the base model output."""
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.model(ids, use_cache=False)
    hidden = out.last_hidden_state[0]      # [seq_len, hidden]
    pooled = hidden[-1]                     # last-token pooling
    emb = pooled / pooled.norm(p=2)         # L2 normalize
    return emb.float().tolist()


def write_configs(out_dir: pathlib.Path, bin_name: str,
                  tokenizer_path: str) -> None:
    config_json = {
        "architectures": ["Qwen3Embedding"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "head_dim": 8,
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
        "model_type": "Embedding",
        "module_config_path": "modules.json",
        "num_to_generate": 1,
        "tokenizer_file": tokenizer_path,
        "is_causal": True,
    }
    with open(out_dir / "config.json", "w") as f:
        json.dump(config_json, f, indent=2)
    with open(out_dir / "generation_config.json", "w") as f:
        json.dump(generation_config_json, f, indent=2)
    with open(out_dir / "nntr_config.json", "w") as f:
        json.dump(nntr_config_json, f, indent=2)


def write_modules(out_dir: pathlib.Path) -> None:
    with open(out_dir / "modules.json", "w") as f:
        json.dump(MODULES_JSON, f, indent=2)
    pooling_dir = out_dir / "1_Pooling"
    pooling_dir.mkdir(parents=True, exist_ok=True)
    with open(pooling_dir / "config.json", "w") as f:
        json.dump(POOLING_CONFIG, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate Qwen3Embedding tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] output dir: {out_dir}")
    print(f"[generate] seed={args.seed}, input_ids={INPUT_IDS}, prompt={PROMPT!r}")

    model = build_model(args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_qwen3_embedding_tiny_fp32.bin"
    convert_weights(model, out_dir / bin_name)

    tokenizer_path = out_dir / "tokenizer.json"
    with open(tokenizer_path, "w") as f:
        json.dump(TINY_TOKENIZER, f, indent=2)

    write_configs(out_dir, bin_name, str(tokenizer_path))
    write_modules(out_dir)
    print("[generate] configs + modules.json written")

    ref_embedding = run_embedding(model, INPUT_IDS)
    with open(out_dir / "reference_embedding.json", "w") as f:
        json.dump(ref_embedding, f)
    print(f"[generate] reference embedding: {len(ref_embedding)} values, "
          f"norm={sum(v * v for v in ref_embedding) ** 0.5:.4f}")

    with open(out_dir / "input_ids.json", "w") as f:
        json.dump(INPUT_IDS, f)

    import transformers
    meta = {
        "seed": args.seed,
        "input_ids": INPUT_IDS,
        "prompt": PROMPT,
        "embedding_atol": 1e-2,
        "cosine_min": 0.999,
        "embedding_atol_q40": 0.1,
        "cosine_min_q40": 0.99,
        "pooling": "lasttoken",
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
