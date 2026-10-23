# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.

## @file generate_embedding_gemma_reference.py
## @brief Generate golden fixtures for the EmbeddingGemma differential test.
##
## Builds a tiny Gemma3 text model (matching the C++ tiny config), saves headless
## weights via the existing gemma3 weight_converter, runs a HuggingFace base
## forward pass, applies mean pooling + L2 normalization (matching nntrainer's
## SentenceTransformer module pipeline), and writes the embedding fixture.
##
## The fixture uses causal attention (is_causal) so the HuggingFace base model
## matches nntrainer numerically without overriding the attention mask. The
## fixture exercises the EmbeddingGemma class' Gemma3 transformer + mean pooling +
## normalize path; the Dense projection heads of the full embeddinggemma model
## are intentionally omitted (linear heads already covered by the round-trip
## test) to keep the reference reproducible.
##
## Usage:
##   python3 generate_embedding_gemma_reference.py [--out <dir>] [--seed <int>]
##
## Requirements: torch >= 2.0, transformers >= 4.50 (Gemma3 support)

import argparse
import json
import pathlib
import sys

import torch
from transformers import Gemma3ForCausalLM
from transformers.models.gemma3 import Gemma3TextConfig

THIS_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[4]
CONVERTER_DIR = REPO_ROOT / "Applications" / "CausalLM" / "res" / "gemma3"
DEFAULT_OUT = THIS_DIR.parent / "embedding_gemma_tiny"

sys.path.insert(0, str(CONVERTER_DIR))
from weight_converter import save_gemma3_for_nntrainer  # noqa: E402

TINY_CONFIG = dict(
    hidden_size=64,
    intermediate_size=64,
    num_hidden_layers=2,
    num_attention_heads=8,
    num_key_value_heads=4,
    head_dim=8,
    # C++ MHA core uses 1/sqrt(head_dim); set query_pre_attn_scalar=head_dim so
    # HF matches the C++ attention scaling.
    query_pre_attn_scalar=8,
    vocab_size=32,
    max_position_embeddings=8,
    rope_theta=1000000.0,
    rms_norm_eps=1e-6,
    sliding_window=4,
    sliding_window_pattern=2,
    tie_word_embeddings=True,
    attention_dropout=0.0,
    hidden_activation="gelu_pytorch_tanh",
)

INPUT_IDS = [1, 4, 2, 3]
PROMPT = "hello tok4 world tok3"

TINY_TOKENIZER = {
    "version": "1.0",
    "truncation": None,
    "padding": None,
    "added_tokens": [
        {"id": 31, "content": "<eos>", "single_word": False, "lstrip": False,
         "rstrip": False, "normalized": False, "special": True}
    ],
    "normalizer": None,
    "pre_tokenizer": {"type": "Whitespace"},
    "post_processor": None,
    "decoder": None,
    "model": {
        "type": "WordLevel",
        "vocab": {
            "<unk>": 0, "hello": 1, "world": 2,
            **{f"tok{i}": i for i in range(3, 31)},
            "<eos>": 31,
        },
        "unk_token": "<unk>",
    },
}

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
    "pooling_mode_mean_tokens": True,
    "pooling_mode_max_tokens": False,
    "pooling_mode_mean_sqrt_len_tokens": False,
    "pooling_mode_weightedmean_tokens": False,
    "pooling_mode_lasttoken": False,
    "include_prompt": True,
}


def build_model(seed: int):
    torch.manual_seed(seed)
    config = Gemma3TextConfig(**TINY_CONFIG)
    model = Gemma3ForCausalLM(config)
    model.eval()
    return model, config


def convert_weights(model, config, bin_path: pathlib.Path) -> None:
    params = model.state_dict()
    with open(bin_path, "wb") as f:
        save_gemma3_for_nntrainer(params, config, "float32", f,
                                  save_lm_head=False)
    print(f"[converter] saved {bin_path} ({bin_path.stat().st_size / 1024:.1f} KB)")


def run_embedding(model, input_ids: list) -> list:
    ids = torch.tensor([input_ids], dtype=torch.long)
    with torch.no_grad():
        out = model.model(ids, use_cache=False)
    hidden = out.last_hidden_state[0]      # [seq_len, hidden]
    pooled = hidden.mean(dim=0)            # mean pooling
    emb = pooled / pooled.norm(p=2)        # L2 normalize
    return emb.float().tolist()


def write_configs(out_dir: pathlib.Path, bin_name: str,
                  tokenizer_path: str) -> None:
    config_json = {
        "architectures": ["Gemma3ForCausalLM"],
        "bos_token_id": 0,
        "eos_token_id": [31],
        "head_dim": 8,
        "hidden_size": 64,
        "intermediate_size": 64,
        "is_causal": True,
        "max_position_embeddings": 8,
        "num_attention_heads": 8,
        "num_hidden_layers": 2,
        "num_key_value_heads": 4,
        "rms_norm_eps": 1e-6,
        "rope_theta": 1000000,
        "sliding_window": 4,
        "sliding_window_pattern": 2,
        "tie_word_embeddings": True,
        "vocab_size": 32,
    }
    generation_config_json = {
        "bos_token_id": 0, "eos_token_id": 31, "do_sample": False,
        "top_k": 1, "top_p": 1.0, "temperature": 1.0,
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
        description="Generate EmbeddingGemma tiny reference fixtures")
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir: pathlib.Path = args.out.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[generate] output dir: {out_dir}")

    model, config = build_model(args.seed)
    print(f"[generate] model params: {sum(p.numel() for p in model.parameters()):,}")

    bin_name = "nntr_embedding_gemma_tiny_fp32.bin"
    convert_weights(model, config, out_dir / bin_name)

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
        "embedding_atol": 5e-2,
        "cosine_min": 0.995,
        "embedding_atol_q40": 0.1,
        "cosine_min_q40": 0.95,
        "pooling": "mean",
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[generate] meta.json written (transformers={transformers.__version__})")

    print(f"\n[generate] done. Commit: {out_dir}")


if __name__ == "__main__":
    main()
