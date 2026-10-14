#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
#
## @file   weight_converter_layergraph.py
## @brief  HF Qwen3.5-0.8B -> nntrainer layer-graph .bin converter.
## @author Seungbaek Hong <sb92.hong@samsung.com>
##
## numpy-only converter: HF Qwen3.5 (model.safetensors, multimodal wrapper --
## text tensors under "model.language_model.*") -> nntrainer layer-graph
## positional .bin in the qwen3.5 graph forward order. Same pipeline role as
## res/qwen3/qwen3-0.6b/weight_converter_layergraph.py (STEP 1 of the QS4CX
## pipeline; see build_qs4cx.sh here).
##
## Graph order (positional .bin is read in graph forward order):
##   embed_tokens                         (as-is [vocab, hidden])
##   for each layer, in layer_types order:
##     full_attention:
##       input_layernorm                  (gamma' = 1 + w)
##       wq      (q half of the wide q_proj; rows permuted; [in, out])
##       q_norm  (gamma' = 1 + w; rows permuted)
##       wk      (rows permuted; [in, out])
##       k_norm  (gamma' = 1 + w; rows permuted)
##       wv      ([in, out])
##       wgate   (gate half of the wide q_proj; rows permuted; [in, out])
##       wo      (input rows inverse-permuted; [in, out])
##       post_attention_layernorm         (gamma')
##       ffn_gate / ffn_up / ffn_down     ([in, out])
##     linear_attention:
##       input_layernorm                  (gamma')
##       linattn_in_qkv                   ([in, 6144]; q|k|v thirds)
##       linattn_conv                     ([4, 6144]; kernel rows REVERSED:
##                                         nntrainer row 0 = current token)
##       linattn_in_z / in_b / in_a       ([in, out])
##       gdelta_A_log / gdelta_dt_bias    (raw FP32, no fold)
##       gdelta_norm_w                    (gated-norm weight, RAW -- the
##                                         layer adds no 1.0; NOT folded)
##       linattn_out                      ([in, out])
##       post_attention_layernorm, ffn_gate / ffn_up / ffn_down
##   model.norm                           (gamma')
##   (lm_head is tied -> not written)
##
## RoPE channel permutation (full attention only). HF rotates the first 64
## channels of each 256-dim head in pairs (j, j+32) with frequency
## theta^(-2j/64); nntrainer mha_core rotates pairs (c, c+128) with
## thetas[c] = theta^(-2c/64) for c < 32 and theta = 0 (pass-through) for
## c in [32, 128). The converter renames HF channel r to nntrainer channel
## pi(r) within each head:
##     r in [0, 32)      -> r
##     r in [32, 64)     -> r + 96
##     r in [64, 160)    -> r - 32
##     r in [160, 256)   -> r
## (generalized to any partial_rotary_factor; the text-only 1D positions
## make Qwen3.5's interleaved mrope a no-op, so a plain partial rope is
## exact). pi is applied to wq/wk output rows and q/k norm gammas ONLY:
## the attention OUTPUT channels stay in HF order (v is unpermuted and the
## attention weights are invariant to a shared q/k permutation), so the
## gate (elementwise with the output) and the wo rows are UNPERMUTED.
##
## Fold: Qwen3.5 RMSNorm computes output*(1.0 + weight); nntrainer rms_norm
## computes output*weight, so gammas are stored as 1 + w. The linear-attn
## gated norm inside gdelta core is the raw-weight exception (see above).
import argparse
import json
import os
import struct

import numpy as np


def load_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    data_offset = 8 + n
    return hdr, data_offset


def open_model_dir(model_dir):
    """Return a tensor-name -> reader(shard) map for single or sharded
    checkpoints. A sharded index.json maps every tensor to its shard file;
    a single model.safetensors covers everything."""
    single = f"{model_dir}/model.safetensors"
    if os.path.exists(single):
        hdr, off = load_safetensors_header(single)
        return {"single": (single, hdr, off)}, None
    idx_path = f"{model_dir}/model.safetensors.index.json"
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]
    shards = {}
    for name, fname in weight_map.items():
        if fname not in shards:
            path = f"{model_dir}/{fname}"
            hdr, off = load_safetensors_header(path)
            shards[fname] = (path, hdr, off)
    return shards, weight_map


def make_reader(shards, weight_map):
    def resolve(name):
        # The multimodal wrapper names text tensors "model.language_model.*;
        # a bare text checkpoint drops the "language_model." segment.
        for cand in (f"model.language_model.{name}", f"model.{name}", name):
            if weight_map is None:
                path, hdr, off = shards["single"]
                if cand in hdr:
                    return path, hdr, off, cand
            elif cand in weight_map:
                path, hdr, off = shards[weight_map[cand]]
                return path, hdr, off, cand
        raise KeyError(f"tensor not found: {name}")

    DTYPE = {
        "F32": (np.float32, 4),
        "F16": (np.float16, 2),
        "BF16": (np.uint16, 2),  # widened to fp32 below
        "I8": (np.int8, 1),
    }

    def read(name):
        path, hdr, off, name = resolve(name)
        meta = hdr[name]
        dt = meta["dtype"]
        shape = meta["shape"]
        beg, end = meta["data_offsets"]
        np_dt, _ = DTYPE[dt]
        with open(path, "rb") as f:
            f.seek(off + beg)
            raw = f.read(end - beg)
        arr = np.frombuffer(raw, dtype=np_dt)
        if dt == "BF16":
            arr = (arr.astype(np.uint32) << 16).view(np.float32)
        else:
            arr = arr.astype(np.float32)
        return arr.reshape(shape)

    return read


def build_rope_perm(head_dim, partial_factor):
    """HF channel -> nntrainer channel, within one head (see module doc)."""
    rot_dim = int(head_dim * partial_factor)
    npair = rot_dim // 2
    half_hd = head_dim // 2
    assert 0 < npair <= half_hd and rot_dim % 2 == 0
    perm = np.arange(head_dim, dtype=np.int64)
    for r in range(npair, 2 * npair):
        perm[r] = r + (half_hd - npair)
    pass_n = head_dim - rot_dim
    for r in range(rot_dim, rot_dim + pass_n // 2):
        perm[r] = r - rot_dim + npair
    # r in [rot_dim + pass_n/2, head_dim) stays identity.
    assert sorted(perm.tolist()) == list(range(head_dim)), "pi must biject"
    iperm = np.argsort(perm)  # nntrainer channel -> HF channel
    assert (perm[iperm] == np.arange(head_dim)).all()
    return iperm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True,
                    help="HF model dir containing model.safetensors + config.json")
    ap.add_argument("--output_name", required=True)
    ap.add_argument("--data_type", choices=["float32", "float16"],
                    default="float32")
    ap.add_argument("--norm_fp16", action="store_true",
                    help="write RMSNorm gammas as FP16 (mixed bin for the "
                         "FP32-weight / FP16-activation quantize path)")
    args = ap.parse_args()

    out_dtype = np.float32 if args.data_type == "float32" else np.float16
    norm_fp16 = args.norm_fp16

    cfg = json.load(open(f"{args.model_path}/config.json"))
    text = cfg.get("text_config", cfg)
    n_layers = text["num_hidden_layers"]
    layer_types = text["layer_types"]
    hidden = text["hidden_size"]
    heads = text["num_attention_heads"]
    kv_heads = text["num_key_value_heads"]
    head_dim = text.get("head_dim", hidden // heads)
    lin_heads = text["linear_num_value_heads"]
    lin_dim = text["linear_key_head_dim"]
    conv_k = text["linear_conv_kernel_dim"]
    intermediate = text["intermediate_size"]
    tied = bool(text.get("tie_word_embeddings",
                         cfg.get("tie_word_embeddings", False)))
    partial = text.get("rope_parameters", {}).get("partial_rotary_factor", 1.0)

    assert len(layer_types) == n_layers
    assert lin_heads == text["linear_num_key_heads"], "k/v head split NYI"

    iperm = build_rope_perm(head_dim, partial)  # nntr channel -> HF channel

    shards, weight_map = open_model_dir(args.model_path)
    read = make_reader(shards, weight_map)

    total = [0]
    f = open(args.output_name, "wb")

    def w(arr, is_rms=False):
        dt = np.float16 if (is_rms and norm_fp16) else out_dtype
        a = np.ascontiguousarray(arr, dtype=dt)
        a.tofile(f)
        total[0] += a.nbytes

    def fold(g):
        """Qwen3.5 RMSNorm: output*(1.0 + weight) -> store 1 + w."""
        return 1.0 + g

    def fold_perm(g):
        """Fold AND apply the per-head rope permutation (q/k norm gammas,
        laid out [heads, head_dim] in HF channel order)."""
        gv = fold(g).reshape(-1, head_dim)
        return np.ascontiguousarray(gv[:, iperm]).reshape(-1)

    def perm_fc(x3):
        """Permute the per-head channel axis of [heads, head_dim, in] by
        iperm (nntr channel c <- HF channel iperm[c]) and return the
        nntrainer FC weight [in, heads*head_dim]."""
        xp = np.ascontiguousarray(x3[:, iperm, :])
        return np.ascontiguousarray(
            xp.reshape(x3.shape[0] * head_dim, x3.shape[2]).T)

    # embedding (as-is)
    emb = read("embed_tokens.weight")
    assert emb.shape == (text["vocab_size"], hidden)
    w(emb)

    for i in range(n_layers):
        lp = f"layers.{i}."
        w(fold(read(lp + "input_layernorm.weight")), is_rms=True)

        if layer_types[i] == "full_attention":
            sa = lp + "self_attn."
            q = read(sa + "q_proj.weight")
            assert q.shape == (heads * head_dim * 2, hidden)
            qv = q.reshape(heads, head_dim * 2, hidden)
            # q half (rotated channels): permute, store [in, out]
            w(perm_fc(np.ascontiguousarray(qv[:, :head_dim, :])))
            w(fold_perm(read(sa + "q_norm.weight")))
            k = read(sa + "k_proj.weight")
            assert k.shape == (kv_heads * head_dim, hidden)
            w(perm_fc(np.ascontiguousarray(
                k.reshape(kv_heads, head_dim, hidden))))
            w(fold_perm(read(sa + "k_norm.weight")))
            v = read(sa + "v_proj.weight")
            assert v.shape == (kv_heads * head_dim, hidden)
            w(np.ascontiguousarray(v.T))
            # gate half: elementwise with the attention output, whose
            # channels are in HF order (v unpermuted; attention weights
            # invariant to a shared q/k permutation) -> UNPERMUTED.
            w(np.ascontiguousarray(
                qv[:, head_dim:, :].reshape(heads * head_dim, hidden).T))
            o = read(sa + "o_proj.weight")
            assert o.shape == (hidden, heads * head_dim)
            # wo consumes the attention output in HF channel order ->
            # UNPERMUTED [in, out].
            w(np.ascontiguousarray(o.T))
        else:
            la = lp + "linear_attn."
            qkv = read(la + "in_proj_qkv.weight")
            assert qkv.shape == (lin_heads * lin_dim * 3, hidden)
            w(np.ascontiguousarray(qkv.T))
            conv = read(la + "conv1d.weight")
            assert conv.shape == (lin_heads * lin_dim * 3, 1, conv_k)
            # Reverse the kernel: nntrainer row 0 applies to the current
            # token; HF F.conv1d w[j] applies to x[t-(K-1)+j].
            w(np.ascontiguousarray(conv[:, 0, :].T[::-1, :]))
            z = read(la + "in_proj_z.weight")
            assert z.shape == (lin_heads * lin_dim, hidden)
            w(np.ascontiguousarray(z.T))
            for nm in ("in_proj_b", "in_proj_a"):
                t = read(la + nm + ".weight")
                assert t.shape == (lin_heads, hidden)
                w(np.ascontiguousarray(t.T))
            a_log = read(la + "A_log")
            assert a_log.shape == (lin_heads,)
            w(a_log)  # raw FP32
            dt_bias = read(la + "dt_bias")
            assert dt_bias.shape == (lin_heads,)
            w(dt_bias)  # raw FP32
            nw = read(la + "norm.weight")
            assert nw.shape == (lin_dim,)
            w(nw)  # gated-norm weight stays RAW (core adds no 1.0)
            outp = read(la + "out_proj.weight")
            assert outp.shape == (hidden, lin_heads * lin_dim)
            w(np.ascontiguousarray(outp.T))

        w(fold(read(lp + "post_attention_layernorm.weight")), is_rms=True)
        gate = read(lp + "mlp.gate_proj.weight")
        assert gate.shape == (intermediate, hidden)
        w(np.ascontiguousarray(gate.T))
        up = read(lp + "mlp.up_proj.weight")
        assert up.shape == (intermediate, hidden)
        w(np.ascontiguousarray(up.T))
        down = read(lp + "mlp.down_proj.weight")
        assert down.shape == (hidden, intermediate)
        w(np.ascontiguousarray(down.T))

    w(fold(read("norm.weight")), is_rms=True)

    if not tied:
        lm = read("lm_head.weight")
        w(np.ascontiguousarray(lm.T))

    f.close()
    n_full = sum(1 for t in layer_types if t == "full_attention")
    print(f"wrote {args.output_name}: {total[0]} bytes "
          f"({total[0]/1e9:.2f} GB), {n_layers} layers "
          f"({n_full} full / {n_layers - n_full} linear), "
          f"dtype={args.data_type}, norm_fp16={norm_fp16}, tied={tied}")


if __name__ == "__main__":
    main()
