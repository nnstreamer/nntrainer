#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare nntrainer's on-device LoRA gradients against a PyTorch reference.

Input is a dump directory written by a Qwen3 LoRA training run with
NNTR_TRAIN_GRAD_DUMP=<dir> (see LORA_TRAINING.md, "Verifying training
numerics"):

  samples.txt                      token ids of each produced sample, in order
  layer<L>_<fc>_it<k>_{A,B,dA,dB}  pre-update loraA / loraB and their grads
  hidden_<k>.bin                   (hybrid runs) the GPU trunk's hidden state

For each of the first --iters steps, the reference rebuilds the step in FP32
on the HF checkpoint: base FC weights optionally replaced by nntrainer's QS4CX
quantize-dequantize (--base qs4cx), the dumped A / B patched into the LoRA
layers, the loss taken on the last token (input ids[:-1], label ids[-1]), and
backpropagated. With --hybrid (default: auto when hidden_0.bin exists) the
dumped trunk hidden state replaces the input of the first LoRA layer, so the
comparison does not depend on how closely the reference reproduces the trunk.

Run the device side with NNTR_TRAIN_FWD_FP32=1 NNTR_TRAIN_DX_Q8=0 for an exact
comparison (FP32 forward and dX); expect rel L2 well below 1% then. With the
default integer kernels expect a few percent (per-row int8 activations).
"""

import argparse
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM

FC_NAMES = {
    "wq": "q_proj",
    "wk": "k_proj",
    "wv": "v_proj",
    "attention_out": "o_proj",
    "ffn_gate": "gate_proj",
    "ffn_up": "up_proj",
    "ffn_down": "down_proj",
}


def qs4cx_roundtrip(w):
    """nntrainer QS4CX quantize + dequantize (rows = output channels)."""
    w = w.astype(np.float32)
    mn = np.minimum(np.float32(0), w.min(1, keepdims=True))
    mx = np.maximum(np.float32(0), w.max(1, keepdims=True))
    scale0 = np.where(mx == mn, np.float32(1),
                      np.float32(15) / (mx - mn)).astype(np.float32)
    v = (w * scale0).astype(np.float32)
    q = np.clip(np.sign(v) * np.floor(np.abs(v) + np.float32(0.5)), -8, 7)
    recip = np.where(scale0 != 0, np.float32(1) / scale0, np.float32(0))
    return (q.astype(np.float32) * recip.astype(np.float32)).astype(np.float32)


def parse_layers(text):
    if "-" in text:
        lo, hi = text.split("-")
        return list(range(int(lo), int(hi) + 1))
    return [int(x) for x in text.split(",")]


def load(path, shape):
    return torch.from_numpy(np.fromfile(path, dtype=np.float32).reshape(shape))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--hf", required=True, help="HF Qwen3 checkpoint dir")
    ap.add_argument("--dump", required=True, help="NNTR_TRAIN_GRAD_DUMP dir")
    ap.add_argument("--layers", default="21-27",
                    help="LoRA layers, e.g. 21-27 or 27")
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--base", choices=["qs4cx", "fp32"], default="qs4cx",
                    help="base FC weights the device model was built from")
    ap.add_argument("--hybrid", choices=["auto", "on", "off"], default="auto")
    ap.add_argument("--verbose", action="store_true",
                    help="print every tensor instead of the summary table")
    args = ap.parse_args()

    torch.set_num_threads(os.cpu_count() or 1)
    layers = parse_layers(args.layers)
    rank, scaling = args.rank, args.alpha / args.rank
    hybrid = (args.hybrid == "on" or
              (args.hybrid == "auto" and
               os.path.exists(f"{args.dump}/hidden_0.bin")))

    model = AutoModelForCausalLM.from_pretrained(args.hf,
                                                 dtype=torch.float32).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if args.base == "qs4cx":
        for blk in model.model.layers:
            for hf_name in FC_NAMES.values():
                m = getattr(blk.self_attn, hf_name, None) or getattr(
                    blk.mlp, hf_name)
                m.weight.data = torch.from_numpy(
                    qs4cx_roundtrip(m.weight.data.numpy()))

    mods = {}
    for L in layers:
        blk = model.model.layers[L]
        for key, hf_name in FC_NAMES.items():
            mods[(L, key)] = getattr(blk.self_attn, hf_name, None) or getattr(
                blk.mlp, hf_name)

    lora = {}

    def patch(mod, key):
        def fwd(x):
            A, B = lora[key]
            return torch.nn.functional.linear(x, mod.weight) + scaling * (
                (x @ A) @ B)

        mod.forward = fwd

    for key, m in mods.items():
        patch(m, key)

    hidden = {}
    if hybrid:
        def pre_hook(_mod, hook_args, kwargs):
            h = hidden["h"]
            if hook_args:
                assert hook_args[0].shape == h.shape, (hook_args[0].shape,
                                                       h.shape)
                return (h,) + tuple(hook_args[1:]), kwargs
            kwargs = dict(kwargs)
            kwargs["hidden_states"] = h
            return hook_args, kwargs

        model.model.layers[layers[0]].register_forward_pre_hook(
            pre_hook, with_kwargs=True)

    samples = [list(map(int, line.split()))
               for line in open(f"{args.dump}/samples.txt")]
    print(f"layers {layers[0]}..{layers[-1]}, rank {rank}, scaling {scaling}, "
          f"base {args.base}, hybrid {hybrid}")

    for it in range(1, args.iters + 1):
        ids = samples[it - 1]
        if hybrid:
            hidden["h"] = load(f"{args.dump}/hidden_{it - 1}.bin",
                               (1, len(ids) - 1, -1))
        for (L, key), m in mods.items():
            stem = f"{args.dump}/layer{L}_{key}_it{it}"
            A = load(f"{stem}_A.bin", (m.in_features, rank)).requires_grad_()
            B = load(f"{stem}_B.bin", (rank, m.out_features)).requires_grad_()
            lora[(L, key)] = (A, B)
        logits = model(torch.tensor([ids[:-1]])).logits[0, -1]
        loss = torch.nn.functional.cross_entropy(logits[None],
                                                 torch.tensor([ids[-1]]))
        loss.backward()
        print(f"iter {it}: tokens {len(ids) - 1}, label {ids[-1]}, "
              f"reference loss {loss.item():.5f}")

        table = {}
        for (L, key), m in mods.items():
            A, B = lora[(L, key)]
            for tag, grad, shape in (("dA", A.grad, (m.in_features, rank)),
                                     ("dB", B.grad, (rank, m.out_features))):
                ref = grad.numpy()
                got = np.fromfile(f"{args.dump}/layer{L}_{key}_it{it}_{tag}.bin",
                                  dtype=np.float32).reshape(shape)
                nr, ng = np.linalg.norm(ref), np.linalg.norm(got)
                rel = np.linalg.norm(got - ref) / nr if nr > 0 else ng
                cos = (float((got * ref).sum() / (ng * nr))
                       if nr > 0 and ng > 0 else float("nan"))
                table[(L, key, tag)] = rel
                if args.verbose:
                    print(f"   L{L} {key:14s} {tag}: |ref| {nr:.3e} "
                          f"|nntr| {ng:.3e} rel_l2 {rel:.3e} cos {cos:.5f}")
        if not args.verbose:
            for tag in ("dB", "dA"):
                print(f"   {tag} rel L2 (%)  " +
                      " ".join(f"{k[:8]:>8s}" for k in FC_NAMES))
                for L in layers:
                    print(f"     L{L:<10d}  " + " ".join(
                        f"{100 * table[(L, k, tag)]:8.2f}" for k in FC_NAMES))


if __name__ == "__main__":
    main()
