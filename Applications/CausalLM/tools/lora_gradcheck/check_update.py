#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Check that the optimizer applied the gradients nntrainer computed.

Uses the same NNTR_TRAIN_GRAD_DUMP directory as ref_grad.py. Each step dumps
the pre-update loraA / loraB and their gradients, so the update actually
applied at step k is W(k+1) - W(k). Two checks:

1. First-step direction. loraB starts at zero, so loraA's first gradient is
   zero and loraB's first Adam step is -lr * sign(dB) wherever dB != 0
   (gradient clipping only scales dB, it does not change the sign). The
   fraction of loraB entries moving against their gradient must be 1.000 for
   every LoRA layer; ~0.5 (random) or 0.0 (opposite) means the gradient the
   optimizer read is not the one the layer computed.
2. Full Adam reproduction for steps 1->2 and 2->3: global-norm clipping over
   every dumped LoRA tensor, then nntrainer's Adam (beta1 0.9, beta2 0.999,
   eps 1e-7, bias correction folded into the learning rate). Expect rel L2 at
   float precision (1e-7 .. 1e-4).

Run the device side without an lr schedule (no --lr_decay_rate), and pass the
same --lr / --clip_grad as the run.
"""

import argparse
import glob
import os
import re

import numpy as np

FC_ORDER = ["wq", "wk", "wv", "attention_out", "ffn_gate", "ffn_up",
            "ffn_down"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--lr", type=float, required=True)
    ap.add_argument("--clip", type=float, default=0.0,
                    help="--clip_grad of the run (0 = no clipping)")
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--eps", type=float, default=1e-7)
    args = ap.parse_args()

    def ld(layer, fc, it, tag):
        return np.fromfile(f"{args.dump}/{layer}_{fc}_it{it}_{tag}.bin",
                           dtype=np.float32).astype(np.float64)

    pat = re.compile(r"(layer\d+)_(\w+?)_it1_A\.bin$")
    fcs = sorted({pat.search(os.path.basename(p)).groups()
                  for p in glob.glob(f"{args.dump}/*_it1_A.bin")},
                 key=lambda t: (int(t[0][5:]), FC_ORDER.index(t[1])
                                if t[1] in FC_ORDER else 99))
    if not fcs:
        raise SystemExit(f"no *_it1_A.bin dumps in {args.dump}")

    print("1) first-step direction: fraction of loraB entries moving "
          "against their gradient (must be 1.000)")
    bad = 0
    for layer, fc in fcs:
        g = ld(layer, fc, 1, "dB")
        d = ld(layer, fc, 2, "B") - ld(layer, fc, 1, "B")
        nz = g != 0
        agree = float(np.mean(np.sign(d[nz]) == -np.sign(g[nz]))) if nz.any() \
            else float("nan")
        flag = "" if agree > 0.99 else "   <-- WRONG"
        bad += agree <= 0.99
        print(f"   {layer} {fc:14s} {agree:.3f}{flag}")
    print(f"   wrong: {bad} / {len(fcs)}")

    print("2) Adam + global-norm clip reproduction")
    keys = [(layer, fc, t) for layer, fc in fcs for t in ("A", "B")]
    m = {k: 0.0 for k in keys}
    v = {k: 0.0 for k in keys}
    b1, b2, eps = args.beta1, args.beta2, args.eps
    for it in (1, 2):
        g = {k: ld(k[0], k[1], it, "d" + k[2]) for k in keys}
        gnorm = np.sqrt(sum((g[k] ** 2).sum() for k in keys))
        s = min(1.0, args.clip / gnorm) if args.clip > 0 and gnorm > 0 else 1.0
        lr_t = args.lr * np.sqrt(1 - b2 ** it) / (1 - b1 ** it)
        rels = []
        for k in keys:
            gc = g[k] * s
            m[k] = b1 * m[k] + (1 - b1) * gc
            v[k] = b2 * v[k] + (1 - b2) * gc * gc
            expected = -lr_t * m[k] / (np.sqrt(v[k]) + eps)
            applied = ld(k[0], k[1], it + 1, k[2]) - ld(k[0], k[1], it, k[2])
            n = np.linalg.norm(expected)
            if n > 0:
                rels.append((np.linalg.norm(applied - expected) / n, k))
        rels.sort(reverse=True)
        print(f"   step {it}->{it + 1}: grad norm {gnorm:.3f}, clip scale "
              f"{s:.4f}, median rel {np.median([r for r, _ in rels]):.2e}, "
              f"worst {rels[0][0]:.2e} ({rels[0][1][0]} {rels[0][1][1]} "
              f"{rels[0][1][2]})")


if __name__ == "__main__":
    main()
