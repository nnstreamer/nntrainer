#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""PyTorch reference for the RLCD decision model (rlcd-decision-only-v1).

Reads questions as JSONL, one per line, in the eve-rlcd row format:

  {"primitive": "choice", "context": "...", "question": "...",
   "choices": ["A option", "B option", ...], "answer": 1}

renders each with the decision prompt (decision.json "prompt"; the whole
prompt is tokenized and cut from the left to --max_len, as eve-rlcd's
training encode does), runs the Qwen3 body, and prints, per question, the
softmax over the 26 letter logits at the last prompt token masked to the
option count -- the readout nntr_decide reproduces.

--check_head also confirms that decision_head.safetensors equals the tied
embedding rows of the letter tokens, so a nntrainer model can take the head
from its own (tied) embedding.
"""

import argparse
import json
import math

import torch
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

LETTERS = [chr(ord("A") + i) for i in range(26)]


def render_prompt(q):
    lines = ["User: Context:", q["context"].strip(), "",
             f"Question: {q['question'].strip()}", "Options:"]
    for letter, choice in zip(LETTERS, q["choices"]):
        lines.append(f"{letter}) {choice}")
    lines.append("Answer with the letter only.")
    lines.append("Assistant: The answer is")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="decision model directory")
    ap.add_argument("--questions", required=True, help="JSONL questions")
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--check_head", action="store_true")
    args = ap.parse_args()

    meta = json.load(open(f"{args.model}/decision.json"))
    letter_ids = meta["letter_ids"]
    tok = AutoTokenizer.from_pretrained(args.model)
    body = AutoModel.from_pretrained(args.model, torch_dtype=torch.float32)
    body.eval()
    head = load_file(f"{args.model}/{meta['head_file']}")
    head = next(iter(head.values())).float()  # (26, hidden)

    if args.check_head:
        emb = body.get_input_embeddings().weight[letter_ids].float()
        diff = (emb - head).abs().max().item()
        print(f"decision head vs tied embedding rows: max |diff| {diff:.3e}")

    for line in open(args.questions):
        if not line.strip():
            continue
        q = json.loads(line)
        k = len(q["choices"])
        ids = tok.encode(render_prompt(q), add_special_tokens=False)
        ids = ids[-args.max_len:]
        with torch.no_grad():
            h = body(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
            logits = head @ h
        p = torch.softmax(logits[:k], -1).tolist()
        best = max(range(k), key=lambda i: p[i])
        ent = -sum(x * math.log(x) for x in p if x > 0)
        out = {"tokens": len(ids), "value": q["choices"][best],
               "confidence": p[best],
               "entropy_confidence": 1 - ent / math.log(k),
               "probs": dict(zip(q["choices"], p))}
        if q.get("primitive") == "score":
            out["score"] = sum(i * x for i, x in enumerate(p)) / (k - 1)
        print(json.dumps(out))


if __name__ == "__main__":
    main()
