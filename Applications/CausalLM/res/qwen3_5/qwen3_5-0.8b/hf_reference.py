#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# HF Qwen3.5-0.8B greedy reference for the nntrainer port check.
# Dumps the argmax token ids and (optionally) top-1 logits of the first step
# so the x86 nntr_causallm run can be compared token-by-token / logit-by-logit.
#
# Usage: hf_reference.py <model_dir> <prompt> [num_tokens]
import json
import sys

import torch
import transformers

model_dir = sys.argv[1]
prompt = sys.argv[2]
num = int(sys.argv[3]) if len(sys.argv) > 3 else 32

tok = transformers.AutoTokenizer.from_pretrained(model_dir)
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(
    model_dir, dtype=torch.float32)
model.eval()

ids = tok(prompt, return_tensors="pt")
with torch.no_grad():
    out = model.generate(
        **ids, max_new_tokens=num, do_sample=False,
        temperature=None, top_p=None, top_k=None,
        return_dict_in_generate=True, output_scores=True)

gen_ids = [int(x) for x in out.sequences[0][ids["input_ids"].shape[1]:]]
first_logits = out.scores[0][0]  # [vocab]
top10 = torch.topk(first_logits, 10)

print(json.dumps({
    "prompt_ids": [int(x) for x in ids["input_ids"][0].tolist()],
    "generated_ids": gen_ids,
    "first_step_top10": [
        [int(i), float(first_logits[i])] for i in top10.indices.tolist()
    ],
}, indent=1))
