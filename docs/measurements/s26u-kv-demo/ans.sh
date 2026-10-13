#!/bin/bash
# the generated text of a demo log, HTP/TIMING lines removed; md5 with -m
t=$(sed -n '/^<|turn>model/,/LLM with NNTrainer/p' "$1" | perl -0pe 's/\[HTP[^\n]*\n//g; s/\[TIMING[^\n]*\n//g' | grep -a -v "LLM with NNTrainer" | tail -n +2)
[ "$2" = -m ] && echo "$t" | md5sum | cut -c1-8 || echo "$t"
