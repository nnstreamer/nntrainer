#!/bin/bash
# memrun2.sh <tag> <p> <C> [extra env]: one run under test/htp/host/measure_memory.cpp (RSS + DMA-buf + model-file pages + swap, 50 ms)
cd $(dirname "$(readlink -f "$0")"); t=$1; p=$2; C=$3; S=R5KL308PALH; D=/data/local/tmp/nntrainer/causallm; R=$D/pr4416_$t; mkdir -p logs/mem
BASE="NNTR_NUM_THREADS=8 NNTR_HTP_E2E=1 NNTR_HTP_E2E_FREE_KVQ=1 NNTR_HTP_KV_DROP_CPU=1 NNTR_HTP_DROP_HOST_FC=1 NNTR_HTP_ATTN_M1_Q8=2 NNTR_HTP_FC_IN_ARENA=1"
adb -s $S shell "ps -A | grep -q nntrainer_causallm" && { echo BUSY; exit 1; }
adb -s $S push measure-memory $R/measure-memory >/dev/null; adb -s $S shell "chmod 755 $R/measure-memory"
log=logs/mem/mm_${t}_p${p}_c${C}.log
adb -s $S shell "cd $R && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH='$R' $BASE NNTR_MOE_CACHE_EXPERTS=$C $4 ./measure-memory $D/mm_${t}_p${p}_c${C}.csv $D/model/gemma4-26b-a4b/nntr_gemma4_26b_a4b_qs2cx_fcqs4cx.bin ./nntrainer_causallm $D/g${p}_i2048" > $log 2>&1
adb -s $S pull $D/mm_${t}_p${p}_c${C}.csv logs/mem/ >/dev/null
echo "$t p$p C$C $4: $(grep -a -o 'prefill: [0-9]* tokens, [0-9]* ms, [0-9.]*' $log | awk '{print $4" ms "$6" TPS"}') | decode last64 $(grep -a -o 'generation(last 64): 64 tokens, [0-9]* ms, [0-9.]*' $log | awk '{print $7}') | md5 $(grep -a -o '<channel|>.*' $log | head -1 | md5sum | cut -c1-8) | $(grep -a '\[memory\]' $log | tail -1)"
