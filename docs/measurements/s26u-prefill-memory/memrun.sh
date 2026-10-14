#!/bin/bash
# memrun.sh <tag> <p> <C> [extra env]: one run with a background memory sampler on the phone; a summary table row + the app's own lines
cd $(dirname "$(readlink -f "$0")"); t=$1; p=$2; C=$3; S=R5KL308PALH; D=/data/local/tmp/nntrainer/causallm; mkdir -p logs/mem
adb -s $S push memsample.sh $D/memsample.sh >/dev/null; adb -s $S shell "chmod 755 $D/memsample.sh"
adb -s $S shell "ps -A | grep -q nntrainer_causallm" && { echo BUSY; exit 1; }
adb -s $S shell "nohup sh $D/memsample.sh $D/memsample_${t}_p${p}_c${C}.txt >/dev/null 2>&1 &" ; sleep 2
./run.sh run $t $p $C "$4" logs/mem/${t}_p${p}_c${C}.log
sleep 2; adb -s $S shell "pkill -f memsample.sh" 2>/dev/null; adb -s $S pull $D/memsample_${t}_p${p}_c${C}.txt logs/mem/ >/dev/null
f=logs/mem/memsample_${t}_p${p}_c${C}.txt; L=logs/mem/${t}_p${p}_c${C}.log
awk -v t="$t p$p C$C" '/^base/ {split($2,a,"="); base=a[2]} /avail=/ {for(i=1;i<=NF;i++){split($i,a,"="); if(a[1]=="avail" && (mina==""||a[2]<mina)) mina=a[2]}; for(i=1;i<=NF;i++) if($i ~ /^(VmRSS|RssAnon|RssFile|RssShmem|VmHWM):$/){k=$i; v=$(i+1); if(v>mx[k]) mx[k]=v}} END {printf "%-14s system peak (MemAvailable drop) %5.0f MiB | process peak: VmRSS %4.0f (anon %4.0f file %4.0f shmem %4.0f) VmHWM %4.0f MiB\n", t, (base-mina)/1024, mx["VmRSS:"]/1024, mx["RssAnon:"]/1024, mx["RssFile:"]/1024, mx["RssShmem:"]/1024, mx["VmHWM:"]/1024}' $f
grep -a "e2e: fc arena\|mem t1\|staging: prefill classes\|attn_m1: registered\|peak memory\|arena: chunks unmapped\|^prefill:" $L | cut -c1-170
