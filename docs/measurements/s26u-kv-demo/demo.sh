#!/bin/bash
# KV-cache demo on the S26 Ultra R5KL308PALH (docs: memory s26u-kv-demo).
#   demo.sh inst <tag>            : push the 7 built binaries to $D/pr4416_<tag>
#   demo.sh cfg                   : push every demo/demo_* config (+ json symlinks)
#   demo.sh run <config> <tag> [extra env] : one run of demo_<config>, log + summary lines
#   demo.sh save <config> <tag> [extra env] : delete the config's dump and run it once (SAVE mode)
#   demo.sh verify <tag> [configs...]       : the whole check, a table and the answers (FULL=1: also the uncached 23k runs)
export ANDROID_SERIAL=${ANDROID_SERIAL:-R5KL308PALH}
D=/data/local/tmp/nntrainer/causallm; MD=$D/model/gemma4-26b-a4b
R=$HOME/workspace/nntrainer; W=$HOME/workspace/s26u-work/demo
BASE="NNTR_NUM_THREADS=${NT:-8} NNTR_HTP_E2E=1 NNTR_HTP_E2E_FREE_KVQ=1 NNTR_HTP_KV_DROP_CPU=1 NNTR_HTP_DROP_HOST_FC=1 NNTR_HTP_ATTN_M1_Q8=2 NNTR_HTP_FC_IN_ARENA=1"   # C comes from demo_<config>/env
case "$1" in
inst)
  t=${2:?tag}; adb shell "mkdir -p $D/pr4416_$t"
  adb push $R/test/htp/build/libnntr_hvx_skel.so \
    $R/builddir/android_build_result/lib/arm64-v8a/{libnntrainer.so,libccapi-nntrainer.so,libc++_shared.so,libsdkl.so} \
    $R/Applications/CausalLM/jni/libs/arm64-v8a/{nntrainer_causallm,libcausallm_core.so} $D/pr4416_$t/ >/dev/null
  adb shell "cd $D/pr4416_$t && chmod 755 nntrainer_causallm && md5sum *.so nntrainer_causallm" | cut -c1-12,34- | tr '\n' ' '; echo ;;
cfg)
  for c in $(cd $W && ls -d demo_*); do
    [ -f $W/$c/nntr_config.json ] || continue
    adb shell "mkdir -p $D/$c"
    adb push $W/$c/nntr_config.json $D/$c/ >/dev/null
    adb push $W/generation_config.json $D/$c/ >/dev/null
    adb shell "cd $D/$c && for f in config.json tokenizer.json tokenizer_config.json; do ln -sf $MD/\$f \$f; done; ls"
  done ;;
run)
  c=${2:?config name without demo_}; t=${3:?tag}; X=$4; OUT=$W/logs; mkdir -p $OUT
  log=$OUT/${c}_$(date +%H%M%S).log
  E=$(cat $W/demo_$c/env 2>/dev/null | tr '\n' ' ')   # the config's own env (mkcfg.py: C)
  adb shell "cd $D/pr4416_$t && LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH='$D/pr4416_$t' $BASE $E $X ./nntrainer_causallm $D/demo_$c" > $log 2>&1
  echo "== $c ($t) -> $log"
  grep -a "KV CACHE SAVE\|kv caches are saved\|size of prompt\|prefill:\|^generation\|first_token_ms\|kv_load\|FATAL\|terminate\|Abort" $log | cut -c1-140
  grep -a -o '<|turn>model.*' $log | head -1 | cut -c1-400 ;;
save)
  c=${2:?config name without demo_}; t=${3:?tag}
  adb shell "rm -f $D/demo_$c/*.bin $D/demo_$c/*.bin.q2"
  "$0" run $c $t "$(cat $W/demo_$c/env_save 2>/dev/null | tr '\n' ' ') $4"; adb shell "ls -la $D/demo_$c/ | grep bin" ;;
row)  # one table row from a log: load ms | prefill tokens / ms | first token ms | ms per token | answer md5
  f=$2
  printf '%-16s %8s %8s %9s %8s %8s  %s\n' "$(basename $f | sed 's/_[0-9]*\.log$//')" \
    "$(grep -a -o 'kv_load_ms=[0-9]*' $f | cut -d= -f2)" \
    "$(grep -a '^prefill:' $f | tail -1 | awk '{print $2}')" \
    "$(grep -a '^prefill:' $f | tail -1 | awk '{print $4}')" \
    "$(grep -a -o 'first_token_ms=[0-9]*' $f | cut -d= -f2)" \
    "$(grep -a -o 'rest_avg_ms=[0-9.]*' $f | cut -d= -f2 | cut -c1-6)" \
    "$($W/ans.sh $f -m)" ;;
verify)
  # demo.sh verify <tag> [configs...]: per cached config, save the cache if the phone has none, then
  # run it, the question alone and (FULL=1, or always for a 2k config) the whole prompt without a cache.
  # The cached run's answer must be byte-identical to the uncached one's. COOL=<s> between runs (default 60).
  t=${2:?tag}; shift 2; CF=${*:-kv2k kv23k_a kv23k_b}; COOL=${COOL:-60}; V=$W/logs/verify_$(date +%m%d_%H%M%S).txt
  last() { ls -t $W/logs/$1_[0-9]*.log | head -1; }
  one() { "$0" run $1 $t > /dev/null; grep -a -q '^generation:' $(last $1) || { echo "FAILED: $1 ($(last $1))"; grep -a 'what()\|FATAL' $(last $1) | cut -c1-200; }; sleep $COOL; }
  for c in $CF; do
    adb shell "[ -f $D/demo_$c/kv.bin ]" || { echo "saving the cache of $c ..."; "$0" save $c $t | grep -a 'size of prompt\|what()\|kv.bin'; sleep $COOL; }
    adb shell "[ -f $D/demo_$c/kv.bin ]" || { echo "FAILED: no cache file for $c (see $(last $c))"; continue; }
    one $c; one ${c}_q
    full=n; grep -q '"max_seq_len": 2304' $W/demo_$c/nntr_config.json && full=y; [ "${FULL:-0}" = 1 ] && full=y
    [ $full = y ] && one ${c}_full
  done
  { printf '%-16s %8s %8s %9s %8s %8s  %s\n' config load_ms prefill_n prefill_ms first_ms ms/token answer_md5
    for c in $CF; do for x in $c ${c}_q ${c}_full; do f=$(last $x 2>/dev/null); [ -n "$f" ] && [ $f -nt $W/demo_$c/nntr_config.json ] && "$0" row $f; done; done
    for c in $CF; do f=$(last ${c}_full 2>/dev/null); [ -n "$f" ] && [ $f -nt $W/demo_$c/nntr_config.json ] || continue
      a=$($W/ans.sh $(last $c) -m); b=$($W/ans.sh $f -m)
      [ "$a" = "$b" ] && echo "OK   $c: the cached answer is byte-identical to the uncached one ($a)" || echo "DIFF $c: cached $a, uncached $b"
    done; } | tee $V
  for c in $CF; do echo "--- $c (cached):"; $W/ans.sh $(last $c) | grep -a -v '^$' | head -3 | cut -c1-400; echo "--- ${c}_q (question only):"; $W/ans.sh $(last ${c}_q) | grep -a -v '^$' | head -3 | cut -c1-400; done | tee -a $V ;;
*) sed -n 2,8p "$0" ;;
esac
