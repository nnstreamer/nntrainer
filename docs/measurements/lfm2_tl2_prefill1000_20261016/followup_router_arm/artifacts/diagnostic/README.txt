Diagnostic only: actual LFM ARM router dispatch and task completion

This patch applies only over frozen production d5604ee3969f985032b607967870f3e8983a4b534572a0e2910a1746839c4796. It changes one app source and no strict arithmetic kernel or normal build flags. Build with NNTR_LFM2_ROUTER_ARM_TRACE=1 and run with NNTR_M0_PROFILE=1. Adding a single #define NNTR_LFM2_ROUTER_ARM_TRACE 1 before the diagnostic source is also valid; preserve that derived source hash separately.

One RAII record is emitted for each app LFM prefill router. enter is actual helper entry, tasks is the submitted count, done increments only after each actual strict kernel returns, rejected counts false kernel results, used is the actual ARM-result selection, fallback is a successfully completed original accelerator route, and finish is successful router completion. A thrown fallback RPC emits finish=0, after the existing worker join. It cannot be accepted as a successful model cohort. These are diagnostic counts, not performance times.

The probe's pointer stays inside the joined parallel_for. Callback atomics are relaxed because the existing pool join establishes completion before the foreground/destructor reads them. There are no normal-build atomics/counters/prints: the TRACE=0 whole-TU instruction stream was independently compared with the frozen production object and is exactly equal. TRACE=0/1 ARM whole TU compilation both pass.

The shared loader warmup calls the backend directly and does not enter this app helper. A fresh single-prompt model invocation therefore requires exactly 38 records, sequences 1..38. No deduplication is performed: duplicated/foreign-process records, missing lines, wrong chronology, errors, mismatched geometry, overflow values and any truncated marker are rejected. ON records may represent a properly joined DSP fallback; use --expect-fast 38 to require all 38 CPU routes for the intended model. OFF requires zero CPU tasks.

Expected all-fast totals: 1024 => 9728 completed tasks; 2048 => 19456. OFF => zero tasks and 38 accelerator fallbacks. The current parser's scope is one prompt, not repeated-prompt logs.

Commands:
python3 parse.py LOG --rows 1024 --enabled 1 --expect-fast 38
python3 parse.py LOG --rows 2048 --enabled 1 --expect-fast 38
python3 parse.py LOG --rows 1024 --enabled 0 --expect-fast 0
python3 parser_check.py
python3 check.py --checked-output HOST_CHECK_OUTPUT --production-tests ../test/htp/host --source candidate --output OWN

Checks: actual helper/caller 12 ON + 12 OFF ASan/UBSan cases cover both M values, success, subnormal input, subnormal weights, nonfinite output, metadata fallback and thrown RPC. All counters are checked against those actual paths. Parser passes 6 valid cohorts and rejects 438 malformed/missing/overflow/order/error/task/truncation variants. The normal candidate and previously tested native f956 binary are unchanged.
