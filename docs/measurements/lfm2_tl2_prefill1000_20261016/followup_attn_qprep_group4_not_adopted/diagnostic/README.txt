Separate diagnostic, never a normal performance build

Apply diagnostic.patch af263c745b23b9215a5707f9bc60c008ef2c168ea0880d9b201adddef4871a39
on production a07332. Recompile hvx_quant_u8.c with
-DNNTR_LFM2_QPREP_GROUP4_TRACE=1 and link the otherwise identical candidate objects.
Do not combine this trace with another optimization. Restore exact normal
candidate objects/runtime before any performance result.

Only the actual group4 helper is instrumented. After the synchronous pool run
has drained all workers, relaxed atomic32 add increments one diagnostic4B seq
and one bounded ERROR log is emitted. No per-worker log, math/dispatch/allocation
change, or new buffer. No-flag compile is object-byte-exact with normal candidate
(codegen/candidate/hvx_quant_u8.o); v81 imports contain no atomic helper. Trace
adds HAP logging imports/formatting and changes helper frame32→64B plus HAP's
280B logger frame. Its timing/memory results are not normal performance evidence.
Maximum constrained payload is79 ASCII bytes including seq UINT32_MAX;
actual groups1152/896, k64/stride2048, bytes1179648/917504. Counter wrap to0 is
well-defined but parser rejects it; one fresh child run cannot approach wrap.

Expected completed single-prompt run:
2048:160 unique helper successes, seq1..160, alternating1152/896 groups,80 each;
     total163840groups and167772160 logical Q-gather bytes skipped.
1024:0 tag records (the trace-enabled library and full capture must independently
     be verified: an absent tag alone cannot prove that the diagnostic was on).
The16 calls/layer come from8 KV heads, each9+7 chunks of128 queries. There are10
attention layers. No fixed-Q8/Gemma or decode call can enter this helper.

Each record proves helper completion, not later HMX/RPC success. Require normal
65 generated IDs, whole-run completion, correct runtime hashes, and a complete
PID-filtered live log separately. Per-process static sequence does not reset at
prompt boundaries: these expectations are for one fresh child and one prompt,
not a repeat harness or several runs concatenated. Hardware log capture can be
incomplete even if the host process succeeds; the strict160 count rejects it.

python3 parse_trace.py raw.logcat.txt --model 2048 --pid CHILD_PID --output result.json
python3 parse_trace.py raw.logcat.txt --model 1024 --pid CHILD_PID --output result.json
Parser requires the Android threadtime PID on every tagged line, rejects mixed
PIDs, truncated records, wrong fields/counts/chunk order, missing/extra/non-
contiguous sequences and conflicting duplicates. Identical payload repeats for
the same seq are transport duplicates and are explicitly counted, not silently
deduplicated. With0 records it cannot independently infer or prove the PID scope.

Host evidence uses the actual instrumented helper, actual SDK arithmetic and
real pool:192 unique entries from the576-case fixture,96 of each chunk size,
no duplicate/gap. This is fixture coverage, not a model dispatcher claim. The
same strict sanitizer/unaligned-type correction as the production tests applies;
initial diagnostic/host logs are preserved and superseded by host-strict.
check_parser.py additionally exercises4 valid and18 negative synthetic captures.
build-validation.json preserves macro-off equality, source/patch/object hashes,
imports and explicit scope. Normal/trace compile commands are archived.
