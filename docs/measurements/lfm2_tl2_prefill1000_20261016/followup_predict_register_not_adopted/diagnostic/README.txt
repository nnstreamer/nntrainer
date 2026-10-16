Diagnostic only, layered over the frozen production patch.
Apply diagnostic.patch and compile htp_compute_ops.cpp with
-DNNTR_LFM2_PREDICT_REGISTER_TRACE=1. No DSP change is needed.
Normal compilation has no counter instructions or state. The diagnostic adds
three process-lifetime atomic uint64 counters (24 bytes), using relaxed increments.
After the existing predictor drain, the existing atClose callback emits one
stderr record: callbacks (the queue's completed count), dots, qualified and fast.
The fast counter is inside the actual NEON helper, not its caller's predicate.
Qualified means valid nonzero request, hidden2048/expert64/non-null sigmoid bias;
caller native TL2 geometry and exact env=1 remain the production guards.
Expected one model process / 64 decode tokens: callbacks=dots=qualified=2368;
fast=0 with env OFF, fast=2368 with env=1. Leave NNTR_HTP_PREDICT_LOG unset:
that existing trace adds three extra dot products per callback, and the parser
rejects this cohort. Multi-session/missing/duplicate/truncated records are also
rejected. No timing claim is based on a diagnostic binary.

parse.py --log DIRECT_STDERR --enabled 0|1 --output OUT.json
parser_check.py OUTPUT: 2 valid and 10 invalid cases PASS.

Full actual Android core TU: normal/off/on compiled successfully. With the
macro absent, 1017/1020 text sections are byte-identical to normal, including
predictor, callback and initializer. The remaining three sdpa sections differ
only in five existing logging __LINE__ immediates (+11 lines), with unchanged
instruction counts; therefore full object identity is not claimed.
Production helper and arithmetic are unchanged by this overlay.
