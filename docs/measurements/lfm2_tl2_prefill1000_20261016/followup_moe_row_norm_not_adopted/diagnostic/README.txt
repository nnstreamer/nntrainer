Diagnostic only, separately applied over frozen production.patch. Build with
NNTR_LFM2_MOE_ROW_NORM_TRACE=1 and run with NNTR_M0_PROFILE set plus the normal
row-norm opt-in. The record is after actual helper/fallback completion and
uses the actual norm_done return, not the requested option or metadata.
The shared loader warmup invokes the backend directly, not this app norm
block; one normal prompt therefore requires 38 actual layer records.
Normal source remains frozen and contains no diagnostic counter or logging.
The diagnostics timing is not normal throughput evidence.
