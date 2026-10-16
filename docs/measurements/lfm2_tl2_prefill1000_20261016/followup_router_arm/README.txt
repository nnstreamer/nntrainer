LFM2 native TL2 prefill ARM router: retained opt-in experiment
2026-10-17, S26 Ultra R3CY70LV96T

Decision: retain NNTR_LFM2_ROUTER_ARM=1 as an opt-in. The default remains the existing DSP router. All final model, repeated-prompt, native, build and diagnostic checks below are complete. This final account supersedes the pending diagnostic/repeated-2048 statuses in the preserved author-freeze README and validation.json.

Normal model results
Twelve normal runs used the same seven runtime binaries for OFF/ON, three balanced pairs per prompt length. Only the runtime router option differed. All 65 dumped IDs (one prefill ID plus 64 generation IDs) matched the established reference in every run. Both lengths also passed two prompts on the same model handle with all 65 IDs identical on each run.

Rows  Prefill median OFF -> ON TPS   Change       Faster pairs
1024  816.587 -> 843.493             +3.295%      3/3
2048  792.570 -> 799.688             +0.898%      2/3

1024 OFF samples: 816.587, 816.587, 821.830.
1024 ON samples:  843.493, 853.333, 842.798.
2048 OFF samples: 816.587, 792.570, 772.830.
2048 ON samples:  799.688, 799.064, 802.508.
The 2048 result is weak: its first pair was 2.07% slower, and the median gain is small relative to run variation. The 1024 result was consistent in these three pairs. This does not establish 1000 TPS prefill or 25 TPS decode.

Decode medians were 17.3819 -> 17.6309 TPS at 1024 and 16.3224 -> 16.1412 at 2048. These are observations from the same runs, not decode improvements attributed to this prefill-only change. No Gemma device benchmark was run for this candidate, and no claim of measured Gemma performance equivalence is made.

Memory and fixed conditions
Maximum sampled whole-child peak, OFF -> ON:
1024: 2731436 -> 2725072 KiB.
2048: 3069164 -> 3018440 KiB.
This metric is process RssAnon plus deduplicated readable fdinfo allocation sizes over the child lifetime, including loading. Sampling requested every 20 ms; the actual intervals and all samples are preserved. It is not total device physical memory and excludes unreported DSP/driver/VTCM allocations and between-sample spikes. The lower observed values do not prove a capacity reduction. The separate app metric, which covers only CausalLM::run, is not the table's metric.

C16, PAGE_CACHE=0, cache growth disabled, preload128, eight CPU workers, six prefill readers, two speculative decode readers and four demand participants (including caller) stayed fixed. The existing predict2/residual0.5, stable slots, early prefill, decode pair read/demand handoff and router-input handoff settings are recorded per run. The model's native TL2 experts, affine int4 projections, CPU FP32 router weights and quantization were not changed. No expert cache, activation/staging buffer, reader count or slot capacity was added.

What changed
For the exact native LFM prefill shape (M1024/2048, K2048, E64, intermediate1536, SiLU/sigmoid), the optional path reads the existing CPU-normalized input and existing CPU FP32 router weights, then writes the existing router_logits tensor. A four-row by sixteen-expert NEON kernel processes four column groups. Each dot visits K0..2047 in the original order with a rounded multiply followed by a rounded add. Existing top-k and MoE computation are unchanged.

The strict kernel lives in its own TU/static library with final -fno-fast-math -ffp-contract=off flags. It rejects an accidental __FAST_MATH__ build. The app's overall flags are unchanged. Actual Meson and independent NDK code generation have no FMLA/FMADD in the strict kernel and no inner-K stack operations. Final dot frame is 64 bytes (callee-save storage outside the loop); weight scan is 0. The app incremental frame remains 3008 bytes. Helper/parallel_for/callback frames are 128/96/32 bytes. These are per-function frames, not a measured total stack peak.

Qualification requires ARM+HTP, exact contiguous NCHW FP32 shapes and extents, compatible contexts/HTP ops, all 128 GU/DN expert metadata entries native TL2, and output disjoint from full input/weight ranges. The original normalized-input path, single input, in_norm, no accelerator norm/out_add/router_norm, and the SiLU/sigmoid policy are required. Presence of NNTR_HTP_ROUTER_U8I8 declines this path, even when its value is 0. Gemma, decode, unsupported types/layouts and OFF use existing code.

Arithmetic failures and v3 fallback
The early finite/hot prescreen was not a general parity proof. Rejected evidence is retained, including the first pragma-only FMA experiment and later source variants:
- v1: 14 native cases, 6 failures. ARM/DSP nonfinite output payloads/signs differed.
- v2: 46 native cases, 8 failures. Nonfinite outputs correctly fell back, but subnormal input/weight operands could produce finite outputs differing by a few ULP. A finite-output-only guard was insufficient.
- v3: scan weights and each task's four input rows for nonzero subnormals; reject any nonfinite output via integer exponent bits. Signed zero is allowed. Any rejection waits for all tasks and invokes the original DSP router, which overwrites the complete logits tensor before selection. Failure/throw/retry and complete overwrite are tested.

v3 adds logical scans of 512 KiB weights plus 8/16 MiB input per call, or 323/627 MiB over 38 layers. These counts are not measured DRAM traffic. It schedules 256/512 tasks through the existing pool with one dispatch/join and a local atomic failure flag. A 32-byte capture uses a 40-byte std::function metadata allocation retained by that existing pool until its next dispatch/destruction; no claim of zero heap activity is made. No new persistent data cache is introduced. CPU completion publishes no DSP router-input handoff stamp, so the following ordinary MoE uses its original staging/fallback contract.

Actual diagnostic dispatch, separate from normal performance
The compile-time-only diagnostic recorded 38/38 CPU routes at each length:
1024: 9728 submitted, 9728 completed, 0 rejected, 0 DSP fallbacks.
2048: 19456 submitted, 19456 completed, 0 rejected, 0 DSP fallbacks.
Both diagnostic runs also matched all 65 IDs and passed memory validation. TRACE=0 has instruction-identical whole-app code to the frozen production TU. The trace library was removed and the normal app/executable were rebuilt byte-identically; artifacts/root/root-diagnostic-removed.json records this restoration.

The profile in these diagnostic runs still reports expert prefetch exposed wait of 35.5 ms at 1024 and 0.1 ms at 2048; registration RPC totals are 2.6 and 3.0 ms. These are individual instrumented observations, not a paired profile establishing a causal effect of the CPU router. They do not justify a claim of zero I/O wait. Diagnostic TPS values are excluded from the normal comparison.

Parser input correction: the frozen diagnostic README's broad statement that foreign-process records are rejected is too strong. The parser assumes one child process's captured stderr. It validates exact 38-call order/geometry/completion and rejects malformed, truncated, duplicate, missing and inconsistent records, but does not authenticate PID. A contrived mixture of disjoint sequence portions from different processes could pass. The actual root captures are per-process child stdout/stderr, not merged logcat; the separate logcat files are ancillary. This clarification supersedes that one author-document claim without rewriting original evidence.

Validation and its limits
Author and independent host executions each pass 120 actual helper/caller cases in five environment settings, ASan/UBSan with no suppressions. They use the strict production kernel through SIMDe, actual HVX router/RMS C through lane emulation, and controlled Tensor/ops/scheduler stand-ins. Ten actual-source mutants each fail the intended assertion. The host emulator alone does not prove device special-value semantics.

The final native binary f956408ffd7f054be2507cedbef1cad72ef12f8f4ac457d4f7fd8975b46aee75 passes 94 cases at OFF/ON and one/eight threads: 376 cases total. It uses the actual extracted app helper/caller, real Tensor/Context/ThreadManager and the actual DSP RPC through an interposer. ON makes 138 RPCs versus 188 OFF, confirming 50 accepted CPU cases in each ON run. Broad exponents, signed zeros, underflowing products, cancellation, NaN/Inf/overflow, last-element subnormal guards, aliases, full source/weight preservation and fallback/retry are covered. This remains bounded empirical testing, not an exhaustive proof for every possible IEEE operand.

Whole app/kernel compilation passed ARM Android, Android x86 and Linux GCC, with independent strict LLVM/codegen checks and deliberate bad-fastmath rejection. Root performed the complete Android app link. The diagnostic separately passes 12 ON plus 12 OFF actual-caller sanitizer checks and 6 valid/438 negative parser cases. Independent diagnostic commentary reviewed counters, joins and lifetime, but did not repeat those diagnostic fixtures; the independent production report lists what it actually executed.

Provenance and reproduction
Production patch SHA256: d5604ee3969f985032b607967870f3e8983a4b534572a0e2910a1746839c4796.
Tests patch SHA256: 403448179df6daf166ef4c3df8ecc7fef0489d530bf253b1f257437fba520d76.
App source103b1e1d / strict kernel d74cacb8 / public header5a79209d are fully hashed in artifacts/production/source-hashes.json. The author base is f7b16e48ca01a650160d823141e86e67c7993a7c, with production code equivalent to ccea0d476; rejected row norm and Tile8 are absent. Deployment manifests retain the later evidence-only base80a8b754 and exact integrated patches separately from the author's minimal patch.

Normal runtime identity is in artifacts/root/root-normal-runtime-identity.json and the three deployment manifests. All seven SHA256 values are retained; no ELF, libraries, model weights or vendor headers are included. Native fixture link-time library provenance and actual device commands are separately recorded, so the native fixture and normal runtime are not represented as the same binary set.

raw/ contains all 12 normal and 2 diagnostic records, logs, 65-ID files and original memory samples. artifacts/repeat/ contains both same-handle runs. artifacts/rejected/ preserves failed arithmetic versions. Author and independent sources, commands, logs, codegen, test patches and prior frozen reports remain separately attributed. Large generated duplicate source/assembly and binaries are omitted where exact frozen source plus recorded commands reproduce them.

Run python3 -B validation/verify.py from this directory to recheck hashes and result semantics without a device or original workspace. See repro/README.txt to materialize source/tests and rerun host checks. manifest.json hashes every payload and its uncompressed bytes, with original source paths retained only as provenance. results.json is recomputed from the archived raw records. No root source, existing evidence or device was modified by curation.
