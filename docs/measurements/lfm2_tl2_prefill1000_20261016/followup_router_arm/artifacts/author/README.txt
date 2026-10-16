Opt-in native LFM prefill ARM router, frozen v3, 2026-10-17

Status: code, caller, native DSP parity and build checks are complete. Production remains opt-in (NNTR_LFM2_ROUTER_ARM=1), based on f7b16e48/ccea0d476 and the original serial normRows. The rejected row-normalization and Tile8 candidates are not included. Root has completed the normal twelve-run same-binary OFF/ON matrix described below. Model dispatch and repeated-prompt completion are being finalized separately; the production default remains OFF.

Production patch: d5604ee3969f985032b607967870f3e8983a4b534572a0e2910a1746839c4796
App: 103b1e1df18efe8de8085f01167f29f2a4502178ac22e5c1aaab68dc38c036a8
Strict kernel: d74cacb8b4c42ea701c37922ff773dff95804e92d0f38d84a09ce3d86000aa79
All five source hashes, including the two Meson changes and public header, are in source-hashes.json. Production source is unchanged since native v3 testing.

Scope and arithmetic
The existing CPU-normalized LFM input and existing CPU FP32 router weights feed a four-row, sixteen-expert NEON kernel. Four expert-column groups produce each existing 4x64 logits slice. For every logit, K visits 0..2047 in order and performs one FP32 multiply followed by one FP32 add. It does not use FMA, a horizontal reduction, reassociation, new quantization or a new output buffer. CPU top-k and all MoE arithmetic remain existing code.

The outer application guard requires incremental prefill, in_norm, no accelerator norm, no out_add or router_norm, SiLU/sigmoid, one input, E64/I1536. The helper additionally requires ARM+HTP, M1024/2048, K2048, contiguous NCHW FP32 input/weights/logits with exact shapes, the same HTP ops and compatible contexts, and all 128 existing GU/DN expert metadata entries native TL2. It rejects output overlap with complete input/weight spans. Presence of NNTR_HTP_ROUTER_U8I8 disables it, including value 0, matching that backend option's current semantics. Gemma, M1, other layouts and OFF retain the existing route. No Gemma timing claim is made.

Strict compilation is essential
Function-scope pragmas, even a pragma placed before arm_neon.h, still allowed FMA under the real app's -ffast-math configuration. Invalid pragma/vector/strict probe sources, commands and disassemblies remain in this directory as rejected experiments. The production implementation therefore uses a separate static-library TU in both app_build and in-tree model builds. Its target appends -fno-fast-math -ffp-contract=off; it errors if __FAST_MATH__ remains enabled. No whole-app flags change. The dot is noinline, including across potential LTO. Root separately built the complete Meson app and checked the final command and zero FMLA output.

Final strict codegen: 16 FMUL and 16 FADD instructions per K/column group, four column groups, zero FMLA, zero inner-K-loop stack operations. Dot frame 64 bytes is callee-saved register storage outside the loop; weight scan frame is 0. App incremental frame remains 3008 bytes. Outlined helper/parallel_for/callback frames are 128/96/32 bytes. These individual frames do not constitute a measured stack peak. Source and exact compile arguments are recorded under codegen/.

Why v3 has explicit fallback
The initial finite-data prescreen is limited to its fixed hot input; its micro speed did not establish general arithmetic parity.
- v1 native 14 cases had 6 failures: ARM and DSP nonfinite outputs differed in NaN payload/sign. Original root-numeric-native logs and invalid-nonfinite/ source/binary manifest are preserved.
- v2 added a final output-exponent check. NaN/Inf/overflow then correctly fell back, but 46 native cases had 8 failures involving subnormal source inputs or weights; some final outputs were finite and differed by a few ULP. root-numeric-native-v2 and numeric-v2/source preserve this rejected version. A finite-output-only guard is insufficient.
- v3 first scans weights for nonzero subnormal bit patterns. Each task scans its four input rows for the same patterns before output writes. Signed zeros are allowed. A post-arithmetic integer exponent check rejects any nonfinite output. False from any task joins all tasks, then the original DSP router overwrites the entire logits tensor before selection. No partial output is consumed. An RPC failure propagates after join.

The guarded native matrix includes broad exponents, signed zero, normal operands whose products/accumulators underflow, cancellation, qNaN/sNaN, infinities, overflow, and last-element source/weight subnormals. These are bounded empirical parity checks, not a claim that a finite prescreen proves every possible IEEE input. The real model must still pass its separate output checks.

Ownership and cost
No activation buffer, expert cache, staging pool, slot/read capacity or quantization changes. The helper writes only existing router_logits. It reads existing normalized input and immutable model weights. It publishes no backend router-input handoff ticket; the subsequent plain MoE uses its original staging and one-shot fallback behavior. Existing prompt, writable-stage and previous-MoE consumption boundaries remain intact.

Each accepted call adds an integer scan of 512 KiB weights plus 8/16 MiB input at M1024/2048: logical extra reads of 8.5/16.5 MiB per call or 323/627 MiB over 38 layers. These are logical reads, not measured DRAM traffic. There are 256/512 four-row tasks, one existing pool dispatch/join, one local atomic bool, and a failure-only relaxed atomic store per rejected task. The join orders the final load. A 32-byte callback capture uses a 40-byte std::function metadata allocation in the existing pool, retained until its next dispatch/destruction; this is not zero heap activity. The retained pointer to the local flag is never used after join. No new persistent tensor/data cache exists. CPU/cache/file-reader contention can offset the isolated arithmetic improvement and is not predicted from micro results.

Validation actually run
- Author host: actual helper/caller and unchanged strict kernel with SIMDe, actual HVX router/RMS C with lane emulation, 120 cases in each of five environments (1, 0, unset, true, and quantized-option presence), ASan/UBSan halt with no suppressions. Work orders include forward/reverse/shuffled with 1/8 workers; alias, context, type/format, extents, task tail, exceptions and complete fallback overwrite are checked. host-v3b.log and checks/host-v3b record this final run. Earlier host-v3 build failure is preserved and is not a passing result.
- Ten actual-source mutations all fail by the intended assertion; checks/mutations-v3-final/results.json records qualification, ownership, last-task, error propagation, fallback overwrite, activation and opt-in guards. An earlier mutation runner pointed at the obsolete host-v3 output and failed to link; it was corrected to host-v3b and is not counted.
- Final whole app ARM, Android x86 and Linux GCC compiles pass; the strict ARM kernel and empty unsupported-platform TUs pass; wrong-fast-math compilation deliberately fails. Root full Meson app build/link also passes.
- Root-only native: frozen binary f956408ffd7f054be2507cedbef1cad72ef12f8f4ac457d4f7fd8975b46aee75 passes all 94 cases at ON/OFF x 8/1 threads (376 cases). It executes the actual extracted app helper/caller with real Tensor/Context/ThreadManager and the real DSP router. An exported RPC interposer forwards to the original stub and counts every invocation. ON138/OFF188 RPCs show exactly 50 accepted fast cases; each case separately checks full output and source/weight preservation. Failed-RPC injection and pool reuse are included. root-native-v3-results.json contains exact device commands, return codes and raw-log hashes. The native fixture source is byte-identical to the frozen executed snapshot.
- Independent agent: /tmp/nntr-lfm2-router-arm-review-20261017/review.txt, validation.json and manifest.json record independently repeated 120x5, ten mutants, six app/kernel compiles, strict IR/codegen and the rehashed root-native logs. No correctness/build blocker was found; this is separate from author tests and root device execution.

Diagnostic and reproduction
The diagnostic/ overlay is separate from production. It records actual helper entry, selected CPU route, completed DSP fallback, submitted/completed/rejected tasks and failed RPC completion. TRACE=0 produces an instruction-identical whole app TU; TRACE=1 has actual-caller 12 ON + 12 OFF sanitizer coverage and 6 valid/438 negative parser tests. See diagnostic/README.txt for exact build/run flags and strict 38-layer cohort parsing. Trace timing is never a normal TPS result.

Host:
python3 test/htp/host/run_lfm2_router_arm_check.py --repo REPO --source candidate --output OWN/host --simde-include SIMDE --sanitize
python3 test/htp/host/run_lfm2_router_arm_mutations.py --checked-output OWN/host --output OWN/mutants
Native build only:
python3 test/htp/host/run_lfm2_router_arm_native.py --source candidate --output OWN/native --compiler NDK/bin/aarch64-linux-android29-clang++ --nntrainer-include CORE/include/nntrainer --nntrainer-lib-dir CORE/lib/arm64-v8a
Run on the already configured root-owned device with NNTR_LFM2_ROUTER_ARM=1 or 0 and argument 8 or 1, using its matching runtime LD_LIBRARY_PATH/ADSP_LIBRARY_PATH. Root commands are archived rather than inferred.

Root normal-model evidence at document freeze
root-normal-summary.json reports all twelve normal runs pass all 65 dumped IDs and memory validation with the same seven binaries OFF/ON. 1024 prefill medians are 816.587 -> 843.493 TPS (+3.295%, all three pairs faster); 2048 medians are 792.570 -> 799.688 TPS (+0.898%, two of three pairs faster; the first pair regressed 2.07%). This is a modest result, especially at 2048; it does not meet 1000 TPS. Decode medians 17.3819 -> 17.6309 and 16.3224 -> 16.1412 are observations, not improvements attributed to this prefill-only change.

Maximum sampled full-child peak is 2731436 -> 2725072 KiB at 1024 and 3069164 -> 3018440 KiB at 2048. The metric is RssAnon plus deduplicated fdinfo allocation sizes over the whole child lifetime. It is not total device memory and excludes unreported DSP/driver/VTCM allocations and between-sample spikes. These lower observations are not evidence of an altered buffer capacity. Per-run values and complete runtime identity are in root-normal-summary.json and root-normal-runtime-identity.json.

Root reports the 1024 two-prompt same-handle test passed; final 2048 repeated-prompt and actual-dispatch diagnostics are still separate pending checks at this document freeze. Root will record final adoption and integrate its model evidence. Current constraints remain C16, PAGE_CACHE=0, fixed preload/readers, no added caches or capacity, and no Gemma device benchmark.
