Diagnostic-only affine FC phases; no production optimization

Base commit8288e3ba3dff986e0c72f37c8db4f3adec4c628c. One file:
nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c
Apply diagnostic.patch and compile DSP with -DNNTR_LFM2_AFFINE_FC_TRACE=1.
Use the normal model path and existing NNTR_HTP_PROFILE=1 (not2). No app/core/IDL change or public runtime knob is needed. Remove the patch/compile flag after the diagnostic. Root alone builds/deploys/runs the phone.

Frozen production-diagnostic patch SHA256:
391e30db92b2d4a53569edc616bece7214315b82fd4d34ac2d6ee49426667d03
Candidate source SHA256:
968aeb6d22d523179873343b4bcba36bbfd067665092e7d574afd7d1c25103ce

Scope
The helper runs only after existing pointer/handle/K validation and successful per-call allocations. Enabled for M>1,K2048, all real int4 affine handles (bits4, offset present, no native TL2 weight pointer), no prepacked/u16 activation or accumulation, and widths2048,512,2048/2048/2048,2048/512/512. Width512 covers load warmups. MoE/dense have separate kernels and receive no counters; tested Gemma4 K2816 and FP32 routerN64 are excluded. All existing arithmetic, buffers, submissions, joins, DMA operations and error returns remain in their original order. No persistent state/cache or heap/VTCM allocation is added.

One FARF(ERROR) record per selected call, [LFM-AFFINE-FC] v1.
- M,K,N,nh,widths: separate prompt vs load warmups.
- rc,stride,bank_tiles,pipeline: detect failure/fallback or a different layout.
- reads,joins,brackets: count actual readouts and pool waits, including final wait and overall-body bracket. Each bracket has two qtimer reads.
- pack_t: foreground parameter scan + optional pre-norm + AH quantize/pack. The already-started first weight DMA overlaps it. It is NOT a pure layout-pack timer.
- sums_t: complete affine_sums worker-pool call.
- read_t: vendor accumulator read call, INCLUDING possible completion of preceding HMX work. Never label this pure int32 store/copy or wholly removable i24 cost.
- join_t: exposed foreground pool waits, including final/error drain. It is NOT complete dequant worker service time.
- drain_t: first weight drain AND each handle-end drain. These are exposed waits, NOT complete DMA transfer times.
- body_t: from after validation/allocations and before initial weight push through final frees. Excludes allocations/layout validation, subsequent calibration and FARF logging; includes instrumentation/counter overhead and ordinary issue/setup/free work. Quantization-error return is also logged after its existing cleanup. Errors are not valid performance samples.
- clock_pairs32/clock_t: sum of32 back-to-back raw-qtimer intervals after body ends. This is only a scale reference. It does NOT exactly measure both timer calls plus branch/counter/register-pressure effects, and is not automatically subtracted.
- hz19200000: raw52ns ticks avoid per-tile1us truncation.

Foreground phases do not nest on enabled paths, so pack+sums+read+join+drain may be subtracted from body to label an unclassified foreground residual. Do not add asynchronous worker or DMA service time to these intervals. Timer overhead can move the overlap boundaries. The hot loop records all readouts: main148480 read brackets alone add296960 qtimer reads. Main totals with all foreground/body brackets and calibration are153840 brackets and312800 qtimer reads. Diagnostic TPS is never a normal performance result. If clock intervals/instrumentation perturbation are material relative to a claimed saving, obtain lower-overhead evidence before implementing a port.

Strict exact-model parser
python3 parse_affine_fc.py /path/to/run.logcat.txt --output phases.json
This parser expects this exact1024-row LFM model/run topology: main80 records (N6144:30,N2048:40,N3072:10), warmup100 records (M512/N6144:1,M64/N6144:29,M64/N2048:50,M64/N512:20). It checks every trace record, required/unique fields, rc0,pipeline1,stride32,bank_tiles32, complete cohort, tile/join/bracket counts and non-overlapping foreground totals. Main expected148480reads,4880joins,153840brackets. Missing/malformed/duplicate/count/layout/error records fail instead of silently disappearing. It deliberately does NOT support extrapolating this topology to2048-row prompts or other models. Validate run exit/65 IDs/settings separately from this parser.

Validation and boundaries
- checks/{before,disabled,enabled}-command.json are exact v81 -O3 -Wall -Werror commands; all compile PASS. Flag-off .text equals baseline bytes. Stack frame656B baseline/off;792B diagnostic-on (+136B); no heap/VTCM growth.
- python3 parser_check.py: exact180 synthetic records and19 malformed/error/count mutations.
- run_guard.py --repo REPO --source candidate/.../hexkl_mm_u8i4_dma.c --output OUT: actual helper included from the full candidate TU,25 cases under ASan/UBSan. Includes positive warmup/projection/supplied-param shapes, Gemma-width/M1/nonaffine/TL2/wrong-handle/prepacked/u16/accumulate exclusions. Tests no table/options/shared-probe mutation or qtimer use on rejected cases.
- run_host_check.py --repo REPO --source candidate/.../hexkl_mm_u8i4_dma.c --reference ../frozen-source/.../hexkl_mm_u8i4_dma.c --output OUT compares the existing scalar whole-FC fixture before/on. Its K64 cases exercise fallback to no diagnostic; they do NOT validate real HMX arithmetic or active large-shape timing. The fixture originally leaked3 main test arrays at process exit: original baseline LSan evidence retained under checks/host-san/. The runner writes only a temporary fixture copy with free(x),free(got),free(want) at final return; no production or assertions changed, leak detection remains enabled. Results under checks/host-san-cleanup/.
- Independent actual whole-FC TU topology/guarded-phase checks are supplied by the reviewer under /tmp/nntr-lfm2-affine-fc-trace-review-20261016:22 cases using no-op arithmetic primitives and a deterministic clock, including quant error and33rd-read error. This proves instrumentation/control-flow count semantics, NOT performance or HMX math. Reviewer additionally checks the parser with actual successful/error records and malformed cases. See their final review before running.

No actual device phase measurements have been collected for this diagnostic yet. It is a measuring tool, not an i24 implementation or speedup claim.
