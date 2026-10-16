Native LFM2 ARM prediction register accumulation — frozen candidate

Production patch: ba6055d746a37aecbb98f38a260ba0671071c285f6a3d426ebf5ec3a6c364f3d
Header: cb9d5e9ac8fd04844b02652e8a782ac69805c93df3aa5c7511935fb54f4e0836
Core: 1c470ac6a9edcc1cbb91f9c1ccd85a5a18132ead41921636dfca4bbf4bf83418
Tests patch: 7c24b11915f44d30ed6933ce42ee79b19f0122b643741135c7f329970c32401e
Baseline snapshot HEAD f7b16e48ca01a650160d823141e86e67c7993a7c, with unchanged
core/header relative to the accepted earlier decode path. Full original hashes
are in baseline.json. No root source, phone or commit was changed by this agent.

Scope and selection
- Only NNTR_LFM2_PREDICT_REGISTER_ACCUM=1 enables the new dot-product path.
  This exact environment value is read once while constructing the existing
  prediction queue. Unset/0/true/01/2/1x use the original implementation.
- initPoolPrefetch validates every expert in every layer has one uniform shape
  before calling initCpuPredictor. The captured option additionally requires
  native TL2 weight bits and K2048/inter1536/N2048. Each call also requires
  AArch64, hidden2048, E64 and non-null sigmoid bias. Legacy QS2, Gemma, other
  dimensions and non-AArch64 use the original scalar-loop source.
- The actual DSP router and mixture weights do not change. The helper only
  predicts which expert reads to issue. Pool sizes, reader count, cancellation,
  handoff, cache policy, pointer ownership and final expert sum order are intact.
- Existing input/weights/bias borrowing and predictor-drain lifetime are intact.
  There is no persistent weight/activation buffer or new allocation. The existing
  score and ID vectors and corrected input are unchanged. The new captured bool
  occupies the padding after the existing float blend: callback size72B unchanged.

Removed work and exact order
The compiled original core loop reloads and stores 64 score floats for every
hidden element, even though they are private to this one prediction. The new
helper retains 16 four-lane accumulators, traversing h=0..2047 exactly once in
ascending order. It uses FMLA(weight,input,sum), matching the actual Android
baseline vector instruction operands, with no lane reduction or across-call
reuse. The same zero initial values, finite rejection, stable sigmoid+bias,
tie-by-lower-ID comparator and partial_sort are retained.

Whole actual AArch64 htp_compute_ops.cpp compiled before/after with current
NDK/Android flags. The optimized hot loop contains 16 FMLA, eight weight LDP,
one input LDR and four loop-control instructions:29 instructions/h. It has no
score load/store or stack spill inside the loop; scores are stored once at the
end. Baseline inner loop has nine instructions repeated eight times per h,
plus outer loop control. Logical score-array traffic removed is
2048*(256B load+256B store)-256B final store =1,048,320B/prediction.
This is logical L1 traffic, NOT measured DRAM traffic. Weight reads and all
131,072 scalar FMA operations per prediction remain necessary and unchanged.

Stack/layout (actual whole-TU .su and clang record layout):
- Predictor function192->192B; callback336->336B; callback object72->72B.
- Initialization-only function288->288B in the final capture layout. No per-job stack growth.
  (The earlier appended-capture draft was304B; it is not the frozen source.)
- Existing std::function stores the same-sized callback, with no new heap call.
See codegen/baseline and codegen/candidate-final. The earlier appended-bool
layout is archived under history/pre-capture-padding and is NOT the final patch.

Validation
1. Actual baseline/candidate headers, actual raw-score prefixes and actual
   prediction callback bodies:130 cases,125 raw score sets,144 callback jobs,
   12 exact-env/native-shape guards PASS under ASan+UBSan with no suppression.
   Finite values including signed zeros compare bit-exact. NaN raw payloads
   are not compared across host instruction emulation; both rankers must reject
   nonfinite scores. Root's real ARM check also passes the same special-value
   cases (subnormals, qNaN/sNaN, infinities, overflow, ULP cancellation/ties).
2. Callback fixtures exercise residual history, blends0/.5/1, gamma fallback,
   trace's additional rankings, resident-key filtering and confidence feedback.
   The callback is extracted verbatim; its queue, pin and model metadata are
   stand-ins. No actual file I/O, DSP scheduling or overlap proof is claimed.
3. Nine actual-source mutants rejected at runtime:reverse h, skipped lane group,
   wrong weight stride, missing hidden/expert/bias/option/native guards and
   accepting a non-exact environment value. Final capture-order edit did not
   change any helper/guard/callback extraction; final whole TU/layout were rebuilt.
4. Actual non-AArch64 fallback ASan/UBSan PASS. Existing expert_prefetch_check.cpp
   ownership/cancellation/rebind/prediction suite PASS against candidate header.
5. Root native ARM binary b234cbf2609e99cdfe972deaa59d8f23421cc9ea1a293e7be1bb666c953730c6
   passed130/125/144/12. Native tested helper and callback body are byte-identical
   to final; only lambda capture layout was subsequently changed and compiled.

Reproduction (paths supplied by caller; no fixed local paths in test runner):
  python3 test/htp/host/run_lfm2_predict_register_check.py \
    --source CANDIDATE_OVERLAY --reference BASELINE_OVERLAY \
    --output OWN_DIR --simde-include SIMDE_ROOT --sanitize
  python3 test/htp/host/run_lfm2_predict_register_mutations.py \
    --source CANDIDATE_OVERLAY --reference BASELINE_OVERLAY \
    --output OWN_MUTANTS --simde-include SIMDE_ROOT
Use --fallback instead of --simde-include for the unmodified non-ARM path.
Use --native-arm --ndk NDK_ROOT --build-only to build ARM check and micro.
Fixtures' environment/affinity controls are Linux/Android only; production
ARM intrinsics are guarded and other platforms keep the scalar source.

Root-run micro (CPU0 only, 31 balanced AB/BA pairs per cohort)
- One warm weight: median18.882242->9.527586us/prediction.
- Sweep of37 separate layer weights:21.877459->14.139568us/prediction.
- All37 top-k outputs match baseline exactly. Micro includes the actual dot,
  finite checks, sigmoid, partial sort and existing vector allocations.
- It excludes residual correction, actual queue/filtering, concurrent model,
  file I/O and DSP. It uses synthetic finite weights, not saved model routes.
  The difference times37 is approximately0.286ms of isolated arithmetic/work;
  it is NOT a decode latency/TPS bound or proof of earlier useful I/O.
  Per-pair raw data and CPU affinity are retained in root-native-micro.log and
  root-native-results.json. No end-to-end performance claim is made here.

Model validation is root-owned and pending. Keep C16, preload128, PAGE0,
GROW0 and all reader/slot counts fixed. Use one compiled core binary with
NNTR_LFM2_PREDICT_REGISTER_ACCUM=0/1 for balanced comparisons. No DSP rebuild is
required. A separate compile-time diagnostic overlay/strict parser is documented
under diagnostic/; it records actual helper calls without affecting normal code.
