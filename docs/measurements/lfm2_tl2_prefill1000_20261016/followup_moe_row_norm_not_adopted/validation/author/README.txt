LFM native TL2 prefill row normalization experiment

Status: production/test source frozen; author host/build checks pass. This is
an opt-in candidate, not an adopted optimization or a measured model speedup.
Root owns native/device/model testing. No root source or phone was modified.
Base f7b16e48ca01a650160d823141e86e67c7993a7c has the same production source as
ccea0d476daf43ddd5dbe201178755e6ff978dfd. All exact hashes are in frozen.json.

The baseline normRows at lfm2_moe_layer.cpp calls the core RMS implementation
over every row, then Tensor.multiply_i(gamma). ARM neon_impl.cpp's RMS outer
H loop is serial. FloatTensor::multiply plus apply_broadcast walks NCHW rows
serially and invokes its ComputeOps ele_mul_fp32 once per 2048-element row,
with alpha=1, beta=0 and both strides=1. HtpComputeOps inherits this stateless
CPU operation from CpuComputeOps. Neither of these primitives starts another
ThreadManager dispatch.

The candidate retains those exact external core functions and arguments. It
submits 16 jobs for M=1024 or 32 for M=2048 to the existing ThreadManager. A job
normalizes 64 consecutive rows, then multiplies those 64 rows by gamma. Each
row retains its RMS sum/reduction, epsilon, scaling and multiply order. Only
the order of independent rows changes. There is no new arithmetic helper,
FP-contract flag, gamma fusion, quantization or vector reduction.

Qualification occurs only in incremental_forwarding's existing in_norm and
not norm_on_accel branch. It requires !out_add, !router_norm, swish rather than
GeGLU, sigmoid rather than softmax, E=64 and intermediate width 1536. The new
helper is compiled only for AArch64 with ENABLE_HEXKL. It requires source and
destination [1,1,M,2048] NCHW contiguous FP32, M=1024 or 2048, a contiguous
[1,1,1,2048] FP32 gamma, the actual HTP singleton ops for both row tensors,
and compatible non-null ContextData identities. Both 64-element expert index
lists must contain TL2CX_WH weights at every index. No weight data is read or
allocated. The exact environment value NNTR_LFM2_MOE_ROW_NORM=1 enables it;
unset, 0 and true remain off. The option is sampled once at the first eligible
call. All checks and expert metadata reads precede the first output write.

The output must be disjoint from the whole source and gamma byte ranges.
Subtraction-based address checks do not construct overflowing end pointers.
Source and gamma may overlap each other because both remain read-only. The
qualified dimensions bound all offsets: each last job ends exactly at M*2048,
and each gamma read is exactly 2048 floats. There is no tail pad or extra row.
The caller holds the Tensor views, weights and backend for the whole call.
ThreadManager::Global initializes its existing process singleton as usual;
parallel_for waits for every worker and its acquire fence before returning.
The router runs only afterwards. No callback starts nested parallel work.
The task slot may retain dead pointer values until the next dispatch replaces
it, as existing ThreadManager call sites do; it cannot invoke them after join.

The real model builder creates HTP residual_add -> sparse MoE with swish,
sigmoid_bias, in_norm=true, I1536 and E64. LayerNode::configureRunContext
attaches that layer's common context to its input, scratch and weights, and
getSharedDataTensor carries the context. Compaction of active experts occurs
later in tryMoeLayerOnAccelerator; the norm gate sees the original 64 experts.
This establishes intended eligibility without claiming a device dispatch
count before root's integration test. Gemma, decode, other sizes/dtypes,
non-HTP and incompatible/overlapping views keep normRows. No Gemma device
benchmark or unchanged-performance claim is made.

Memory/code generation
There is no new activation tensor, ION arena, expert cache, reader or worker
capacity. Logical source/output/gamma read and write bytes are unchanged;
the intended benefit is scheduling and locality. A 64-row working region is
512 KiB per row tensor. Baseline and candidate nominally perform the same six
full-row transfers per result (two source reads, RMS write, gamma/output
reads, final write), before actual cache effects.

Under the existing NDK r30 app flags, the incremental caller frame is 3008 B
before and after. The original normRows function's 268 machine-code bytes are
identical (SHA 02867b0fcd7e55d92dbcacb7956c23dd875afbbc87b18c9bcd5cd9e8a376ee47).
The new outlined helper has a 256 B frame, parallel_for specialization 112 B,
and callback 48 B, excluding the unchanged core callee frames. The callback
captures 40 B and libc++ requests a 48 B task object in the existing task slot;
assignment can temporarily overlap the previous task object's allocation.
The enabled bool and initialization guard name 9 B of static state; object
merged-BSS growth including layout is 16 B. These small overheads are not a
claim of zero additional memory. The codegen directory records commands,
stack usage, object symbols and small relevant disassembly slices. Linux GCC
non-ARM .text was identical; Android x86 .text was not byte-identical despite
the helper being compiled out. Compilation and fallback correctness are the
supported cross-platform claims, not unchanged whole-app code or performance.

Validation
Actual helper plus actual incremental caller block were extracted, not
recoded, for 89 cases under each of env=1/0/unset/true. ASan and UBSan halt
on error with no suppressions. The host fixture uses extracted ARM RMS and
elementwise NEON bodies through SIMDe; Tensor/ops metadata, fixed broadcast
and ThreadManager are stand-ins. It checks normal/special FP32 bits, untouched
source/gamma/canaries, row coverage, 1/8 workers, ascending/reverse/shuffled
work stealing and completion, actual caller fallbacks, shape/type/format/
contiguity/HTP/context/expert-list guards, all-null pointers and overlapping
output ranges. Each of eight mutated production guards/offsets/arithmetic
calls fails its intended runtime assertion. Early mutation draft failure
from an unused-parameter warning was corrected to preserve the reference;
final-mutations.log/results.json contain runtime rejections for all eight.

Whole production TUs compiled before/after for NDK AArch64, NDK x86 and Linux
GCC. Changed C++ lines and both C++ fixtures match clang-format 14. No complete
Windows/Tizen/Yocto build or device arithmetic claim follows from these builds.

The native fixture uses the actual linked libnntrainer Tensor, RunLayerContext,
HTP CPU ops, core RMS, Tensor.multiply_i reference and real ThreadManager. It
has 30 checks per process, with M1024/M2048, finite and special values, caller
fallbacks, real Tensor alias/context and final-expert dtype guards. It does
not call DSP or load expert weights; weight objects use valid ZEROS initializer
metadata with alloc_now=false, and each one asserts isAllocated()==false.
The first native harness attempt failed before the helper because NONE is not
a legal Weight initializer even with deferred allocation; root-native-1-8.log
retains that failure. Native-v2 fixes only the test initialization. Its final binary SHA is
e936c5201916b2f43f0bc86ece3b45b71563117d1ad2ef941f8bd3ac79f59943.
Root ran native-v2 with env=0/1 and argument 1/8: all four combinations
passed 30 checks each, including six exact numeric cases. The root-native-fixed
logs and root-native-fixed-results.json retain commands, runtime locations and
log hashes. Author did not run the phone. Normal model timing is still pending.

The preliminary root CPU-only micro used the original extracted normRows and
the same core RMS/multiply functions. Six normal/special cases were exact.
At 1024 it measured 426.146 -> 200.729 us; at 2048, 1176.354 -> 545.312 us.
These are fixed-input CPU measurements, excluding expert I/O and DSP. The
per-call deltas sum arithmetically to about 8.6/24.0 ms over 38 layers but do
not predict or bound real model gain. Cache/memory bandwidth contention and
pool scheduling can offset them; the rejected conv Tile8 experiment already
demonstrated this limitation. Only root's balanced normal model A/B can decide
whether to retain this candidate. C16/PAGE0/preload128 and all existing data
paths stay fixed. No further row/chunk-size tuning is included.

Reproduction (all output directories are separate from the checkout)
  python3 test/htp/host/run_lfm2_moe_row_norm_check.py \
    --repo CHECKOUT --source CANDIDATE --simde-include EXTERNAL_SIMDE_0.8.2 \
    --output NEW_HOST_OUTPUT --sanitize
  python3 test/htp/host/run_lfm2_moe_row_norm_mutations.py \
    --repo CHECKOUT --source CANDIDATE --simde-include EXTERNAL_SIMDE_0.8.2 \
    --output NEW_MUTATION_OUTPUT
  python3 test/htp/host/run_lfm2_moe_row_norm_native.py \
    --source CANDIDATE --output NEW_NATIVE_OUTPUT \
    --compiler NDK/bin/aarch64-linux-android29-clang++ \
    --nntrainer-include INSTALLED_CORE/include/nntrainer \
    --nntrainer-lib-dir INSTALLED_CORE/lib/arm64-v8a
The native script only builds; deployment/execution is deliberately external.
Exact local compiler inputs and core hashes are retained in commands.json.
production.patch changes one app source; tests.patch adds the five fixtures/
runners. No canonical repository files were changed by this author.
