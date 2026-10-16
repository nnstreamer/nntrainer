LFM2048 convolution chunk view, isolated helper v2

Frozen production:94ae618667ce4f9d822b68d7d03814d0f1100110f7685be012cad17ea7331342
Frozen app source:846921eaa9b7da92bb20b7ee891e918b317017e471b0048a1e2b3a05b4708de3
All source/test/native binary hashes:source-hashes.json.
One production file:Applications/CausalLM/layers/conv_block_layer.cpp.
production.patch applies to original baseline; v1-to-v2.patch applies to2392af87v1.
Both actual local-tree apply checks pass. No root files or phone were modified.

Purpose and separation
V1 removed redundant2048 projection host copies, but normal phone measurements
also showed small consistent1024 PF and2048 decode decreases. V2 is a NEW
candidate to isolate unchanged paths; it does not inherit a v1 speed claim.
The old six-argument full helper and lambda are restored verbatim. The range
helper has a distinct name and is defined only for AArch64, called only by the
2048 chunk helper. Chunk helper is noinline when the compiler reports a GNU or
MSVC noinline attribute. No shared portable macro existed; this follows the
feature-tested local attribute convention in avx2_impl.cpp68-76, with the actual
noinline feature probe and an empty fallback for unsupported compilers.

The actual Android compiler restores the full callback's59instructions/236B
machine bytes exactly, including row setup and hot loop. Full helper is inlined
as before. Common app frame is1760B again(v1:1824B); first9 prologue instructions
match baseline exactly. Whole app still grows7860->8212B from the added2048
route, so full M1 byte identity is NOT claimed. V1 app was10052B. V2 chunk has
its own464B frame, only reached at2048; known app+helper frames are2224B versus
v1's1824+192=2016B, an additional208B on that2048 call chain. No large scratch
or activation allocation is added. These are static codegen facts,not timings.

2048 route semantics unchanged fromv1
Exact K=C=N2048,M2048,native affine input/output weights,FP32 contiguous tensors,
!compare,supported backend,not(HTP-owned state and from>0). Every mutable host
range is pairwise disjoint and disjoint from full input/conv weights before
first RPC. Unsupported or overlapping inputs use the original fallback before
new work. Each retained view is also checked against still-live inputs and
workspaces; missing/unusable views are materialized by existing sync_to_host.

The existing1920+128 projection RPC shapes remain. Pregate then conv/postgate
blocking row work consumes each retained view before another engine call.
Global gated history preserves rows1919/1918 for tail1920 and1920/1919 for1921;
projection gate rows are chunk-local(r-first). Multiply,two FMA,postmultiply FP
order and precise pragmas remain. State saves global last2rows only after both
chunks succeed. RPC/sync errors propagate with state unchanged; retry recomputes
correctly. Output projection and ownership/from rules are unchanged.

The existing strided invokeLayer(to_host=false) does sampledCopy+noteIonAlias
without creating copied_out.kept, so a later same-class stage does not restore
an already-consumed projection in full. All view readers have joined first.
Host_view remains a read-only single-owner view until next engine call.

Capacity/copy/metadata evidence
Actual extracted stage/fit helpers make identical requests and allocations:
 input requests15728640+1048576B,capacities15732736+1052672B;
 output requests47185920+3145728B,capacities47190016+3149824B.
Fit allocation is page-rounded request+one page; size class is a map key,not
proof of spare capacity. No class/request grows. Existing model buffers can
already be larger; these figures are not whole-model peak memory.
Logical full staging copies per conv64->16MiB;48MiB projection copy removed.
Both sampled windows remain4224B/layer;30layers remove1440MiB of full-copy
accounting,1439.879MiB net copy payload after windows. Not measured DDR traffic.
QKV_TO_HOST or profile>=2 forces full copies as before; normal/profile1 only.

No new activation/weight/content-cache buffer,reader,preload,quant or core/DSP
API. Small metadata allocations remain fromv1 and are NOT claimed zero:
five1element argument vectors32B and extra handle/chunk vectors32B held during
RPC. Geometric growth is11additional allocations/88cumulative payload B perRPC,
excluding allocator overhead. This is a source/ABI estimate,not an allocation
profiler. Existing ThreadManager callbacks also allocate metadata; full1024
callback returns to48B baseline allocation while ranged callback remains56B.
Callbacks may remain in existing task_ until replacement but all workers finish
before return; no projection content is kept by a new cache. Root monitor is
required for actual complete process peak.

Validation
-82 actual app/range/staging-helper cases pass ASan+UBSan(SIMDe0.8.2+host FMA):
 finite,signedzero,subnormal,Inf,NaN;1/4concurrent lanes;same-ops repeat;default
16/17/31/64/1024/2048 and tiny range tail;dtype/from/owns/alias fallback;null or
bad view sync;either RPC/sync failure then retry;state/canaries. Dead first ION
and nonsampled host projection bytes are poisoned. Actual baseline fullhelper
literal identity is required by the runner before compilation.
-6/6 range-history,projectionoffset,input-overlap,sampled-host,state-before-
 success,missing-from source mutants rejected for intended assertions.
-Whole Android ARM64 TU,Android x86_64 TU,Linux GCC TU pass. No new unused helper
 or noinline warning. clang-format14 roundtrip is identical.
-Same82cases built as native ARM NEON(libc++static),binary
 checks/native-arm-final/check SHA874e77d22921aa273d2decd9612d10343649a904b96787d2f5b5ee8e9fd5f5ca.
 Root executed this exact SHA on S26:82PASS. Evidence:root-native-arm.log/json.
 Host/native fixtures use fake projection RPC and allocator; they do not prove
 DSP arithmetic,FastRPC coherence,model IDs or TPS.
-Independent reviewer reports82ASan/UBSan,6mutants,wholeAndroidTU PASS; final
 retained-input/codegen cross-check is reported separately. No production changes are
 planned after final846921ea freeze. Normal v2 A/B,65IDs,repeat and peak remain
 root-owned. Preserve v1 evidence separately.

Reproduction
python3 test/htp/host/run_lfm2_conv_chunk_check.py --repo REPO --source CANDIDATE --baseline BASELINE --simde-include SIMDE_ROOT --output OUT --sanitize
python3 run_mutants.py --repo REPO --simde-include SIMDE_ROOT --output MUTANT_OUT
Native:replace SIMD argument with --native-arm --build-only --cxx NDK_AARCH64_CLANGXX;
run OUT/check with no arguments. No user-local paths in committed fixtures.
Compiler/extract/source manifests:checks/{host-san-final,native-arm-final}/commands.json.
Android TU:android-command-final.json;codegen:codegen/summary-final.json and analyze.py.
SIMDe0.8.2 remains external,never a vendored repository edit.
