LFM2048 convolution: consume the existing1920+128 projection ION views

Frozen production.patch
  4a6661df1b353699c22151b4d36e60ba5f341811c5c8766e5a615f8fbdec85a7
Frozen tests.patch
  2845812902815c03a8b35460314f08b5c2125caed7f9600549f145428353906c
Only production file: Applications/CausalLM/layers/conv_block_layer.cpp
  before bc019abdbf0cadb48e0aaa4ecc3898bc9b0a37e11f28c306d9762f766292869a
  after  2392af8742e5ac2c0679bbb26364d50e0d12e0d0254ebc0ed03c17459f89f8f9
Base7a6f088c13f80f2e6975d76c2cc35416babd8abe; no core/DSP/IDL/API changes.
Author did not edit root/original worktrees, use the phone, commit or push.

Scope and arithmetic
The new path requires ARM64, native affine input/output weights, FP32 contiguous
input/projection/gated/conv_output, K=C=N=2048 and exactly2048 prompt rows. It
retains the existing two projection RPCs:1920 then128. The existing DSP internal
norm_ld split therefore stays1728+192 then128; handle order, affine weights,
per-row quantization and projection math do not change. The caller passes zero
post_chunk values rather than a null table; both mean no pre/post norm or RoPE.

For each chunk, its view is consumed before the next engine call: pre-gate
parallel_for completes, then the convolution/post-gate parallel_for completes.
The existing helper now accepts a global first/last row range and a projection
pointer local to that range. Tail row1920 reads gated1919/1918; row1921 reads
1920/1919. It does not read the first projection view after another RPC. The
current source ThreadManager waits all worker threads and acquires completion
before returning. State is copied from global gated rows2046/2047 only after
both chunks succeed, then the unchanged output projection runs.

The same multiply, FMA, FMA, post-multiply rounding order and precise pragma
remain. NDK ARM -O3 row assembly has the same fmul/fmla/fmla/fmul/store sequence,
and its worker lambda remains stackless. Existing full-range helper calls use
first0,lastrows, including1024. This is semantic/arithmetic preservation, not
an identical1024 object-code claim. Decode, Gemma width, compare and other
unsupported cases continue through the old code.

Ownership and fallback
All mutable host ranges(projection,gated,output,state) must be mutually disjoint
and disjoint from input and convolution weights before any new-path RPC. This
is stricter than the old sequence: CPU writes between RPCs must not modify an
unread tail input. Alias/in-place or partial overlap declines before work and
uses the existing fallback. The app invokes this only for its own graph tensor
storage. Engine host_view is a read-only view valid only until the next engine
call; arbitrary external const-cast writes to engine-owned ION remain outside
that pre-existing contract.

A null or overlapping returned view is materialized with sync_to_host before
CPU consumption. Backend/sync failure propagates; state has not been changed.
A retry starts from local row0 and recomputes both chunks. from is an external
incremental-forward offset, not a local convolution-history offset. The old
htp_owns_state_&&from>0 rejection still falls through to the old error path;
from0 can initialize a second prompt after decode, as before.

The strided invokeLayer path used here does sampledCopy+noteIonAlias. It does
NOT record copied_out.kept. Therefore the later stage of the same class does
not restore this dead projection to the host. The view is already consumed,
so only the sampled host windows are ever needed. This proof does not reuse
trueSource/kept behavior for these projections. Actual kept input subranges
are a separate existing stageInput behavior, additionally checked by review.
NNTR_HTP_QKV_TO_HOST=1 or profile>=2 still forces full projection copies through
the existing API. Those modes are not copy-saving evidence.

Capacity and copy proof
The two requests are identical to old invokeFc. Actual extracted stage/fit
on fresh class buffers gives:
  rows  input request/allocated      output request/allocated
  1920  15728640/15732736 B          47185920/47190016 B
   128   1048576/1052672 B            3145728/3149824 B
No new staging class or larger request, activation array, expert cache, reader,
preload, weight format or DSP scratch is introduced. Existing model pools may
already hold larger same-class tensors: the claim is equality of requests and
maximum capacity requirements, not that this table is a whole-model peak.
The shared app gated/projection/output tensors retain their existing sizes.

One2048 conv layer can omit48MiB of full projection copy, retaining the16MiB
input copy. Sampled windows write33*64bytes per chunk,4224bytes/layer. Across
30conv layers this is1440MiB fewer stagedMemcpy bytes(1439.879MiB fewer actual
copy payload bytes after sample writes). These are logical copy bytes, not
measured DDR bandwidth or a predicted wall-time/TPS gain. No work is moved
outside the prefill timer and no backend kernel count is reduced.

Small metadata costs are not zero
The existing batch-norm API creates five one-element argument vectors each
RPC: weight pointer8B,scale pointer8B,destination pointer8B,width4B,post_chunk4B
on ARM64(total32B payload). Internally it creates handles[3] and chunks[3]
(maximum capacity4 each in the used libc++,16B each), an empty gamma vector,
and slices[3](OutSlice24B,max capacity4=96B). The slices and invokeLayer's
out_off/out_ld vectors also exist in the old contiguous-conv route; the
argument/handle/chunk vectors are additional. With geometric1/2/4 growth,
those extra vectors request88 cumulative payload bytes in11 allocations/RPC,
with64B held extra at the RPC, excluding allocator overhead. These are source
and ABI counts, not a whole-app allocator measurement or an extra-heap0 claim.
All these vectors die when that RPC returns; no projection content is cached.

The existing ThreadManager task_ also owns callable metadata. Splitting CPU
work makes4 submissions instead of2; the postgate callable capture is enlarged
by the first-row reference. It may stay in the existing task_ until replaced,
but has no live worker after the barrier. NDK codegen reports app forwarding
stack1760->1824B and a192B range-helper frame; the latter can be nested under
the app frame. Worker row stack remains0B. Thus stack/metadata are disclosed
separately from unchanged activation/ION capacity. Root's complete-run memory
monitor, not these static counts, decides actual peak impact.

Validation
-82 actual app/range/staging-helper cases under ASan+UBSan, SIMD through
 SIMDe0.8.2 with host FMA. Patterns include finite,signedzero,subnormal,Inf,NaN;
1/4concurrent lanes,repeat generations,full-range16/17/31/64/1024/2048,tiny
 final subrange,actual app dtype/from/owns-state/alias guards,invalid view sync,
 failure on either RPC/sync and successful retry. Future output canaries are
 checked. First ION is poisoned when the second RPC starts, proving no old-view
 read. The full host projection is poisoned between sampled windows.
-Actual stageInput/stage/fit/host_view/sync/ionAlias helpers are extracted from
 the unchanged core. RPC arithmetic is a deterministic projection-byte stand-in;
 numeric convolution/gating uses actual app helpers. No DSP/FastRPC arithmetic
 or performance claim follows from that stand-in.
-The SAME82cases build with real arm_neon.h, no SIMDe, NDKr30/libc++static.
 Root ran the binary on S26 with SHA8293faf7b7bed96be45d9adb23e8cebfea551d40fc7c3023f81fa961e89ee69c and reported82PASS; root-native-arm.{log,json}.
-Existing retained-route48cases PASS. Six source mutants are all rejected:
 chunk-local previous row,wrong projection offset,missing input overlap guard,
 sampled-host consumption,premature state copy,missing owns/from guard.
-Whole app TU builds: Android ARM64,Android x86_64,Linux GCC PASS. clang-format14
 leaves frozen source unchanged. Existing header warnings are unchanged.
-Independent review reran82cases and6mutants;12additional actual kept-input
 cases(out_pool/x2_pool,poisoned host,1/4lanes,nullview,secondRPCfail+retry)PASS.
 Review artifacts: /tmp/nntr-lfm2-conv-chunk-view-review-20261016/.
-Normal model65IDs,complete memory,repeat prompts and end-to-end A/B belong to
 root and are pending at author freeze. Native fixture is not a model replay.

Reproduction(choose external build/dependency paths; no user-local paths baked
into committed tests):
  python3 test/htp/host/run_lfm2_conv_chunk_check.py --repo REPO --source CANDIDATE --baseline BASELINE --simde-include SIMDE_ROOT --output OUT --sanitize
BASELINE and CANDIDATE contain Applications/CausalLM/layers/conv_block_layer.cpp.
SIMDe0.8.2 source tarball SHAed2a3268658f2f2a9b5367628a85ccd4cf9516460ed8604eed369653d49b25fb. Ubuntu0.7.2 lacks requiredvfmaq and was not used.
Native: replace SIMD option with --native-arm --build-only --cxx NDK_AARCH64_CLANGXX;
run OUT/check without arguments on Android. Only libc/libm/libdl are needed.
Commands/extraction hashes are in checks/{host-san,native-arm}/commands.json.
Full patch/source/test hashes are in source-hashes.json; report hashes in manifest.json.
