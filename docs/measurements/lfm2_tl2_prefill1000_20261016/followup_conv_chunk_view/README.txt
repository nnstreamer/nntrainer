LFM convolution at 2048 rows retained chunk views -- isolated v2 retained

The native affine LFM prefill at 2048 rows now consumes each existing 1920+128
projection ION view before the next RPC. It omits the full projection copy
to the host, while keeping the same FC requests, DSP kernel shapes, exact
arithmetic, tensor sizes and actual staging capacities. State is published
only after both chunks succeed. The previous two gated rows survive the
chunk boundary in the existing tensor. Invalid views use sync_to_host;
unsupported shapes, aliases and backends retain their existing path.

The final patch is 94ae618667ce4f9d822b68d7d03814d0f1100110f7685be012cad17ea7331342,
source 846921eaa9b7da92bb20b7ee891e918b317017e471b0048a1e2b3a05b4708de3.
Only libcausallm.so changes in the seven-file deployed runtime. Core,
executable and the retained GU DSP skeleton are identical to the control.
Model bytes and quantization are unchanged: native TL2 experts, affine
int4 dense weights, FP32 router/norm/convolution weights.

Normal full-model results

S26 Ultra R3CY70LV96T, C16, cache growth 0, PAGE_CACHE=0/Direct I/O, preload 128.
Expert slots 608, decode spares 32 and reader counts are unchanged. Both sides
use predict 2, residual 0.5, stable slots, early prefill, pair reads, demand
handoff and router-input handoff. See raw/*.json for exact environments.
Each length uses A1 B1 B2 A2 A3 B3 and 64 generation steps, profiling off.
All 12 final runs match all 65 saved output IDs (prefill output plus64 steps).

                     Control median   Isolated v2 median
1024 prefill TPS          823.151            842.798
1024 decode TPS            17.6260            17.6357
2048 prefill TPS          801.566            844.188
2048 decode TPS            16.7715            16.7979

2048 prefill median is 5.32% higher; paired changes are +8.07%,+3.27%,-0.21%.
This is a local repeated measurement with run variation, not a guaranteed
gain. The new chunk path runs only at 2048, so the observed 1024 difference
is not attributed to its copy saving. Decode differences are small and are
not claimed as a decode optimization. The 1000 prefill/25 decode targets
remain unmet. Same-handle two-prompt checks at both lengths also match all
reference IDs on both executions.

Largest whole-run sampled peak(KiB), control -> isolated v2:
1024:2733944 ->2733832;2048:3054044 ->3037816.
Final candidate peaks are 2.607/2.897 GiB;2048 is 15.848 MiB below control.
The monitor samples the supervised child from before exec/load through exit,
requested every 20 ms(about 21 ms observed). It sums RssAnon and same-sample
fdinfo allocation sizes deduplicated by mount/inode. All selected peak
samples have complete readable size/identity accounting and no read errors.
This is process allocation accounting, not total physical device memory:
unreported DSP/driver/VTCM allocations and between-sample spikes may be
absent, and allocated fd sizes need not all be resident. Do not add RSS,
VmHWM or the separate app-run-only peak to this composite.

Why v1 was revised

The first candidate extended a shared helper with a row range. Its initial
normal A/B found 2048 prefill+3.55%, but 1024 prefill-0.81% and 2048 decode-2.12%.
Those complete runs remain in this archive. Inspection found extra setup
instructions and an outlined helper on 1024, plus an app frame increase.
The profile also showed increased expert miss wait; it did not establish
the cause of the normal regressions. We do not retrospectively call them
noise or assign all changes to the helper.

Final v2 restores the original full helper literally and isolates the range
helper and noinline chunk function. The old full callback's236 machine-code
bytes and the app's first 9 prologue instructions match the baseline. App
frame 1760 B is restored. The whole app object is not identical: it still
contains the new 2048 guard and branch. The earlier regressions did not recur
in the final paired measurements. This is not proof of their original cause.

Mechanism and memory bounds

PROFILE=1 is separate from normal TPS. The v1 control/candidate pair reports
staging site 5861 at 1735.5 ->295.5 MiB and 120 ->60 full copies; all other
printed copy-site byte/count totals and RPC shape/count/row totals match.
The final v2 profile again reports 295.5 MiB/60 copies. The expected reduction
is 30 *2048 *6144 *4 =1509949440 bytes =1440 MiB. These are logical full-copy
bytes, not measured DDR traffic. Counters print to 0.1 MiB and omit the small
sampled windows. With 30 main convolution calls this is strong aggregate
path evidence, not a direct per-layer branch counter.

Both v1 profiles and final v2 have 2304/2304 experts ready when needed, with
exposed wait printed as 0.0 ms. Final registration totals 2.1 ms. The precision
of those counters does not imply a universal zero-wait guarantee.

Actual extracted fit/stage helpers prove equal requests and capacities:
activation 15732736+1052672 bytes; projection 47190016+3149824 bytes.
There is no larger staging request, extra activation/weight cache, reader,
preload or VTCM allocation. These are per-class bounds, not whole-model peak.
The existing batch API adds small transient argument/handle/chunk vectors:
64 B additional live payload at each RPC in the tested ABI, excluding
allocator overhead. Callback metadata also changes. Final chunk frame is
464 B, nested only at 2048; known app+helper frames total 2224 B, versus 2016 B
in v1. No zero-extra-heap or zero-extra-stack claim is made.

Validation and limits

Author and independent reviewer ran 82 actual app/staging/helper cases with
ASan/UBSan, six assertion-detected mutants, and Android ARM/x86 plus Linux
GCC whole-TU builds. Review adds 12 kept-input cases with poisoned host bytes,
null views, RPC failure and retry. Root repeats the integrated 82-case host
check and executes the final 82-case native NEON binary on S26. These use
actual source helpers with mock projection RPC/allocator; they are not DSP
arithmetic or concurrency simulations. Full-model output/repeat checks above
provide the separate actual device evidence. Full Android app build,
clang-format14 and diff whitespace checks pass. Full Windows/Tizen/Yocto
and remote PR CI were not run. Gemma was not benchmarked again as requested;
the LFM-only guard preserves fallback arithmetic, without an unmeasured
Gemma throughput-invariance claim.

Run test/htp/host/run_lfm2_conv_chunk_check.py with --repo, --source,
--baseline, --simde-include(SIMDe 0.8.2) and --output plus --sanitize.
For native ARM use --native-arm --build-only --cxx with the Android compiler.
The runner needs an explicit baseline source tree, not an inferred git parent.
Exact commands, dependency hash and source hashes are in validation/.
These standalone checks do not add a SIMDe dependency to the general suite.

results.json keeps the two experiments separate. raw/ preserves full commands,
IDs, logs and memory samples; artifacts/ binds each binary set to its source.
validation/ contains frozen patches, source, build/check logs, independent
reviews and same-handle runs. manifest.json hashes every archived file and
also records the original bytes' hash for compressed inputs. The following
gate-tile and router experiments are outside this convolution change.
