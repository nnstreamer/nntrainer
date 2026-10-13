# Gemma 4 memory profile: PR 4416

The memory profile keeps the measured peak below 4 decimal GB with model page
cache disabled and below 6 GB with it enabled. It was measured on 2026-10-13 on
a Galaxy S26 Ultra (SM-S948N, Android 16, Hexagon v81), using PR revision
`87f8580f5916d07cbb75dec660009edfaa407df1` plus the memory changes in this tree.

## Memory-only measurements

The workload is Gemma 4 26B-A4B, QS2CX_WH experts, QS4CX FC weights, Q4_0
embedding and untied LM head, batch 1, 2,048 input tokens and 128 generated
tokens. In this application's counting convention, prefill produces the first
output token and `num_to_generate: 127` produces the remaining 127. Decode TPS
below uses those 127 decode iterations, including the first iteration's setup.

| Run | Expert cache C | Sampled total peak (GB) | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| Original PR, legacy page cache | 16 | 8.377* | 645.243 | 9.948 |
| Memory profile, cache OFF | 12 | 3.586 | 682.439 | 7.027 |
| Memory profile, cache ON | 12 | 5.312 | 687.479 | 8.388 |
| Memory profile, OFF after ON | 12 | 3.645 | 663.427 | 6.961 |

These are individual runs, not confidence intervals. The two OFF runs give
headroom below 4 GB, including a run immediately following a buffered run.
The smaller expert cache and storage reads reduce decode throughput relative to
the original C=16 configuration. C is `NNTR_MOE_CACHE_EXPERTS`: it contributes
12 expert pairs per MoE layer to a shared pool, giving 360 slots for 30 layers.
It changes retention, not the router's top-8 selection or the model's weights.

All 128 generated token IDs match the original PR in all three optimized runs.
All runs used self-generated tokens, with identical decode NLL sum
`30.548928186064231` over 127 scored positions. This establishes equivalence for
this workload, not a general accuracy benchmark.

Raw CSV samples, token IDs, compact runtime logs, exact workload configuration,
source and binary hashes are in
[`docs/measurements/pr4416-memory-87f8580`](../measurements/pr4416-memory-87f8580/).
The model file was 7,540,724,856 bytes. Its contents were not read to compute a
hash during measurement, to avoid warming its cache.

## Final decode configuration

The optional `--decode-fast` profile starts prefill at C12 and grows the shared
expert pool by 22 pairs per layer after the first decode token, reaching C34
(1,020 pairs). It also discards expired CPU prefill scratch and prefetches
predicted experts directly into reusable DSP arena slots. Prediction only
schedules reads; the real DSP router still selects all experts and weights.
The final Android binaries include all memory and throughput changes.

| Page cache | Sampled total peak (GB) | Prefill tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| OFF | 3.871 | 656.831 | 12.146 |
| ON | 5.478 | 672.137 | 12.072 |

These are individual full-length runs on the same 2,048 + 128 workload. Decode
includes all 127 steps and first-token setup, not just a faster final window.
Both use normal greedy generation with `NNTR_GENERATED_IDS` to record IDs,
without `NNTR_PPL_DECODE`; all 128 IDs match the original PR. The memory-only
table above used NLL scoring, which requests full logits and adds CPU work, so
the tables should not be treated as a controlled throughput speedup ratio.
A separate final OFF run after ON scored all 127 decode steps: NLL sum remained
`30.548928186064231`, all 128 IDs matched, and its sampled peak was 3.873 GB.
The final OFF/ON/check runs had no swap; maximum sampling gaps were 78.6,
116.3 and 110.3 ms respectively.
The requested 15 tok/s decode target was not reached; tuning stopped at the
user's request.

The read path coalesces each weight's code and adjacent metadata, using a
bounded 2 MiB direct-IO buffer. Arena growth allocates only the remaining slots
at its tail instead of another full 256 MiB chunk. CPU prediction borrows the
existing router weights and applies gamma to its input; it does not retain an
additional approximately 43 MB folded-weight copy. Outstanding prediction and
read jobs are drained before inference returns.

The CPU scratch change reduced resident memory by approximately 69 MB in the
paired C34 measurements. Its log reports 295 MiB of virtual pages advised away,
which must not be confused with resident bytes saved. Persistent tensors and
whole pages shared with them are preserved; prior inference outputs must have
been consumed before calling `releaseInferenceScratch()`.

Final samples, generated IDs, runtime logs, settings and artifact/source hashes
are in [`decode-speed`](../measurements/pr4416-memory-87f8580/decode-speed/).
One final launch failed during the first DSP expert registration with
`0x80000402` (DSP `AEE_ENOMEMORY`). The same binaries and settings succeeded on
retry; the cause was not established. The failed launch is retained in the
evidence and is not counted as a successful performance measurement.

## Using the profile

Deploy the HTP-enabled application, core libraries and matching v81 skeleton,
then copy [`run_gemma4_memory.sh`](../../Applications/CausalLM/run_gemma4_memory.sh)
to the application directory. Use the device's FastRPC library, not the SDK's
link-time `libcdsprpc.so` stub.

```sh
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  sh ./run_gemma4_memory.sh --page-cache off ./nntrainer_causallm MODEL_DIR

LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  sh ./run_gemma4_memory.sh --page-cache on ./nntrainer_causallm MODEL_DIR

# Final throughput configuration; change off to on to enable the model cache.
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  sh ./run_gemma4_memory.sh --decode-fast --page-cache off \
  ./nntrainer_causallm MODEL_DIR
```

The saved workload has `init_seq_len: 2048`, `max_seq_len: 2176`,
`num_to_generate: 127`, `skip_prefill: false`, greedy generation and EOS disabled
for a fixed output length. Adjust its absolute model/tokenizer paths on another
device. The script overrides the saved `moe_cache_experts: 16` with C=12. It
enables E2E decode, FC weights in the shared arena, int8 decode attention and
release of CPU KV history after seeding DSP decode. Default CPU threads: 8.

The profile is for one prompt per process. Its existing `NNTR_HTP_KV_DROP_CPU`
mode releases history needed by a subsequent prefill; start a fresh process for
another prompt. `NNTR_HTP_KEEP_ARM_WEIGHTS` is rejected because retaining host
copies defeats this profile. Prepared TL2 files are not supported by the new
page-cache reader; unset `NNTR_HTP_TL2_PREPARED_FILE` when using the profile.

The underlying `NNTR_HTP_PAGE_CACHE` setting is optional:

| Value | Weight-file behavior |
| --- | --- |
| Unset | Existing loading behavior |
| `0` | Linux/Android `O_DIRECT`; no buffered fallback |
| `1` | Buffered positioned reads with an approximately 2 GB cache budget |

Both explicit modes first advise eviction of old clean pages for the model
file. They never change the system-wide page-cache setting. OFF controls model
weights, not shared libraries, tokenizer files or unrelated applications. ON
tracks 1 MiB file ranges with a process-wide LRU and advises eviction with
`POSIX_FADV_DONTNEED`. Kernel eviction is advisory: observed model cache peaked
at 2.074 GB in the memory-only profile and 2.125 GB in the final fast profile.
Other processes mapping the file can retain additional pages.

## Removing retained duplicate tensors

- Q4_0 embedding reads only requested token rows instead of retaining the full
  embedding table. The separate tied-embedding implementation is unchanged.
- DSP-resident FC and dense weights skip unnecessary CPU repacking. Their raw
  host pages are discarded after registration only when there is no CPU
  consumer of that tensor.
- Prefill and decode share the same 16 DSP LM-head handles. Registration uses
  bounded slices; the old 396 MiB host source is released after binding.
- Prefill quantized KV buffers are registered on demand and released after each
  layer finishes. The authoritative history remains available until DSP decode
  seeding, when the corresponding CPU history is released.
- Direct IO uses a bounded 2 MiB scratch buffer per reader, because filesystem
  direct IO cannot pin some FastRPC mappings. It does not retain another tensor
  or silently fall back to buffered IO.

Temporary conversion and transfer buffers are still necessary. The changes
remove unused retained representations while preserving CPU consumers and
tensor lifetimes.

## Measurement and validation

[`measure_memory.cpp`](../../test/htp/host/measure_memory.cpp) samples from process
creation through loading, prefill and decode. At each sample it adds:

```text
CPU RSS (RssAnon + RssFile + RssShmem)
+ unique exported DMA-buffer bytes from the process's fdinfo
+ resident model-file pages observed by mincore without faulting them in
+ VmSwap
```

DMA allocations are deduplicated by inode, including the process's exported
FastRPC allocations. The total is the maximum simultaneous sum, not the sum of
independent component maxima. It excludes unrelated OS/app memory. ON's peak
includes swapped memory; maximum VmSwap in that run was 74,711,040 bytes. Both
OFF runs had zero VmSwap. OFF after ON initially observes leftover cache before
the model opens, then reports zero model-file residency after eviction.

The final sampler sleeps 50 ms between samples; maximum observed gaps were
75.0, 104.8 and 125.2 ms. Shorter peaks may be missed. *The legacy baseline used
an earlier 100 ms sampler without VmSwap, and its mmap file RSS can overlap
model cache accounting. It is a diagnostic comparison, not an exact physical
memory subtraction against the optimized runs.*

Build the sampler with the Android NDK:

```sh
"$ANDROID_NDK/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android26-clang++" \
  -std=c++17 -O2 -static-libstdc++ test/htp/host/measure_memory.cpp -o measure-memory
```

On the device, from the deployed application directory:

```sh
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. \
  ./measure-memory off.csv /path/to/model.bin \
  sh ./run_gemma4_memory.sh --page-cache off ./nntrainer_causallm MODEL_DIR
```

For token/NLL validation, set `NNTR_PPL_DECODE` to a **new, nonexistent** output
path. An existing path instead enables teacher forcing and is not a check of
independent generation. The evidence files retain `source=self` for every run.

Validation completed:

- Android NDK r30 core and CausalLM builds; Hexagon SDK 6.4.0.1, tools 19.0.04,
  HexKL 1.0 beta.2 (6.4.0.2 library directory), v81 skeleton.
- `bash test/htp/host/run_host_checks.sh`: exit 0, including graph, attention,
  Q4, MoE and new weight-reader checks. Optional AArch64 NEON emulation was
  skipped because its cross-compiler/QEMU were unavailable.
- Weight-reader checks on host and Android: unaligned/parallel reads, EOF and
  overflow, direct mode without cache population, bounded buffered eviction,
  invalid-policy rejection. Non-Linux fallback compiled on the host with the
  Linux preprocessor macro disabled; this is not a Windows build.
  The Android fixture directory is `/data/local/tmp` (f2fs), passed as the
  check executable's argument. Android `/tmp` is tmpfs and does not provide
  the direct-IO residency behavior this check requires.
- Tiny CPU-only inference with cache unset/OFF/ON produced identical
  `tok14tok12` output.
- Shell syntax, clang-format 14 on changed C/C++ lines and `git diff --check`.
- The final TensorPool Android suite passed all 50 tests, including preservation
  of persistent aliases, partially shared pages and non-CPU allocations during
  scratch release. Prefetch ownership/cancellation and the updated reader host
  checks also passed.

Remote CI and the full cross-platform build matrix have not been run.
