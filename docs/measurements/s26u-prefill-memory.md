# S26 Ultra: memory at C = 12 and 16 after the prefill commits, page cache off

Measured 2026-10-14 on the Galaxy S26 Ultra `R5KL308PALH` (SM8850, Hexagon
v81), PR 4416 at `e7b64c8c6` (the five prefill commits on `788e629f9`),
Gemma 4 26B-A4B QS2CX_WH experts with QS4CX FC weights, batch 1, 128
generated tokens. One run per cell. The environment is the prefill
performance one, not `run_gemma4_memory.sh`'s profile (doc
`htp_attention/61`): no `NNTR_HTP_RELEASE_PREFILL_SCRATCH`, no
`NNTR_MOE_CACHE_GROW`, `NNTR_HTP_PAGE_CACHE=0` (O_DIRECT model reads):

```sh
NNTR_NUM_THREADS=8 NNTR_HTP_E2E=1 NNTR_HTP_E2E_FREE_KVQ=1 NNTR_HTP_KV_DROP_CPU=1 \
NNTR_HTP_DROP_HOST_FC=1 NNTR_HTP_ATTN_M1_Q8=2 NNTR_HTP_FC_IN_ARENA=1 \
NNTR_HTP_PAGE_CACHE=0 NNTR_MOE_CACHE_EXPERTS=<C>
```

## Peak

`measure-memory` is [`test/htp/host/measure_memory.cpp`](../../test/htp/host/measure_memory.cpp)
(CPU RSS + the process's DMA-buf bytes + resident model-file pages + swap,
50 ms samples, decimal GB as it prints). The second column is an
independent check: the drop of `/proc/meminfo` `MemAvailable` from before the
process started to its minimum, 0.3 s samples (`memsample.sh`).

| Cell | measure-memory peak (GB) | MemAvailable drop (GiB) | Peak at | Decode, mean (GB) | Prefill tok/s | Decode tok/s (last 64) |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| 1024 tokens, C = 16 | **3.686** | 3.27 | 7.0 s, prefill | 3.250 | 587 | 8.8 |
| 1024 tokens, C = 12 | **3.335** | 3.19 | 7.2 s, prefill | 2.808 | 584 | 8.1 |
| 2048 tokens, C = 16 | **3.901** | 3.67 | 8.8 s, prefill | 3.268 | 669 | 8.7 |
| 2048 tokens, C = 12 | **3.563** | 3.13 | 8.5 s, prefill | 2.853 | 670 | 7.9 |

The peak is during the prefill in every cell, once the loader has released
its host copies and the arena, the DSP heap and the staging buffers are all
mapped. C = 16 costs 0.33-0.35 GB more than C = 12: 120 more expert slots.
The answers are the same bytes as every other run of this build (md5
`dc6144f6` at 1024 tokens, `b30eae5a` at 2048).

With the page cache off the decode runs at 8-9 tok/s instead of 15: every
expert miss is an O_DIRECT read. The prefill is within 2-5 %. The same four
cells with the default loader (page cache on) peak at 4.4-4.9 GiB by the
MemAvailable measure (`meminfo_summary_page_cache_on.txt`), with the
loader's host copies (RSS 2.0-2.6 GiB) and the DSP heap (590-680 MiB,
against 140-200 here) both larger.

## Where it is

At the measure-memory peak (1024 tokens) the CPU side is 244 MiB of
anonymous memory and 38 MiB of file pages; all the rest is DMA-buf: 3233
MiB at C = 16, 2899 at C = 12. The DMA-buf bytes, from the application's
own counters (`[HTP] arena chunk`, `[HTP] e2e: fc arena`, `[HTP] mem`):

| DMA-buf, MiB | C = 16 | C = 12 | What |
| --- | ---: | ---: | --- |
| Expert pool (arena chunks 0-2) | 704 | 576 | 1.45 MiB a slot x 480 / 360, chunks rounded to 64 MiB |
| Prefill FC, WH layout (arena chunks 3-6 / 3-5) | 1024 | 768 (+54 in the pool's last chunk) | 206 weights, 786 MiB of codes and scales; at C = 12 the pool's last chunk has room for the tail, at C = 16 it does not |
| Decode FC, Q4M1 layout (`e.arena`, 2 chunks) | 448 | 448 | 396 MiB used: the M = 1 GEMV's own copy of the same FC set |
| Staging pools | 177 | 177 | 290 at 2048 tokens; 94 / 242 released after the prefill |
| DSP heap and other FastRPC allocations | ~880 | ~930 | `s1_heap` 190 / 184 MiB after graph init (the int8 KV cache is 220 MiB of it at 1024 tokens); the rest is not itemized by any counter |

So the FC weights are on the NPU twice, 786 + 396 MiB, besides the host
copies the loader holds until it drops them (786 MiB at load, 396 at the
first token). The 448 MiB decode copy and the 238 MiB the WH chunks leave
empty at C = 16 (1024 mapped for 786) are the two slack items; the latter is
the 256 MiB chunk size in `registerRm`'s `place` call.

## Reproducing

From `docs/measurements/s26u-prefill-memory/`, with the binaries in
`$D/pr4416_<tag>` as the prefill scripts install them and the configs
`$D/g1024_i2048`, `$D/g2048_i2048`:

```sh
./memrun2.sh <tag> 1024 16 "NNTR_HTP_PAGE_CACHE=0"   # measure-memory (build it first, doc 61)
./memrun.sh  <tag> 1024 16 "NNTR_HTP_PAGE_CACHE=0"   # MemAvailable sampler on the phone
```

`memrun2.sh` prints one line: prefill, decode, md5 and the sampled peak;
the CSV lands in `logs/mem/`. `memrun.sh` prints the MemAvailable drop, the
process's RSS peaks (anon / file / shmem, VmHWM) and the application's own
memory lines. Raw data of the runs above: `measure-memory_*.csv` and
`.metrics.log`, `meminfo_*.txt`, the two `*_summary.txt`.

Caveats: single runs, +-0.15 GB between runs of one cell by either sampler
(the 2048-token C = 16 cell read 3.67 GiB by MemAvailable against 3.1-3.3
for the other three); `peak memory: N KB` in the application's output is
`ru_maxrss`, the CPU side only, and reads 1.6-2.7 GB because of the loader;
the DMA-buf residual is a sum nobody prints, derived here.
