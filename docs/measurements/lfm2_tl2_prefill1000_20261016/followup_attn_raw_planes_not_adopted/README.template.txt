RAW4PLANE QK attention experiment — not adopted

The bounded full-model confirmation failed the declared adoption rule. The
attention source was restored; the separate fixed-Q8 copy-tail correctness fix
remains. No 1000 TPS, decode speedup, or unchanged Gemma performance is claimed.
This archive is evidence, not a patch proposed for production.

Normal measurements (prefill TPS)
@@TABLE@@

All 24 runs are retained. The first 12 and confirmation 12 remain separate from
the pooled summaries. Initial order was AB/BA/AB; confirmation was BA/AB/BA at
each length. The frozen confirmation plan required a nonnegative pooled median
and at least 4/6 paired wins at BOTH lengths, plus correctness/memory checks.
1024 fails; the favorable, variable 2048 observations do not rescue that rule.
There was no length narrowing, filtering, or further outcome-driven sampling.

All 24 runs preserve all 65 dumped token IDs: one first prefill token plus 64
generated tokens. The six ARM runtime binaries were identical between arms;
only the controlled DSP build differed. C16, PAGE_CACHE=0, GROW=0, preload128,
eight host threads, six prefill readers, two speculative decode readers and
four demand participants (including the caller) were unchanged. Model/config,
environment, fresh runtime hashes, individual TPS and all memory samples are
retained. Pooled decode medians are 17.60175 -> 17.5343 TPS at 1024 and
16.55885 -> 16.5038 at 2048; this prefill change does not establish a decode
kernel effect. Device temperatures do not prove constant clocks.

Memory scope and observed peaks
The whole-child monitor requests samples every 20 ms from loading through
termination. Its composite is RssAnon plus deduplicated accelerator allocation
sizes from fdinfo, not VmRSS or total physical device memory. Scans are not
atomic; DMA residency is not verified; unreported driver/DSP/VTCM allocations
and between-sample peaks can be absent. The separate application report covers
CausalLM::run and has a different sampling interval. Do not add these metrics.
Pooled maximum sampled composite KiB, control -> candidate:
  1024: 2714740 -> 2724868
  2048: 3018984 -> 3065796
Both candidate outliers remain included. Independent peak samples have no
identity/read errors, but other scans contain errors retained in raw reports.
The 2048 r6 peak has a temporary +49 MiB fd-size aggregate relative to its
paired control peak, with lower RssAnon; it does not identify an allocator or
prove extra persistent storage. The 1024 r5 peak has higher RssAnon and lower
fd sizes. One or two contiguous terminal RSS-less samples are documented rather
than treated as missing mid-run measurements. See analysis/memory-outlier-
readback.json.gz and the final independent review for exact neighboring samples.

Candidate and measurement boundaries
The v2 patch is d8d1827242aab3e698e7c69eefb73711305d37b843ba3798ba2b2e251e75fbb3;
candidate hexkl_attn_q.c is
8c54a3c86509c17b0c5cda01c0a2de6abf3de335fca6e268cb74b7dfed26d4e5.
It replaces QK SDK int32 materialization with four accumulator planes in the
same existing 8 KiB L.acc/Qx slots, reconstructing signed int32 in registers.
The exact v81/full LFM dynamic-Q8 geometry is limited to 1024/2048 rows,
32 query/8 KV heads, HD64, Br128/Bc256. It validates the required first 4 KiB
of the SDK's 16 KiB configuration. PV retains the SDK reader. No cache,
activation capacity, worker, reader, or persistent buffer was added.

The initial source-only v1 was replaced before model measurements because its
generic dequant path added an inner-loop mode branch and an out-of-line call.
V2 restores the generic body and selects raw/generic outside the row loop.
Context size remains 768 bytes (flag in existing padding); main, QK and worker
frames remain 1536/512/384 bytes; the qualifier has a 32-byte frame. Generic
fallback mode selection still has a cost; source/assembly parity does not prove
Gemma performance is unchanged. No Gemma model benchmark was performed.

Six instrumented model runs are separate from normal TPS: two actual-dispatch
runs and four stage runs (one control/candidate pair per length). Actual
dispatch verifies all ten model attention calls:
  1024: 102400 serial QK readouts, no deferred readouts.
  2048: 81920 serial and 286720 deferred QK readouts.
Each count is one complete four-plane, 8 KiB readout, excluding PV.
The four stage runs have ten-layer inner totals 151.201 -> 137.643 ms at 1024
and 363.843 -> 350.656 ms at 2048. Their QK buckets fall 23.423 -> 5.817 and
84.276 -> 21.387 ms; exposed softmax at 2048 rises 44.322 -> 93.080 ms.
Shorter QK can expose previously overlapped softmax work. Deferred dequant begins
after softmax/P-quant and overlaps PV; it does not overlap that softmax job.
Do not sum overlapping buckets or equate QK changes with whole-model savings.
The configuration qualifier is outside the inner-body timer. The 1024 stage
pair also has exposed expert wait 20.7 -> 150.3 ms; this is context, not proof
of the cause of normal-run variation. No causal correction is applied to TPS.

Correctness coverage and its limits
Author and independent strict ASan/UBSan host tests cover 2064 signed row
blocks, 4129 configuration/geometry checks and 568 pipeline cases, with three
deliberate mutations detected. Actual HVX/libnative reconstruction and existing
dequant math are compared; HMX, DMA, softmax/PV and hardware fences are stand-ins
in that fixture. v81 compilation/codegen and unsupported-architecture fallback
checks are separate. Three actual-device corrected-KV arms then pass 36 outputs
and four signed probes each, including A/B/A reuse, dynamic Q8/Q4, fixed Q8
HD64/HD256 and unsupported-path coverage. All three 231908352-byte outputs are
SHA256 e64b0a2342d8132bd09e23763ca8958270c62a8e2a99dd720e1cb27f4478c394.
An independent reviewer compared the complete bytes at collection time. The
archive omits those large dumps and ELF/vendor files; hashes and frozen reports
cannot independently recreate that full-byte comparison without recollection.
The device logs, execution records and complete sampled-memory reports remain.
Fixture memory has fd-read errors in some scans and is not a performance or
memory-capacity claim.

Separate correctness repair and artifact incident
The first real control fixture exposed an existing fixed-Q8 HD64 copy-tail
overflow/race, before running this candidate. Rows may legally be rewritten.
The narrow XOR tail fix was applied identically to all three arms; its source,
tests, original failure and corrected proof are committed independently as
e3eed8f40b504caa314b5329775e82d831f3b903 under
docs/measurements/lfm2_tl2_prefill1000_20261016/followup_fixed_q8_copy_tail.
See references/kv-fix.json. This archive does not duplicate that full record.
Dynamic-Q8 LFM does not use the corrected helper.

Before any device fixture/model execution, reviewer objcopy --dump-section
commands omitted an explicit output ELF and rewrote two parent build files.
The deployment hash check caught the mismatch; zero device computations used
those files. Root preserved the incident, restored exact bytes from untouched
repeat links and reverified all seven phone files. Source, objects and SDK
inputs were unchanged. The incident is in incident/ and independent post-
recovery records; no claim that the original artifacts were never modified is
made. Later controlled links and the common KV fix use exact frozen inputs.

Controlled builds and final restoration
The initial controlled pair compiles 63 objects once, reuses 62 and replaces
only the attention TU, with repeated links/import/dependency/external-library
identity records. The common-KV build uses 61 original common objects plus the
same corrected KV object in all arms, leaving only attention different:
  control    2d693821c60863c530508cd9249c285be0df18311923eeda0081e21401fe0bd5
  candidate  f39929135978faf78194973df76aee9bafbaa7f41b75ec6da75b646813b67fc8
  diagnostic 9750194852b6f3af2e885355fb373da4b7ad3e8430c0ae0521583008fdb376f6
The no-op KV overlay reproduces its original object; shared libraries are
hashed before/after; repeated links are identical. Historical 3baddb92 DSP
identity is not asserted for these rebuilt objects. Source and runtime restore
records select the corrected control as lfm-kv-tail-fix-1, with all seven phone
hashes verified. This equals the six already tested controls; no fresh model
run is presented as necessary or performed. The attention candidate is absent.

Archive layout and replay
summary.json and root/ keep initial, confirmation, pooled plan/decision records.
runs/ retains all 24 normal and six profile inputs. analysis/ includes strict
independent recomputation, both peak investigations and phase interpretation.
device/ holds three fixture inputs/reports and dump hashes. source/, author/,
independent/, host/, codegen/, diagnostic/ and build/ retain frozen patches,
tests, source identities and commands. related-audit/moe-input-prep/ records
the separate source audit that ruled out duplicate input quantization. The
later TL2-to-VTCM feasibility investigation is deliberately outside this archive.

  python3 -B verify.py
  python3 -B reproduce_host.py --repo /path/to/nntrainer \
    --hexagon-tools /path/to/Hexagon_SDK/6.4.0.1/tools/HEXAGON_Tools/19.0.04 \
    --output /path/to/new-host-results

verify.py needs only Python's standard library, writes temporary files outside
the archive, checks every payload hash, reruns strict raw-log TPS/ID/memory,
stage and dispatch parsers, and rechecks the declared decision. It does not
rebuild, contact a phone, or infer omitted dump bytes. reproduce_host.py requires
a compatible repository checkout (the recorded e3eed8f40 state has the required
stubs/aligned HVX host shim), Linux, a C compiler and the licensed SDK libnative
tools; it expands frozen source/tests only into a temporary directory. Original
absolute command paths are provenance, not a promise those paths exist elsewhere.
There is no automated device rerun in this archive.

Frozen historical C/C++/headers/patches and raw logs are gzip-compressed with
deterministic timestamps, preserving original bytes without changing them to
satisfy repository formatting. manifest.json records stored and decompressed
hashes; provenance.json maps original inputs. No ELF, object, vendor library,
large tensor dump, or later production optimization is included. This bounded
experiment is not a claim that the repository's full CI matrix passed.
