LFM affine FC prefill: foreground phase measurements
==================================================

Diagnostic evidence only. The normal production baseline is 7b3db6d189f,
which includes native TL2 expansion load removal. No FC i24 conversion,
affine-sum fusion or issue-loop optimization is included in this record.

Two S26 Ultra 1024+64 runs with the version2 diagnostic pass the strict
180-call cohort (360 paired records). Warmup100 calls are separated from
the real prefill80: N6144 x30, N2048 x40 and N3072 x10, all K2048/M1024.
The main calls contain148480 accumulator reads and4880 worker joins.

Main prefill foreground, ms            Run1         Run2
Body                                  136.138      135.760
Parameters/pre-norm/quantize/AH pack     34.409       33.916
Affine row sums                         0.939        0.927
Accumulator read call                  27.514       27.476
Exposed dequant worker joins            9.364        9.406
Exposed weight DMA drains               3.388        3.587
Other foreground                       60.525       60.446

These intervals do not overlap on the calling thread. Body starts after
validation/allocations and before initial weight DMA; it includes final
frees and instrumentation but excludes calibration and FARF output. Pack
includes parameter scanning and optional pre-norm, not just layout copying.
Read includes completion of preceding HMX work, not just accumulator stores.
Joins/drains measure exposed waits, not full asynchronous service time.
Other foreground includes HMX issue/setup/ordinary work and instrumentation;
it is not all overhead that can be removed. Worker times are not added.

The affine-sum region is under1ms, a small upper bound for removing only
that region. The27.5ms read region is a loose upper bound when changing only
readout, not an i24 saving estimate: the HMX completion portion must still
happen, and a new worker could change the9.4ms exposed joins. These data do
not establish a normal-model TPS gain. The next bounded experiment concerns
the FC issue loop's per-iteration branch/descriptor/stack work, confirmed in
v81 assembly. Alignment masks themselves are already hoisted by the compiler.

Instrumentation and recovery
-----------------------------
The original verbose v1 FARF record exceeds the device transport length and
is truncated near225-228 payload characters. The strict parser rejects all
phase analysis from that run; the raw180 truncated records and rejection are
retained. Its inference still passes, but no partial-field timing conclusion
is drawn. V2 changes only formatting: two lines tied to the existing start
tick, lossless hex durations and a required terminal marker. Full-width
integer tests bound the geometry line at188 and timing line at167 bytes.
The strict parser requires unique IDs, one geometry and one timing record,
all fields, exact counts, complete cohorts and valid non-overlapping phases.

Main instrumentation performs312800 timer reads including32 calibration
pairs per call. The empty-interval scale is not a measurement of complete
timer/branch/register overhead and is not subtracted. Added timing calls can
alter overlap. Diagnostic TPS remains excluded from normal comparisons.
The diagnostic adds136 bytes to the v81 scalar frame (656->792) and no heap,
VTCM, cache, reader or expert allocation. It does not enable shared MoE probes.

The full v2 DSP build passes. Actual whole-FC instrumentation topology is
checked with ASan/UBSan and deterministic primitive/clock stand-ins; this
validates accounting and error drains, not real HMX math or timing. Parser
and formatted-length checks include malformed/duplicate/missing/truncated
records and maximum integers. Independent v81 compilation and source review
are retained with their exact limits. Flag-off .text matches the baseline.
The whole-FC legacy fixture initially leaks3 test-owned arrays at exit; the
original LSan failure is retained and only a temporary fixture copy frees
those arrays before its successful rerun. Production was not changed.

All three runs use C16, CACHE_GROW=0, PAGE_CACHE=0/Direct I/O, preload128 and
the normal router handoff. All65 generated IDs match the1024 reference; no
Gemma phone run was repeated. The largest diagnostic sampled peak is
2734584KiB (about2.608GiB), with complete readable-size fd identities.
Memory covers child exec/load through exit, sampled about every21ms, as
RssAnon plus fdinfo allocations deduplicated by(mnt_id,ino). It excludes
unreported DSP/driver/VTCM memory and may miss inter-sample spikes; allocation
sizes are not necessarily resident. It is not total physical device memory.

Root reversed each diagnostic patch and restored the exact normal DSP
b30be327d815af865452bb6f4a18c1d80c785a4c6abbdc0c1f06840345219273.
All diagnostic binaries remain separately named on the phone. The ordinary
benchmark harness is unchanged; the versioned driver immediately retrieves
only the new diagnostic tags from the same logcat start timestamp.

raw/ retains inference logs, normal-harness logcat, additional phase records,
IDs, metadata and memory reports. validation/ retains frozen patches, full
builds, strict parsers, commands, codegen, host checks and independent review.
artifacts/ records exact binaries/source hashes; no binaries/models are
committed. The diagnostic source author starts at8288e3ba3, and the root
uses7b3db6d18; the intervening expansion patch does not change the FC file.
manifest.json hashes all retained files except itself.
