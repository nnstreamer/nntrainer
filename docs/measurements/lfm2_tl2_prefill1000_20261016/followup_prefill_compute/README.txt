LFM fixed-cache prefill compute follow-up, 2026-10-16

Implementation through c2bf2b8ee (attention 0fcdfc1f9, convolution
82eed4516, TL2 down epilogue c2bf2b8ee), based on pushed e58b8f8dc.

The expert cache remains C16, with growth and page cache disabled,
Direct I/O, preload128, six prefill readers, two speculative decode
readers, four demand readers, and the existing32 decode spare slots.
NNTR_LFM2_DECODE_DEMAND_HANDOFF=1 is enabled for both sides.
The native TL2 model and configurations are the parent report's fixtures.
Gemma device benchmarks were not repeated, as requested. The new kernel
branches exclude fixed-Q8 Gemma and preserve generic fallback arithmetic.

Changes
-------
* Full dynamic-Q8 LFM attention uses the existing fused parameter/pack
  helper and parallel output stores on the existing worker pool.
  Zero-row quantization initialization and arithmetic order are retained.
  Optional12 stage counters extend the existing stats sequence only when
  its length is at least20; normal calls still request8 entries.
* The LFM ARM prefill convolution applies its post gate before storing,
  removing a complete intermediate write/read and one worker-pool pass.
  Precise multiply/FMA rounding is preserved despite app fast-math flags.
* Native LFM TL2 M=1 down panels use the existing four-tile dequantizer,
  sharing broadcasts and calls. Other shapes/palettes keep their fallback.
  This saves isolated kernel cycles; no full-model decode gain is claimed.

Normal full-model measurements
------------------------------
Three repeats per side/model, A1 B1 B2 A2 A3 B3. A is demand-handoff,
B is the combined changes above. TPS values below are medians.

Input   Prefill A -> B       Decode A -> B       Max sampled peak A -> B
1024    801.252 -> 824.477    17.6503 ->17.5728    2738748 ->2738728 KiB
2048    725.469 -> 735.897    16.8244 ->16.5975    3052668 ->3053724 KiB

1024 prefill improves2.90% in this comparison. The2048 ranges overlap
widely (A707.427..809.486, B719.354..754.050), so this does not establish
a reliable2048 speedup. Decode medians are0.44% and1.35% lower in the
combined measurements; no decode TPS improvement is established here.
The1000 prefill/25 decode targets are not reached.

All12 normal runs match all65 reference generated IDs (one prefill
output plus64 decode steps). The same-handle2x1024 repeat also matches
the reference. The21 attention/KV/router replay remains byte-exact:
50306f9804fc358e45a9ad110cd7cd2ff985ccc740b73ee7db6c5050035ba54d.

One2048 B2 attempt failed before generation with AEE_ENOMEMORY during
stable arena registration. Device logs showed a separate
lfm2_snapshot_check using the DSP at that time. This failed attempt is
retained in excluded-runs.json and raw/. The harness now waits for and
checks overlap with that fixture as well; its retry has a distinct tag.
Earlier valid runs ended before that fixture opened its DSP session.

Stage evidence, not additional full-model gains
-----------------------------------------------
Instrumented1024 attention profiles, ten layers combined:
                       Original    Store-only    Store+Qprep
Q preparation us       36839       37042         24704
Output store us        18024        2994          2942
Inner kernel total us 179042      163426        150131
The kernel total contains the stages; do not add these values together
or add overlapping softmax work. Instrumented TPS is not in the normal
comparison. Profile source/runtime hashes are in artifacts/.

An app-only2048 crossover keeps core and DSP identical, changing only
libcausallm.so. Convolution totals fall579.944 ->515.213ms, while MoE
rises1173.821 ->1222.509ms and other layers vary. Full prefill is
726.757 ->722.909 TPS in this instrumented pair. This isolates a lower
convolution cost without establishing a total2048 performance gain.

Memory
------
Peak is the maximum same-sample process RssAnon plus fdinfo allocation
sizes deduplicated by mount/inode, sampled throughout loading to exit
with a requested20ms interval. Max candidate values are2.612/2.912GiB
for1024/2048. All selected peak samples have complete identities and
no fd read errors. Private DSP heap, driver and VTCM allocations may
be absent. Allocated fd sizes are not necessarily resident physical
bytes. This is not total physical device peak; do not add RSS or arena
size again. No new expert cache or persistent scratch is introduced.

Validation and reproduction
---------------------------
Android core/app and Hexagonv81 builds passed. The final C/C++ source
uses clang-format14. Independent review found no blocking issue:
Qprep360 cases plus2 rejected mutations, store120 exact/write-coverage
cases, stats54 length/dispatch/error cases, all with ASan/UBSan.
The simulator and production-worker TL2 down checks passed. The ARM
convolution checker covers finite/special data and dispatch/alias guards.
Windows/Tizen full builds and remote PR CI were not run in this phase.

Use ../bench.py with the variant, model/model2048, --memory and the
exact environment retained in results.json. Parent repro/ has toolchain,
model configs and the device replay/repeat-check build instructions.
Run tools/lfm2_attn_qprep_check.py --hexagon-tools "$DEFAULT_HEXAGON_TOOLS_ROOT"
--output-dir /tmp/lfm2-qprep-check --sanitize --mutations for the production
Qprep/HVX helper fixture. Run validation/dn4-run-check.sh REPO OUTPUT
for the production TL2 down fixture. Convolution checker compiler flags
are in validation/conv-build-command-v2.json; paths must match the local
Meson/NDK installation. Independent review logs state their host-only
limits. artifacts/*.source.patch.gz retain the precise staged variants
from e58, while source-sha256.json identifies the committed implementation.
