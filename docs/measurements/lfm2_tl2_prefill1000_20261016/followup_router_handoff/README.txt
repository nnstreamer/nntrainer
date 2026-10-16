LFM prefill router-to-MoE input reuse
===================================

The FP32 LFM router already copies its input into DSP scratch. The following
plain MoE call can now consume those exact bytes instead of copying the same
input again. This is opt-in with NNTR_LFM2_ROUTER_INPUT_HANDOFF=1 and requires
matching ARM and DSP builds. Default/off keeps the original wire values,
scratch order and input copy. An older DSP does not negotiate the new ticket
word and rejects it; deploy the core and full DSP skeleton together.

The implementation reuses the existing scratch allocation. It adds 48 bytes
of ARM ownership state and 48 bytes to DSP scratch metadata, 96 fixed bytes
in the tested ABIs. It does not add an activation or expert cache, VTCM, I/O
scratch or reader. C16, cache growth=0, PAGE_CACHE=0/Direct I/O, preload128,
608 expert slots, 32 decode spare slots and all reader counts were held fixed.
The model bytes, quantization, arithmetic and accumulation order are unchanged.
Gemma was not benchmarked again; its dimensions/GeGLU/norm paths do not select
this native LFM/SwiGLU reuse path.

Ownership and fallback
----------------------
ARM binds a one-shot ticket to the actual staged ION input, session, byte
extent and write epoch. Input/output writes, buffer replacement and prompt
boundaries invalidate it. Caller-writable input aliases and overlapping
output restoration decline reuse. DSP separately binds that ticket to this
session's scratch allocation, byte extent and scratch epoch. Every attempt
consumes the ticket, including errors and mismatches. Tickets never wrap.

FastRPC maps one ION buffer to different DSP virtual addresses in successive
RPCs on this device. The initial pilot incorrectly compared those transient
addresses and skipped no copies. The final implementation keeps source
identity on ARM and scratch identity on DSP. This is a trusted paired-client
protocol; it does not authenticate arbitrary clients fabricating tickets.
Existing single-owner and read-only host_view lifetime contracts still apply.

Reuse requires active compact native TL2 weights, K2048/I1536/N2048, SwiGLU,
M128..4096, no norms/direct cached output and sufficient existing scratch.
A valid hit carves input at the existing prefix and skips its DMA. Scratch
growth or any other decline uses the original layout and copy. Both paths
use the same reserve-size expression. The normal on/off comparison therefore
includes copy removal and the hit-only layout change; it does not isolate
their individual timing effects.

Normal S26 Ultra measurements
-----------------------------
One binary; only HANDOFF=0/1 differs. For each prompt length, three pairs
follow A1 B1 B2 A2 A3 B3. Profiling and DSP trace are off. All runs generate
64 decode tokens after prefill and match all 65 saved output IDs.

                    1024 OFF -> ON          2048 OFF -> ON
Median prefill TPS   812.054 -> 847.682       778.116 -> 810.768
Median decode TPS    17.5103 -> 17.6114       16.8687 -> 16.7627
Max sampled KiB     2733820 -> 2734356      3052316 -> 3053300
Prefill median delta        +4.39%                  +4.20%
Faster paired prefill       3 / 3                   2 / 3

The candidate peaks are approximately 2.608 / 2.912 GiB. Execution ranges
and every raw run are retained in results.json; 2048 varies substantially
(ON 788.602..898.640 TPS). These are local repeated measurements, not a
guaranteed gain. Decode is outside the reuse path: the +0.58% and -0.63%
changes are reported, not claimed as a decode optimization. Prior rounds'
metadata-pair results are not the denominator for these percentages.
The 1000 prefill / 25 decode TPS targets remain unmet.

Separate mechanism diagnostics
------------------------------
Build DSP with -DNNTR_LFM2_ROUTER_HANDOFF_TRACE=1 for diagnostic runs only.
There are 38 real router producers and 39 matching plain-MoE calls (one load
warmup plus 38 real layers). The broader state tag also logs three unrelated
dense calls: 42 state lines do not mean 42 handoff requests. The parser
matches the 38 requested tickets across producer, publish, state and consumer.

1024: all 38 input copies skipped, 304 MiB, with unchanged scratch capacity.
2048: 35 input copies skipped, 560 MiB. The first three real calls matched
ownership but used the original copy because scratch needed to grow:
  55,976,704 -> 61,799,296 -> 62,969,984 -> 63,099,648 bytes.
The unchanged reserve expression, compiled with actual v81 types/macros,
reproduces the last three sizes for active-expert counts 54, 63 and 64.
Those expert counts are inferred from capacities, not separately logged.
Unused experts are removed before RPC; the old reserve bound therefore
changes with the layer's active-expert count. All later hits keep capacity.

The earlier two address-bound pilots and detailed state pilot are archived
as superseded diagnostics, not speedup evidence. The final trace runs match
all output IDs. Normal timing uses a different, trace-free DSP binary.

Validation
----------
The actual ARM staging/selection and DSP RPC/carve helper fixture passes
176 cases under ASan/UBSan, independently repeated by a second agent and
root. It includes three MAP_SHARED virtual mappings of one backing file,
distinct M128/1024/2048 inputs, writer invalidation, aliases, restart, capacity
fallback, errors and one-shot consumption. Eighteen targeted mutants fail
at their intended assertions. Arithmetic and DMA are host stand-ins; device
ID parity supplies the actual model-output check.

Before committing, the fixture's inner-tile divisor was corrected from 64
to the actual N_INNER=32 macro and all 176 cases passed again. Production
did not change. Host job/context sizes still intentionally differ from v81
(64/256 versus 32/64 bytes); those tests check ordering and input equality,
not absolute DSP allocation sizes. The separate v81 capacity proof above
uses actual types. Both old and corrected fixture patches are retained.
The final fixture is validation/tests-remapped-ninner32.patch; the affected
prefix-carve mutant was rerun and rejected after that correction.

Both 1024 and 2048 same-handle, two-prompt checks match the reference IDs on
both executions. ARM NDK r30 and full Hexagon v81 builds pass, including the
separate trace build. clang-format 14 and diff whitespace checks pass. The
new ownership fixture is wired into run_host_checks.sh. The full integrated
host suite exits successfully, including the corrected 176-case fixture.
The optional ARM NEON abs-max execution was skipped because the cross
compiler or qemu-aarch64 was unavailable. The complete log is retained;
local checks do not replace the full cross-platform CI matrix.

Native TL2 model SHA256:
ceebe2a4e9b4e931c28955b2e223801f352829cca567cb9697744a6f7a12f27d
1024 generated-ID SHA256:
b6c1b801f95caded5399ec685cce0f900a20c3f9df760482dd7b5c9a62a5b92e
2048 generated-ID SHA256:
2bc7afc3c7d8e4387c8d9c7378d6792a1ef0fb56c2635f73b95e8650ac7fef33

Memory and reproduction
-----------------------
Memory covers child exec/load through exit, requested every 20 ms (about
21 ms observed): maximum same-sample RssAnon plus fdinfo allocation sizes
deduplicated by (mnt_id, ino). Retained peak scans have complete readable fd
identities. Allocated bytes need not be resident; unreported DSP/driver/VTCM
memory and brief spikes may be absent. Do not add RSS, VmHWM or arenas to
this metric. It is not total device physical memory.

raw/ contains full run commands, environment, IDs, compressed logs and
memory samples. artifacts/ binds each deployed binary to its exact source
patch and base commit. validation/ retains build logs, independent review,
mutants, job order, ownership diagnostics and same-handle checks. The parser
checks actual copy skips and legitimate capacity fallbacks, not just ticket
requests. manifest.json hashes all retained files except itself.
