Native LFM M1 gate/up direct epilogue -- retained work removal

Base: 207d299c41393bfffbab23d40692d8fdeb25dcfd (normal source from7b3db6d).
Production patch SHA256 d86b15ae408261f17728b3928dc29dd72fb375d5a772213234338c5c16f94d74.
Candidate normal DSP SHA256 3baddb92d14d1d23fe9b6c6a7cacfe3a6cd4708001c110bb8b15e1b0b7276f7c.
Only three production/test files are changed. No GU trace or FC candidate is
retained in production. Artifacts record exact client/DSP hashes and patches.

The unsplit native TL2 accumulator already has contiguous gate/up planes.
For native LFM K2048/I1536/N2048, SILU, split1, row1 and magnitude1, call the
existing dequant/SwiGLU worker directly using exactly the old tile ownership.
This removes each pair's stack copy and serial wrapper. Nonfused and fused
requant paths retain arithmetic, weights, quantization, LUT preparation and
joins/drains. Unsupported shapes, Gemma, split/magnitude variants keep fallback.
No extra heap, VTCM, cache, expert slots, readers or preload capacity. The two
v81 worker frames remain640/896 bytes. Normal builds contain no trace state.

Normal device measurements: S26 Ultra R3CY70LV96T, native TL2 experts and
QS4CX_WH_AFFINE dense weights, C16, Direct I/O/page cache0, cache grow0,
preload128, predict2, stable slots/early prefill/pair reads/demand handoff/
router input handoff1. Same64 output tokens; 65 saved IDs include first token.
OrderA1 B1 B2 A2 A3 B3 for each prompt length, same20ms memory monitor.

                  control median  candidate median      paired wins
1024 prefill TPS         846.981           846.281           2/3
1024 decode TPS           17.5198           17.7089          3/3
2048 prefill TPS         810.127           794.723           1/3
2048 decode TPS           16.7452           16.8288          1/3
Decode median changes+1.08%/+0.50% overlap run variability. No resolved end-to-end
speedup is claimed. Prefill is outside this M1 guard and also varies widely.
This is retained for proven removal of redundant work with identical output,
not as evidence of reaching1000 prefill or25 decode TPS.

Largest normal sampled peaks(KiB), control->candidate:
 1024:2734944->2733732; 2048:3053112->3052760 (about2.607/2.911GiB).
Whole supervised child lifetime from beforeexec through exit, RssAnon plus
same-sample fdinfo allocation sizes deduplicated by mount/inode. Every reported
peak sample has complete readable-size identities/no read or identity errors.
This is allocated process accounting, not device physical memory: unreported
DSP/driver/VTCM memory and transients between samples are excluded. Do not add
hostRSS/VmHWM to it. Small observed differences are not memory-saving claims.

Validation: full DSP build; actual SDK libnative HVX dequant/quant/LUT parity
544cases,2664lane checks,13guards,4mutants; independent rerun and176concurrent
lane pipelines. Tests include mixed experts/splits/magnitudes/Gemma fallback.
All12 normal runs have exact65 IDs; same-handle two-prompt runs also match at
1024 and2048. A separate normal profile pair has netMoE9516->9471us/token;
45us is too small here to establish a speedup. No Gemma phone reruns were made.
Host fixtures do not emulate QuRT/DMA/cache visibility or silicon timing;
full repository/Windows/Tizen/Yocto CI was not run for this scoped DSP change.

Actual dispatch evidence is diagnostic only. V1 transport repeats identical
payloads; it has no call identity, so aggregate counts are rejected. V2 adds a
monotonic sequence with20bytes total diagnostic globals and logs after all
workers drain. Repeated identical seq/payloads are reported, not mistaken for
new calls. Conflicts/missing seq/truncation/incomplete workloads are rejected.
V2 raw21510 records contain4345unique calls(seq1..4345),9728active experts and
466944tile pairs. All pairs select direct; fusedcalls2076/unfusedcalls61216.
Old7296 worker calls/token become988.9375; serial wrappers removed7296/token.
Logical stack store+load bytes removed3.5625MiB/token, not DDR traffic or a
measured bandwidth gain. Instrumented TPS is excluded. All65IDs still exact.
The exact normal source/DSP were restored after diagnostics.

Canonical test: test/htp/host/run_tl2_gu_epilogue_checks.py --repo <repo>
 --reference <reference tree> --output <tmp> --hexagon-tools <tools19.0.04>
 --sanitize --mutations. Reference, commands and source hashes are archived
under validation/. Gzip files are lossless; unpack snapshots when rerunning.
manifest.json hashes every evidence file except the manifest itself.
