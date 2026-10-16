LFM affine FC issue leaf -- rejected normal throughput experiment

Control is retained GU source36a75df1a (binary built before its commit on207d).
Frozen candidate patch e1e45c09574d0b131fea13d2de3822b413effb47b550d22e81c718dfe93e9e0e.
Normal DSP9056191eed9d8766c033cccc379227e3f5d3e672a2e99b4c5301965b3b0fcf83.
The patch and both diagnostic variants are evidence ONLY, removed from production.
Exact source/client/DSP hashes and deployment patches are under artifacts/.

Only native LFM affine K2048 prefill geometry selects the proposed small issue
leaf. It preserves HMX packet order, math, clear/readout, worker and DMA ordering.
v81 assembly has six packets/issue in the old loop and two in the leaf, but
alignment mask calculation was already hoisted. Caller stack656->672B (+16B),
leaf0; no heap/VTCM/cache/reader/preload capacity changes. Assembly packet counts
are NOT cycle counts: stalls, call overhead and changed overlap can dominate.

S26 Ultra R3CY70LV96T; native TL2 experts / affine int4 dense weights. C16,
page cache0 (Direct I/O), cache grow0, preload128, predict2, fixed readers,
stable slots/early prefill/pair reads/demand handoff/router handoff1.
Order A1 B1 B2 A2 A3 B3 independently at1024 and2048,64 decode tokens.

                 control median  candidate median  paired prefill wins
1024 prefill TPS       840.033           827.141               1/3
2048 prefill TPS       888.889           785.276               0/3
1024 decode TPS        17.6309           17.6649
2048 decode TPS        16.8377           16.8200
Prefill median changes -1.53% / -11.66%; the candidate is not adopted.
Decode is outside its M>1 guard; the small differences are not speed claims.
All12 normal runs have exactly the reference65 saved IDs.

Largest normal measured peaks(KiB), control->candidate:
1024:2734508->2731308;2048:3052264->3065180. The larger2048 candidate peak
is recorded without claiming a memory reduction (about2.923GiB).
The monitor covers the entire supervised child from beforeexec through exit,
requested20ms samples of RssAnon plus same-sample allocated fdinfo sizes,
deduplicated by mount/inode. All peak identities/readable sizes are complete.
This is allocated process accounting, not total physical device memory;
unreported DSP/driver/VTCM and between-sample transients are excluded.

Validation: fullv81 DSP build; author and independent actual-caller46
ASan/UBSan cases plus8 guard/order mutants;7 numerical scalar cases with
233472 byte-identical FP32 output. Host primitives emulate arithmetic and
logical queue retirement, not silicon concurrency/cache behavior or timing.
No Gemma phone repeats or full repository/Windows/Tizen/Yocto CI were run.

A separate normal profile pair had total prefill2470->2512ms while aggregate
FC host timings decreased. These aggregates include100 load warmups, so a
second controlled phase diagnostic separates main work and actual dispatch.
The diagnostic instruments both variants identically and is excluded from
normal TPS. Its parser requires exact source-derived G/T/B cohorts, arena,
banks, counters, serial timing and complete records. The main2048 cohort has
190 kernel calls and296960 readouts;100 load warmups are separate. Existing
ARM1920 and DSP1728 row caps produce nested splits for conv FC. The reports
in diagnostic/root/ contain strict raw-device comparison and interpretation.
Readout time includes HMX completion; other includes required issue work and
instrumentation. Neither is a directly removable cycle budget. A timing-only
diagnostic cannot establish the cause of the uninstrumented12-run regression.

Source/DSP were restored exactly before starting the separate MoE candidate.
Reproduction commands and fixture limits are in validation/author/review.txt
and diagnostic/README.txt. Gzip files are lossless snapshots. manifest.json
hashes every file in this evidence bundle except the manifest itself.
