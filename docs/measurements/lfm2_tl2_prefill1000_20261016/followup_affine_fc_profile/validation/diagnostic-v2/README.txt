Affine FC diagnostic v2: complete bounded packets, no kernel change

Frozen full patch: diagnostic-v2.patch
SHA256 d09b1ce6beb56321796c9bb8ecc8f4cbb1b82bd67fded37e3890b7e597d21639
Candidate source SHA256605b4a6d60c35d3c6868b981942e242acff273fb5fcae95187e1b9cefdd3eec3
Parser SHA256bb16bf4b7e495b84afbfe5d7c4de27d920bc6b65b7c20912944aa4471add60f2
Compile flag remains -DNNTR_LFM2_AFFINE_FC_TRACE=1.
The full patch applies to normal generic FC. v2-over-v1.patch is only for an existing v1 overlay; do not apply both. Root has already built/run v2 and restored normal source/DSP.

V1 is rejected, never used for phase analysis
The original diagnostic/ directory remains unchanged. ../v1-rejected/ archives exact source/patch/parser/metadata and the180 physically truncated device lines with hashes. Phone v1 payloads ended at225..228 characters. Neither the apparently complete leading phase fields nor partial clock fields are considered valid measurements.

Only two FARF print calls changed relative to v1. All numeric capture/timers, brackets, guard, existing start tick, math, output order, error returns and lifetimes remain exactly in place. No persistent sequence counter was added. Both prints occur after body timing and after the same32-pair calibration.

Wire format (all fields mandatory, terminal ! mandatory):
[LFM-AFC2G] s=hex_start d=M,2048,N,nh,width0,width1,width2 x=rc,stride,bank_tiles,pipeline c=reads,joins,brackets !
[LFM-AFC2T] s=hex_start t=hex_body,hex_pack,hex_sums,hex_read,hex_join,hex_drain k=19200000,32,hex_clock !
Start and all timing values are unsigned64-bit hexadecimal, with no precision loss. Geometry/counts are decimal. Start identifies a call; no adjacency/order assumption is needed. IDs must be unique across the input run, each with exactly one G and one T packet. Reused/colliding IDs, missing/duplicate parts, malformed/unknown fields, bad cardinalities/ranges or any v1 line fail. Terminal! detects even truncation midway through the last numeric field.

String-size proof uses the ACTUAL C formatter via FARF shim, not a duplicate format estimate. At full UINT32/INT32/UINT64 printable widths, G188bytes and T167bytes. Adding64prefix bytes+NUL gives253/232, fitting256 and leaving margin below the observed225..228-byte payload ceiling. Actual phone v2 packets are at most88bytes. Parser's payload ceiling is192bytes.

Strict parser:
python3 parse_affine_fc.py /path/to/continue-affine-fc-trace-v2-r1.affine.logcat.txt --output phases.json
It requires180 unique calls/360 packets, all expected warmups, main80calls/148480reads/4880joins/153840brackets, pipeline1/stride32/bank32, rc0, exact clock fields and disjoint foreground intervals. This is the exact1024-row LFM2 topology, not a generic2048-row parser. Validate run exit/65 IDs/settings separately.

Phase semantics are unchanged from v1:
body starts after validation/allocations and before first weight push, ends after final frees, excludes calibration and BOTH log calls. pack includes parameter scan/optional norm/AH packing with initial weight DMA in flight. sums includes its pool run. read includes vendor accumulator read AND possible completion of preceding HMX work. join and drain are exposed foreground waits, not worker/DMA service time. Do not add overlapping service times. Read is not pure int32-store cost or expected i24 savings. Empty-pair clock data is only a scale reference; no automatic overhead subtraction. There are still312800 raw qtimer reads for the main80 calls including calibration; the new log call does not add timers, but additional logging between model layers can alter overall overlap. Diagnostic TPS is not a normal benchmark.

Validation
- v81 enabled/disabled -O3 -Wall -Werror PASS. Disabled.text matches normal baseline byte-for-byte. Stack remains656B normal/disabled,792B diagnostic; no new heap/VTCM/persistent allocation.
- python3 parser_check.py: valid180calls/360packets, reverse/shuffled pairing PASS;43 malformed/missing/count/range/error cases and all170 prefix truncations of both representative packets rejected. Actual invalid v1 log rejected without phase output.
- python3 run_guard.py --repo REPO --source candidate/nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c --output OUT:25 actual-helper guards, unchanged shared probes/options/table, no timer reads for rejected guards, actual full-width printf bound under ASan/UBSan PASS. It includes the full source TU; unused arithmetic functions are removed by link-time section GC. This is guard/logging validation, not HMX arithmetic or timing evidence.
- Independent reviewer: /tmp/nntr-lfm2-affine-fc-trace-review-20261016/v2/{review.txt,validation.json}; actual complete TU22 mock-clock topology/error cases, actual formatter limits, independent parser/missing/truncation cases and both real device logs PASS. No source/format blocker.
- Device phase summaries and source/evidence hashes are in checks/ and validation.json. Root alone used the device. See measured-bounds.txt for interpretation. No i24 port or other production optimization is implemented here.
