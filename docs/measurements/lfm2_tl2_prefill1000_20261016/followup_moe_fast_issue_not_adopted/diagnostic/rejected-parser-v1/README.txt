Actual native MoE leaf dispatch diagnostic (separate from frozen production)

Apply diagnostic.patch only on the e8bbf0ae candidate, then build with
-DNNTR_LFM2_MOE_ISSUE_TRACE=1. Without that flag the v81 whole object is
byte-identical to normal production. Normal production itself is unchanged.

One [LFM-MISSUE] packet is emitted after the call's cleanup:
q= per-module increasing uint32 sequence, assigned only in native LFM prefill scope
s= raw qtimer at that sequence assignment (identity, NOT elapsed time)
d= M, compacted expert count
b= actual once-per-call leaf predicate (-1 means never reached)
t= total GU/DN accumulator-tile issue entries
f= actual GU/DN leaf call entries, incremented at the call sites
r= final kernel return code; ! is mandatory terminal marker

Counts are issue entries, not successfully read-out tiles. Only rc==0 may be
aggregated. Each successful GU entry issues 64 packets; each DN entry issues 48,
in increasing kt. Full-caller address/order parity establishes this relation;
the logger is not a new HMX simulator. Main calls must have f==t in BOTH phases,
nonzero GU count divisible by96, DN divisible by64, matching block counts, b=1.
No global probe_on/phase timer mutation. Counters are foreground local variables;
only the diagnostic sequence is an atomic uint32 (4 B static metadata). No heap,
VTCM, readers or cache additions. The instrumented caller frame remains17152 B.
The added increments/atomic/logging may perturb scheduling; never use this run
as normal TPS evidence.

Scope is computed after input/weight/row validation: M>1, K2048/inter1536/N2048,
compact E1..64, nonempty all-native TL2, no affine, SiLU. Dense/Gemma/M1/GeGLU are
not emitted. An early validation return before sequence assignment is not emitted;
the required38 complete main records catch missing main calls. A post-assignment
return can create a missing sequence, which is rejected. Start a fresh process
and collect one process's complete log from warmup through completed generation.

Strict parser, for either input size:
python3 parse.py /path/to/logcat.txt --rows 1024 --output /tmp/counts1024.json
python3 parse.py /path/to/logcat.txt --rows 2048 --output /tmp/counts2048.json
Main is exactly38 calls with the requested M. Other M cohorts are reported
separately. Complete identical SDK replay packets are deduplicated and counted;
conflicting duplicates, multiple Android PIDs, mixed prefix formats, missing or
noncontiguous sequence1..N, nonincreasing/colliding raw start ticks by logical
sequence, numeric overflow, partial/zero/fallback/error main calls, truncated
packets and terminal suffixes are rejected. Delivery order may change due to SDK
replays, so logical sequence order rather than arrival order governs tick checks.
Bounds use observed record count, not a range allocated from an untrusted seq.
No parser can prove an unadvertised final non-main record existed; expected main
count and warmup-before-main order are the relevant completeness contract here.

Validation:
- Actual whole production TU, diagnostic-on: eleven calls, six emitted scope
  records (five successes plus a deliberate read37 error), 688128 output bytes
  exactly match normal candidate. Bad-row validation returns before scope and
  is not falsely claimed as a logged error. ASan/UBSan; same limited pre-existing
  scalar shift-base exception as the production fixture.
- Actual FARF format with maximum uint32/uint64/int32 arguments:146 payload chars;
  64-char SDK prefix plus terminating NUL totals211, below256.
- Parser:6 valid1024/2048/replay/PID fixtures and264 negative cases, including180
  all-prefix truncations, huge numeric values, missing/conflicting/replayed data,
  PID mixing and raw-tick collision/order violations.
- v81 -O3 -Wall -Werror normal/off/on compile+assemble PASS. Off object hash equals
  normal. The source/commands/results are in manifest.json, codegen and checks.

Reproduction after production whole fixture:
python3 run_check.py --checks /path/to/whole-check-output --output /tmp/trace-check
python3 parser_check.py /tmp/parser-result.json
Full v81 commands: codegen/{normal,off,on}-command.json.
