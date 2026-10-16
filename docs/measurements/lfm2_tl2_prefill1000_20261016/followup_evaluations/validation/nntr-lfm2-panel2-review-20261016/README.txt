# Independent native TL2 paired-panel review

Reviewed production SHA256 `bcacf49dece0aa16410ca3b4c75667d0be625289a6b3f5b13d3e21666b36255a` and tests SHA256 `9a1b44f86bcf992d5ee956f9948b0871d0e2a8992b9f3ce3314e54fc50ad26d6` from `/tmp/nntr-lfm2-decode-opt-20261016/panel2`. Candidate source is `/home/jungone-lee/workspace/nntrainer-lfm2-tl2-panel2-20261016`. No parent/original production files or device changed. No correctness blocker found.

## Exactness and bounds

Each activation LUT value combines three uint8 activations with ternary weights and has magnitude <=765. The paired kernel combines all eight groups in a halfword before widening: at most24*255=6120, including intermediate partial sums. Thus replacing two separate low/high sums with one sum cannot overflow int16 or change the integer result. Across K, production requires K/inter<=8192 (max2,088,960), and specifically DN1536/GU2048 are <=391,680/522,240. No int32 overflow. Each panel maintains its own even/odd accumulators and sign predicates. The actual predicated-sign branch was tested with libnative, not only its scalar fallback.

DN remains per-column integer GEMV then the already verified four-tile FP dequantizer. There is no cross-panel floating-point reduction or reassociation. GU only pairs with split=1; split2/4 retain the exact original partial-output ordering and reduction path. Pairing across the gate/up boundary is safe because these remain separate integer columns and the existing epilogue consumes them after the worker join.

K tails are supplied by the existing hvx_tl2_prepare zero-padding and native packed layout. K2048 consumes86 blocks, the last with16 active values. The function's K argument and src1 stride use the same ceil(K/24) block count. Full N=2048 makes DN panels complete; non-full N and other shapes fail the guard and retain the clamped single-panel path.

## Ownership and lifecycle

The worker range partitions contiguous panel jobs. Both `unit+1<end` and `panel+1<panels` must hold, so a pair cannot cross a worker or expert boundary. Incrementing unit inside the body then at the loop increment consumes exactly two jobs. GU's split=1 condition makes the next job the next panel. Existing feed mode gives one expert per synchronous pool run; the expanded test also covers multi-expert and nonzero-first ranges.

acc_dn is carved separately for every expert and separately from LUTs, residual outputs, and DMA slabs. Its existing64tiles*512B=32KiB allocation exceeds the paired path's contiguous2048int32=8KiB use. The bytes are fully overwritten by panel_pair before dequantization. Disjoint workers touch disjoint128-column spans; subsequent scratch reuse follows existing pool joins. No allocation is added. However, at most8KiB/expert of a previously idle allocation is newly read/written (32KiB for top4); that is memory traffic/page-touch, not evidence of a32KiB allocation or measured RSS increase.

The LFM guard requires nativeTL2, K2048, I1536, N2048, M1, and unit-magnitude GU and DN palettes. Gemma's dimensions fail the guard. v81 enables DN by default; GU is compile-time opt-in. Earlier architectures retain the single-panel implementation through the function fallback. No new platform headers or host-only APIs enter production.

## Independent validation

`primitive.log`: original author's primitive fixture compiled against actual v81 branch of production hvx_gemv_u8tl2.c using Hexagon libnative, ASan/UBSan. Ten cases cover K1/24/25/704/1536/2048/8192, both extreme all-positive/all-negative weights with activation255, and a third unpaired panel. Exact against two original panels and scalar dense sums.

`workers.log`: actual worker functions and actual moe_m1 structs extracted unchanged, real hvx_worker_pool on its repository pthread shim, real v81 TL2 path and real FP dequantizers on libnative, ASan/UBSan. 160 cases:1..8 total lanes;4/5 experts, first1/count3 and first3/count1 ranges; GU split1/2/4; one expert with magnitude7 to mix fallback with paired neighbors. Atomic per-element claims require a single owner for each GU/paired-DN scratch span and every DN output. Whole-buffer parity, unused-expert/stride regions, and acc_dn redzones pass. This adds real concurrent ownership coverage absent from the author's original single-expert sequential-worker fixture.

`build_workers.py`, `workers_check.c`, `workers_under_test.h`, `native_v81.h`, and `build_primitive.py` reproduce this audit. These are temporary audit scripts with explicit local paths, not proposed portable repository tests. Author's v81 simulator logs and stack reports were independently read: primitive/worker exact PASS, DN worker896B unchanged, paired callee384B, GU caller160B vs168B baseline. The existing pool stack is32KiB; these are static per-function compiler frames, not a measured full call-chain high-water mark.

## Limits

No new silicon timing or model-output claim comes from this review. libnative checks cannot emulate HMX/FastRPC/QuRT cache behavior, and host ASan cannot instrument the proprietary library internals. Parent-controlled LFM replay must confirm unchanged65 generated IDs before a performance claim. The original single-expert fixture is useful but multi-expert concurrency should be retained as regression coverage if this implementation is adopted. No Gemma device benchmark was run.
