# S26 Ultra 다음 최적화 라운드 — handoff (2026-10-12)

PR [nntrainer#4416](https://github.com/nntrainer/nntrainer/pull/4416)(head `aac2acce8`)의
prefill·decode 코드를 읽어 **아직 하지 않은** 최적화 후보를 찾고, 그중 무엇을 할지 정한 기록이다.
`s26u-optimization-log.md`(된 것·안 된 것)의 다음 장에 해당한다. 이 문서를 쓰는 시점에 **구현은 하나도
시작하지 않았다.** 아래 수치는 코드와 기존 측정에서 유도한 추정이고, 기기 측정값이 아니다.

판정 규칙(결정자 2026-10-12): **코드만 보고 효과가 따라오는 것은 적용, 가설이거나 측정해야 크기를 아는
것은 측정 뒤 결정.** 전 항목의 게이트는 답 문장 md5 동일(p1024 `fe44d523`, p2048 `aff26409`,
p4096 `1b462136`, log §6).

## 1. 적용 확정 — prefill (8건)

p2048 prefill 3,308 ms(619 TPS) 기준. 경로는 qkv `mm_u8i4_layer_norm_ld` → attention
`attn_q2_step_fq`(int8 KV, f32 Q alias, u16 ctx) → o-proj `mm_u8i4_layer_res_add_u16` → dense →
router `router_logits_f32`(top-k는 DSP) → MoE `mm_u8i4_moe_layer_norm_add`.

| # | 항목 | 지금 코드 | 바꿀 것 | 추정 이득 | 선행 조건 |
|---|---|---|---|---|---|
| P2 | attention의 CPU 몫을 DSP로 | 층마다 CPU가 (a) K/V f32→fp16 변환·저장을 단일 스레드로(`Applications/CausalLM/layers/mha_core.cpp:1535-1541`, 층당 48 MiB), (b) `calibrate_q2_scales`(`:1143-1190`)가 K/V 16 MiB + Q 32–64 MiB를 다시 훑어 \|max\|. 합 294 ms(p2048, log §3.4) | f32 K/V 행을 Q처럼 ION alias로 `attn_q2_step_fq`에 넘기고 DSP가 max·fp16 cache 행 쓰기·양자화를 한 패스로. 대안: qkv epilogue(`htp_compute_ops.cpp:4790-4845`)에 접기 | −200~290 ms | HVX f32→fp16이 NEON RNE와 **비트 일치**하는지 호스트 단위 테스트(denormal 포함) 먼저 |
| P3 | o-proj 뒤 norm+add 패스 통합 | FC 뒤 별도 M×N 패스 `hvx_rmsnorm_add_blocks_f32`(`test/htp/nntr_hvx_mm_u8i4.c:978`) | `hexkl_mm_u8i4_dma.c:753` row-block 루프의 `row_done` 훅(`:809`)에서 행이 캐시에 있을 때 처리, `out_f32` 직접 기록 | −75 ms (층당 2.5) | 없음 (행 단위 qf32 순서 동일) |
| P6 | KV append bake 병렬화 | `hexkl_kv_q.c:491-509`: 풀이 양자화를 끝낸 뒤 HMX 스레드가 (head, tile) 전부를 직렬로 bake, master를 DDR에서 재독, `xor80_copy`(`:458,:463`) 별도 패스 | 32행 정렬로 나눠 각 워커가 자기 타일을 핫 캐시에서 bake | bake_us ≈2 ms/층 → −60 ms | 없음 (permutation) |
| P7 | Q f32→u16 DDR 왕복 제거 | `q16_worker`(`test/htp/nntr_hvx_attn_q.c:393-460`)가 16 MiB u16을 DDR에 쓰고 `q_dma_push`(`hexkl_attn_q2.c:211`)가 재독 | f32 행을 DMA로 Q staging에 넣고 `stage_qprep`(`:226`) 안에서 양자화 | stats[2] 크기(추정 2–4 ms/층) | 없음 (u16 값 동일) |
| P8 | down 세 번째 chunk 선큐잉 | gu batch 2 vs dn chunk 3(88 tiles/32). lookahead(`hexkl_mm_u8i4_moe.c:2478`)가 0–1만 큐잉, chunk 2는 `hexkl_moe_expand_chunk`(`:2630`)를 `DRAIN_DN` 아래 동기 호출, 호출당 128회 | 마지막 gu batch 시점에 남은 dn chunk 전부 큐잉 | 3–5 ms/호출 추정 | 없음 (정수) |
| P9 | 헛 futex wake 제거 | `hvx_worker_pool.c:324-346` submit마다 wake. 워커는 ~100 µs 스핀(`:191`) 중이라 대부분 아무도 안 잠. attention 호출당 512, MoE 호출당 ~800회 | sleeper 카운터, 0이면 wake 생략 | 1–3 ms/호출 | 없음 |
| P10 | router | (a) `hvx_router_rows_f32.c:116-124`: 256행 세그먼트마다 rs 합산 뒤 x를 DDR에서 재독(46 MB/호출), acc store/reload 11회. (b) `:211-220` 스칼라 분기 top-k, `:221-227` 스칼라 weight 체인 | (a) r0-outer, K 전체 레지스터 유지. (b) vmax/vcmp 라운드, 동점은 낮은 인덱스 유지 | (a) ~1 ms/층, (b) ≤1.5 ms/호출. dot 자체는 FP issue 한계(10.7 중 ~3.5–4 ms가 바닥) | 이 엔트리에 `_timed` 쌍이 없음 → 계측 추가(적용 조건은 아님) |
| P11 | RoPE 테이블 DSP 상주 | `htp_compute_ops.cpp:1455-1460` 층마다 M×2×hd f32(4–8 MiB)를 `rope_pool_`에 memcpy + 매핑, prefill당 ~140 MB | 위치별 고정 테이블이므로 한 번 올려 두기(decode 그래프는 이미 그렇게 함) | ~10 ms | 없음 |

## 2. 적용 확정 — decode (6건)

p1024 C=32 decode ≈80 ms/토큰 기준. "80 vs 63 ms" 갭의 회계: 가중치 28 ms는 전부 53 GB/s 가정인데
S25 측정(`267-dma-bypass.md`)은 FC 42 GB/s, lm_head 50 GB/s, **expert(miss 대기 제외) 14.5 GB/s,
router 13.5 GB/s**. 후자 둘이 바닥 대비 10–12 ms. 단 S26 로그의 `MOE(net)` 열은 #289 이후 dense-early를
포함(`hexkl_graph.c:214`)하므로 DENSE 몫을 빼고 봐야 한다. op 사이 dispatch 손실은 없다(ops/wall
0.999), ARM 토큰 간 0.4–1.5 ms, miss 핸드셰이크 합 ≈1 ms.

| # | 항목 | 지금 코드 | 바꿀 것 | 추정 이득 | 선행 조건 |
|---|---|---|---|---|---|
| D2 | expert M=1 파이프라인 double buffering | `moe_m1_q_worker`(`hexkl_mm_u8i4_moe.c:1225-1255`)가 슬라이스 issue→run→`hexkl_dma_lane_wait` 후 반환. 층당 17회 풀 run(A×8, B, C×8), join마다 DMA 엔진이 빔(깊이 = 행렬 1개, 10–20 µs 갭) | `fc_m1_worker`(`:2928-2986`) 패턴의 레인별 double buffer: stage A를 8×22 유닛 한 run, C를 8×88 한 run, B는 그대로 | MoE → ≈7 ms(바닥 6.6), 3–8 ms/토큰 | 없음 (int32 column sum, expert 순서 scatter `:2118-2135` 불변). `test/htp/host/moe_layer_host_check.c` 갱신 |
| D3 | router 가중치 레이아웃 | `hvx_m1_ops_f32.c:395-416` `router_sm_lane`: 4레인(6 중 2 유휴), 레인당 의존 vmpy→vadd 체인, 512 B 간격 128 B 컬럼, l2fetch lead 16 KB(`ROUTER_PF_ROWS 128`, `:246`). 1.44 MB/층 107 µs | `hexkl_graph_set_param`에서 block-major `[4][K][32]`로 저장, 레인당 360 KB 연속 스트림 + 긴 lead | 107 → ~45 µs/층 ≈ 1.8 ms/토큰 | 없음 (값·k 순서 동일) |
| D4 | miss 층의 호출당 파이프라인 채우기 | `graph_moe_miss`(`hexkl_graph.c:171-241`) 층당 최대 3회 `layer_run`, 각각 `hexkl_dma_ring_reset`(`hexkl_mm_u8i4_moe.c:1817`), 첫 gate_up push 노출(`:1877-1886`, 991 KB), 11 KB rpcmem 복사 2회 full drain(`act_direct/out_direct` 강제 off `:1667-1668`, `moe_dma_copy` `:1830`, `:2150`). 토큰당 60–90회 | `graph_op_router_topk`(`:530`, arena 유휴)에서 `ids[0]` gate_up 선push, M=1에서 act/out direct 허용 | 0.6–2 ms/토큰 (C 작을수록 큼) | 없음 |
| D5 | id 경로의 1 MiB logits 할당·복사 | `nntrainer/models/neuralnet.cpp:1757-1771` 토큰마다 `new float[262144]` + memcpy, `causal_lm.cpp:898`에서 `delete[]`(누수 아님). `want_logits=false`면 DSP가 쓰지도 않음(`htp_compute_ops.cpp:5887-5907`), `generate()`는 `htpDecodeTokenId`(`causal_lm.cpp:410`) | HTP id 경로면 할당·복사 생략 | 0.2–0.4 ms | 없음 |
| D8 | embedding 행 fault | `nntrainer/layers/embedding.cpp:81,152-168` `NNTR_EMBEDDING_MMAP`은 MAP_PRIVATE lazy fault. 다음 행은 argmax 뒤에 알므로 cold 행 = 토큰 경로 위 4 KiB flash 읽기 | 로드 시 `madvise(WILLNEED)` | 0–0.3 ms | 없음 |
| D9 | rebind tail | `hexkl_mm_u8i4_dma.c:373-376` `tail_from_arena`가 scale+colsum을 uncached arena에서 로드, 24 µs/라운드 ≈0.5 ms/토큰. ARM은 이미 값을 가짐(`htp_compute_ops.cpp:7990-7999`) | ANS 페이지(1024 B, 지금 자리 없음)에 실어 보내기 → mailbox 레이아웃 변경 | ≤0.5 ms | 없음. 후순위 |

## 3. 보류 — 측정 뒤 결정 (6건)

| # | 항목 | 왜 보류 | 결정에 필요한 측정 |
|---|---|---|---|
| P1 | staging 버퍼 static 매핑 | 로그 §3.3: `RPCMEM_TRY_MAP_STATIC`은 −169 ms(p2048)였지만 NPU 쓰기가 CPU에 안 보여 기각. 그때 CPU 쪽 CIVAC만 시도했고 **DSP L2 flush는 안 함**. 트리의 mailbox 경로는 `qurt_mem_cache_clean`을 씀(`hexkl_token.c:62,76`). 층당 매핑 ≈460 MiB × 27 µs/MiB ≈ 12 ms, prefill당 ≈370 ms가 상한 | 가설. prefill skel 엔트리에 출력 FLUSH / 입력 INVALIDATE를 넣고 static 매핑 → md5 유지 여부 |
| P4 | MoE down epilogue 재구성 | 2-bit 호출에서 `dn_fg = exp_bg`(`hexkl_mm_u8i4_moe.c:2292`)라 `moe_dn_fg_worker`(`:2753` → `hvx_dequant_i32.c:139`)가 전경에서 23 MB `out_c`의 랜덤 행 load+add+store, `memset`(`:1833`)·residual/norm 패스(`nntr_hvx_mm_u8i4.c:1953`) 별도. 제안: (expert,row) slab 순차 기록 + expert 순서 reduce/add/norm 한 패스(heap +184 MB p2048). 로그의 "scatter 128 ms"는 `a1b2bcc8c` **전** 값 | epilogue의 DDR store를 뺀 probe로 지금 남은 비용 |
| P5 | softmax 분할 3 → 5 | `hexkl_attn_q2.c:655` `n_units=3`(`SMX_PARTS=3`), 워커 3·4 유휴. 효과는 softmax가 PV/QK를 기다리게 할 때(stats `wait` > 0)만 | `test/htp/nntr_hvx_attn_q.c:600-616`이 버리는 `us_qprep/us_dma/us_submit/us_total`(커널은 `hexkl_attn_q2.c:881-888`에서 채움) 출력 → wait 확인 |
| D1 | expert DMA 큐 1 → 4 | `htp_moe_opts.h:71` `HTP_MOE_DMA_QUEUES_DEFAULT 1u` → `hexkl_mm_u8i4_moe.c:1602`. expert 357 MB/토큰이 `hexkl_dma_ring` 하나. FC는 레인별 큐라 42 GB/s. 로그 §3.2의 "DMA_QUEUES 2·4 효과 없음"은 TPS(±1, miss 흔들림 10 ms) 판정 | `NNTR_MOE_DMA_QUEUES=4` 번갈아 1셀, **`MOE(net)` − DENSE 몫**으로 판정 |
| D6 | miss 리더 4 → 6 | `readMisses`(`htp_compute_ops.cpp:7379-7392`) expert당 2 job, 리더 `min(4,n)`, 전부 join 후 응답 → 3 miss = 6 job/4 리더, 꼬리 25 % 유휴. flash-bound면 0(S25 8리더 무변화) | S26 miss가 flash 출처인지 페이지 캐시 출처인지 |
| D7 | arena CPU 매핑 cached | `htp_compute_ops.cpp:8340-8346` `HTP_RPC_FLAGS_UNCACHED`; 복사 3.7–4.9 GB/s(`:7960`, 282-C) = 0.7–0.9 ms/라운드. cached + `a.seq` 전 sync + rebind 시 DSP invalidate(`hexkl_token.c:244-257`, 지금은 포인터 스왑). 페이지 캐시 출처면 ≈8 ms, flash면 0 | D6과 같은 측정 + 일관성 probe |

D1·D6·D7은 `NNTR_HTP_PROFILE=1` 1회로 재료가 나온다(`round us:` `htp_compute_ops.cpp:5757`,
per-kind `:5661`). 기타: `NNTR_HTP_PREDICT=1`이면 ROUTER 2배(`hexkl_graph.c:527`, +1.6 ms) → config of
record에서 꺼져 있는지 확인.

## 4. 비추천 / 벽

- `NNTR_HTP_ROUTER_U8I8`: logits 0.17–0.39 차이, md5 깨짐.
- softmax 세 번째 패스 제거(`hvx_softmax_q.c:766-808`): PV 반올림 변경.
- router f16/bf16 저장(`hvx_m1_ops_f32.c:429` ponytail): 파일의 f32 router 값이 bf16-exact일 때만 무손실, 미확인.
- `HVX_SOFTMAX_Q_MAX_TILES 128`(`hexkl_attn_q2_plan.h:160`)이 가시 컬럼 4096 상한 — global 층 p4096이 딱 걸려 있음 (성능이 아니라 벽).

## 5. 이미 바닥이라 다시 볼 필요 없는 것

- prefill: router 입력 staging(dense가 먼저 돌고 `out_pool_` 스왑, 복사 없음 — Gemma 빌드 순서 `gemma4_causallm.cpp:476-490`), qkv 출력 sample-only·Q/ctx ION alias, expert 스트리밍(글로벌 960슬롯, 백그라운드 리더, 층 0만 동기), top-k DSP, attention tile skip·softmax/epilogue↔PV/QK 파이프라인·K/V double buffer·zero-point int16, MoE weight DMA chunking(다음 gate_up이 DN(n−1) 아래)·row-quantize-once·slot 순서 gather, router 4×4 블로킹·W VTCM 1회/호출.
- decode: op dispatch(`hexkl_graph_forward` `:1376-1405` barrier 없음), worker pool 400×pause 스핀·토큰당 park 1회, miss post 시점(ROUTER_TOPK 직후, L+1은 router 의존으로 더 못 당김), miss poll 20 µs·notice 1 µs·LRFU 5–7 µs·rebind 포인터 스왑, lm_head 50 GB/s + softcap/ban/argmax DSP(`hexkl_graph.c:677-706`) + id만 반환, FC WH `fc_m1_worker` double buffer(42 vs 53은 레인당 첫 블록 노출 ≈2 ms 이론치), expert GEMV 연산부 >150 GB/s 상당, ARM 토큰 간 작업(`arm_us` 0.4–1.5 ms와 일치).

## 6. 진행 규칙

- 항목당 커밋 하나, 빌드를 기기에 나란히 두고 번갈아 3회, 2분 냉각, 답 md5 비교(log §6).
- 추천 순서: P2 → P3 → D2 → D3 → 작은 것들(P6–P11, D4, D5, D8, D9). P2는 변환 비트 일치 테스트가 먼저.
- 보류 6건은 §3의 측정이 나온 뒤 이 문서에 결과와 결정을 덧붙인다.

## 7. 2026-10-13 적용 결과

기준은 이 문서를 쓴 뒤 PR에 올라온 작성자 커밋 3개(`4b50de470` decode overhead, `525aa04af` TL2 experts,
`48a6bbb22` expert prefetch)를 포함한 head `48a6bbb22`이다(기기 디렉터리 `pr4416_h48`). 그 위에 항목당 커밋
하나씩 올렸고, 전부 host check 통과 + 기기에서 답 문장 md5 동일(기본 env p1024 `fe44d523`, p2048 `aff26409`;
FAST env p1024 `46a84cf6`). env는 §5.2의 config of record에 사용자 지시(10-13)로 `NNTR_HTP_ATTN_M1_FAST=1
NNTR_EMBEDDING_MMAP=1`을 더한 것(§7.2부터); §7.1은 그 전의 기본 env.

### 7.1 항목별 상태

| # | 상태 | 커밋 | 비고 |
|---|---|---|---|
| P3 | 적용·검증 | `0377eb9c4` | 첫 구현은 1 ulp 차이가 8 %의 원소에 났다: **컴파일러가 `Q6_Vsf_*`를 qf32 연산으로 낮추고 따라갈 수 있는 변환을 지우기 때문에**, 기존 `addb_worker`는 handle(블록) 경계마다 누적기를 sf로 한 번 반올림한다. 융합 경로가 같은 지점에서 같은 반올림을 하도록(store를 거쳐 강제) 고쳐 동일. 기기 진단(35호출 비트 비교) bad=0 |
| P6 | 적용·검증 | `8e49f2776` | (head, tile) 단위 풀 분배. permutation |
| P7 | 적용·검증 | `ba1a09bfc` | f32 Q를 DMA로 staging에 넣고 `stage_qprep` 안에서 같은 HVX 연산 순서로 양자화. `q16_worker`·DSP heap의 u16 사본 삭제 |
| P8 | 적용·검증 | `37d36e941` | 마지막 gate_up batch가 남은 dn chunk 전부 큐잉 |
| P9 | 적용·검증 | `a015f2ab7` | sleepers 카운터, seq_cst Dekker 쌍 |
| P10 | (a) 적용, 기기 확인 §7.3 | `555f6e173` | 행 블록 outer / k 블록 inner로 루프만 교환(같은 rows_block 호출, 같은 저장 지점). (b) 벡터 top-k는 미착수 |
| P11 | 적용·검증 | `4ed071a61` | `stageInput`로 보관: 같은 테이블 포인터·바이트가 층마다 오고 sliding/global은 size class가 달라 서로 안 밀어냄 |
| P2 | 1단계 적용, 기기 확인 §7.3; 2단계 코드만 | `c02772db3` | 1단계: 캘리브레이션(K/V \|max\|, Q min/max, scale·encoding 유도, fixed scale 설정)을 `attn_q2_step_fqc` 한 호출 안으로. max는 순서 무관이라 비트 동일. 2단계(K/V f32→fp16 변환·cache 행 쓰기를 DSP로): CPU cache가 rpcmem 블록이라 DSP가 직접 쓸 수 있고, HVX 정수 RNE 변환기(`hvx_f16_rne.h`)는 호스트 에뮬레이터에서 `hexkl_cvt_f32_to_hf`와 126만 케이스(fp16 전 패턴·동점·subnormal 경계) 비트 동일. 엔트리 `attn_q2_step_fqk`·ARM·mha_core 배선까지 작성했으나 **미빌드·미검증**(stash) |
| D3 | 적용·검증 | `03e16fab5` | block-major `[Ep/32][K][32]`, `hvx_router_sm_pack_f32` |
| D5 | 이미 됨 | `4b50de470` (작성자) | `decode_model->incremental_inference`가 tensor를 돌려주어 `new float[]`를 안 탄다 |
| D8 | 이미 됨 | — | `Tensor::activate()`가 mmap 직후 `madvise(WILLNEED)` |
| D2 | 미착수 | — | 작성자 커밋이 M=1 스케줄을 레인별 다중 큐 DMA(#117/#177/#185)로 바꿔 handoff의 분석(`:1225-1255`)이 더는 맞지 않는다. decode 벽시계가 같은 빌드에서 11.2–15.3 TPS로 흔들려(§7.2) 3–8 ms/토큰은 `NNTR_HTP_PROFILE=1`의 MoE us/토큰으로만 판정 가능 |
| D4, D9 | 미착수 | — | 같은 이유(decode 소음 안) |
| P1 P4 P5 D1 D6 D7 | 미측정 | — | — |

### 7.2 r1 = P3+P6+P7+P8+P9+P11+D3 (기본 env, G128, 번갈아, 2분 냉각)

| 설정 | 빌드 | prefill ms (회차) | decode 마지막 64 TPS (회차) |
|---|---|---|---|
| p1024 | h48 | 2134, 1872, 1936 | 13.4, 14.5, 14.4 |
| p1024 | r1 | 1650, 1689, 2281* | 14.9, 14.3, 12.1* |
| p2048 | h48 | 3371, 3361, 3436 | 11.2, 14.4, 15.3 |
| p2048 | r1 | 3213, 3141, 3304 | 14.0, 14.1, 13.1 |
| p4096 | h48 | 7518 (1회) | 11.9 |
| p4096 | r1 | 7068 (1회) | 13.3 |

읽는 법: (1) prefill은 p1024 중앙값 1872→1689(−10 %), p2048 평균 3389→3219(−5 %), p4096 7518→7068(−6 %).
\* r1 p1024 3회차는 배터리 28.8 °C에서 나온 이상치(같은 빌드 1·2회차의 +35 %). (2) **decode 마지막 64 TPS는
같은 빌드에서 11.2–15.3으로 흔들려 D3·P9·P8의 토큰당 몇 ms는 벽시계로 못 가른다** — 기기가 다른 작업자와
공유되고(같은 날 `/data/local/tmp/gemma4_decode_prefetch_*`) 측정 중 배터리 26→30 °C. (3) p4096은 FAST env로
돌아 md5 기준(`1b462136`)과 못 비교했고 두 빌드의 본문이 달랐다(FAST p4096은 §2.7대로 near-tie 형식) → §7.3의
기본 env 재확인.
