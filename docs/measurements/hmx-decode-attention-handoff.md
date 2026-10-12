# HMX decode attention 평가 — handoff (2026-10-12)

다른 세션이 이 문서만 보고 "decode attention을 비트 동일 제약 없이 빠르게 할 수 있는가, 정확도 비용은 얼마인가"를 평가할 수 있도록 적는다. 대상은 Gemma-4 26B-A4B(2-bit expert, QS4CX FC, untied head)의 one-PD decode 토큰, 기기는 S26 Ultra(SM8850, V81).

## 1. 지금 상태 (측정값)

| 항목 | 값 | 출처 |
|---|---|---|
| decode 토큰 | 80–86 ms (p1024, C=32) | `[HTP] graph per-kind us/token` |
| 그중 ATTN_M1 | **10.3 ms/토큰** (p1024, 30층 합) | 같은 줄, `ATTN_M1=10265` |
| ATTN_M1 구조 | int8 K/V 캐시(`NNTR_HTP_ATTN_M1_Q8=2`), hf(fp16) 레인 8체인, CPU `fmla .8h` 반올림 순서를 그대로 재현 | `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c`, `hvx_attn_m1_hf.h` |
| 효율 | 378 MMAC를 32 GMAC/s → HVX hf 피크의 약 1/15 (작업자 계측: scores 30 + PV 31 Mcyc, 8레인 포화, ≈3 MAC/cycle) | `docs/measurements/s26u-prefill-handoff.md` §9 D3 |
| 왜 느린가 | int8 K/V를 hf로 풀어(`Q6_Wh_vunpack_Vb`) fp16 곱, 8개 누산 체인과 트리 축소 순서가 CPU와 같아야 해서 레인 폭·누산 형식을 바꿀 수 없음 | `hvx_attn_m1_hf.h` 머리 주석 |
| 기대 | 제약을 풀면 토큰당 3–4 ms(u8×i8 `vrmpy` 내적, f32 누산). decode −6~7 ms ≈ **+8 %** | 추정, 기기 미측정 |

decode attention의 수치가 바뀌는 변경은 지금까지 "비트 동일(답 md5 동일)" 규칙 때문에 전부 보류됐다. 이 평가는 그 규칙을 풀고 **속도와 정확도 비용을 함께 재는 것**이다.

## 2. 후보 세 가지

### A. PR #4418의 `NNTR_HTP_ATTN_DECODE_HMX` 이식
- 내용: M=1 attention을 prefill용 fp16 flash-attention HMX 커널(`hexkl_attn_f16.c`, Q·Kᵀ와 P·V를 HMX 행렬곱, online-softmax 재스케일을 diag 행렬곱으로)에 태움. host 스위치는 `htp_compute_ops.cpp`의 `decode_hmx`(PR 4418 트리 9016–9025행), DSP 엔트리 `nntr_hvx_attn_f16_decode`.
- 적용 조건이 LFM2 형상에 고정: `n_q == 1 && head_dim == 64 && n_head_q == 32 && n_head_kv == 8 && cache_to >= 256 && !window && softcap == 0 && !sinks`. Gemma(hd 256/512, 25층 윈도우 1024)는 그대로는 켜지지 않음 → 조건 일반화 + hd 256/512 타일 경로 확인 필요.
- 그들의 A/B(`docs/measurements/lfm2_24ba2b_s26_call_fusion.json`): HMX=1 strict 15.2–15.6 TPS vs HMX=0 hybrid 16.6–17.1. **decode 이득이 확인되지 않음.** 결론 문구: "no clear throughput improvement from fusion alone".
- 관련 커밋: `f4f24a65e` "[HTP] Optimize resident KV caches and decode attention" (18 files, +1612): `hexkl_attn_f16.c/_plan.h`, `hexkl_attn_q.c`, `hexkl_kv_q.c`, `hvx_attn_decode_f16.c`, `hvx_attn_decode_q.c`, device test `nntr_hvx_attn_q.c`. 우리 트리에도 `hvx_attn_decode_f16.c`와 `nntr_hvx_attn_f16_decode`(`test/htp/nntr_hvx_attn_f16.c:505`)가 있고 host는 `kDecodeMaxRows=5`일 때 그 경로를 쓴다(`htp_compute_ops.cpp:9265`) — 단 one-PD 토큰(E2E)에서는 ATTN_M1 그래프 op가 돌므로 이 경로는 쓰이지 않는다.
- 참조 fetch: `git fetch https://github.com/nntrainer/nntrainer.git pull/4418/head:refs/pr/4418` (head `faeede212`).

### B. ATTN_M1을 제약 없이 다시 쓰기 (추천)
- one-PD 그래프의 `HTP_OP_ATTN_M1`(`hexkl_graph.c` `graph_op_attn_m1` → `hvx_attn_m1_forward`)을 그대로 두고 커널 내부만 교체: scores = Q(int8 또는 u8로 양자화)·Kᵀ(int8)를 `Q6_Vw_vrmpy`로(4배 밀도), f32 softmax, PV = P(u8)·V(int8) 같은 방식. 캐시 레이아웃(`kt8` [tiles][hd][64], `v8` [seq][hd])은 유지.
- 옵트인 env 제안: `NNTR_HTP_ATTN_M1_FAST=1`. 기본 경로와 같은 바이너리에서 A/B 가능.
- 검증: 기존 `attn_m1_host_check`는 비트 비교라 그대로 못 씀 → f64 참조 대비 SNR/최대 오차 한계(기존 `ATTN M1 GEMMA BIT-IDENTICAL` 줄이 f64 대비 SNR을 이미 출력: 35.5 dB가 현재 값)로 바꾼 검사 하나 추가.
- 규모: 하루.

### C. producer-quantized Q8 캐시 (seed 제거)
- #4418의 "producer-quantized Q8 cache": prefill attention이 decode 캐시를 바로 씀. 우리는 prefill 끝에 CPU fp16 캐시 → `attn_m1_kv_append_f16`(80–160 ms). 수치가 바뀌는 이유: prefill의 양자화 스케일(헤드당)과 decode의 스케일(차원당)이 다름. 이득은 첫 토큰 −80~−160 ms뿐 → 우선순위 낮음.

## 3. 측정 방법

### 3.1 환경 (이 워크스테이션)
- 워크트리 `~/nntrainer-4416`(브랜치 `htp/4416-remaining` = PR #4416 head, `Seunghui98:htp/s26-prefill-opt`), 빌드 스크립트 패턴: `HEX_ARCH=v81; source tools/htp/env.sh; HEXKL_ROOT=~/Qualcomm/hexkl-1.0-beta.2/hexkl_addon` → stub(`generate_stub.sh`) → skel(`test/htp/build.sh`) → `ninja -C builddir && meson install --no-rebuild --skip-subprojects` → `Applications/CausalLM/build_android.sh --htp --cache`. builddir는 `-Dhexkl-lib-subdir=6.4.0.1/armv8_android26`.
- 기기 S26 Ultra `R5KL309ALRK`, `$D=/data/local/tmp/nntrainer/causallm`. 최신 빌드는 `$D/pr4416_mem2`(head `24a2fe91a`), 기준 `$D/pr4416_base`(`c93cdc158`). 설정 `$D/g{1024,2048,4096}_i2048`(G=128), 매트릭스용 `$D/m_p{P}_g{G}`(P∈1024/2048/4096, G∈64/512/1024).
- 실행 env: `NNTR_NUM_THREADS=8 NNTR_HTP_E2E=1 NNTR_HTP_E2E_FREE_KVQ=1 NNTR_HTP_KV_DROP_CPU=1 NNTR_HTP_DROP_HOST_FC=1 NNTR_HTP_ATTN_M1_Q8=2 NNTR_HTP_FC_IN_ARENA=1 NNTR_MOE_CACHE_EXPERTS=32` (+ `NNTR_MOE_CACHE_GROW=8` 권장). 2분 냉각, 번갈아 3회.
- 속도: `generation(last 64)` TPS와 `NNTR_HTP_PROFILE=1`의 `[HTP] graph per-kind us/token`의 `ATTN_M1=`.

### 3.2 정확도
1. **기준 continuation**: 기준 빌드(`pr4416_mem2`, 비트 동일 경로)를 greedy로 돌려 생성 id 열을 `$D/ppl_gemma_p{P}.ids`로 저장(`NNTR_PPL_DECODE` 경로가 `ppl_*.ids`를 쓰는 형식; `Applications/CausalLM/models/causal_lm.cpp`의 `ppl_dec`/`ppl_forced` 참고). 지금 md5 기준값: p1024 `fe44d523`, p2048 `aff26409`(답 줄 `grep -a -o '<channel|>.*' log | head -1 | md5sum | cut -c1-8`), G=128 설정.
2. **후보 빌드**: 같은 continuation을 강제(`NNTR_PPL_DECODE=<ids>`)해 teacher-forced NLL/token과 top-1 일치율을 낸다. 기준 빌드 자기 자신은 NLL 차 0, top-1 100 %.
3. **판정 한계(사용자와 합의 필요)**: 예 NLL +0.01/token 이내, top-1 98 % 이상. 그 안이면 생성 문장을 사용자가 읽고 최종 허용.
4. 보조: `attn_m1_host_check`의 f64 대비 SNR(현재 35.5 dB). 후보 커널의 SNR을 같은 자리에서 출력.

### 3.3 함정
- 문장 md5가 바뀌는 것은 정상(near-tie 토큰). md5가 아니라 §3.2로 판정.
- p4096 fp16 KV는 폰이 재부팅됨(int8 유지). `init_seq_len ≤ max_seq_len`이어야 함.
- C=32는 max_seq ≥ 2560(p2048 G≥512)에서 DSP 매핑 한도로 expert swap 실패(`0x80000402`). 긴 G는 C 24–28.
- 1회 측정은 ±0.5–1 TPS 흔들림. decode 비교는 번갈아 3회.

## 4. 참고 파일
- 우리 커널: `nntrainer/tensor/htp_backend/hvx/hvx_attn_m1_f32.c`(forward: `p1_unit_q8`, `pv_walk_q8_*`; seed: `append_tile_q8`), `hvx_attn_m1_hf.h`, 스펙 `attn_m1_det.h`, host check `test/htp/host/attn_m1_host_check.c`, 그래프 호출 `hmx/hexkl_graph.c` `graph_op_attn_m1`.
- 작업자 분석: `docs/measurements/s26u-prefill-handoff.md` §9(D3), `docs/measurements/s26u-optimization-log.md` §5.
- #4418: 위 §2-A의 파일들, 측정 `docs/measurements/lfm2_24ba2b_s26_*.md/json`, 최종 env `docs/measurements/lfm2_htp_pr_validation_20261012/environment.json`.

## 5. 결과 (2026-10-12, 후보 B 구현: `NNTR_HTP_ATTN_M1_FAST=1`)

커밋 `011a5a315`. 캐시 바이트·레이아웃은 그대로, 산술만 교체: q는 헤드당 int16(K의 차원별 스케일 포함), scores는 int8 타일 행을 `vunpack` 후 `Vxx.w += vmpy(Vu.h, Rt.h)`로 int32 누산 → f32, softmax f32(`hvx_exp_det_sf`), P는 15비트 정수로 확률 행에, PV는 같은 vmpy로 256 위치 블록마다 int32 → f32 flush, 끝에 `out *= sv / l`. P3의 lane 단위는 (kv head, 128-dim chunk)(gqa 8은 half까지)라 V 바이트가 DDR을 1~2회만 지난다. 기본값은 기존 비트 동일 경로이고 env로만 켜진다(int8 캐시 필요, hd % 256, gqa 1/2/4/8). 호스트 체크 `ATTN M1 FAST`: f64 대비 SNR 27.8–42.1 dB(스펙 경로 최저 35.5), 워커 수 무관 동일 바이트.

### 5.1 S26 Ultra, 같은 바이너리(`$D/pr4416_fast`), env만 다름, 번갈아, 2분 냉각, G=128, C=32, GROW=8

| 회차 | 설정 | prefill ms | last64 TPS | ATTN_M1 us/token | MOE us/token | 답 md5 |
|---|---|---|---|---|---|---|
| p1024 1 | 기존 비트 동일 경로 | 1831 | 12.08 | 10383 | 39529 | fe44d523 |
| p1024 1 | fast 켬 | 2209 | 10.91 | 6390 | 48200 | 46a84cf6 |
| p1024 2 | 기존 비트 동일 경로 | 1905 | 12.74 | 10318 | 39671 | fe44d523 |
| p1024 2 | fast 켬 | 1813 | 13.38 | 6236 | 37276 | 46a84cf6 |
| p1024 3 | 기존 비트 동일 경로 | 1761 | 12.91 | 10300 | 36999 | fe44d523 |
| p1024 3 | fast 켬 | 1748 | 13.84 | 6231 | 37946 | 46a84cf6 |
| p2048 1 | 기존 비트 동일 경로 | 3346 | 11.70 | 12018 | 46989 | aff26409 |
| p2048 1 | fast 켬 | 3272 | 14.80 | 6888 | 37680 | 405ed974 |
| p2048 2 | 기존 비트 동일 경로 | 3801 | 13.06 | 11904 | 39357 | aff26409 |
| p2048 2 | fast 켬 | 3415 | 14.54 | 7013 | 38763 | 405ed974 |

ATTN_M1: p1024 10.3 → 6.2–6.4 ms/token, p2048 11.9–12.0 → 6.9–7.0. 전체 TPS 차이는 MOE miss 대기의 흔들림(p1024 1회차 fast 셀의 MOE 48 ms)이 attention 이득보다 크므로 ATTN_M1 열로 판정. fast 경로의 답은 실행마다 같다(결정적).

### 5.2 분해 (phase words, `NNTR_HTP_PROFILE=1`, kpcyc/token, cache 0 = sliding 25층, cache 1 = full 5층)

| | append | scores | softmax | pv | pool | call us |
|---|---|---|---|---|---|---|
| p1024 기존, sliding | 5634 | 30275 | 10703 | 32376 | 10430 | 7677 |
| p1024 fast, sliding | 1328 | 30388 | 4446 | 34878 | 9604 | 5242 |
| p1024 기존, full | 801 | 9167 | 1914 | 23344 | 4621 | 2582 |
| p1024 fast, full | 174 | 5055 | 863 | 5186 | 1689 | 897 |
| p2048 기존, full | 832 | 15885 | 3214 | 45959 | 8556 | 4461 |
| p2048 fast, full | 189 | 7469 | 1606 | 11893 | 2939 | 1495 |

- sliding 층의 scores·PV lane-cycle은 두 커널이 같다: 토큰당 K 52 MB·V 52 MB를 24–27 GB/s로 흘리는 중이고, 이것이 이 SoC의 HVX l2fetch DDR 한계(`docs/measurements/117-m1-gemv-vtcm-feed.md` 21–27 GB/s). 커널로는 더 못 줄인다. 이득은 직렬 APPEND(스칼라 q×scale, 2.7 ms → 0.6 ms)와 full 층(gqa 8, hd 512: 컴퓨트 바운드였고 PV의 V 중복 읽기 제거)에서 나왔다.
- 스칼라 q 양자화 첫 버전은 APPEND 8.2 M pcyc(3.9 ms)였다. 벡터화 전에는 이득이 0이었다 — 분해를 먼저 보지 않았으면 커널 탓을 했을 것.
- 더 내려가려면 KV 바이트 자체를 줄이거나(4비트 K/V) DMA로 VTCM에 넣어야 한다(#267 계열). p2048 이상에서 sliding 층 비용은 창 1024로 고정이라 full 층만 자란다.

### 5.3 안 한 것
- §3.2 정확도 게이트(강제 continuation NLL / top-1): 사용자 지시로 이번엔 생략. 채택 전 필수.
- 기본값 전환: env 옵트인 그대로.
