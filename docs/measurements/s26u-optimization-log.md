# S26 Ultra Gemma-4 MoE 최적화 기록 — 된 것, 안 된 것, 그 이유

2026-10-10 ~ 10-11, Galaxy S26 Ultra(SM8850, Hexagon V81), Gemma-4 26B-A4B 2-bit MoE(30층, 층당 expert 128개, top-8).
코드는 PR [nntrainer#4416](https://github.com/nntrainer/nntrainer/pull/4416)(head 브랜치 `htp/s26-prefill-opt`, 지금 `6c9692c74`)에 있다.
환경·빌드·실행 방법과 측정 원본은 브랜치 `htp/s26-prefill-handoff`의 `docs/measurements/s26u-prefill-handoff.md`에 있고(PR에는 넣지 않음), 이 문서는 그 결과만 추린 것이다.

## 1. 결과

| | 시작 | 지금 | 개선 |
|---|---|---|---|
| prefill 1024토큰 | 2,313 ms (443 TPS) | 1,815 ms (**564 TPS**) | +27 % |
| prefill 2048토큰 | 4,650 ms (440 TPS) | 3,308 ms (**619 TPS**) | +41 % |
| prefill 4096토큰 (C=24) | 9,246 ms (443 TPS) | 7,270 ms (**563 TPS**) | +27 % |
| decode, 1024 프롬프트 | 10.83 TPS | **11.93 TPS** | +10 % |
| decode, 2048 프롬프트 | 10.44 TPS | **12.16 TPS** | +16 % |
| decode, 4096 프롬프트 (C=24) | 8.0 TPS | 9.6 TPS | 편차가 커서 참고만 |

읽는 법:

- prefill의 "시작"은 10-11 아침의 빌드(`1148f40f5`)다. 그 전날 밤의 attention 작업 전은 p1024 2,254–2,514 ms, p2048 4,466–4,530 ms였다.
- decode TPS는 **128토큰을 생성했을 때의 마지막 64토큰**(`generation(last 64)` 줄)이다. `generation: 128 tokens` 줄은 첫 토큰 시간(1.5–4.7 s)이 들어가 더 낮다.
- decode의 "시작"은 10-11 저녁의 값이다. 폰의 `generation_config.json`이 그날 15:41에 바뀌어 문장 끝 뒤에도 128토큰을 다 만들게 됐고, 같은 코드가 그 전(12–13 TPS)보다 낮게 읽힌다. 전후를 같은 설정에서 쟀으므로 개선 폭은 유효하다.
- expert 풀은 C=32(4096만 24). 값은 빌드를 번갈아 돌린 3회 평균(4096은 2회), 2분 냉각. 한 번 잰 값은 prefill ±5 %, decode ±0.5 TPS 흔들린다.
- **정확도**: PR에 넣은 것은 전부 생성 문장이 기준 빌드와 바이트 동일하다(답 줄 md5 비교). 예외는 §2.1의 attention 커밋 하나로, 그때 기준 문장이 한 번 바뀌었다.

## 2. 성공한 것 (PR에 있음)

### 2.1 prefill

| 커밋 | 무엇을 | 왜 효과가 있었나 | 측정 |
|---|---|---|---|
| `7d9490881` `f5a70c4d5` `66e023786` | attention의 Q·context를 CPU가 변환·복사하지 않고 NPU 안에서 u16로 넘김 | CPU 변환·복사가 30층 합 106 ms였다 | 그 106 ms → 0. 문장 md5가 바뀜(p1024 `1fd0625a` → `2ec61a41`): near-tie 토큰에서 갈린 것이고 요약 내용은 일관. 두 경로의 context 차이는 최대 8 step |
| `fb6e71850` | pre-norm scratch를 M×K에서 레인당 4행으로 | NPU heap 15–24 MB를 쓰던 것이 0.6 MB | p4096 C=24가 `AEE_ENOMEMORY` 없이 돈다 |
| `e1e4ffca2` | FC slice를 1 MiB로(K ≥ 4096) | o-proj가 한 번의 호출로 끝남 | o-proj 호출 95 → 65 (p1024) |
| `1148f40f5` | 블록의 텐서를 NPU 쪽 버퍼에 그대로 두고 CPU에는 표본 2 KB만 복사 | 층 사이를 오가던 memcpy가 사라짐 | staging memcpy 143 → 22 ms (p1024), 402 → 39 ms (p2048) |
| `a1b2bcc8c` | MoE: 행을 한 번만 양자화, down 결과를 행 단위로 scatter | MoE의 병목은 HMX 행렬곱이 아니라 HVX 워커의 메모리 접근이었다(호출당 226 ms 중 scatter 128, pack 63) | p1024 2,313 → 1,985 ms, p2048 4,650 → 4,034 ms |
| `ecb2c87db` | MoE: pre norm을 scan 안에서, 결과를 제자리에 누적 | 같은 23 MB 텐서를 여러 번 훑던 패스를 없앰 | ↓ 세 커밋 합 |
| `cebe70803` | qkv·o-proj의 행 chunk를 NPU 쪽이 돌게 해 RPC를 1회로 | FastRPC가 넘긴 버퍼를 호출마다 통째로 매핑·sync한다(버퍼 MiB당 ≈27 µs). chunk 2–4회가 그대로 2–4배였다 | ↓ |
| `935a718af` | decode용 KV seed를 ION 버퍼로 전달 | f32 K/V 1 GB를 일반 메모리로 보내고 있었다 | p1024 → 1,955 ms, p2048 → 3,526 ms (세 커밋 합). 첫 토큰의 seed 1,055–1,097 → 973 ms |
| `f29642e3e` | qkv의 head별 norm+RoPE를 역양자화 패스에 합침 | qkv의 NPU 시간 46 %가 행렬곱 뒤의 별도 패스였다(층당 5.9 of 12.7 ms) | p1024 → 1,815 ms, p2048 → 3,308 ms, p4096 → 7,270 ms |

### 2.2 decode

토큰 하나(약 86–98 ms)의 절반 가까이가 **expert miss 대기**(풀에 없는 expert를 flash에서 읽는 동안 NPU가 기다리는 시간, 26–43 ms)다. 성공한 것은 둘 다 이 대기를 겨냥한다.

| 커밋 | 무엇을 | 왜 효과가 있었나 | 측정 (마지막 64, 3회 평균) |
|---|---|---|---|
| `ca51dd761` | miss를 기다리는 동안 그 층의 dense 분기를 먼저 계산 (`NNTR_HTP_DENSE_EARLY`를 기본값으로) | 대기 시간 자체는 flash 속도라 못 줄인다. 그 시간에 NPU가 할 일을 주는 것은 된다 | p1024 10.83 → 11.53, p2048 10.44 → 10.82 TPS |
| `353daf3a4` | expert 교체 정책을 LRU에서 LRFU(반감기 32토큰)로 (`NNTR_MOE_LRFU`를 기본값으로) | miss 개수는 거의 같다(50.9 → 54.4, 57.9 → 56.9개/토큰). miss가 나는 **층**이 줄어 읽기를 기다리는 횟수가 준다 | p1024 → 11.93, p2048 → 12.16 TPS. 대기 25–35 → 24–29 ms |
| `6c9692c74` | miss 요청을 감시하는 CPU 스레드가 `sched_yield` 대신 pause 명령으로 대기 | S26에서 요청을 알아채는 데 86–150 µs가 걸렸다(토큰당 17–24회) | 0 µs. 토큰당 약 2 ms라 TPS로는 잡음 안 |
| `9873fdcdf` | LRFU가 내보낼 expert를 고를 때 점수를 한 번만 훑음 | 내보낼 것 하나마다 풀 전체(960개)를 훑으며 점수를 다시 계산하고 있었다 | 요청을 받고 읽기를 시작하기까지 96–109 → 36–39 µs/회. 토큰당 약 1.5 ms라 TPS로는 잡음 안. 고르는 결과는 같다(miss 개수 동일) |

끄는 법: `NNTR_HTP_DENSE_EARLY=0`, `NNTR_MOE_LRFU=0`.

### 2.3 첫 토큰 (10-12, dlwlzzero 세션)

첫 토큰(`[TIMING] first_token_ms`)은 p1024 1,463 ms, p2048 2,206 ms였고, 그중 decode 토큰 자체는 220–240 ms다. 나머지를 재서 셋을 없앴다. 전부 답 문장 md5 동일(p1024 `fe44d523`, p2048 `aff26409`), decode·prefill TPS 변화 없음.

| 커밋 | 무엇을 | 왜 효과가 있었나 | 측정 (p1024 / p2048) |
|---|---|---|---|
| `321fb63a2` | KV seed의 int8 양자화를 HVX 벡터로 (amax, 양자화, K의 64×128 바이트 전치를 vshuff 네트워크로) | 원소마다 스칼라 `q8_of`, K 바이트마다 64 B 간격 scatter(바이트당 캐시 라인 하나)였다 | `kv_seed_rpc_ms` 429 → 93 / 900 → 183 |
| `970655c18` | seed 행을 CPU 캐시의 fp16 비트 그대로 DSP에 (`attn_m1_kv_append_f16`), DSP가 정수 연산으로 정확히 widening | CPU 풀에서 fp16→f32 변환 145 / 288 ms, staging 바이트 2배 | 변환 0 ms; seed 합계(mha_core 타이머) 240 → 96 |
| `746888e13` | RoPE 테이블을 "같은 테이블 공유"(`graph_share_param`)로 바인딩: 30층 중 2개만 전송 | 층마다 2 MiB를 heap 인자로 보내고 DSP가 memcmp로 같은지 비교했다 — 층당 14.6 ms, 합 439 ms. 첫 토큰의 가장 큰 미설명 덩어리 | attention 노드 532 → 수 ms |
| 합 | | | **첫 토큰 1,463 → 506 ms, 2,206 → 636 ms** |

남은 첫 토큰 구성(p1024 506 ms): decode 토큰 227(expert 풀이 비어 있어 평소 82보다 +145), pool_sync 77(host FC 사본 해제), graph_init 71, seed 89, 나머지 ≈40.

host check: `ATTN M1 SEED VEC`(스칼라 정의와 바이트 비교, 전치 매핑 돌연변이가 잡힘), `ATTN M1 SEED F16`(fp16 전 클래스), graph_host_check의 share 케이스.

### 2.4 메모리 (10-12, dlwlzzero 세션)

어디에 얼마나 쓰는지부터 잤다(p2048, `[HTP] mem` 로그 + 0.5초 RSS 샘플링). DSP가 매핑한 것은 arena 2,176 MiB(expert 슬롯 32×30×1.45 = 1,392 + FC·dense WH 768) + lm_head Q4M1 448 + heap 686(prefill 끝) ≈ 3.3 GB로 매핑 한도에 걸려 있고, host는 prefill 중 RSS 2.7–3.1 GB(대부분 prefill activation 텐서), prefill 끝 mem_available **1.5 GB**였다.

| 커밋 | 무엇을 | 측정 (p2048) |
|---|---|---|
| (이 절) | QS4CX FC·dense의 CPU 사본(786 MiB)을 첫 decode 토큰이 아니라 **로드 직후** 해제 (`dropHostFc(qs4cx_only)`): prefill은 WH 핸들, decode는 arena를 읽으므로 그 뒤 아무도 읽지 않음. Q4_0 lm_head(396)는 prefill의 CPU lm_head가 읽으니 첫 토큰까지 유지 | prefill 끝 mem_available 1,523 → 2,150 MiB. 페이지 캐시가 그만큼 expert 파일을 더 담아 pgpgin 8,809 → 5,154 MiB/실행(1회) |
| | prefill의 staging 버퍼(23 MB 행이 32 MiB class, 6개 pool, 370 MiB ION)를 첫 토큰이 끝난 뒤 해제(16 MiB 초과 class) | −320 MiB (p1024 −128) |
| | 첫 토큰의 pool_sync가 396 MiB만 해제 | 첫 토큰 p1024 506 → 440 ms (1회) |

| `212afcd21` | `NNTR_MOE_CACHE_GROW=<n>`: 첫 토큰이 끝난 뒤(staging 해제 뒤) 풀을 층당 n개 키움. 슬롯은 miss가 채우며 `place()`로 매핑 | p2048 C32 +8, 번갈아 3회: 마지막 64 **12.29 → 12.79 TPS**, miss 56.9 → 42.3/토큰, 대기 27.1 → 22.8 ms, page-in 6.3 → 5.2 GB. 기본값 0(p4096은 여유가 작아 실행 중 매핑 실패 가능) |
| (다음 커밋) | seed의 fp16 행을 staging 복사 없이 rpcmem 캐시에서 그대로 DSP에; RoPE 테이블 host 사본(30층 ≈78 MB)을 바인딩 뒤 해제 | seed RPC p1024 89 → 83, p2048 173 → 157 ms |

문장 md5 동일, decode TPS 변화 없음(GROW 제외). 남은 후보: (a′) C 상향은 GROW로 가능해졌고 p4096용 안전한 상한 계산이 남음; (a) expert 슬롯은 decode용·prefill 양자화 KV 캐시(≈190–490 MiB heap)는 prefill용인데 둘이 동시에 매핑돼 DSP 피크가 합이다 — 슬롯을 prefill 뒤에 매핑하면 C를 +4–10 올릴 여지(검증 필요); (b) CPU fp16 KV 캐시는 rpcmem이라 seed가 staging에 복사하지 않고 그대로 넘길 수 있음(복사 2×8 MiB 제거); (c) RoPE 테이블 host 사본 30개 ≈ 78 MB는 바인딩 뒤 불필요; (d) embedding Q4_0 ≈ 400 MB host 상주는 파일 mmap으로; (e) prefill activation 텐서 ≈ 2 GB(host 피크의 대부분)는 all-NPU prefill에서 CPU가 읽지 않는 것이 많다 — 로더·그래프 메모리 계획 쪽.

## 3. 실패한 것과 원인

### 3.1 decode — 미리 읽기 계열 (전부 실패, 원인이 같다)

| 시도 | 결과 | 원인 |
|---|---|---|
| 다음 **층** router를 미리 돌려 예측한 expert를 페이지 캐시에 올리기 (`NNTR_HTP_PREDICT=1 NNTR_MOE_PREFETCH=4`) | 12.0 TPS, 기본과 차이 없음. flash 읽기량 +30 % | 예측은 쓸 만하다(top-8 겹침 67.7 %, 1위 적중 91 %). 그런데 읽을 시간이 한 층(≈2.5 ms)뿐이고, 예측 읽기가 실제 miss 읽기와 flash를 나눠 쓴다 |
| 같은 예측을 풀의 여분 칸에 미리 읽어 두기 | miss 대기 25.7 → 33.2 ms로 **악화** | 위와 같음. 토큰당 17개가 쓸모 있었지만 추가 읽기의 flash 경합이 더 컸다 |
| 다음 **토큰** 예측: 각 층 router의 9–16위를 미리 읽기 (구현 전에 순위 로그로 시뮬레이션) | 다음 토큰 miss의 20–23 %만 그 안에 있음. miss가 전부 덮이는 층은 14–17 %. 풀에 넣으면 miss가 47 → 48–58개/토큰으로 **증가** | 맞히는 것보다, 틀려서 쓸 expert를 밀어내는 것이 많다. 방금 쓴 expert는 이미 풀에 있으므로 miss는 본래 "최근에 안 쓴 것"이라 예측이 어렵다 |
| 읽은 expert를 CPU RAM에 보관 | 토큰당 28개 적중, 메모리 +1.1 GB, 대기 변화 없음 | 한 층의 miss 일부만 RAM에서 나와도 나머지를 flash에서 읽는 동안 똑같이 기다린다. 메모리 비용 때문에도 기각 |
| expert 파일을 RAM에 고정 (`NNTR_MOE_PIN`) | mlock 30/30 실패, 첫 토큰 +4.5 s, 대기 악화 | 폰 RAM 11.4 GB에 expert 전체 5.4 GB를 NPU 쪽 3.3 GB와 함께 둘 수 없다. mlock이 실패한 직접 원인은 확인하지 않았다 |
| miss를 읽는 스레드 4 → 8개 | 12.4 TPS, 차이 없음 | 4개로 이미 flash 속도다 |
| LRFU 반감기 32 → 8, 4 (10-12, 번갈아 3회) | miss는 54.4 → 49.3개/토큰으로 줄지만 **더 느림**: p1024 11.84 → 11.27, p2048 11.79 → 10.95 TPS | 대기는 miss 개수가 아니라 flash에서 읽은 바이트를 따라간다. 반감기가 짧으면 페이지 캐시에 없는 expert를 더 자주 놓친다(flash 읽기 4.5 → 5.2 → 6.6 GB/실행) |
| miss 한 회의 파일 범위를 한꺼번에 미리 요청 (`posix_fadvise(WILLNEED)`, 번갈아 4회) | 11.63 → 11.29 TPS, 차이 없음 | 읽기 스레드 4개가 이미 같은 일을 한다 |
| 페이지 캐시 조언 `NNTR_MOE_FADVISE=1` | decode 차이 없음, **prefill 1,840 → 3,068 ms** | 풀에서 나가는 expert를 다시 읽어 달라고 요청해 flash 읽기가 늘어난다 |

**공통 원인 (측정)**

1. miss 읽기는 이미 flash 속도다. 읽기 스레드 4개로 2.8 GB/s, 1개면 1.5 GB/s(대기 2배). 같은 파일을 `dd`로 읽으면 1.9–2.7 GB/s.
2. 한 층의 miss는 병렬로 읽는다. 그 층의 miss가 **전부** 준비돼 있어야 대기가 줄고, 일부만 맞히면 그대로다.
3. 그래서 miss 대기 = 토큰당 놓친 바이트 ÷ flash 속도다. 줄이는 길은 놓치는 양(C, 교체 정책)과 기다리는 동안의 일(dense 선행) 둘뿐이고, 성공한 두 커밋이 그것이다.
4. (10-12 추가) 놓친 expert의 절반쯤은 flash가 아니라 **페이지 캐시**에서 나온다. decode 중 폰의 남는 RAM 약 2.1 GB를 커널이 expert 파일 캐시로 쓰고 있어서, 풀(960개) 뒤에 약 1,400개짜리 2차 캐시가 이미 있는 셈이다. 폰 전체 RAM(11.1 GB)은 decode 중 여유가 0.25 GB라 이 캐시를 더 키울 방법은 없다.

### 3.2 decode — 그 밖

| 시도 | 결과 | 원인 |
|---|---|---|
| 계산부 설정 10종 (`FC_LANES` 4가지, `DMA_QUEUES` 2·4, `GEMV_LEAD_KB` 0·384, `GEMV_FEED=0`, `FC_FEED=l2`) | 기본값보다 나은 것 없음. feed를 끄면 2배 느림 | FC·dense·lm_head는 이미 메모리 대역폭 한계(토큰당 가중치 1.5 GB)에서 돈다 |
| C를 최대로 올리기 | p1024는 40, p2048은 36, p4096은 28까지만 됨 | NPU가 매핑할 수 있는 메모리가 약 3,136 MiB다. expert 한 칸 1.45 MiB × 30층 × C에 고정분 약 0.8 GB, decode FC 448 MiB, heap 610–1,060 MiB가 더해진다. 긴 프롬프트일수록 heap이 커서 C가 줄어든다 |
| NPU가 miss 답을 기다릴 때 잠들지 않고 계속 확인 (`NNTR_HTP_E2E_SPIN_US=2000`, 번갈아 4회) | 11.63 → 11.25 TPS, 차이 없음 | 답이 늦게 보이는 것이 아니었다. 답이 먼저 와 있고 NPU가 dense 선행을 아직 하고 있는 회가 섞여 지연처럼 보였다 |
| decode 중 prefill 전용 FC 가중치를 내려 expert 칸으로 쓰기 | 하지 않음 | 전제가 틀렸다. 고정분 768 MiB는 prefill 전용이 아니라 decode도 같은 핸들로 쓰는 FC·dense 가중치다(§4) |
| expert 파일을 더 촘촘히 저장(2비트 코드 중 하나가 3 %만 쓰여 1.64비트까지 가능, 읽기 −18 %) | 하지 않음 | CPU에서 풀면 토큰당 약 9 ms가 들어 이득이 사라지고, NPU에서 풀려면 커널 입력 형식을 바꿔야 한다 |

### 3.3 prefill

| 시도 | 결과 | 원인 |
|---|---|---|
| staging 버퍼를 미리 매핑 (`RPCMEM_TRY_MAP_STATIC`) | p2048 3,139 ms(652 TPS)까지 빨라지지만 **문장이 깨짐** | NPU가 쓴 내용이 CPU에서 안 보인다. CPU 쪽 캐시를 비워도(`DC CIVAC`) 그대로였다. 근본 원인은 밝히지 못했다. 캐시 일관성을 문서로 확인하기 전에는 쓰면 안 된다 |
| staging 버퍼 크기를 2의 거듭제곱 대신 1/4 옥타브로 | p1024 −66 ms, p2048 **+86 ms**, p4096 −210 ms | 방향이 길이마다 달라 확정하지 못했다. p4096에서는 두 번 다 빨랐다 |
| attention 층별 보정을 8스레드로 | prefill당 −27 ms (0.8 %) | 효과는 있으나 코드 30줄 값을 못 한다 |
| uncached staging, futex wake 생략 | 벽시계로 구분 안 됨 | 호출당 3–8 % 수준의 비용이었다 |
| FC slice를 K=2816까지 넓히기 (qkv도 1회 호출) | 뒤의 MoE 호출이 `AEE_ENOMEMORY` | 원인 미확인 |
| dense+router, attention+o-proj 호출 합치기 | 하지 않음 | RPC 60회 ≈ 20 ms뿐이라 |
| staging memcpy 4스레드, router 행 블록 4 → 6, DCVS MAX corner | 효과 없음 | 원인을 따로 분석하지 않았다 |

### 3.4 정확도 때문에 PR에서 뺀 것

| 시도 | 결과 | 상태 |
|---|---|---|
| qkv 패스에서 Q를 고정 scale의 u16로 (브랜치 `htp/s26-static-q-wip`) | attention의 CPU 몫 294 → 217 ms | 문장이 바뀌고 어색한 구간이 있다. 판정할 정확도 게이트가 없다: decode NLL로는 수치가 같은 두 경로끼리도 +0.04–0.07 차이가 나서 가르지 못한다 |
| `NNTR_HTP_KV_SEED_Q=1` | seed 1,055 → 773 ms | 문장이 깨진다(p2048 빈 출력, p1024 반복). 기존 문제 |
| `NNTR_MOE_SPLIT` | — | 기준 빌드에서도 깨진 문장을 낸다. 검증용으로 못 쓴다 |

## 4. 작업하면서 틀렸던 것

- **`NNTR_HTP_PROFILE=2`의 분해를 믿고 MoE를 분석했는데 그 경로는 실제 경로가 아니었다.** 계측용 엔트리 + CPU norm으로 돌고, "타이머 밖" 시간의 대부분은 계측 자체의 비용이었다. 워커 종류별로 다시 재서 진짜 병목(scatter, pack)을 찾았다.
- **FastRPC 비용을 "호출 횟수 × 고정비"로 봤는데 실제로는 버퍼 크기에 비례했다.** 그래서 호출 합치기는 효과가 작다고 봤다가, chunk 호출을 1회로 줄이자 크게 줄었다.
- **decode miss를 "읽기가 느려서"로 보고 미리 읽기를 세 가지 만들었는데, 읽기는 이미 flash 속도였다.** `dd`와 읽기 스레드 수 실험을 먼저 했으면 만들지 않았을 것이다.
- **C=40이 모든 길이에서 된다고 안내했는데 p2048에서는 실패한다.** 길이별로 다시 재서 고쳤다.
- **NPU arena의 고정분 약 780 MiB를 "prefill 전용 가중치"라고 추정하고 decode 때 내리는 안을 1순위로 냈는데, 재 보니 decode도 쓰는 가중치였다.** chunk별 사용량을 출력해 확인했다(expert 칸 1,408 MiB + FC·dense 768 MiB, 빈 곳 없음). 만들기 전에 재서 하루치 작업을 아꼈다.
- **LRFU 반감기 8이 한 번 잰 값으로는 +6 %였는데 번갈아 3회에서는 −5 %였다.** 한 번 잰 decode 값은 믿지 않는다.

## 5. 남은 것

| 항목 | 크기 | 걸림돌 |
|---|---|---|
| decode attention 커널 | 토큰당 11.8 ms | CPU의 fp16 합산 순서를 따르는 구조라 느리다. 바꾸면 반올림이 달라져 정확도 게이트가 먼저 필요 |
| decode 작은 op 융합 (norm·add·RoPE) | 3.8 ms | 비트 동일 가능. NPU 그래프 코드 변경 |
| 첫 토큰 시간 | 남은 것 ≈ 290 ms (§2.3) | 첫 토큰의 expert 전부 miss(+145 ms: prefill이 쓴 expert로 풀을 미리 채우기), pool_sync 77, graph_init 71 |
| lm_head·FC 비트 수, router fp16 | 대역폭 한계 | 정확도 결정 필요 |
| prefill attention softmax | 호출당 5.5 ms (p2048 168 ms) | 미착수 |
| prefill o-proj 뒤 패스, router | 층당 2.5 ms, 호출당 10.7 ms | 미착수 |

decode의 하한 추정: 가중치 1.5 GB/토큰을 메모리 대역폭 53 GB/s로 읽으면 28 ms(35 TPS). miss·attention·dispatch를 다 잡았을 때 현실적인 상한은 약 50 ms(20 TPS)로 본다 — 추정이며 기기 미측정.

## 6. 측정 규칙 (이 문서의 숫자가 따르는 것)

- 비교할 빌드를 기기에 나란히 두고 번갈아 3회, 실행 사이 2분 냉각.
- 정확도는 같은 env에서 답 줄 md5 비교: `grep -a -o '<channel|>.*' log | head -1 | md5sum | cut -c1-8`. 지금 기준은 p1024 `fe44d523`, p2048 `aff26409`, p4096 `1b462136`.
- 다른 길이를 돌린 직후의 첫 p4096 실행은 16–19 s가 나온다(순서 효과). 버린다.
- 프로파일 env를 켠 실행의 md5는 의미가 없다(stderr가 섞인다).
