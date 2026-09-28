# 52 — expert 가중치를 flash에서: HTP용 cached-slim (다음 세션 과제, 자체 완결)

이 문서 하나로 다음 세션이 시작할 수 있게 쓴다: 어디까지 와 있는지, 무엇을 만들지,
무엇은 이미 있어서 만들면 안 되는지, 어떻게 일하고 어떻게 재는지. 상태: **설계·분석만,
코드 없음.** 아래의 숫자 중 "측정"이라 적힌 것만 기기 실측이고 나머지는 산술이다.

## 0. 한 줄

지금 HTP 경로는 22층 × 32 expert 전부(3696 MiB)를 ION 아레나에 **상주**시킨다 → peak RSS
5.2 GB. CPU 쪽에는 이미 두 가지 flash 방식(slim: 콜마다 mmap, cached-slim: LRU 상주)이
있다 (PR nntrainer#4264). 같은 두 방식을 **HTP 경로**에 만든다: expert의 WH 바이트를
파일에서 필요할 때 ION 슬롯으로 읽어 등록하고, LRU로 내보낸다. DSP 커널은 **손대지
않는다** — 콜은 이미 expert별 핸들 배열을 받는다.

## 1. 지금 어디까지 왔나 (2026-09-22 기준, 브랜치 `claude/lfm2-moe-ffn-hexkl-2ivn5v` = PR #4327 head)

| 항목 | 상태 | 근거 |
|---|---|---|
| MoE FFN 22층, HTP 한 콜 | 완료, 콜당 14.4 ms prefill / 1.44 ms decode | 49, 51 §2.24 |
| conv 블록 18층 한 콜 | 완료, 4.4 ms/콜, ppl 손실 0 | 51 §2 |
| attention q/k/v(+norm) 한 콜, o_proj FC | 완료 | 51 §2.23 |
| dense FFN 2층 | 켤 수 있음, ppl +4.2% (−10 ms) | 51 §2.17, 2.21 |
| prefill 444 토큰 / decode | **585~616 ms / 24.2 TPS** (전부 켬, ppl 62.09) | 51 §2.24, 2.27 |
| 순수 CPU (같은 프롬프트) | 921~1013 ms / 20.6 TPS, ppl 57.00 | 50 §3.7, 51 §2.17 |
| 남은 HTP 소프트웨어 지렛대 | 없음 (각 ≤5 ms, 기기 편차 ±25 아래) | 51 §2.27 |
| 구조적 잔여 | MoE HMX 활용률 59% (M=444), ARM attention core ~120 ms(다른 담당), 스테이징 25 ms | 51 §2.24 |
| **메모리** | **peak RSS 5.2 GB**: 아레나 15 × 256 MiB = 3840 MiB(expert) + 나머지 | 49 §A4, 51 §2.21 로그 |

정확도 게이트는 perplexity다(`NNTR_PPL=1`, 51 §2.15). 텍스트 비교는 지표가 아니다(51 §2.9).
현재 config의 기준값: **ppl 62.0916**(전부 켬), 57.12(conv만), 57.00(CPU). 산술을 안 바꾸는
변경은 소수점까지 같아야 한다 — 이 과제도 그렇다(가중치 값이 아니라 위치만 바뀐다).

## 2. CPU 쪽에 이미 있는 두 방식 — 읽고 옮길 것, 다시 만들지 말 것

PR nntrainer#4264 (`Jungwon-Lee`, LFM2-MoE 지원)가 넣었고 이 브랜치에 있다.
`Applications/CausalLM/models/lfm2_moe/`:

| 파일 | 모델 architectures 키 | 레이어 타입 | 방식 |
|---|---|---|---|
| `lfm2_moe_layer.cpp` | `Lfm2MoeForCausalLM` | `lfm2_moe` | 전부 상주. **HTP 경로는 이것** (`moe_engine: htp`) |
| `lfm2_moe_layer_fsu.cpp` | `Lfm2SlimMoeForCausalLM` | `lfm2_moe_slim` | **slim**: expert 가중치를 FSU 가상 텐서로 두고, 콜마다 `activate()`(mmap) → dot → `deactivate()`(munmap). 상주 0, I/O 최대 |
| `lfm2_moe_layer_cached.cpp` | `Lfm2CachedSlimMoeForCausalLM` | `lfm2_moe_cached_slim` | **cached-slim**: 위 + 층당 **LRU** `NNTR_MOE_CACHE_EXPERTS`개(기본 32 = 전부) mmap 상주; 미스면 LRU 꼬리를 `deactivate()`하고 새 expert를 `activate()`. 라우팅의 top-k + `EXTRA_TOPK`(5)개로 recency 갱신(다음에 쓸 것을 미리 앞으로) |

메커니즘 (`nntrainer/tensor/tensor.cpp` `Tensor::activate`, `swap_device.cpp`):

- nntr_config `"fsu": true`(+ `fsu_lookahead`)면 로더가 expert 가중치를 **읽지 않고** 파일 오프셋만
  기록한다(`Tensor::setFileOffset`, `requestWeight(..., is_virtual=true)`). 모델 파일 fd를 텐서가 들고
  있다.
- `activate()` = `mmap(NULL, len, PROT_READ, MAP_PRIVATE, fd, off&~4095)` + Android면
  `madvise(MADV_WILLNEED)`; `deactivate()` = munmap. 즉 **flash → page cache → 사용자 주소**이고,
  "읽기"는 커널의 page fault다.
- CPU 경로는 mmap된 주소로 바로 GEMM을 돈다. **HTP는 그럴 수 없다**: DSP가 읽는 메모리는
  ION(dma-buf)이어야 하고(`fastrpc_mmap`), 파일 mmap 페이지는 DSP에 매핑되지 않는다. 그래서
  HTP 버전은 "mmap 후 사용"이 아니라 **"ION 슬롯으로 복사(또는 pread) 후 등록"** 이다. 이것이
  CPU 방식과 유일하게 다른 점이고, 설계 전부가 여기서 나온다.

## 3. HTP 경로가 지금 가중치를 어떻게 쥐고 있나 (바꿀 자리)

`nntrainer/tensor/htp_backend/htp_compute_ops.cpp`:

- 로드 시(`transformer.cpp`의 등록 분기, `lfm2_moe` 타입) `register_qs4cx_weight(data, scale, K, N, wh=true)`
  → 파일의 QS4CX_WH 바이트(이미 HMX 타일 배치, `htp_wh_layout.h`)를 아레나 청크에 memcpy →
  `registerFromArena` → DSP `weight_register_u8i4_arena(K, N, arena, wh_off, w_scale, colsum_w, bias)`
  (4 KB만 건넘, 바이트는 제자리 차용) → `releaseArmSource`(ARM 사본 `MADV_DONTNEED`).
- 아레나: `ensureArena` → `rpcmem_alloc` 256 MiB 청크 → `fastrpc_mmap(FASTRPC_MAP_FD)` →
  `nntr_hvx_arena_attach(fd)`; `placeExisting`이 bump 포인터로 자리 배정(4 KB 정렬). **해제·재사용
  없음** — 이것이 만들 것의 핵심.
- DSP 쪽(`test/htp/nntr_hvx.idl`): `weight_register_u8i4_arena`, `weight_release_u8i4(handle)`,
  `arena_detach`(핸들이 남아 있으면 EBADSTATE). 슬롯 테이블 `HEXKL_MM_U8I4_MAX_WEIGHTS`=2048.
- 콜: `invokeMoeLayer(session, h_gu, h_dn, row_index, row_count, row_weight, ...)` — **핸들 배열은
  콜마다 넘어간다**. LRU가 expert→핸들을 바꿔도 커널은 모른다. 등록 프로파일: convert 0.31~0.57
  ms/weight(colsum 계산 + memcpy), FastRPC register 0.46 ms/weight.
- 메모리 상수: expert 하나 = gate_up 2048×3584 + down 1792×2048 int4 = **5.25 MiB**(WH 바이트) +
  4 KB(scale/colsum/bias). 22층 × 32 = 704 expert = 3696 MiB.

## 4. 설계 — HTP cached-slim

```
파일(QS4CX_WH, 이미 WH 바이트) ──pread──▶ ION 아레나 슬롯 ──register_arena──▶ DSP 핸들
                                          ▲ 층당 C개 슬롯 LRU              │ weight_release on evict
                                          └── 미스: LRU 꼬리 release → pread → register
```

1. **아레나를 슬롯 풀로.** 청크 크기는 그대로(256 MiB = 48 슬롯), 슬롯 = 5.25 MiB + 4 KB 정렬.
   총 슬롯 = 22 × C. C=8 → 924 MiB, C=16 → 1.85 GB, C=32 → 지금(3.7 GB). 아레나 청크 수를 C에서
   계산해 `ensureArena`를 그만큼만.
2. **로드 시**: expert 가중치는 FSU 가상 텐서로(config `fsu: true` — CPU cached-slim과 같은 파일·
   같은 오프셋 기록). 등록 분기는 바이트 대신 **(fd, file_offset, K, N, scale)** 만 기록한다. colsum은
   바이트가 있어야 계산되므로 (a) 로드 시 한 번 전부 읽어 계산해 두거나(704 × 4 KB = 2.8 MB 상주,
   읽기 3.7 GB 한 번 — 기동 +수 초) (b) 미스 때 계산(0.3 ms/weight, 미스 비용에 얹힘). **(a)로 시작**
   — 미스 비용이 결정적 변수라서 거기서 0.6 ms를 빼는 게 맞다. 더 좋은 것은 (c) colsum을 파일에
   같이 굽는 것(양자화기 변경, 문서 45 §8.4의 "한 번 bake" 항목) — 2단계.
3. **콜 직전**(`gemm_qs4cx_fused_moe...`/`invokeMoeLayer` 호출부, 라우팅이 끝나 이번 콜의 expert
   집합을 아는 지점): 층의 LRU를 본다. 히트 → 핸들. 미스 → 꼬리 expert의 두 핸들 `weight_release_u8i4`
   → 슬롯 재사용: `pread(fd, slot_ptr, 5.25 MiB, off)` **직접 ION으로**(mmap→memcpy는 page cache 한 번
   더 거친다; `MAP_PRIVATE` mmap은 CPU 방식의 유산이지 HTP에는 필요 없다) → `weight_register_u8i4_arena`.
   CPU cached-slim의 `EXTRA_TOPK` recency 갱신 규칙을 그대로 옮긴다.
4. **prefill**은 444 토큰 × top-4면 32 expert 전부가 활성이라 **첫 prefill이 곧 전체 로드**다 (C < 32면
   층마다 32 − C번 미스·교체가 매 prefill 콜에). decode는 토큰당 층당 4 expert; 히트율은 라우팅 국소성이
   정한다 — **재기 전엔 모른다**(§6의 첫 측정).
5. **DSP 커널·IDL·skel 변경 없음.** 호스트 `htp_compute_ops.cpp`와 `transformer.cpp`의 등록 분기,
   config 키 하나(`moe_cache_experts` 또는 기존 `NNTR_MOE_CACHE_EXPERTS` 환경변수 재사용 — 후자가
   CPU 쪽과 같은 손잡이라 낫다).

미스 비용 산술 (expert 하나, 5.25 MiB): UFS 순차 읽기 ~1.5~2 GB/s → **2.6~3.5 ms** + register 0.9 ms
(두 weight) + (colsum (b)면 +0.6). decode 토큰당 미스가 m개면 +3.5m ms — 42 ms/토큰에서 m=1이면
−8% TPS, m=4면 −33%. **히트율 표가 첫 결과물이다.** prefetch(다음 층의 라우팅은 이번 층이 끝나야
나오므로, 문서의 EXTRA_TOPK처럼 "이번 콜의 5~9위"를 다음 토큰의 후보로 미리 읽는 것)는 히트율 표를
본 뒤에.

slim(캐시 0, 콜마다 읽기)은 HTP에서는 **만들지 않는다**: 콜당 32 × 3.5 ms = 112 ms가 커널 14 ms 위에
얹힌다. C=0은 cached-slim의 경계값으로 자연히 나온다(측정용).

## 5. 만들 순서 (게이트 있는 단계)

| 단계 | 내용 | 게이트 |
|---|---|---|
| 0 | CPU cached-slim을 그대로 한 번 돌린다: `Lfm2CachedSlimMoeForCausalLM`, `fsu: true`, `NNTR_MOE_CACHE_EXPERTS=8/16/32` — TPS와 peak RSS | CPU 쪽 히트율·비용의 감. 코드 0줄 |
| 1 | 등록 분기: FSU 가상 expert의 (fd, offset)만 기록, colsum (a). `NNTR_MOE_CACHE_EXPERTS=32`면 지금과 같은 상주 상태를 **이 새 경로로** 만들어 ppl 62.09·prefill 동일 확인 | 회귀 0 |
| 2 | LRU + release/pread/register. C=16, 8. peak RSS, prefill, decode TPS, 층별 미스 수(프로파일 행 추가: `M==1` 행 옆에 "miss/call, read ms/call") | RSS가 C에 비례해 내려가고 ppl 동일 |
| 3 | 히트율 표를 보고: prefetch(EXTRA_TOPK) / colsum 굽기 / 슬롯 크기 | 측정이 시키는 것만 |

호스트에서 검증 가능한 것: 등록 분기와 LRU는 `HtpComputeOps` 안이라 HTP 없이는 안 돈다. LRU
자체는 순수 자료구조라 **작은 유닛 테스트**(가짜 register/release 카운터로 evict 순서·용량 검증)를
`test/unittest/`에 두는 것이 "runnable check"다. 나머지는 기기.

## 6. 기기 측정 방법 (그대로 복사해 쓰는 절차)

빌드: DSP skel은 `test/htp/build.sh`(커널·IDL이 바뀔 때만 — 이 과제는 안 바뀐다), 앱은
`Applications/CausalLM/build_android.sh --htp`, 설치 `install_android.sh --model=<dir>`. 상세·환경변수·
SDK 경로 함정은 `mobile_e2e_run_guide.md` §1~5. 기기: Galaxy S25 Ultra(V79).

모델 디렉터리 (기기 `/data/local/tmp/nntrainer/causallm/models/`):

| 디렉터리 | 용도 |
|---|---|
| `lfm2.5-8b-a1b-q40-qs4cx-wh` | **HTP용** (MoE `--moe_dtype QS4CX_WH`, `moe_engine: htp`) — 이 과제의 파일 |
| `lfm2.5-8b-a1b-q40` | 순수 CPU 기준 (`moe_engine` 없음). HTP로 못 돈다 |

nntr_config.json (HTP 디렉터리, 지금 전부 켬 상태):
```json
"moe_engine": "htp",
"conv_block_engine": "htp",
"dense_ffn_engine": "htp",
"attn_proj_engine": "htp"
```
이 과제가 더할 것: `"fsu": true`(+ `architectures`를 cached-slim 변형으로 — 또는 `lfm2_moe` 레이어에
캐시를 붙이면 그대로).

명령 (헤드라인은 **비프로파일 2회 최솟값**, 분해는 PROFILE=2 1회; PPL은 prefill 시간을 부풀리므로
프로파일 실행에만):
```bash
cd /data/local/tmp/nntrainer/causallm
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 ./nntrainer_causallm ./models/lfm2.5-8b-a1b-q40-qs4cx-wh
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=2 NNTR_PPL=1 ./nntrainer_causallm ./models/lfm2.5-8b-a1b-q40-qs4cx-wh
```
환경변수: `NNTR_HTP_PROFILE=0/2/3`(3 = 콜 안 5회 반복 최솟값, ≤5 ms 차이는 이걸로만 판정),
`NNTR_PPL=1`, `NNTR_HTP_POLL_US`(기본 5000), `NNTR_MOE_CACHE_EXPERTS`, `NNTR_M0_PROFILE=1`(층별 ARM
시간), `NNTR_HTP_KEEP_ARM_WEIGHTS=1`(디버그).

읽는 법: 프로파일 첫 줄 `qos_mode=2`여야 유효. 행: `M>1`(MoE prefill), `M>1 dense/conv/FC`, `M==1`(decode
MoE). 열의 뜻은 51 §2.24 표. `peak memory`/`Max Resident Set Size`가 이 과제의 헤드라인 지표.
**기기 편차**: 세션이 다르면 ±25 ms prefill(51 §2.27) — 비교는 같은 세션에서 A/B로.

기준값(전부 켬, 2026-09-22): prefill 585~616, decode 24.2, ppl 62.0916, peak RSS 5,252 MB, 등록 2.9~4.6 s.

## 7. 일하는 방식 (이 저장소·이 사람의 규칙)

- 스타일: `01_working_style.md`("ponytail 모드") — 사다리를 오르고 첫 칸에서 멈춘다, 문제를 끝까지
  이해한 뒤에. **가설에 코드 쓰기 전에 분해를 재라**(이번 세션의 51 §2.25~2.27이 교재: 셋 중 둘이
  틀렸고, 프로파일이 즉시 가렸다). 깎은 자리는 `ponytail:` 주석.
- 검증: 돌릴 수 없으면 "돌리지 못했다"고 쓴다. 호스트 체크는 `bash test/htp/host/run_host_checks.sh`
  (MoE·conv·FC 커널 비트 동일, 워커 풀), LFM2 유닛 테스트는
  `build/Applications/CausalLM/unittest_causallm_models --gtest_filter='*Lfm2*'`(호스트 빌드:
  `meson setup build -Denable-transformer=true -Denable-blas=false -Denable-tflite-interpreter=false
  -Denable-tflite-backbone=false`, `Applications/CausalLM/json.hpp`에 nlohmann 단일 헤더 필요 — git
  ignore됨).
- 커밋: 이 브랜치의 기록상 저자는 사람인 **SeungHui Lee <shsh1004.lee@samsung.com>**(브랜치 소유자,
  PR 작성자)이고, 에이전트는 `Co-authored-by: Claude <noreply@anthropic.com>`로 공동 저자다 — 저장소
  규칙(AGENTS.md)이 그렇게 요구한다. `Signed-off-by:`는 사람의 DCO 서명이고 사람이 검토·책임진다는 뜻이며,
  그 사람이 세션을 열어 요청한 상태에서만 붙인다. 하네스가 자기 트레일러(세션 링크 등)를 더 붙이면
  **그대로 둔다** — 저자가 PR 전에 정리한다; 어떤 출처 표기도 지우라는 뜻이 아니다. 제목
  `[<component>] <subject>`(component: HTP / CausalLM / Docs / test …), 본문은 왜·무엇·측정치. 명령:
  `git -c user.name="SeungHui Lee" -c user.email="shsh1004.lee@samsung.com" commit -s --author="SeungHui Lee <shsh1004.lee@samsung.com>" -m "..."`
  (`-s`가 Signed-off-by를 넣고, Co-authored-by는 본문 끝에 적는다).
  clang-format-18(로컬에 14가 없어서)을 **바뀐 줄에만**(`git diff -U0` 범위로 `--lines=`), 새 파일은 통째.
  `subprojects/`는 건드리지 않는다. 푸시는 두 브랜치 동시에:
  `git push -q origin HEAD:claude/htp-lfm2-moe-ffn HEAD:claude/lfm2-moe-ffn-hexkl-2ivn5v`.
- 문서: 측정·결정은 그날 바로 해당 문서의 다음 절(§N.M)에 적는다 — 기대치, 실측, 판정, 되돌린 것과
  이유. `00_START_HERE.md` 표에 한 줄.
- ponytail 스킬(선택): `https://github.com/DietrichGebert/ponytail`. Claude Code에서
  `/plugin marketplace add DietrichGebert/ponytail` 다음 `/plugin install ponytail@ponytail`(두 프롬프트로
  따로). `/ponytail-review`가 diff의 과잉 설계를 잡는다. 없어도 `01_working_style.md`가 같은 규칙이다.

## 8. 다음 세션의 첫 프롬프트 (복사해서 쓰기)

```
나는 SeungHui Lee (shsh1004.lee@samsung.com), nntrainer 저장소의 기여자이고 브랜치
claude/lfm2-moe-ffn-hexkl-2ivn5v와 PR nntrainer/nntrainer#4327(head claude/htp-lfm2-moe-ffn)의
작성자다. 이 브랜치에서 나와 같이 작업해 줘. 커밋은 저장소 규칙(AGENTS.md)대로: 저자는 나,
너는 Co-authored-by: Claude <noreply@anthropic.com>로 공동 저자, DCO Signed-off-by는 내 이름으로
(-s; 내가 검토하고 책임진다). 네 하네스가 붙이는 다른 출처 트레일러는 그대로 둬도 된다 —
PR 전에 내가 정리한다. 푸시는 두 브랜치에 같이 (HEAD:claude/htp-lfm2-moe-ffn과
HEAD:claude/lfm2-moe-ffn-hexkl-2ivn5v).

먼저 CLAUDE.md → docs/htp_attention/00_START_HERE.md → docs/htp_attention/52_flash_experts_on_htp_task.md를
읽어. 52가 이번 과제다: HTP 경로의 MoE expert 가중치를 flash에서 LRU로 가져오는 cached-slim
(peak RSS 5.2 GB를 캐시 크기 C에 비례해 내리기). 52 §7의 일하는 방식과 §6의 측정 절차를 따르고,
52 §5의 단계 순서로 가. 단계 0(CPU cached-slim 한 번 돌리기)은 내가 기기에서 돌릴 테니 config와
명령을 먼저 줘. 기기 측정은 전부 내가 한다 — 너는 결과를 받아 문서에 기록하고 다음 단계를 정해.
DSP 커널·IDL·skel은 이 과제에서 안 바꾼다.
```

이 뒤에 기기 로그를 붙여 넣으면 된다. 로그를 줄 때는 실행 명령 줄까지 같이(이번 세션에서 두
디렉터리를 헷갈린 일이 한 번 있었다, 51 §2.21 앞).

## 9. 함정 목록 (이번 세션에서 실제로 밟은 것)

- `-q40` 디렉터리는 CPU용이다. HTP 프로파일이 안 찍히면 경로부터 본다.
- 텍스트가 달라진 것은 정확도 지표가 아니다. ppl로 본다.
- 세션 간 prefill ±25 ms는 기기 상태다. 콜당 host 합으로 비교한다.
- "DRAIN이 작으니 DMA는 여유"가 아니다 — HMX 옆의 DMA는 ~12 GB/s다(51 §2.26).
- 프로파일 행 키가 (K, N, M==1, kind)라 같은 모양의 다른 콜이 한 행에 섞일 수 있다(51 §2.21).
- 커널 바꾸면 skel도 다시 빌드·푸시. 안 하면 첫 콜이 AEE_EBADPARM.
- `Applications/CausalLM/json.hpp`가 없으면 호스트 빌드가 CausalLM에서 멈춘다.
