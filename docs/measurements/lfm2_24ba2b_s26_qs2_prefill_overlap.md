# S26 Ultra LFM2-24B-A2B: QS2 prefill 확장과 HMX 중첩

2026-10-09. QS2CX_WH expert의 **2→4bit 확장을 VTCM에서 유지하면서**,
prefill에서 background HVX 확장을 HMX 연산과 겹치도록 변경했다.
동일 소스 기준 3회 교차 측정에서 prefill 평균은
**2478.0 → 2343.0 ms (5.4% 감소)**였다.
ARM 확장 방식 대비로도 3.4% 짧아졌다.

## 구현과 적용 범위

PR [4412](https://github.com/nntrainer/nntrainer/pull/4412)의
`4c3953bb161d0f7d799a72ddf9bdf9462e3ed482`에서 background chunk 확장
스케줄링을 이식했다. 다음 gate/up chunk를 앞선 HMX batch에서 준비하고,
이전 block의 down 가중치는 gate/up 연산 중, 다음 expert의 첫 gate/up
chunk들은 down 연산 중 확장한다. 각 chunk는 DMA 완료 후 확장을 제출하고,
HMX가 읽기 전에 해당 background job만 기다린다. Gate/up는 한 job으로 묶는다.

Chunk별 job/context/LUT를 호출이 끝날 때까지 유지하며 오류 경로도 남은
job을 회수한다. 현재 LFM2 형상은 gate/up 3개, down 2개 chunk이다.
활성 expert가 모두 QS2일 때 적용하며, QS2/QS4 혼합 또는 어느 한쪽이
8개 chunk를 넘는 형상은 기존 동기 확장을 사용한다. Decode의 HVX GEMV와
VTCM double buffer 경로는 그대로다. 새 ARM API나 환경 변수는 추가하지 않았다.

가중치 본체는 DRAM과 DMA에서 계속 2bit다. **FC는 QS4CX_WH, expert는
QS2CX_WH**이며 embedding/tied head의 기존 Q4_0 fixture는 유지했다.

## 측정 조건과 결과

S26 Ultra SM-S948N (`R5KL30G6MCX`), Hexagon v81, SDK/HexKL 6.4.0.1.
입력 1024 / 생성 64 tokens, cache 8, early prefetch 1, readers 6,
ARM threads 8, `NNTR_HTP_LMHEAD_CPU=0`. 동일 앱·ARM 라이브러리·모델을 사용했다.
ARM 확장 → VTCM 동기 확장 → VTCM 비동기 확장을 3회 반복했다.
강제 냉각·캐시 비우기는 없었고 온도는 각 실행 전 배터리 온도이다.

| 방식 | 평균 prefill ms | 합산 decode tok/s | Expert arena MiB |
| --- | ---: | ---: | ---: |
| ARM 확장 | 2426.7 | 9.183 | 1536 |
| VTCM 동기 확장 | 2478.0 | 9.737 | 832 |
| VTCM 비동기 확장 | 2343.0 | 9.676 | 832 |

Decode 수치는 `총 생성 tokens × 1000 / 총 decode ms`다. 이번 변경은
prefill 최적화이며 decode의 작은 차이를 개선으로 단정하지 않는다.
Arena는 DSP에 매핑된 전체 arena 크기로 RSS가 아니다.
후보 prefill은 2256~2509 ms로 변동이 있었으며, 느린 세 번째 실행도
평균에 포함했다. 3회 측정의 관측 결과로 모든 실행에서 같은 개선을 보장하지 않는다.

| 실행 | 배터리 °C | Prefill ms | Decode ms | Decode tok/s |
| --- | ---: | ---: | ---: | ---: |
| arm 1 | 27.1 | 2439 | 6979 | 9.17037 |
| sync 1 | 27.1 | 2427 | 6640 | 9.63855 |
| overlap 1 | 27.7 | 2256 | 6639 | 9.64001 |
| arm 2 | 28.6 | 2422 | 7009 | 9.13112 |
| sync 2 | 28.9 | 2514 | 6536 | 9.79192 |
| overlap 2 | 29.4 | 2264 | 6582 | 9.72349 |
| arm 3 | 29.6 | 2419 | 6920 | 9.24855 |
| sync 3 | 29.8 | 2493 | 6542 | 9.78294 |
| overlap 3 | 30.0 | 2509 | 6621 | 9.66621 |

이전 측정 문서의 바이너리는 현재 소스의 attention/DSP queue 변경을
포함하지 않아 초기 비교에서 생성 문자열도 달라졌다. 해당 비교는 제외했다.
위 기준 바이너리는 **현재 전체 소스에 이번 작업 직전의 MoE 파일만 사용해
재빌드**했다. 다른 소스·SDK·옵션·ARM 런타임은 후보와 같다.

## 프로파일과 정확성

타이밍 반복과 별도로 `NNTR_HTP_PROFILE=2 NNTR_OP_TIME=1`로 측정했다.

- Prefill MoE DSP: 32346.9 → 26527.7 µs/call, 약 18.0% 감소.
- Decode MoE DSP: 317.5 → 317.1 µs/call.

상세 profile 행은 JSON에 보관했다. `drain`은 DMA와 확장 준비를 기다린
노출 시간을 포함한다. 새 DMA trace의 wait 구간에서는 HVX 확장을 제외했으므로
이전 trace의 wait 수치와 직접 비교해 DMA 엔진 성능을 해석하면 안 된다.

실제 모델 MoE 2,471회 호출의 입력·출력 및 routing manifest,
**4,943개 파일 전체가 동기/비동기 방식 사이에서 바이트 단위로 일치**했다.
모든 반복의 생성 문자열 hash도 같았다. 기존 반복적인 `cred` 출력은
여전히 남아 있으며, 이 검사는 모델 품질이나 PPL 검증을 대신하지 않는다.

- S26 실제 장치: compact/expanded MoE, M1 GEMV, FC, conv prefill/continuation,
  fused SwiGLU 5개 테스트 통과. MoE에 M=193 및 빈 expert·1/65/129/193행의
  불균등 routing을 추가했다.
- 전체 host checks 통과. Native HVX/ISS 및 NEON 검사는 도구 부재로 생략.
- Host compact 검사는 여러 chunk, mixed precision, DMA byte 수 및
  background 작업을 실제 wait까지 지연한 경우의 context/buffer 수명을 확인한다.
- 집중 sanitizer 검사에서 ASan finding은 없었으나, 큰 형상의 기존 scalar
  SwiGLU reference에서 `swiglu_det.h:152`의 음수 left shift를 UBSan이 보고했다.
  따라서 sanitizer 전체 무경고 통과로 기록하지 않는다.
- 변경 C/C++에 clang-format 14 적용, `git diff --check` 통과.

## 재현과 아티팩트

후보 실행 경로는 `/data/local/tmp/nntr_qs2_prefill/bin`, 기준은
`/data/local/tmp/nntr_qs2_prefill_sync/bin`이다.

```sh
cd /data/local/tmp/nntr_qs2_prefill/bin
NNTR_HTP_LMHEAD_CPU=0 NNTR_HTP_QS2_VTCM=1 \
  sh run_lfm2_strict.sh /data/local/tmp/nntr_qs2_pr4415_20261009/model
```

- `/tmp/nntr-qs2-prefill/sync-current-skel.so`: SHA256 `566264a4de23264645e72dee55bb1cc88c4b7c3c15613984a920908b55050c07`
- `test/htp/build/libnntr_hvx_skel.so`: SHA256 `d727f2a879d4e1c30175c66389d413e67f8e3e992a1504efd2c1e51d7f60d804`

원시 로그, 비교 스크립트, 호출별 덤프 및 변경 전 파일은
`/tmp/nntr-qs2-prefill/`에 있다. 초기의 비교 조건이 섞인 로그는
`initial-mixed-baseline/`에 분리했다. 영구 요약은 이 문서와
[JSON](lfm2_24ba2b_s26_qs2_prefill_overlap.json)에 보관한다.
