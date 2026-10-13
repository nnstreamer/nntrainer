# S26 Ultra LFM2-24B-A2B: QS2 expert의 VTCM 확장

> 후속 개선: [prefill background 확장 측정](lfm2_24ba2b_s26_qs2_prefill_overlap.md).
> 아래는 동기 VTCM 확장을 도입한 시점의 결과다.

2026-10-09, nntrainer 작업 트리. PR [4412](https://github.com/nntrainer/nntrainer/pull/4412)
`4c3953bb161d0f7d799a72ddf9bdf9462e3ed482`의 palette 확장 커널을 현재
LFM2 expert 캐시에 적용했다. 같은 최종 바이너리에서 ARM 확장과 VTCM
확장을 3회씩 교차 측정한 전체 모델 decode는 **8.86 → 9.41 tok/s,
6.2% 개선**이다.

## 구현

```text
QS2CX_WH 파일 → ARM 파일 읽기 → DRAM의 2bit expert LRU
  → DMA: VTCM 각 행의 상단 절반에 2bit 전송
  → HVX: 같은 VTCM 행 안에서 palette를 조회해 4bit로 확장
  → decode HVX GEMV / prefill HMX matmul
```

ARM에서의 2→4bit 확장은 기본 경로에서 제거했다. Expert의 gate/up와 down
가중치 본체가 DRAM과 DMA에서 2bit를 유지한다. Scales와 column sums는
기존과 같고 palette 네 바이트를 arena tail에 저장한다. 캐시 hit에서도
VTCM에 새로 옮긴 압축 가중치를 DSP가 확장한다. 재양자화는 하지 않는다.

`hexkl_weight_u8i4` registry가 bits/palette를 보관하고 캐시 교체 시 함께
갱신한다. 새 single/batch FastRPC 메서드는 IDL 끝에 추가해 기존 메서드
번호를 유지했다. DSP는 범위·alignment·palette와 retired handle을 확인한
뒤 교체하며, palette cache line도 교체 때 invalidate한다.

각 VTCM 행을 한 worker만 확장하므로 입력과 출력이 겹쳐도 읽기 전 데이터를
덮지 않는다. 기존 expert double buffer를 유지해 다음 expert의 DMA와 현재
expert의 확장·연산을 겹친다. VTCM 할당 크기는 기존 4bit 경로와 같다.
Prefill은 도착한 chunk만 펼치고 이후 같은 expert의 행 블록에서 재사용한다.

2bit를 DRAM의 int4 GEMV로 잘못 읽지 않도록 VTCM feed가 꺼지거나 slab이
맞지 않는 작은 decode는 HMX 경로로 처리한다. 압축 expert가 있는 호출의
선택적 HVX prefill tail도 HMX로 처리한다. FC/conv API에는 압축 handle을
허용하지 않으며 **FC는 QS4CX_WH, expert는 QS2CX_WH**로 분리한다.
PR의 native 2bit GEMV나 token-resident graph 전체를 이식한 것은 아니다.

## 측정

SM-S948N (`R5KL30G6MCX`), Hexagon v81, HexKL/SDK 6.4.0.1.
입력 1024 tokens, 생성 64 tokens, expert cache 8, early prefetch 1,
readers 6, ARM threads 8. LM head도 HTP (`NNTR_HTP_LMHEAD_CPU=0`).
Embedding/tied head의 기존 Q4_0 fixture는 그대로 사용했다.

ARM/DSP/앱을 동일 소스 기준으로 빌드한 뒤 환경 변수
`NNTR_HTP_QS2_VTCM=0/1`만 바꿨다. ARM→VTCM 순서를 세 번 반복했고
강제 냉각이나 캐시 비우기는 하지 않았다. 온도는 시작 시 배터리 온도다.

| 실행 | 온도 °C | Prefill ms | Decode ms | Decode tok/s |
| --- | --- | --- | --- | --- |
| arm 1 | 25.6 | 2435 | 7243 | 8.83612 |
| vtcm 1 | 26.0 | 2464 | 6785 | 9.43257 |
| arm 2 | 26.0 | 2430 | 7261 | 8.81421 |
| vtcm 2 | 27.6 | 2554 | 6841 | 9.35536 |
| arm 3 | 27.6 | 2379 | 7158 | 8.94104 |
| vtcm 3 | 28.5 | 2474 | 6775 | 9.44649 |

합산 decode 처리량은 `192000 / 세 실행 decode ms 합`이다.
Prefill 평균은 **2414.7 → 2497.3 ms, 약 3.4% 증가**했다.

가중치 본체 DMA는 **절반**이다. 예를 들어 이 모델의 top-4 MoE 한 층은
18 MiB의 int4 payload 대신 9 MiB를 전송한다. Scales, activation, 출력
전송까지 절반이라는 뜻은 아니다. 같은 수의 expert를 유지하면서 전체
arena mapping은 **1536 → 832 MiB**가 됐다. 이는 RSS가 아닌 DSP에 매핑된
shared arena 크기이며 FC와 chunk 반올림을 포함한다.

별도 `NNTR_HTP_PROFILE=2 NNTR_OP_TIME=1` 실행에서도 decode MoE 2432회가
모두 HVX GEMV + VTCM feed를 탔다. 가중치 DMA 카운터는 실제 기기에서
**18432 → 9216 KiB/call**이었다. MoE DSP 시간은 **436.9 → 317.9 µs/call**,
DMA wait는 **218.4 → 53.6 µs/call**이었다. 이 프로파일은 위 합산 TPS에
섞지 않았다. 전체 모델에는 FC/conv HMX, attention, RPC 등 다른 비용도 있다.

여섯 실행의 생성 텍스트 SHA-256은 모두
`a93eebddb2ba37072508dcc2d9333f268861e2e35b2cbf25a8d436b107809f69`였다.
기존의 `cred` 반복 출력은 계속된다. 커널 동등성과 이 fixture의 성능을
확인했으며 모델 품질, PPL, token ID 동등성은 검증하지 않았다.
다양한 routing이나 장시간 발열 성능도 이 짧은 측정으로 일반화하지 않는다.

앞선 [DMA bypass 기록](lfm2_24ba2b_s26_decode_dma.md)의 7.91 tok/s와는
앱 빌드가 다르다. 기존 앱과 새 라이브러리를 섞은 초기 실행은 로딩에
실패했고, 앱까지 clean rebuild한 뒤 정상 실행됐다. 초기·중간 빌드의
수치는 위 최종 A/B 통계에 포함하지 않았다.

## 검증과 재현

- 최종 v81 DSP 빌드/runtime import 검사, Android ARM library/app 빌드 통과.
- Device 5개 테스트 통과: compact MoE, 기존 M1 GEMV, FC, conv, fused SwiGLU.
  실제 K=2048/I=1536/N=2048, 5 experts, M=1/4/65, 두 palette 세대의
  cache rebind, feed on/off, DMA bypass on/off를 포함하며 출력은 bit-exact.
- Compact host 18개 조합: 4bit/2bit/mixed, decode/prefill, DMA byte count 통과.
  ASan/UBSan도 통과했다(하네스 전역 할당에 대한 leak 검사는 끔).
- Registry swap host: 압축 tail, bounds, invalid palette, 4↔2 rebind 통과.
- 전체 `test/htp/host/run_host_checks.sh` exit 0. Native Hexagon/ISS 검사는
  `libnative.a`, abs_max의 ARM NEON 검사는 cross compiler/qemu가 없어 skip됐다.
- clang-format 14 변경 줄 검사 및 `git diff --check` 통과.

```sh
adb -s R5KL30G6MCX shell '
  cd /data/local/tmp/nntr_qs2_vtcm/bin &&
  NNTR_HTP_LMHEAD_CPU=0 NNTR_HTP_QS2_VTCM=1 \
    sh run_lfm2_strict.sh /data/local/tmp/nntr_qs2_pr4415_20261009/model
'
```

`NNTR_HTP_QS2_VTCM`을 생략하면 VTCM 확장이 기본이다. `0`은 같은 QS2 파일을
ARM에서 펼치는 비교용 경로다. IDL이 확장됐으므로 ARM stub와 DSP skel을
함께 생성·빌드해야 한다. 측정 raw logs와 이번 변경만의 patch는
`/tmp/nntr-qs2-vtcm/`, 수치와 binary hash는
[JSON 기록](lfm2_24ba2b_s26_qs2_vtcm.json)에 있다.
