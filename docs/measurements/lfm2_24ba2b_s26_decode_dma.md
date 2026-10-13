# LFM2 24B A2B S26 Ultra decode DMA 최적화

후속 구현: [QS2 expert의 VTCM 확장과 새 A/B 측정](lfm2_24ba2b_s26_qs2_vtcm.md).
아래는 ARM에서 2bit를 펼치던 초기 DMA bypass 단계의 기록이다.


2026-10-09. FC `QS4CX_WH`, expert 파일 `QS2CX_WH`인 LFM2-24B-A2B에서
가중치 DMA의 DSP L2 우회를 FC와 convolution까지 연결했다. S26 Ultra에서
1024-token prefill과 64-token decode를 각 버전 두 번 실행한 결과,
decode 처리량은 합산 기준 **7.36 → 7.91 tok/s (+7.4%)**, 평균 prefill은
**7100.5 → 2890 ms**였다. 원시 측정값과 조건은
[측정 JSON](lfm2_24ba2b_s26_decode_dma.json)에 있다.

기준선은 로컬 `bdc40c2ab`와 작업 시작 전부터 있던 strict NPU, QS2 가중치,
RPC fusion 변경을 포함한다. 기존 변경을 보존하고 이번 변경을 추가했다.
양쪽에서 같은 ARM 실행 파일·라이브러리·모델·설정을 사용하고 DSP skel만
바꿨다. 기존 모델 파일은 양쪽 모두 `cred`를 반복 생성한다. 이번 결과는
연산 경로의 동등성과 성능 비교이며 모델의 생성 품질을 보증하지 않는다.

## PR 4412에서 구현된 기능

[PR 4412](https://github.com/nntrainer/nntrainer/pull/4412)의
`4c3953bb161d0f7d799a72ddf9bdf9462e3ed482`를 기준으로 확인했다.
PR 전체를 병합한 결과가 아니라 아래 기능 중 현재 LFM2 경로에 맞는
DMA 변경을 선별 적용한 결과다.

| 기능 | PR 구현 | 현재 LFM2에 대한 판단 |
| --- | --- | --- |
| DSP에서 decode op 연속 실행 | `htp_graph_desc.h`, `hexkl_graph.c`, `hexkl_token.c`, `htp_compute_ops.cpp`의 token driver와 DSP queue | 로컬에도 이전 LFM2 graph hook이 있지만 현재 WH FC 설정은 기존 Q4_0 weight binding으로 연결되지 않음 |
| WH FC 단일 토큰 GEMV | `hexkl_mm_u8i4_fc_m1_run`, `HTP_GRAPH_FEED_WH`; worker별 VTCM double buffer와 DMA queue | PR은 Q4_0 전용이 아님. 현재 FC RPC에 단순 GEMV를 붙이는 것만으로 같은 전송 구조가 되지는 않음 |
| MoE 단일 토큰 연산 | HVX u8 × i4 GEMV, VTCM feed, top-k expert 처리 | 기존 로컬 경로에서 이미 사용 중 |
| QS2 expert 유지와 DSP 확장 | registry의 `bits`와 palette, `hvx_expand_i2i4`, 2-bit weight DMA와 VTCM 확장 | 로컬은 파일에서 읽은 QS2를 ARM에서 i4로 확장한 뒤 arena에 저장. PR의 압축 arena 경로는 이번에 이식하지 않음 |
| expert 캐시와 읽기 병렬화 | pool 기반 slot 재사용, prefetch, token miss round의 `NNTR_MOE_MISS_READERS` | 로컬의 기존 LRU·prefetch와 token driver의 miss round는 별도 경로. 스위치만 추가해서 적용할 수 없음 |
| heap 가중치 DMA 우회 | 등록 시 cache flush 성공을 기록하고 arena 또는 clean heap만 `src_bypass` 허용 | 이번에 이식 |
| Gemma 모델 의미 | softmax router, GeGLU, dense와 MoE 결합, Gemma attention 설정 | LFM2의 sigmoid+bias router, SwiGLU, convolution은 유지해야 함 |

WH FC와 2-bit expert의 구체적인 PR 구현은
[MoE 및 FC 커널](https://github.com/nntrainer/nntrainer/blob/4c3953bb161d0f7d799a72ddf9bdf9462e3ed482/nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c),
[가중치 registry](https://github.com/nntrainer/nntrainer/blob/4c3953bb161d0f7d799a72ddf9bdf9462e3ed482/nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_dma.c),
[host backend](https://github.com/nntrainer/nntrainer/blob/4c3953bb161d0f7d799a72ddf9bdf9462e3ed482/nntrainer/tensor/htp_backend/htp_compute_ops.cpp)에 있다.

## 반영한 변경

`hexkl_weight_u8i4`에 immutable heap image의 flush 성공 상태를 추가했다.
공통 predicate는 arena 가중치 또는 flush에 성공한 heap 가중치만 DMA
우회를 허용한다. flush 실패 시 기존 캐시 경로를 유지한다.

PR의 MoE 적용에 더해, 현재 LFM2가 사용하는 FC RPC의 첫 가중치 전송과
다음 가중치 prefetch, fused FFN, convolution의 a/b/c/out projection에도
같은 정책을 연결했다. activation·output·convolution state의 DMA 정책과
수치 연산 순서는 유지한다. 기존 `NNTR_MOE_DMA_BYPASS` 옵션으로 제어한다.

FC와 expert의 파일 포맷은 각각 `QS4CX_WH`, `QS2CX_WH` 그대로다.
현재 fixture의 embedding 및 tied lm_head는 기존 `Q4_0`이며 이번 변경은
그 가중치를 변환하지 않는다. expert의 arena 표현도 기존의 확장된 i4를
유지한다. 따라서 이 결과를 native 2-bit arena decode 성능으로 읽으면 안 된다.

## S26 Ultra 측정

기기 `SM-S948N`, serial `R5KL30G6MCX`, Hexagon v81, HexKL 6.4.0.1.
strict NPU 경로, cache 8, early prefetch 1, prefetch readers 6,
ARM threads 8, 입력 1024 tokens, 생성 64 tokens, max sequence 4096.
측정 순서는 기준선 → 개선 → 기준선 → 개선이다. 강제 냉각이나 캐시
비우기는 하지 않았으며 시작 배터리 온도는 25.6–28.1°C였다.

| 실행 | 시작 배터리 온도 | Prefill ms | Decode ms | Decode tok/s |
| --- | --- | --- | --- | --- |
| 기준선 1 | 25.6°C | 7123 | 8940 | 7.15884 |
| DMA 개선 1 | 25.6°C | 2876 | 8086 | 7.91491 |
| 기준선 2 | 26.9°C | 7078 | 8450 | 7.57396 |
| DMA 개선 2 | 28.1°C | 2904 | 8106 | 7.89539 |

합산 decode 처리량은 `128000 / (두 실행의 decode ms 합)`으로 계산했다.
Prefill 평균 처리 시간은 59.3% 감소했다. 두 번씩의 실행이므로 장시간
발열이나 다양한 expert 선택 패턴의 성능을 대표하는 수치는 아니다.
네 실행의 진단 메시지를 제외한 출력 텍스트가 byte 단위로 같았다.
token ID와 PPL은 이 비교에서 수집하지 않았다.

별도 `NNTR_OP_TIME=1` 프로파일의 decode 시간은 다음과 같다.
ARM에서 관찰한 노드 시간으로 RPC·스케줄링을 포함한다.

| 노드 | 기준선 ms/token | 개선 ms/token |
| --- | --- | --- |
| MoE | 46.340 | 46.147 |
| Convolution | 28.000 | 25.689 |
| Attention | 19.739 | 19.418 |
| Dense FFN | 6.789 | 6.238 |
| QKV | 5.080 | 4.966 |
| Attention output FC | 4.565 | 4.418 |

FC에 기존 whole-weight VTCM double buffer를 그대로 둔 채 HVX GEMV를
붙인 별도 실험은 6.83 tok/s로 기준선 프로파일의 7.57 tok/s보다 느려서
최종 코드에서 제거했다. 이 실험은 PR의 worker별 tiled DMA feed 전체를
이식한 것이 아니므로 PR의 FC 커널 자체가 느리다는 결론은 내릴 수 없다.

## 검증과 재현

v81 DSP 빌드 및 runtime import 검사를 통과했다. FC host 검사에서
M=1/150, 여러 폭의 가중치, accumulate, output offset/stride를 검증했다.
arena·clean heap·dirty heap을 섞어 DMA 우회가 허용된 가중치에만 적용되는지
확인했다. convolution host 검사는 state continuation과 DMA 범위를 검증했다.
전체 host 스크립트는 attention 검사까지 통과한 뒤 swap harness의 기존
미연결 entry point 네 개에서 링크가 실패했다. 실행 시 즉시 실패하는
link-only stub을 보완했고 swap과 뒤의 abs-max 검사를 개별 재실행해 통과했다.
FC harness에 누락되어 있던 fused quantization의 scalar stand-in도 보완했다.

S26 Ultra에서 다음 네 device test가 통과했다. FC와 convolution 검사에는
DMA 우회 전후 출력 및 state의 `memcmp` 검사를 추가했다.

- `HmxMmU8I4Layer.ThreeWeightsMatchPerWeightReference`
- `HmxMmU8I4Layer.ConvBlockNativeWhPrefillAndContinuation`
- `HmxMmU8I4Layer.FusedSwigluMatchesTwoCallReference`
- `HmxMmU8I4Layer.MoeLayerM1GemvMatchesHmx`

검증 런타임은 기기의 `/data/local/tmp/nntr_lfm2_decode_4412/bin`, 모델은
`/data/local/tmp/nntr_qs2_pr4415_20261009/model`에 있다. 실행 예:

```sh
adb -s R5KL30G6MCX shell 'cd /data/local/tmp/nntr_lfm2_decode_4412/bin && sh run_lfm2_strict.sh /data/local/tmp/nntr_qs2_pr4415_20261009/model'
```

로컬 원시 로그, A/B 실행 스크립트, 이번 작업만의 patch는
`/tmp/nntr-lfm2-decode-4412/`에 있다. 모델·ARM 바이너리를 유지하고
`HEX_ARCH=v81 bash test/htp/build.sh`로 생성한 DSP skel을 별도 runtime에
배치하면 재현할 수 있다. SDK와 HexKL 경로는 해당 빌드 스크립트의
환경변수로 지정한다.

## 남은 최적화 경로

첫째, QS2 데이터를 arena에서도 압축 상태로 유지하고 DSP VTCM에서
확장하는 PR 경로가 필요하다. 현재 ARM 확장·복사와 i4 arena 크기를
줄일 수 있지만 registry, swap, pool capacity, prefill 및 decode를 함께
이식해야 한다. 같은 cache 8 설정에서도 slot 크기와 메모리 예산을
다시 계산하고 palette·colsum·expert 교체의 동등성을 검증해야 한다.

둘째, WH weight handle을 상주 graph에 연결하고 convolution state와 KV
seed를 넘겨야 여러 작은 RPC를 줄일 수 있다. 프로파일에서 residual add와
RMSNorm도 각각 약 9 ms/token을 차지한다. 모델 의미를 유지한 fusion 또는
graph 실행이 별도 GEMV 교체보다 우선할 근거가 된다.

마지막으로 기존 반복 생성 문제를 CPU 참조·prefill logits·teacher-forced
decode로 분리해야 한다. 정상적인 텍스트와 다양한 expert routing에서의
성능·품질 검증이 끝나기 전에는 이 fixture의 tok/s를 모델 전체의 최종
성능으로 취급하지 않는다.
