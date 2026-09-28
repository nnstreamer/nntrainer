# 53 — arXiv 2511.11248: int32 → u8 직접 재양자화, 우리 커널에 되나 (별도 세션 과제, 자체 완결)

상태: **닫힘 (2026-09-23, §8)** — 논문은 T-MAN(가중치 LUT 역양자화)이고 int32 → u8 재양자화 논문이
아니다; 우리 세 커널에는 int32에서 u8로 직행할 텐서가 없고, 이득 산술은 게이트(−5 ms) 미달. 구현 안 함.
남은 것은 SDK `hexkl_micro.h`의 acc_read 변형 확인 한 줄(§8.3)이고, 그것은 논문과 무관한 별개 지렛대다.
§1~§3은 우리 커널 쪽 사실(측정·코드), §4는 논문에서 채운 것, §8이 판정.

## 0. 한 줄

질문은 "HMX int32 누산기 → f32 → u8인 지금의 에필로그를 int32 → u8로 줄이면 얼마나 빨라지나".
답의 절반은 이미 측정돼 있다: **그 에필로그는 §2.22(51) 이후 HMX 아래에 숨어 있어서, 노출된
몫은 콜당 ~0.35 ms(2.4%)다.** 그러니 이 과제는 "dequant를 빠르게"가 아니라 (a) 숨은 워커 시간이
정말 여유가 있는지 재고, (b) int 경로가 **파이프라인 구조**(행 스캔 배리어, VTCM의 f32 gate_off
448 KB)를 없애 주는지, (c) 정확도(ppl)가 버티는지를 가리는 일이다.

## 1. 지금의 에필로그 — f32 홉이 어디에 있나

`nntrainer/tensor/htp_backend/hmx/hexkl_mm_u8i4_moe.c` (MoE 층 콜, prefill 14.4 ms / decode 1.44 ms):

```
gate_up:  HMX u8×i4 → int32 타일(acc_read, VTCM)
          ├ hvx_dq_swiglu_worker (hvx_dequant_i32.c): f32 = (acc − zp·colsum)·act_scale·w_scale + bias   [gate, up 두 타일]
          │                                          silu(gate)·up  f32 (hvx_swiglu_det_sf)  → gate_off f32 [64 × 1792], VTCM 448 KB
          └ moe_dn_worker의 rq 유닛: 행마다 min/max 스캔(hvx_quant_rows_u8_params) → scale, zp
                                   f32 → u8 RNE(hvx_quant_pack_u8_ah_rows)          → mid u8 AH 타일
down:     HMX u8×i4 → int32 → hvx_dequant_acc_tile_to_f32 → res_f32 → scatter-add f32 (라우팅 가중치) → out_c f32
```

- 활성화 양자화는 **행별 동적**(per-row u8, min/max → scale/zp). 가중치는 열별 int4(QS4CX_WH).
  이 레시피가 ppl 62.09를 준다(51 §2.21). 정적/평탄화/SmoothQuant 레시피는 51 §2.11~2.14에서
  전부 졌다(재기준이 Q4_0라 SNR 바닥 ~17 dB; 최종 판정은 ppl).
- 행별 동적이 **파이프라인을 정한다**: 한 행의 scale은 그 행의 1792열이 전부 나와야 나온다 →
  requant는 gate_up 배치 4개가 다 끝난 뒤에만 → 이걸 숨기려고 §2.22에서 down을 한 블록 뒤로 보냈다.
- down 쪽 출력은 f32여야 한다(residual stream이 ARM f32) — int32 → u8은 여기엔 해당 없음.

FC 콜(`hexkl_mm_u8i4_dma.c` `hexkl_mm_u8i4_layer_run`)과 conv 블록(`hexkl_conv_block.c`)도 같은
조각(dequant f32, conv는 phase 2에 requant)이고, 둘 다 풀 에필로그로 숨겨져 있다(51 §2.22~2.27).

## 2. 측정된 것 (51 §2.24, poll 5000, 전부 켬)

| MoE 콜 14.45 ms | us | 비고 |
|---|---:|---|
| mm (HMX 발행) | 9,102 | 타일당 18.5 ns, **천장** |
| acc (acc_read, int32 타일 VTCM 착지) | 2,954 | 타일당 0.37 us, **벤더 함수, 타일당 1회가 최소**(46 §3080) |
| dequant (노출) | 243 | 마지막 에필로그 대기 |
| requant (노출) | 62 | rq를 실은 잡의 대기 |
| scatter (노출) | 44 | |
| **숨은 워커 시간** | **미측정** | MoE 행의 SWIGLU 열은 0. conv 행만 `(hidden)`으로 찍는다(3.0 ms/콜) |

즉 int32→u8이 워커 일을 반으로 줄여도 **노출 0.35 ms 안에서만** 준다 — HMX가 병목인 한 prefill은
안 움직인다. 예외 셋:

1. **acc_read 2.95 ms (21%)**: int32 타일 8 KB/타일. HexKL이 u8/i8 출력 acc_read(고정소수점 스케일 내장)를
   가지고 있다면 4배 작은 읽기 — 이게 있으면 이 과제의 가장 큰 항목. 이 트리의 스텁은 `acc_read_int32`뿐
   (`test/htp/host/stub/hexkl_micro.h`). **SDK의 `hexkl_micro.h`를 열어 `acc_read_*` 변형을 확인하는
   것이 첫 조사**(이 컨테이너엔 SDK 없음).
2. **VTCM**: gate_off f32 448 KB가 u8 112 KB로. 지금 레이아웃 7.03 MB/8.3 — 여유가 생기면 staging
   타일 수나 활성화 슬롯 2벌(51 §2.26이 못 한 것)에 쓸 수 있다.
3. **decode(M=1)**: 콜 1.44 ms 중 dequant 1.1 us, gather 128, acc 259, mm 764 — 에필로그는 0. 해당 없음.

## 3. 정확도 — 이 과제의 진짜 게이트

int32 → u8 직행은 스케일을 **미리** 알아야 한다(고정소수점 곱-시프트). 선택지와 우리 데이터:

| 스케일 | 정확도 근거 | 파이프라인 |
|---|---|---|
| 행별 동적(지금) | ppl 62.09 | 행 스캔 배리어 있음 |
| 행별 동적, int32 도메인에서 스캔 | 수학적으로 지금과 같은 격자(u8 = round((x−min)/scale))를 int32 곱-시프트로 근사 → RNE와 다른 반올림 가능. **ppl로 확인** | 배리어 그대로(스캔이 필요) — 구조 이득 없음, 워커 시간만 감소 |
| 채널별/텐서별 정적(캘리브레이션) | 51 §2.12~2.14의 정적 레시피는 전부 손실. SwiGLU 뒤 활성화는 outlier가 커서 per-row가 필요했다 | 배리어 사라짐, 타일 단위로 u8 즉시 기록 가능 — **구조 이득 최대** |
| 배치(열 묶음)별 동적 | 격자 변경 — 새 레시피, ppl 필요 | 배리어가 배치 단위로 줄어듦 |

논문이 어느 칸인지가 §4다. 정확도 판정은 **ppl(`NNTR_PPL=1`) 하나**: 62.0916 기준, +0.5% 이내면
통과(51 §2.15의 규약). 텍스트 비교 금지(51 §2.9).

## 4. 논문에서 채운 것 (2026-09-23)

**출처 주의.** 이 컨테이너도 arxiv.org와 미러 전부(alphaxiv, semanticscholar, researchgate, huggingface
papers, paperswithcode)가 막혀 PDF 본문은 못 읽었다. 아래는 (a) 검색 엔진이 인용한 논문 본문 조각(초록,
구현 절, 표 4), (b) 논문이 공개한 **커널 소스**(`github.com/kaleid-liner/executorch` @817bd29,
`backends/qualcomm/runtime/op_packages/TMANOpPackage/src/ops/TMANLinear.cpp` — github.com은 열린다),
(c) `microsoft/T-MAC/t-man`의 README·`docs/build.md`를 합친 것. 코드에서 읽은 것은 확정, 논문 조각은
인용 문장 그대로이며 표의 나머지 칸은 모른다.

**논문**: *T-MAN: Enabling End-to-End Low-Bit LLM Inference on NPUs via Unified Table Lookup*
(USTC · Microsoft Research · Tsinghua · Microsoft, 2025-11-14). 한 줄: 저비트(1.58/2/4-bit) **가중치**를
NPU에서 **테이블 조회로 역양자화**하는 프레임워크(T-MAC의 NPU 판). **int32 누산기 → u8 재양자화
논문이 아니다** — §0의 전제("논문 = 누산기 직행 재양자화")가 틀렸다.

- [x] **하드웨어·누산기**: Snapdragon 8 Gen 3 / 8 Elite(HTP v75/v79), HMX + HVX — 우리와 같은 칩. prefill은
      HMX, decode는 HVX. 그러나 HMX에 넣는 dtype이 다르다: 가중치를 LUT로 **INT8**(BitNet, per-tensor) 또는
      **FP16**(Qwen3/Llama, per-block g64/g128)으로 풀고, 활성화는 **INT16**(`LType = int16_t`, QNN
      `--ptq 16a4w`), 출력은 **f32**(`CType = float`), 스케일은 fp16. 우리 u8×i4 네이티브 경로와 달리
      **가중치 역양자화 단계가 존재**하고, 그것이 그들의 병목이었다("NPUs have poor performance on
      computations other than GEMM, like dequantization").
- [x] **스케일 단위·결정 시점**: 가중치 per-block(fp16 스케일·오프셋을 LUT에 "bake"), 활성화는 **정적**
      (PTQ 캘리브레이션, 16a) — 코드의 `ACT_GROUP_SIZE = 256`은 K축 256 단위 그룹 스케일. 런타임
      min/max 스캔 없음.
- [x] **비선형**: 커스텀 op는 Linear 하나(`TMANLinear`, 보조 `TMANPrecompute`/`TMANFinalize`)이고 SwiGLU 등은
      QNN 그래프의 일반 op(f16/int16). 정수 LUT 활성화 함수 아님. "두 matmul 사이"를 다루지 않는다.
- [x] **이득의 정체**: prefill 1.4×, decode 3.1×, 에너지 −84%(vs 기준 NPU 방법 = QNN); 2-bit 커널 1.8–2.5×
      vs QNN, 4-bit는 비슷; 최대 8× vs QNN-W_FP16A_FP16. 이득은 **가중치 역양자화를 DMA → HVX(LUT) → HMX
      3단 파이프라인으로 숨긴 것**과 decode의 비트 직렬 LUT GEMV(T-MAC)다. 에필로그(누산기 → 출력)
      얘기가 아니다 — 출력은 f32로 나가고 dtype 변환은 QNN 그래프가 한다. 즉 §2의 핵심 질문("벽시계
      이득의 전제가 에필로그 노출인가")의 답: **아니다, 전제가 가중치 dequant 노출**이고 우리 경로에는
      그 단계 자체가 없다.
- [x] **정확도**: WikiText2 ppl. QNN W_INT4(per-channel) A_INT16: Llama-3.1-8B 18.62 / Qwen3-8B 25.37;
      T-MAN W_INT2(per-block) A_INT16: 12.81 / 13.14. 주장은 "per-block 2-bit가 QNN의 per-channel 4-bit보다
      낫다"이고, f32 대비 손실은 인용 조각에 없다.
- [x] **§3 표의 칸**: **어느 칸도 아니다.** 활성화 쪽만 보면 "정적(캘리브레이션)" 행이지만 16비트이고,
      재양자화 산술이 없다(f32 출력). 스캔 배리어를 없앤 대가로 활성화 비트를 두 배 쓴 것이고, HMX u8
      경로에는 못 옮긴다 — int16 활성화면 활성화 바이트 2배 + mm dtype 변경이고, u8 정적 레시피는 51
      §2.12~2.14에서 전부 졌다. 기대 이득 = 0 + (acc_read 변형 유무, §8.3).

## 5. 만들 순서 (게이트 있음)

| 단계 | 내용 | 게이트 |
|---|---|---|
| 0 | §4 채우기 + SDK `hexkl_micro.h`에서 acc_read 변형 확인 | 이득 산술이 −5 ms 미만이면 **여기서 멈추고 문서에 이유** |
| 1 | MoE 행에 숨은 워커 시간 프로브 추가(conv의 `cb_stage_probe_add` 패턴, `HEXKL_PROBE_SWIGLU` 원자 가산) — 기기 1회 | 워커 시간 vs HMX 발행 시간 비율. 워커가 발행보다 짧으면 "int 경로로 워커를 줄인다"는 효과 0 |
| 2 | (논문 칸에 따라) int32→u8 rq 유닛을 **호스트 스칼라 스텁으로 먼저**(`test/htp/host/`) — 지금 경로와 u8 바이트 비교 | 바이트 차이의 분포(동적-int32 칸이면 RNE 차이 ≤1 LSB인지) |
| 3 | HVX 구현, 기기 ppl + 프로파일 | ppl ≤ 62.4, 노출 dequant/requant 변화, VTCM 절약 |

## 6. 일하는 방식·측정·커밋

52 §6~§7과 동일 — 그 문서의 §6(기기 절차, 디렉터리, 환경변수, 기준값), §7(ponytail 규칙, 호스트 체크,
커밋 형식·명령, 두 브랜치 푸시), §9(함정). 여기 다시 쓰지 않는다. 이 과제는 **커널을 바꾸므로 skel도
재빌드**(`test/htp/build.sh`, 52 §9 마지막 항목). 호스트 체크 `bash test/htp/host/run_host_checks.sh`가
MoE·conv·FC 커널의 비트 동일을 지킨다 — 격자를 바꾸는 변경은 그 참조(`hvx_scalar_stubs.c`의
`quant_row`)도 같이 바뀌어야 하고, 그러면 그 체크는 "구조 검증"으로 돌아간다(51 §2.14의 교훈: 참조가
바뀌면 SNR은 의미를 잃고 ppl만 남는다).

## 7. 다른 세션의 첫 프롬프트 (복사해서 쓰기)

```
나는 SeungHui Lee (shsh1004.lee@samsung.com), nntrainer 저장소의 기여자이고 브랜치
claude/lfm2-moe-ffn-hexkl-2ivn5v와 PR nntrainer/nntrainer#4327(head claude/htp-lfm2-moe-ffn)의
작성자다. 이 브랜치에서 나와 같이 작업해 줘. 커밋은 저장소 규칙(AGENTS.md)대로: 저자는 나,
너는 Co-authored-by: Claude <noreply@anthropic.com>로 공동 저자, DCO Signed-off-by는 내 이름으로
(-s; 내가 검토하고 책임진다). 네 하네스가 붙이는 다른 출처 트레일러는 그대로 둬도 된다 —
PR 전에 내가 정리한다. 푸시는 두 브랜치에 같이 (HEAD:claude/htp-lfm2-moe-ffn과
HEAD:claude/lfm2-moe-ffn-hexkl-2ivn5v).

먼저 CLAUDE.md → docs/htp_attention/00_START_HERE.md → docs/htp_attention/53_int_requant_task.md,
그리고 53이 가리키는 52 §6~§7(측정 절차, 일하는 방식)을 읽어. 과제: arXiv 2511.11248(양자화된
행렬곱의 int32 누산기를 f32를 거치지 않고 u8로 재양자화하는 방법)을 우리 HTP MoE 커널
(hexkl_mm_u8i4_moe.c)에 적용할 수 있는지 분석하고, 된다면 구현. 순서는 53 §5 그대로: 논문을
읽고 53 §4의 빈칸을 먼저 채우고(§3 표의 어느 칸인지), SDK hexkl_micro.h에서 acc_read 변형이
있는지 확인하고, 이득 산술이 −5 ms 미만이면 구현하지 말고 문서에 이유만 남겨. 정확도 게이트는
ppl(NNTR_PPL=1, 기준 62.0916, +0.5% 이내)이고 텍스트 비교는 지표가 아니다. 기기 측정은 전부
내가 한다 — 너는 config·명령을 주고 결과를 받아 문서(53의 다음 절)에 기록해. 가설에 코드 쓰기
전에 분해를 재라 — 51 §2.25~2.27이 왜 그런지의 기록이다.
```

## 8. 판정: 구현하지 않는다 (2026-09-23)

§5 0단계의 게이트에서 멈춘다. 이득 산술이 −5 ms에 못 미치고, 그보다 먼저 **구조가 안 맞는다**.

### 8.1 우리 세 커널에는 int32 → u8로 직행할 텐서가 없다

§1의 f32 홉을 재양자화의 **입력**이 무엇인지로 다시 보면, 어디에서도 누산기가 아니다:

| 커널 | 재양자화 입력 | 그 앞에 있는 것 |
|---|---|---|
| MoE gate_up → down (`hexkl_mm_u8i4_moe.c`) | gate_off = silu(g)·u | g, u 각각 dequant(f32) → 비선형 |
| conv 블록 phase 2 (`hexkl_conv_block.c`) | conv 출력 | dequant → a·c 곱 → 3탭 conv, 전부 f32 |
| FC(q/k/v/o), MoE down (`hexkl_mm_u8i4_dma.c`, down) | 없음 | 출력이 f32여야 한다(residual stream, attention core) |

int32에서 나가려면 SwiGLU(또는 곱·conv)를 정수로 해야 하고, 정수 SwiGLU는 g·u의 스케일을 **미리** 알아야
한다(LUT 인덱스) = 정적 스케일 = 51 §2.12~2.14에서 진 레시피(SwiGLU 뒤 활성화의 outlier 때문에 per-row가
필요했다). 동적 per-row를 유지하는 한 g·u는 f32로 풀려야 하고, 그러면 §3의 2행("int32 도메인에서
스캔")조차 성립하지 않는다 — **스캔할 int32 텐서가 없다.** 이 아이디어가 맞는 자리는 attention이 HTP로
올라올 때의 q/k/v 투영 → KV 블록 u8(30번 과제의 블록 양자화기) 하나뿐이고, 이 과제 밖이다.

### 8.2 이득 산술 (51 §2.24, poll 5000, 전부 켬)

- 노출 에필로그: dequant 243 + requant 62 + scatter 36 = **341 us/콜(2.4%) × 22 = 7.5 ms**. 산술을 무엇으로
  바꾸든 이것이 상한이고, 그중 콜 양끝의 구조적 대기(rq(0) 앞의 마지막 gate_up 에필로그, 마지막 down
  에필로그)는 안 사라진다. 워커 시간이 반으로 줄어 노출이 반으로 준다고 쳐도 ≤ 3.7 ms.
- VTCM: gate_off f32 448 KB → u8 112 KB로 336 KB가 남으면 활성화 슬롯 2벌 — 상한 −0.2 ms/콜 = −4.4 ms
  (51 §2.26이 그 이유로 안 한 것).
- decode(M=1): 에필로그 0(§2). 해당 없음.
- **둘 다 정적 스케일 전제**다(8.1). 정적이면 ≤ 8 ms에 ppl 손실, 동적 유지면 **0 ms**. 어느 쪽도 게이트
  미달. §5 1단계(숨은 워커 시간 프로브)도 이 과제에서는 안 만든다 — 그 수치가 어떻게 나와도 8.1을 못
  뒤집는다.

### 8.3 열려 있는 것 하나: acc_read의 폭 (논문과 무관)

acc 2,954 us(21%)는 타일당 0.36 us에 int32 8 KB를 VTCM에 내려놓는 벤더 함수 한 번씩이다. 이 트리와 이
컨테이너에는 SDK `hexkl_micro.h`가 없고(`test/htp/host/stub/`은 `acc_read_int32`만), 트리가 아는 변형은
`acc_read_int32`(int 경로)와 `acc_read_f16`(fp16 matmul 경로, ref_13)뿐이다. **확인은 기기 빌드 머신에서
한 줄**(`HEXKL_ROOT`는 `test/htp/build.sh`가 쓰는 beta2 addon):

```bash
grep -n 'acc_read\|requant\|scale\|shift' "$HEXKL_ROOT"/include/hexkl_micro.h
```

판정 규칙: int 누산기를 **더 좁은 폭으로** 읽는 변형(int16/i8/u8/f16 출력 + 스케일이나 시프트 인자)이
있으면 **별개 지렛대**("acc_read 폭")로 연다 — 읽기가 바이트 비례라면 반폭에 −1.5 ms/콜 × 22 ≈ −30 ms가
상한이라 게이트를 넘는다. 단 (a) 스케일을 읽기 전에 알아야 하므로 gate_up에만 해당(down은 f32 출력이
필요), (b) f16 출력은 K=2048의 u8×i4 합(최대 ~4.2M)이 f16의 정수 정밀도(2048)를 넘어 비트 동일이
깨지므로 ppl 게이트를 다시 통과해야 하고, (c) 정적 스케일을 요구하는 변형이면 8.1의 정확도 문제가 그대로다.
그런 변형이 없으면 이 과제는 여기서 닫힌다.

### 8.4 재검토: HVX 워커 안의 int32 → f32 → u8를 논문처럼 정수로 (2026-09-23, 프로브 코드)

질문(작성자): "HVX에서 int32 → f32 → u8 변환이 일어날 텐데, 이걸 논문처럼(스케일을 미리 구워 넣은 정수
산술·테이블) 하면?" — 8.1은 "그 텐서가 없다"였고, 이번엔 워커 안의 산술 자체를 본다.

**지금 워커가 하는 일, 벡터(32열) 1개 기준** (`hvx_dequant_i32.c`, `hvx_swiglu_det.h`, `hvx_quant_u8.c`):

| 단계 | f32 op 수 | 정수로 가능한가 |
|---|---:|---|
| dequant g: cvt(int32→f32), zp·cs 곱, 빼기, ×s_r, ×w_c, +b_c | 6 | **가능** — (acc − zp_r·cs_c)는 int32 그대로(2 op), ×(s_r·w_c)는 열별 Q15 곱셈기 + 행별 시프트로 `vmpyo_VwVh_s1_rnd_sat` 1 op. 논문의 "2단계 bake"가 정확히 이것(열 상수를 미리 구움). 3~4 op |
| dequant u | 6 | 같음 |
| SwiGLU det: exp(≈26) + recip(10) + 3 | ≈40 | **f32 입력이 필요**. 정수판은 LUT 시그모이드인데 인덱스가 되려면 g의 범위(스케일)를 **미리** 알아야 한다 = 정적, 또는 g·u를 먼저 스캔(배리어 2개 추가, 곱의 상한은 느슨해 u8 범위 낭비). 그리고 이 함수는 ARM과 **비트 동일 계약**(44 §L2: 1 LSB 뒤집힘이 토큰 스트림을 바꿨다) — 바꾸면 ppl 재게이트 |
| requant 스캔(min/max) + 양자화(×1/s, RNE cvt, +zp, pack) | 2 + ≈2.5 | 입력이 f32라 정수 형식 없음. 스캔 배리어는 per-row인 한 산술과 무관 |

즉 "논문처럼" 바꿀 수 있는 부분은 **dequant 12 op 중 8 op**(전체 ≈57 중 14%)이고, f32 홉은 SwiGLU가 f32인 한
사라지지 않는다. 논문의 1단계(vlut16으로 비트 재포장)는 입력이 ≤4비트일 때의 기술이라 int32 누산기에
맞을 자리가 없다.

**그 14%가 벽시계를 움직이려면** 워커 시간이 HMX 그림자(발행 시간 × 워커 3)에 가까워야 한다. 노출
341 us/콜은 "가깝지 않다"고 말하지만, 워커 시간 자체는 §2가 미측정이라고 적은 그대로다 — 51 §2.25~2.27의
규칙대로 가설에 코드를 쓰기 전에 그 수를 잰다. **§5 1단계를 만들었다**(이 커밋):

- `hexkl_mm_u8i4_moe.c`: `moe_tail_probe_add` → `moe_worker_probe_add`로 일반화. gate_up 에필로그
  (`moe_gu_worker` = `hvx_dq_swiglu_worker` 타이밍 래퍼)와 down 에필로그(`moe_dn_worker`: dequant + scatter
  + 실려 있는 rq 유닛)의 워커 슬라이스 시간을 원자 가산으로 SWIGLU 열에 합산. conv의 `cb_stage_probe_add`와
  같은 읽기: **워커 시간의 합**이지 벽시계가 아니다. 프로브 off면 0 비용.
- `htp_compute_ops.cpp`: 버킷에 `swiglu_hidden` 플래그 — MoE 층 커널이 채우는 행(MoE·dense·conv)은 SWIGLU를
  mm 잔차에서 빼지 않고 `swiglu(hidden)`으로 찍는다(전엔 conv만). fused 경로는 그대로 뺀다.
- 산술 불변(타이머만): 호스트 체크 `MOE KERNEL MATCHES REFERENCE`, conv, FC, 풀 전부 통과. **기기에서는
  안 돌렸다** — skel과 앱 둘 다 재빌드 필요(52 §9 마지막 항목).

**측정(작성자)**: 전부 켬 config 그대로, `NNTR_HTP_PROFILE=2 NNTR_PPL=1` 1회. 볼 것은 `M>1` MoE 행과
`M>1 dense` 행의 `swiglu(hidden)` 값과 `mm` 잔차(hidden을 뺐으니 §2.24의 9,102 근처여야 한다; 크게 다르면
플래그가 잘못 붙은 것). 판정:

| swiglu(hidden) / (mm + acc ≈ 12.1 ms × 워커 3 = 36 ms) | 뜻 | 정수 dequant의 값어치 |
|---|---|---|
| < 30% | 워커는 대부분 놀고 있다; 노출 341 us는 파이프라인 양끝 | **0** — 산술을 반으로 줄여도 벽시계 불변. 과제 닫힘 |
| 30~70% | 여유는 있으나 배치 단위로 밀리는 구간이 있음 | 노출 341 us의 일부(≤ 3.7 ms/prefill, 8.2) — 게이트 미달 |
| > 70% | 그림자가 거의 찼다 — 51 §2.22의 가정이 틀렸다는 뜻 | 정수 dequant(−14%)가 아니라 **SwiGLU(70%)**가 대상; 그때도 비트 동일 계약과 ppl이 먼저 |

어느 칸이든 "int32 → u8 직행"은 답이 아니고, 셋째 칸만 다른 과제(SwiGLU 비용)를 연다.

**실행 1 (2026-09-23, 작성자, 전부 켬, PROFILE=2 PPL=1)**: 프로브가 **바이너리에 없었다** — MoE 행이
`swiglu 0.0`으로 찍혔다(프로브가 들어간 skel이면 gate_up 에필로그만으로 ≥ 5 ms가 나와야 한다; conv 행의
`swiglu(hidden) 3190.7`은 예전 코드도 같은 표기라 재빌드의 증거가 아니다). 명령 프롬프트의 브랜치가
`claude/eager-keller-f91z9o`였다 — 6af3834(이 브랜치)의 skel·앱이 아니다. 같은 세션 기준값으로는 쓸 수 있다:

```
ppl 62.0916 (그대로)   prefill 1766 ms (PPL 켬이라 헤드라인 아님)   decode 23.1   peak RSS 5,129 MB   등록 3.78 s
  M>1 MoE   calls=23  host 14489  dsp 13818  transport 672   [mm 9056 acc 2938 stage 388 drain 266+45 quant 278 dequant 276 alloc 138 gather 74 requant 74 scatter 42 rest 190]  blocks=1039
  M>1 dense calls=3   host 9679   dsp 9097   transport 581   [mm 5830 acc 1927 dequant 322 requant 63]
  M>1 conv  calls=19  host 4524   dsp 3974   transport 550   [swiglu(hidden) 3191 mm 2082 acc 665 dequant 372 gather 268]
  M>1 FC N=3072 host 2137 dequant 5.0 | N=2048 host 1639 dequant 8.6   M==1 host 1439 dsp 1351
```
§2.24와 같다(MoE 14448 → 14489, dequant 243 → 276, requant 62 → 74: 세션 편차). 다시 돌릴 것: 두 브랜치 중
하나를 6af3834로 받아 `test/htp/build.sh`로 skel을 다시 빌드·푸시하고(52 §9: 안 하면 이전 skel이 그대로 돈다),
앱도 `build_android.sh --htp`로 다시 빌드한 뒤 같은 명령. 판정 기준은 MoE 행의 `swiglu(hidden)`이 0이 아닌 것.
