# 54 — 2-bit MoE expert (W2A8): 모델을 굽고 VTCM에서 4-bit으로 펼친다

상태: **Step 0·1 코드 들어감, 기기 측정 0.** 아래에서 "측정"이라 적힌 숫자만 기기 실측이고
나머지는 산술이다. 논문(arXiv 2511.11248, T-MAN)은 읽었다 — §2에 요약.

## 0. 한 줄

expert 가중치를 2-bit으로 굽고 DSP가 **VTCM 안에서 int4로 펼쳐** 기존 HMX u8×i4 커널에
먹인다. 노리는 것은 prefill의 weight DMA이고, 상주 메모리(3696 → ≈1900 MiB)는 덤이다.
**이 과제의 값어치는 두 개의 값싼 측정(§5 Step 0, Step 1)이 정한다** — 둘 다 코드는 이미 있다.

## 1. 하드웨어가 강제하는 세 가지

1. **HMX 시간은 비트폭과 무관하다.** `mm` 9102 + `acc` 2954 µs = 14.45 ms 콜의 84%이고 둘 다
   **타일 수**(K·N/1024)에 비례한다 (51 §2.24). 2-bit이어도 타일 수는 같다.
   **2-bit이 줄이는 것은 DDR→VTCM DMA 바이트와 상주 메모리뿐이다.**
2. **열 하나의 스케일이 K 전체를 덮어야 한다.** dequant는
   `f32 = (acc − zp·colsum)·act_scale·w_scale + bias` 한 번뿐이므로 (hvx_dequant_i32.h),
   논문의 block=64 그룹 스케일은 **k-그룹마다 accumulator를 비우는 것**을 뜻한다 —
   `acc_read`가 이미 콜의 21%(0.37 µs/타일, 벤더 함수, 타일당 1회가 최소, 46 §3080)라
   그 방향은 산술적으로 죽는다.
   ⇒ 포맷은 **2-bit 인덱스 + 4-엔트리 int4 팔레트**. 펼친 뒤에는 colsum·dequant·SwiGLU·
   requant가 **지금과 비트 단위로 동일**하다. 바뀌는 것은 오프라인 양자화기와 펼침 한 단계뿐.
3. **prefill MoE는 DMA 포화로 *보인다*.** 콜당 weight 176 MB / dsp 13.82 ms = **12.7 GB/s**,
   그리고 "HMX 옆에서 DMA는 유휴 33이 아니라 ~12 GB/s"가 51 §2.26에서 독립적으로 관측됐다.
   HMX 12.06 < DMA ~14 → DMA가 벽. **다만 두 숫자가 맞는 것은 하나가 다른 하나를 기다리는
   것의 증명이 아니다** (44 §11.5). Step 0이 그걸 잰다.

## 2. 논문과 우리의 차이

| | T-MAN (2511.11248) | 우리 |
|---|---|---|
| 커널 | prefill: LUT dequant → **fp16** matrix core / decode: 순수 **VLUT16 LUT-GEMV** | prefill·decode 모두 **HMX u8×i4**, int32 누산 |
| 양자화 | GPTQ **비대칭 per-block, block=64** (W_INT2/W_INT4) | per-column, 그룹 없음, zero-point 없음(대신 colsum) |
| 이득의 원천 | 런타임 dequant를 **없앤 것** (decode 3.1×, prefill 1.4×, 에너지 −84%) | 우리 에필로그는 **이미 HMX 아래 숨어 있다** — 노출 몫 0.35 ms/콜 = 2.4% (53 §2) |
| 저장 | bit-serial, prefill에 on-the-fly repack (12× 연산 감소) | 오프라인 WH bake, repack 불필요 |
| 정확도 | Qwen3-8B 2-bit ppl 13.14 (QNN 25.37 대비 −48%) | 8B **dense**의 숫자다. 우리는 활성 1B의 작은 expert라 민감도가 다를 수 있다 |

**논문에서 가져올 수 있는 것은 커널이 아니라 "저장·전송 폭을 줄인다" 한 가지뿐이다.**
논문 이득의 원천(런타임 dequant 제거)은 우리에게 이미 존재하지 않는다.

## 3. 펼침 — 하드웨어 지원은 없다, HVX 소프트웨어다

HMX의 weight 포트는 i4/i8뿐(`hexkl_micro_hmx_mm_u8i4` / `_u8i8`). 이 트리의 스텁에 i2 변형은
없다. SDK `hexkl_micro.h` 확인은 53 §5 단계 0의 `acc_read_*` 확인과 같이 처리할 것.

**레이아웃 사실 두 개** (`htp_wh_layout.h` `whSlot`을 풀고 `hvx_gemm_u8i4_wh.c:57-65`로 교차 확인):
`whSlot`은 **nibble** 인덱스이고 `byte = c*4 + (r%4)` — **연속 바이트 4개가 한 열**, 128 B 벡터
하나가 타일의 32개 열을 정확히 한 번씩 덮는다. 2-bit 코드를 같은 nibble 순서로 담으면
**소스 바이트 j → 출력 바이트 2j, 2j+1** 로 완전히 연속이다.

| | 방식 | 출력 벡터당 op | 층당 (168 MiB, 5 워커, ~1.4 GHz) | 예산 1.4 ms |
|---|---|---:|---:|---|
| **A** | (열,그룹)별 팔레트 — 코드를 바이트 레인으로 전개 → 128-엔트리 테이블(`vlut32` 4개 체인) ×4 → nibble 재팩 | ≈17 | ≈3.3 ms | ✗ |
| **B** | **그룹별 공용 팔레트 + 열 변동은 `w_scale[n]`이 흡수** — 코드 **쌍**(4비트) 인덱스, **16-엔트리 `vlut32` 한 번** ×2, `vshuff` | **3** | **≈0.6 ms** | ✓ |

B의 핵심은 테이블을 코드 하나가 아니라 **코드 쌍**으로 잡는 것이다:
`T[(c1<<2)|c0] = pal[c0] | (pal[c1] << 4)` → 16엔트리 → `vlut32` 한 op으로 출력 바이트가 나온다.

```c
const HVX_Vector lo = Q6_V_vand_VV(v, m0f);
const HVX_Vector hi = Q6_V_vand_VV(Q6_Vh_vlsr_VhR(v, 4), m0f);
const HVX_VectorPair o = Q6_W_vshuff_VVR(Q6_Vb_vlut32_VbVbI(hi, tab, 0),
                                         Q6_Vb_vlut32_VbVbI(lo, tab, 0), -1);
```

성립 조건은 **팔레트가 벡터 안에서 상수**인 것. 그래서 열별 팔레트 *모양*은 포기하고 열별
크기는 이미 있는 `w_scale[n]`이 가져간다 — QLoRA NF4의 "전역 코드북 + 블록 스케일"과 같은
구조이고, 가우시안 가중치의 Lloyd-Max 4레벨 비율(1.51/0.4528 = 3.33)이 int4 격자의
`{-7,-2,2,7}`(3.5)과 거의 맞는다. 팔레트는 k-그룹마다 교체 가능(벡터 안에서 상수이므로 비용 0).

**A는 아낀 DMA보다 비싸서 후보가 아니다. 정확도 대조군으로만 만든다.**
(`Q6_Vb_vlut32_VbVbI`의 match 필드 동작은 SDK 헤더로 확인할 것 — 문서를 쓴 컨테이너에
Hexagon 툴체인이 없다.)

## 4. 기대값

| | prefill/콜 | prefill 22층 |
|---|---:|---:|
| 지금 (측정) | 14.45 | 585~616 |
| 2-bit, 펼침 비용 0 가정 (DMA 14 → 7.4, HMX 12.06이 새 벽) | ≈13.0 | ≈554 (−5%) |
| − 펼침 비용 (B 견적 0.6) | ≈13.6 | ≈567 (−3%) |
| 2-bit + HMX 패딩 59% 해결 (별도 과제, 51 §2.24) | ≈8 | ≈420 |
| **메모리** | — | **3696 → ≈1900 MiB, peak RSS 5,252 → ≈3,400 MB** |

**정직하게: prefill 이득은 작고 불확실하며, 메모리 이득만 확실하다.** decode는 더 작다 —
M=1 콜은 mm 764 + acc 259 = 1.02 ms로 HMX가 DMA(1.19 ms)와 거의 같은 벽이라 −0.2 ms/콜
(≈ +2~3 TPS)에 그친다 (48 §3). 논문의 진짜 decode 레버는 LUT-GEMV로 **64행 패딩과 acc_read를
동시에** 없애는 것이고(세 벽 중 둘), 그건 이 과제가 아니다.

펼침이 예산을 못 맞추면 **되돌아갈 자리가 있다**: 펼침을 호스트 등록 시점
(`get_or_register_wh`)으로 옮기면 DMA 이득은 사라지지만 파일 크기 절반과 ppl 결과는 남고,
2-bit 모델 자체가 52(flash-experts)의 미스 비용을 절반으로 만든다.

## 5. 단계와 게이트

| 단계 | 내용 | 상태 | 게이트 |
|---|---|---|---|
| **0** | DMA 포화 확인 (`HEXKL_MOE_HALF_DMA`) | **코드 있음** (`f3874b0`) | dsp −1.4 ms/콜이면 예산 확정. ≤0.3이면 **prefill 이득 0**, 메모리 과제로 재정의 |
| **1** | 팔레트 ppl 표 (`--moe_palette_*`) | **코드 있음** (`1a69287`) | §5.2 표. B 행이 살아남는가 |
| 2 | 숨은 워커 시간 프로브 + 펼침 마이크로벤치 | 미착수 | 펼침 ≥120 GB/s. 워커 슬랙 있음 |
| **3a** | `QS2CX_WH` 포맷 + 양자화기 | **코드 있음** (`996ce0c7`) | 파일이 절반, 펼치면 4-bit 대역과 바이트 동일 — **확인됨** |
| 3b | HTP 등록 경로 (메모리 이득) | 미착수 | peak RSS ≈3,400 MB, ppl은 1의 해당 칸과 소수점까지 동일 |
| 4 | VTCM 펼침 (DMA 이득) | 미착수 | `drain`↓, **`mm`·`acc` 불변**, dsp 순감소 |

3과 4를 나눈 이유: 배선 실수를 펼침 성능 문제와 섞지 않기 위해서다.

### 5.1 Step 0 — 돌리는 법

`HEXKL_MOE_HALF_DMA`는 weight 청크의 k-타일 절반만 push하고 나머지 VTCM은 직전 expert의
바이트를 그대로 둔다. **산술은 쓰레기, 타이밍은 정확**: 같은 HMX 발행 루프, 절반의 트래픽.

```bash
source ~/workspace/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
cd ~/workspace/nntrainer

# (a) 기준선 skel
bash test/htp/build.sh
#     push + 기존 앱으로 NNTR_HTP_PROFILE=3 1회

# (b) 절반 DMA skel — 같은 세션에서 바로 이어서
HEXKL_EXTRA_CFLAGS=-DHEXKL_MOE_HALF_DMA=1 bash test/htp/build.sh
#     push + 같은 명령 1회
```

기기:
```bash
cd /data/local/tmp/nntrainer/causallm
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=3 \
  ./nntrainer_causallm ./models/lfm2.5-8b-a1b-q40-qs4cx-wh
```

읽는 법 — `M>1 MoE` 행만 본다. `NNTR_PPL`은 켜지 않는다(이 실행의 출력은 무의미하다).

| 열 | 기준선 | 기대 |
|---|---:|---|
| `dsp` | 13822 | **≈12400이면 DMA가 벽** |
| `drain` | 251+53 | 내려감 |
| `mm` (control) | 9102 | **안 움직여야 정상** |
| `acc` (control) | 2954 | **안 움직여야 정상** |

`mm`/`acc`가 움직였으면 실험이 잘못된 것이다 (44 §10). 끝나면 (a)의 skel로 되돌려 push.

### 5.2 Step 1 — 돌리는 법 (커널 변경 없음, 기존 skel 그대로)

팔레트는 *값*의 제약일 뿐이라 **파일은 그대로 QS4CX_WH**다. 즉 오늘의 커널이 내일의
가중치로 오늘의 ppl을 찍는다.

```bash
build/Applications/CausalLM/nntr_quantize_stream <fp32_model_dir> \
  -o <out_dir>/lfm2.5-8b-a1b-q40-qs4cx-wh-pgmax \
  --fc_dtype Q4_0 --moe_dtype QS4CX_WH --embd_dtype Q4_0 --lmhead_dtype Q4_0 \
  --isa ARM --moe_palette_g max
```

**`max`여야 한다.** `QS2CX_WH`는 팔레트를 텐서당 하나만 담으므로(§6), `g 256`은 실제
2-bit 파일과 *통계적으로만* 같고 바이트로는 다른 모델이다. §5.3이 G가 SNR을 안 바꾼다고
했으니 `max`로 잃는 것은 없고, 얻는 것은 **잰 것과 굽는 것이 같은 모델이라는 보장**이다
(tiny fixture에서 확인: `--moe_palette_g max` QS4CX_WH를 펼치면 QS2CX_WH와 바이트 동일).

경로가 도는 것은 tiny fixture(`test/unittest/models/causallm_reference/lfm2_moe_tiny`)로
호스트에서 확인했다 (2026-09-23): 출력 크기 71,648 B로 **불변**, 달라진 바이트는
`27088..43472`와 `55008..71392` 두 구간뿐 — MoE 2개 층의 expert payload
(4 expert × (gate_up 2560 + down 1536))와 정확히 일치하고 dense FFN·conv·attention·
embedding·lm_head는 바이트 동일. 그 안의 코드는 16개 → **4개**
(`gate_up {-5,-2,1,4}`, `down {-5,-2,1,5}`).

`nntr_config.json`은 기존 HTP 디렉터리와 같게(전부 켬). 기기:
```bash
LD_LIBRARY_PATH=. ADSP_LIBRARY_PATH=. NNTR_NUM_THREADS=8 NNTR_HTP_PROFILE=2 NNTR_PPL=1 \
  ./nntrainer_causallm ./models/lfm2.5-8b-a1b-q40-qs4cx-wh-pgmax
```

채울 표 (기준 **ppl 62.0916**):

| scope | G | 적용 | ppl | 판정 |
|---|---|---|---|---|
| group | `max` | both | | **이것 하나가 판정한다** (§5.3) |
| group | `max` | gate_up | | 위가 떨어졌을 때만. 혼합 정밀도 후퇴선 (I/O 1.6×) |
| ~~group~~ | ~~256/128/64~~ | | | **굽지 않음** — §5.3, group scope에서 G는 SNR을 소수점까지 안 바꾼다 |
| ~~colgroup~~ | ~~any~~ | | | **굽지 않음** — §5.3, +0.4 dB에 펼침 6배 |
| ~~refit off~~ | | | | **굽지 않음** — §5.3, −0.35 dB |

| ppl | 판정 |
|---|---|
| ≤ 62.4 | 표준 게이트 통과 (51 §2.15) |
| 63~65 | 게이트 재설정 논의. dense FFN이 +4.2%로 꺼져 있는 것이 전례 (51 §2.17) |
| > 70 또는 발산 | 혼합 정밀도로 후퇴, 그래도 안 되면 **중단** |

**Step 0과 Step 1은 서로 독립이다.** 둘 중 하나라도 떨어지면 Step 2~4는 하지 않는다.

### 5.3 §5.2 표는 **한 칸으로 줄었다** — 실제 가중치 SNR (2026-09-23, 호스트)

기기 실행을 여섯 번 쓰기 전에, 실제 fp32 체크포인트에서 expert 텐서를 직접 읽어 후보들을
호스트에서 먼저 줄 세웠다. **SNR은 게이트가 아니다**(51 §2.14, 기준이 Q4_0이라 바닥이 ~17 dB) —
후보의 *순서*를 정하는 데만 쓴다. 판정은 여전히 ppl 하나다.

방법: fp32 레이아웃(`writeLfm2Moe`의 순서)을 그대로 재현해 텐서 오프셋을 계산하고, 합계가
파일 크기 **33,871,427,328과 정확히 일치**하는 것으로 오프셋을 자체 검증했다. 양자화기
(`__fallback_quant_nxk_qs4cx_f32`)는 harness에 그대로 옮겨 libnntrainer 링크 없이 돌렸다.

`layer2_expert0_gate_up` (K=2048, N=3584), SNR dB:

| config | G=K | 512 | 256 | 128 | 64 | 32 |
|---|---:|---:|---:|---:|---:|---:|
| **4-bit (지금)** | | | **16.73** | | | |
| 2-bit group, refit on | 7.81 | 7.81 | **7.81** | 7.81 | 7.81 | 7.81 |
| 2-bit group, refit off | 7.43 | 7.43 | 7.43 | 7.45 | 7.45 | 7.46 |
| 2-bit colgroup, refit on | 8.04 | 8.10 | 8.19 | 8.35 | 8.68 | 9.30 |

층·expert를 가로질러 균일하다 (`gate_up`/`down` × layer 2/12/23 × expert 0/17):
4-bit 16.30~17.28, 2-bit colgroup G=64 8.47~8.85.

**읽히는 것 넷:**

1. **group scope에서 G는 아무것도 하지 않는다** — G=32와 G=K가 소수점까지 같다. 코드 히스토그램이
   어디서나 같은 모양이라 DP가 같은 네 레벨을 고른다. ⇒ §5.2 표의 `max`/256/128/64 네 줄은
   **같은 모델 한 줄**이다. **지금 굽고 있는 G=256 하나가 group 행 전체를 판정한다.**
2. **colgroup은 G=256에서 +0.4 dB, G=32에서도 +1.5 dB뿐**이다. §3의 A는 펼침이 6배 비싼데
   돌려주는 게 이것뿐이라 **후보가 아닌 것이 확인됐다** — 대조군으로도 구울 값어치가 없다.
3. **refit은 +0.35 dB.** 켜둔 채로 간다. 끈 모델을 따로 구울 이유는 없다.
4. **4-bit → 2-bit 간격은 ≈8.9 dB이고 어디서나 같다.** 층별/텐서별 혼합 정밀도를 고를
   SNR 손잡이는 없다 — 고른다면 ppl로 골라야 한다.

**그래서 기대를 낮춰 잡아야 한다.** 비교 전례: dense FFN의 재양자화는 mean_abs_err
0.0451 vs 0.0333, 즉 **≈2.6 dB** 나빠진 것으로 ppl +4.2%를 냈다 (51 §2.19, §2.17). 여기는 그
3배가 넘는다. G=256 모델의 ppl이 62.4를 넘을 가능성이 높고, 넘는다면 **남은 레버는 팔레트
조정이 아니다** — 우리는 지금 RTN(최근접 반올림)이고, 논문이 2-bit에서 쓸 만한 수를 낸 것은
**GPTQ** 포맷이기 때문이다. 2비트에서 GPTQ와 RTN의 차이는 팔레트 스코프 차이보다 훨씬 크다.
그건 캘리브레이션 데이터와 expert별 Hessian이 필요한 별개 과제이고, 이 과제의 진짜 분기점이다.


## 5.4 `QS2CX_WH` 포맷 (2026-09-23, 호스트)

```
[K*N/4 codes][4 palette bytes][N f32 scales][N f32 colsums]
```

코드는 **whPack의 nibble 순서 그대로**, 4개씩 한 바이트에 낮은 비트부터. 그래서 소스 바이트
j가 출력 바이트 2j·2j+1로 떨어지고 섞을 것이 없다 — §3의 B가 성립하는 조건이다.

**팔레트는 텐서당 하나다.** §5.3이 그룹 크기를 실제 체크포인트에 대고 재봤을 때 SNR을
전혀 안 바꿨기 때문이다. 코드 히스토그램이 어디서나 같은 모양이라 적합이 같은 네 레벨을
고른다. 그룹 축을 포맷에 남기면 DSP 펼침에도 테이블 교체로 따라 들어온다 —
아무것도 변하지 않는 축을 위해. *ponytail: GPTQ처럼 그룹마다 분포가 달라지는 레시피가
오면 그때 축을 넣는다. 지금 레시피에는 없다.*

tiny fixture 확인 (2026-09-23):

| | |
|---|---|
| 파일 | 71,648 → **59,424 B** (−12,224 = 8 expert × (gate_up −1024, down −512, palette +8)) |
| 펼침 | `--moe_palette_g max` QS4CX_WH와 **layer 1의 8개 텐서 전부 바이트 동일** (weights·scales·colsums) |
| `moe_layer_dtype` | `"QS2CX_WH"` |

유닛 테스트 `wh_pack2_expands_back_to_wh_pack`가 `whPack2 → whExpand2 == whPack`을 지킨다.
`whExpand2`는 DSP HVX 구현이 만족해야 할 **명세**이기도 하다 — 같은 16-엔트리 테이블,
출력 바이트당 한 번의 조회.

**아직 못 돌린다.** 파일은 있고 텐서로 로드도 되지만, `get_or_register_wh`·IDL·DSP 펼침이
없어서 HTP에 올릴 수 없다(3b, 4). CPU는 `FloatTensor::dot`이 `QS4CX_WH`와 같이 throw하도록
막아뒀다 — 조용히 틀린 답을 내는 것보다 낫다.

## 6. 들어간 코드

| 커밋 | 무엇 |
|---|---|
| `f3874b0` | `HEXKL_MOE_HALF_DMA` (측정 전용 빌드), `build.sh`의 `HEXKL_EXTRA_CFLAGS` |
| `1a69287` | `nntrainer/tensor/htp_wh_palette.h` (팔레트 DP + 적용), `quantize_stream.cpp`의 `--moe_palette_*`, 유닛 테스트 `wh_palette_is_optimal_and_restricts` |
| `996ce0c7` | `QS2CX_WH`: `whPack2`/`whExpand2`/`whPaletteTable`, `QS2CX_WH_Tensor`, dtype 배관, `--moe_dtype QS2CX_WH`, 유닛 테스트 `wh_pack2_expands_back_to_wh_pack` |

팔레트 적합은 k-means가 아니라 **DP로 정확**하다: 후보 코드가 16개뿐이라 최근접 분할이
연속 구간이 되고, 네 구간 위의 DP가 최적이다. 유닛 테스트가 그것을 1820개 4-부분집합
전수 탐색과 그룹마다 비교한다 (호스트에서 돌아간다).

`--moe_palette_refit`은 제한된 코드에 대한 최소제곱 `w_scale`이다. 기본 on이지만, 하나의
ppl 숫자에 두 변경이 같이 타는 것이라 끌 수 있게 해뒀다.

## 7. 일하는 방식·측정·커밋

52 §6(기기 절차, 디렉터리, 환경변수, 기준값), §7(ponytail 규칙, 호스트 체크, 커밋 형식),
§9(함정)과 동일. 여기 다시 쓰지 않는다. 덧붙일 것 둘:

- **Step 0·4는 커널을 바꾸므로 skel 재빌드**, 안 하면 첫 콜 `AEE_EBADPARM`.
  Step 1은 **바꾸지 않는다** — 양자화기만 바뀐다.
- 호스트 체크: `bash test/htp/host/run_host_checks.sh`(통과 확인됨),
  `build/test/unittest/unittest_nntrainer_cpu_backend --gtest_filter='*wh_*'`(통과 확인됨).

기준값 (전부 켬, 2026-09-22): prefill 585~616, decode 24.2, ppl 62.0916, peak RSS 5,252 MB.
세션 간 ±25 ms는 기기 편차 — A/B는 같은 세션에서. decode TPS < 30이면 스로틀이므로
`transport`/`host`/`total`은 비교에 쓰지 않는다 (44 §11.3).

## 8. 다른 세션의 첫 프롬프트 (복사해서 쓰기)

```
nntrainer 저장소, 브랜치 claude/htp-lfm2-moe-ffn. 먼저 CLAUDE.md →
docs/htp_attention/00_START_HERE.md → docs/htp_attention/54_w2a8_experts_task.md,
그리고 54가 가리키는 52 §6~§7(측정 절차, 커밋 형식, ponytail 규칙)을 읽어.
과제: MoE expert를 2-bit으로 굽고 VTCM에서 int4로 펼치는 것(54 §0). Step 0과 Step 1의
코드는 이미 들어가 있으니 §5.1·§5.2의 명령을 나에게 주고, 내가 기기에서 돌려 로그를
주면 §5의 표를 채우고 다음 단계를 정해. 둘 중 하나라도 게이트를 못 넘으면 Step 2~4는
하지 말고 문서에 이유만 남겨. 정확도 지표는 ppl 하나이고 텍스트 비교는 지표가 아니다.
가설에 코드 쓰기 전에 분해를 재라 — 51 §2.25~2.27과 44 §10~§11이 왜 그런지의 기록이다.
```
