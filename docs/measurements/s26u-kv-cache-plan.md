# S26 Ultra KV cache 저장·로드 데모 — 구현 계획 (2026-10-13)

목표(사용자): prefill을 한 번 돌려 KV cache를 파일로 저장하고, 데모에서는 그 파일을 로드한 뒤
필요하면 짧은 prefill을 더 하고 60토큰을 생성한다. 비교 대상은 같은 질문을 cache 없이 60토큰
생성한 것. 저장해 둘 cache는 **2k 하나, 22–23k 두 개**(문서 세 편). NPU는 로드된 cache를
FastRPC 공유 메모리로 읽어야 한다.

실행자: 다음 세션(Opus). 이 문서는 코드를 읽어 쓴 것이고 **기기 측정값은 하나도 없다.**
"기기 미측정"이라고 적힌 숫자는 전부 코드와 기존 측정에서 유도한 추정이다.

기기: `R5KL308PALH` (SM-S948N, user build, 세 대 중 새 폰). 모델은
`/data/local/tmp/nntrainer/causallm/model/gemma4-26b-a4b/` (md5 `426437624d1aaee9abd3fb242b2e8b6b`,
10-13 push 완료). 브랜치 `htp/s26-kv-dump` = PR 4416 head `48a6bbb22`. 빌드·측정 규칙은
`s26u-optimization-log.md` §6, 환경은 메모리 `workstation-htp-env`.

## 0. 이미 있는 것 — 다시 만들지 말 것

| 단계 | 코드 | 비고 |
|---|---|---|
| 저장 | `Applications/CausalLM/models/causal_lm.cpp:665` SAVE 모드. config `system_prompt.kvcache.pre_computed_cache_path`의 파일이 **없으면** `head_prompt`만 prefill → `kv_cache.save(path, rows)` → 종료 | `prompt_ = system_prompt`만 인코딩 |
| 로드 | `:701 load_kvcache()` → `KVCacheManager::load()` (`kv_cache_manager.cpp`) 층별 fp16 K/V 슬랩에 fread, `setKVCachePosition(rows)` | 파일이 **있으면** USE 모드 |
| NPU가 읽음 | 슬랩은 `installKVCacheSharedAllocator()`(`:155`)로 **rpcmem(ION)** 할당. prefill attention은 `sdpa_q2_kvcache`가 그 포인터의 행을 DSP에 append(`mha_core.cpp:1288`), decode 첫 토큰은 `decode_kv_seed_f16`이 rpcmem 행을 복사 없이 seed(`24a2fe91a`) | 파일을 직접 DSP에 매핑하는 길은 없다: fastrpc_mmap은 DMA-BUF만 받는다. rpcmem으로 한 번 읽는 지금 구조가 맞다 |
| 로드 뒤 추가 prefill | `setCacheIndex()`(`mha_core.h:337`)가 DSP int8 미러를 stale로 → 다음 attention 호출이 행 전부(SYS+n)를 다시 append, q2 scale도 그 전체로 보정(`mha_core.cpp:1283-1300`). `prefill_from = SYS_PROMP_LEN`, INIT_SEQ_LEN 단위 chunk(`:713`) | |
| 토큰 생성 | 평소 E2E decode | |

토크나이저는 `add_special_tokens=false`(`huggingface_tokenizer.cpp:45`)라 프롬프트 안의 `<bos>`
문자열이 토큰이 된다. 뒷부분을 따로 인코딩해도 BOS가 덧붙지 않는다. `sys_prompt_token_size`는
안 적어도 된다(USE 모드가 같은 토크나이저로 `head_prompt`를 센다, `:584`).

작업 트리에 아직 커밋 안 된 한 줄: `causal_lm.cpp` USE 모드의 `[TIMING] kv_load_ms=… rows=…`
(stderr). Phase 1 커밋에 넣는다.

## 1. Phase 1 — 2k (기존 경로 검증, 코드 거의 없음)

### 1.1 코드 (커밋 1개, `[CausalLM]`)

- `KVCacheManager::load`: 파일 크기 ≠ Σ층(rows × width × 2 B × 2)이면 예외. 지금은 다른
  모델·길이의 파일을 조용히 읽는다. 신뢰 경계라 생략하지 않는다.
- `[TIMING] kv_load_ms` 한 줄 (위).

### 1.2 기기 검증 — 한 번도 안 돌린 경로 둘

1. **SAVE 모드가 HTP E2E 세션을 연 채 decode 없이 종료**하는 경로. 걸리면 `return` 전에
   세션 정리를 호출하는 것으로 끝나야 한다.
2. **M = 20–60행, from = 2048인 HTP prefill.** 지금까지 최소 M은 1024. qkv·MoE·attention 커널의
   행 타일 가정(32행 등)에 걸릴 수 있다. 걸리는 커널에서만 고친다.
3. 2048행 재append 30층의 비용(`NNTR_HTP_ATTN_TRACE=1`로 `q2 calibration rows=` 확인), 첫 토큰
   seed 정상, 생성 문장이 문서를 참조하는지.

### 1.3 데모 자산 (이미 있음, `~/workspace/s26u-work/`)

- `demo/demo_kv2k/nntr_config.json`: `head_prompt` = `docs/measurements/260-prompt2048.txt`의
  `</Text>`까지 + `\n`, `sample_input` = 질문, `tail_prompt` = `<turn|>\n<|turn>model\n`,
  `kvcache.pre_computed_cache_path` = `$D/demo_kv2k/kv2k.bin`, `num_to_generate` 60,
  `max_seq_len` 2304, `init_seq_len` 2048. 덤프 440 MiB.
- `demo/demo_nokv`: `<bos><|turn>user\n질문<turn|>\n<|turn>model\n`만. `demo/demo_full2k`: 문서+질문 통째.
- `demo/generation_config.json`: greedy, eos 비활성(정확히 60토큰).
- `demo.sh inst <tag>` / `cfg` / `save <tag>` / `run <kv2k|nokv|full2k> <tag> [env]`. 기본 env는
  `s26u-prefill-handoff.md` §5.2와 같고 C=32.

순서: `demo.sh inst kv` → `demo.sh cfg` → `demo.sh save kv`(SAVE) → `demo.sh run kv2k kv`(USE) →
`run nokv kv` → `run full2k kv`. 번갈아 3회, 2분 냉각. 보고: `kv_load_ms`, `prefill:`,
`first_token_ms`, `rest_avg_ms`(60토큰이면 `generation(last 64)`가 첫 토큰을 포함하므로 TPS는
이것으로 읽는다), 답 문장. 정확도 게이트는 full2k와 내용 일치(토크나이저 이음새와 HTP attention의
누적 순서 때문에 바이트 동일은 요구하지 않는다).

## 2. Phase 2 — 22–23k 두 개: 왜 그냥은 안 되나

`S = 23,552`(23k + 추가 prefill·생성 여유, 64 배수)로 잡으면 지금 코드는 세 곳에서 **모든 층에
S행**을 잡는다. Gemma-4는 30층 중 25층이 sliding window 1024(`config.json layer_types`,
`sliding_window 1024`), 5층(5·11·17·23·29)이 full attention(kv 2헤드 × hd 512 = 폭 1024, sliding은
8 × 256 = 2048)이다.

| 할당 | 지금(S행 × 30층) | 필요한 것(sliding은 window 크기만) |
|---|---|---|
| CPU fp16 cache, rpcmem (`KVCacheManager::allocate`, 모든 층 `max_seq_len`행) | **5,060 MiB** | full 5 × S × 1024 × 4 B = 460 + sliding 25 × R_s × 2048 × 4 B = 625 (R_s = 3,200) → **1,085 MiB** |
| prefill DSP int8 미러 (`kv_cache_q_register(max_rows = cache_key_dim.height())`, `mha_core.cpp:1239,1272`) | 2,530 MiB | 545 MiB |
| decode DSP attn_m1 cache (`htp_compute_ops.cpp:3844`, `max_seq` = `graph_words_[6]`, 두 cache 모두) | 1,265 MiB (int8) | full 115 + sliding 25 × R_s × 2048 = 156 → 271 MiB |
| 덤프 파일 (fp16, 살아 있는 행만) | — | 460 + 25 × 1024 × 2048 × 4 = 200 → **660 MiB**, 로드 ≈ 0.4 s (기기 미측정) |

폰 RAM 11.1 GB, DSP 매핑 한도 ≈ 3,136 MiB(log §3.2)라 왼쪽 열은 불가능하고 오른쪽 열이 목표다.
R_s = window 1024 + INIT_SEQ_LEN 2048 + 여유 128 = 3,200: 덤프를 **만드는** 실행이 2048행 chunk를
window 뒤에 이어 써야 해서 이 크기가 필요하다(데모 실행만이면 1,216이면 된다; 하나로 간다).

그리고 prefill attention에 벽이 하나 있다: `hexkl_attn_q2_plan.h:160`
`n_blk > HVX_SOFTMAX_Q_MAX_TILES(128)` → 32행 블록이 보는 컬럼이 4096을 넘으면 `EBADPARM`.
sliding 층은 window + 2048 = 3,072 컬럼이라 안 걸리고, **full 층 5개는 cache_to > 4096부터
HTP attention이 거절**한다 → `try_quantized_attention`이 `fail()`로 false → mha_core의 CPU fp16
attention(이 층은 영구 CPU). 덤프 생성은 1회성이라 느려도 되지만 §2.3의 확인이 필요하다.

### 2.1 설계: "층별 행 수 + 행 base" (ring 아님)

sliding 층의 cache를 ring으로 만들면 DSP 커널 셋(`hexkl_attn_q2`, `hvx_attn_m1`, seed)의 행
주소 계산이 전부 바뀐다. 대신 **행 base 오프셋**을 둔다: 층 l의 절대 위치 a는 행
`a − base_l`에 있고, base_l은 chunk 사이에서만 바뀐다. 커널은 상대 행 번호만 받으므로 causal·window
마스크는 그대로 맞는다(질의 행과 키 컬럼이 같은 양만큼 이동). **RoPE 위치는 절대 위치**를 써야
한다 — `mha_core.cpp:1542,1565,1658-1665`의 `apply_rotary_emb_tensor_v2(..., cache_index/_from, ...)`는
그대로 두고, 슬롯 계산만 `cache_index − base_`로 바꾼다. decode의 ROPE op는 `call->pos`(절대)를
쓰고 ATTN_M1만 `pos − base`를 쓰면 된다.

바뀌는 곳:

| # | 어디 | 무엇 |
|---|---|---|
| A | `KVCacheManager` | `allocate(..., kv_widths)`에 층별 행 수 `rows_l` 추가(sliding = R_s, full = S). `getPosition/advance`는 절대 위치 그대로. write/read view는 `pos − base_l`. 층별 `base_l` 멤버 + `setBase(l, b)`. `save/load`: 헤더 {magic, n_layers, 층별 (rows_live, width), dtype} + 살아 있는 행만(sliding: 마지막 min(pos, 1024)행, full: pos행). 로드 뒤 `base_l = pos − rows_live_l` |
| B | `Gemma4CausalLM::allocateAndBindKVCache` (`gemma4_causallm.cpp:1017`) | 층별 rows: `isSlidingAttentionLayer(l) ? R_s : S`. placeholder 차원(`createGemma4KVCachePlaceholders`)도 같은 행 수 |
| C | `MHACoreLayer` | `cache_base` 멤버(기본 0), `setCacheIndex`처럼 세터. 슬롯을 쓰는 모든 자리(`b_cache_key_step` 뷰, `try_quantized_attention`의 `cache_from/to/off`, `:1239 max_rows` 검사, CPU 경로의 `end_row` 계산 `:2051…`)에서 `cache_index − cache_base`. RoPE 인자는 절대 유지. `htpDecodeAttn` seed(`:712`)는 행 [0, pos−base) |
| D | `CausalLM::prefill` 람다 (`causal_lm.cpp:713`) | chunk k를 시작할 때 sliding 층마다 `new_base = max(0, p0 − 1024)`(p0 = chunk 시작 절대 위치); base가 바뀌면 마지막 1024행을 슬랩 앞으로 memmove(25층 × 1024 × 4 KiB = 100 MiB, 수 ms)하고 그 층 `setCacheIndex`(→ `q_cache_synced = 0`, 미러 재append 1024 + 2048행). full 층은 base 0 고정 |
| E | DSP decode: `hvx_attn_m1_ctx`에 `row_base`(cache별, full은 0). `nntr_hvx_attn_m1_register`에 행 수를 cache별로(sliding R_s, full S), `hvx_attn_m1_forward_prof(…, pos, …)`와 `kv_append(_f16)(…, kv_from, …)` 안에서 `pos − row_base`. IDL(`test/htp/nntr_hvx.idl`) 인자 추가 → `generate_stub.sh` | 호스트 쪽 `htp_compute_ops.cpp:3844`에서 두 cache의 행 수와 base를 넘김. base는 데모 실행에서 로드 직후 한 번 정해지면 생성 중 바뀌지 않는다(R_s 안에서 끝남) |
| F | config | `max_seq_len` S, `init_seq_len` 2048, `moe_cache_experts`는 §2.4 예산에 맞춰 16(또는 8) |

E가 유일한 DSP 변경이고 뺄셈 두 곳 + 등록 인자다. A–D는 호스트.

### 2.2 덤프 생성 실행 (문서당 1회, 오프라인)

`demo_kv23k_a`, `demo_kv23k_b` config의 SAVE 모드. **SAVE 모드는 chunk를 안 쓴다**(`:681`
`incremental_inference(..., 0, input_len)` 한 번) → §2.1 D의 `prefill()` 람다를 쓰도록 바꾼다(한 줄).
예상 시간(기기 미측정): HTP 쪽 23.5k / ≈600 TPS ≈ 40 s + full 층 5개 CPU fp16 attention
O(n²) ≈ 11 TFLOP → 수십 초–수 분. 결과 파일 660 MiB × 2.

### 2.3 확인해야 하는 위험 (만들기 전에)

| # | 위험 | 확인 방법 |
|---|---|---|
| R1 | full 층의 CPU attention fallback이 **HTP o-proj와 맞물리는가.** o-proj는 u16 context를 DSP 버퍼에서 읽는다(`attn_out_u16`, `mha_core.cpp:1353,1386`). attention이 CPU로 떨어지면 `io.out`(f32)에 쓰는데 o-proj가 그걸 읽는 경로가 살아 있는지. `fail()` 뒤 `attn_out_u16` 층의 처리를 읽고, 안 되면 그 층만 `attn_out_u16=false`로 빌드(`gemma4_causallm.cpp` 빌더에서 full 층 분기) | 호스트 코드 읽기 → 기기 p4096+ 실행 |
| R2 | `0x80000402`(원격 ENOMEM) — log §2.6: `max_seq_len ≥ 2560`·C ≥ 24에서 호스트 RAM 바닥에 따른 상태 의존 실패. S = 23.5k는 rpcmem 1.1 GB + DSP cache 0.8 GB가 더 든다 | §2.4 예산으로 C 결정, `NNTR_MOE_CACHE_GROW` 끔 |
| R3 | decode 그래프 `max_seq`(`graph_words_[6]`) = S: RoPE 테이블 `[max_seq][64]` f32 = 6 MiB, `pos >= max_seq` 검사(`hexkl_graph.h:346`) — 문제 없을 것 | 로그 `attn_m1: registered … max_seq=` |
| R4 | `kv_cache_q_register(max_rows ≤ 0xFFFF)` (`htp_compute_ops.cpp:9454`) — 23.5k OK | — |
| R5 | prefill RoPE 테이블 층당 M × 2 × hd f32(`htp_compute_ops.cpp:1455`), chunk마다 다시 → 변화 없음 | — |
| R6 | 로드 뒤 추가 prefill(짧은 질문)에서 sliding 층 미러 재append는 1024 + n행, full 층은 **23.5k행** × 5층 → 4096 컬럼 벽에 걸려 full 층은 CPU attention(R1과 같은 경로, 질문 n행 × 23.5k 컬럼이라 빠름) | R1이 풀리면 자동 |
| R7 | decode의 full 층 attention이 23.5k 행을 읽음: 5 × 23.5k × 1024 B = 118 MiB/토큰 → 토큰당 +3–6 ms (기기 미측정) | `rest_avg_ms` 비교 |

### 2.4 메모리 예산 (DSP 매핑 ≈ 3,136 MiB 한도, log §3.2)

| 항목 | C=32 | C=16 | C=8 |
|---|---|---|---|
| expert 슬롯 1.45 MiB × 30 × C | 1,392 | 696 | 348 |
| FC·dense WH + lm_head Q4M1 | 768 + 448 | 768 + 448 | 768 + 448 |
| decode attn_m1 int8 (§2) | 271 | 271 | 271 |
| prefill 미러 int8 (첫 토큰에 `FREE_KVQ`로 해제, prefill 중엔 공존) | 545 | 545 | 545 |
| heap(p2048 기준 686, 길이에 따라 ↑) | ≥ 686 | ≥ 686 | ≥ 686 |
| 합 | **4,110** ✗ | **3,414** ✗(빡빡) | **3,066** |

→ Phase 2는 **C=8로 시작**, 로드 성공하면 16을 시험. decode TPS는 C 8→32에서 1 TPS 안이다(log §2.6
읽는 법 3). 호스트 쪽은 rpcmem fp16 cache 1.1 GB가 더 들고 `NNTR_HTP_KV_DROP_CPU=1`이 seed 뒤
돌려준다.

### 2.5 입력

22–23k 토큰 문서 두 편은 **사용자가 준다.** 각각 `head_prompt`에 `<bos><|turn>user\n…문서…\n`
형태로 넣는다(JSON 문자열 100 KB, 문제 없음). 질문은 `sample_input`.

## 3. 순서와 게이트

1. Phase 1 끝까지(§1). 커밋 1개. 데모 숫자 3 × 3회.
2. §2.3 R1 확인(코드 읽기 → p4096 config로 full 층 fallback 1회 실행).
3. A·B·C(호스트, 층별 행 수 + base) — 커밋 1개. **게이트: 2k 데모의 답 md5가 1번과 동일**
   (base = 0, 행 수 = 종전이면 바이트 동일해야 한다).
4. D(chunk 사이 compaction) + SAVE 모드 chunk — 커밋 1개. 게이트: p4096 config(2 chunk)의 답이
   종전 md5 `1b462136`과 동일(sliding 층은 base가 2048−1024로 바뀌지만 보는 행은 같다).
5. E(DSP base) — 커밋 1개. host check: `test/htp/host/`에 attn_m1 base ≠ 0 케이스 하나(base 0과
   같은 결과). 게이트: 2k·4k md5 불변.
6. 23k 덤프 생성 2회(§2.2), 로드 데모, 번갈아 3회.

커밋 형식: 히스토리대로 `[CausalLM] …` / `[HTP] …` / `[docs] …`, 본문에 무엇·왜와
`Validation:` 줄, `git commit -s`, `Co-authored-by:`만, 세션 링크는 넣지 않는다.
`clang-format-14`는 바뀐 줄만. `subprojects/`는 건드리지 않는다.

## 4. 하지 않는 것

- ring buffer: base 오프셋이 같은 메모리 효과를 호스트 코드 위주로 낸다.
- 파일 mmap을 DSP에 직접 매핑: fastrpc가 일반 파일을 받지 않는다.
- softmax 타일 한도(128) 상향: VTCM 예산의 벽이고 full 층 5개뿐이라 CPU fallback으로 간다
  (R1이 막히면 그때 다시 본다).
- 두 23k cache를 한 프로세스에서 바꿔 끼우기: config 둘, 프로세스 재시작. 로드 0.4 s.
