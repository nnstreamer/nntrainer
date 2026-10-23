# CausalLM Reference Tests — Android Verification Report

**Branch:** `test/reference`  
**Date:** 2026-06-18

| Item | Value |
|------|-------|
| Device | Samsung `R3KYC06DR6R` (arm64-v8a) |
| Android API | 29 |
| NDK | r26d |
| ABI | arm64-v8a (`armv8.2-a+fp16+dotprod+i8mm`) |
| FP16 | Enabled (`-DENABLE_FP16=1`) |
| Host | Ubuntu 22.04 x86-64 |

---

## Summary

55/55 tests pass on Android after two bug fixes. See [TESTING.md](../Applications/CausalLM/TESTING.md) for the full run procedure.

| Result | Count |
|--------|-------|
| **PASSED** | **55** |
| SKIPPED | 0 |
| FAILED | 0 |
| **Total** | **55** |

---

## Background

The `test/reference` branch adds `unittest_causallm_models` (55 gtest tests). These tests were **not reachable on Android** before this work:

- Root `meson.build:736` skips `subdir('Applications')` entirely when `platform == 'android'`, so the meson test target never builds.
- `Applications/CausalLM/meson.build:250` lists `'android'` in `model_unittest_platforms`, but that code is **dead** — it is never reached from a meson Android build.
- `jni/Android.mk` only builds `causallm_core`, `nntr_quantize`, `test_api`, and `nntr_safetensors_info` — no gtest, no test sources.

To run the tests on Android, three changes were required:

1. Vendor `googletest` into `jni/` and add a `unittest_causallm_models` ndk-build target to `jni/Android.mk`.
2. Fix a DeBERTa-v2 FP16 overflow bug in `nntrainer/layers/layer_normalization_layer.cpp`.
3. Fix a tokenizer path resolution bug in `Applications/CausalLM/quantize.cpp`.

### Android runtime constraints

These constraints affect test execution and are handled by environment variables at run time. No test source changes were needed.

| Constraint | Detail | Resolution |
|------------|--------|------------|
| `findFixtureDir()` resolves via `__FILE__` | Host build path; invalid on device | Set `NNTRAINER_CAUSALLM_FIXTURE_DIR` |
| `runQuantize()` calls `std::system()` | Subprocess needs binary on device | Set `NNTR_QUANTIZE_BIN` |
| `std::filesystem::temp_directory_path()` | May return inaccessible path | Set `TMPDIR` |
| `nntr_config.json` `tokenizer_file` | Stored as absolute host path | Fixed in `quantize.cpp` (see Bug 2) |

---

## Test Results

### Passed — FP32 differential (10 tests)

Compare nntrainer outputs against HuggingFace reference logits.

| Test | Model |
|------|-------|
| `Gemma3DifferentialTest.FP32MatchesHFReference` | Gemma3 |
| `Gemma4DifferentialTest.FP32MatchesHFReference` | Gemma4 |
| `Qwen3MoeDifferentialTest.FP32MatchesHFReference` | Qwen3 MoE |
| `Qwen2DifferentialTest.FP32MatchesHFReference` | Qwen2 |
| `Qwen3DifferentialTest.FP32MatchesHFReference` | Qwen3 |
| `Qwen3EmbeddingDifferentialTest.FP32MatchesHFReference` | Qwen3 Embedding |
| `Qwen2EmbeddingDifferentialTest.FP32MatchesHFReference` | Qwen2 Embedding |
| `KalmEmbeddingDifferentialTest.FP32MatchesHFReference` | KaLM Embedding |
| `EmbeddingGemmaDifferentialTest.FP32MatchesHFReference` | Embedding Gemma |
| `DebertaV2DifferentialTest.FP32MatchesHFReference` | DeBERTa-v2 |

### Passed — Q4_0 differential (9 tests)

On-the-fly quantization via `nntr_quantize`, then cosine/logit error bound check.

| Test | Model |
|------|-------|
| `Gemma3DifferentialTest.Q40CloseToFP32Reference` | Gemma3 |
| `Gemma4DifferentialTest.Q40CloseToFP32Reference` | Gemma4 |
| `Qwen2DifferentialTest.Q40CloseToFP32Reference` | Qwen2 |
| `Qwen3DifferentialTest.Q40CloseToFP32Reference` | Qwen3 |
| `Qwen3EmbeddingDifferentialTest.Q40CloseToFP32Reference` | Qwen3 Embedding |
| `Qwen2EmbeddingDifferentialTest.Q40CloseToFP32Reference` | Qwen2 Embedding |
| `KalmEmbeddingDifferentialTest.Q40CloseToFP32Reference` | KaLM Embedding |
| `EmbeddingGemmaDifferentialTest.Q40CloseToFP32Reference` | Embedding Gemma |
| `DebertaV2DifferentialTest.Q40CloseToFP32Reference` | DeBERTa-v2 |

### Passed — tiny-model deterministic (36 tests)

Random weights; checks argmax token, logit values, and weight round-trip.

| Test class | Model | Dtypes |
|------------|-------|--------|
| `Gemma3TinyModelTest` | Gemma3 | FP32, Q4_0-FP32 |
| `Gemma4TinyModelTest` | Gemma4 | FP32, Q4_0-FP32 |
| `Qwen3MoETinyModelTest` | Qwen3 MoE | FP32 |
| `Qwen3SlimMoETinyModelTest` | Qwen3 Slim MoE | FP32 |
| `Qwen3CachedSlimMoETinyModelTest` | Qwen3 Cached Slim MoE | FP32 |
| `GptOssTinyModelTest` | GPT-OSS | FP32 |
| `GptOssCachedSlimTinyModelTest` | GPT-OSS Cached Slim | FP32 |
| `Qwen2CausalLMTinyModelTest` | Qwen2 | FP32, Q4_0-FP32 |
| `CausalLMTinyModelTest` (Qwen3) | Qwen3 | FP32, Q4_0-FP32 |
| `EmbeddingGemmaTinyModelTest` | Embedding Gemma | Q4_0 bidirectional |
| `Qwen25EmbeddingTinyModelTest` | Qwen2.5 Embedding | Q4_0 bidirectional |
| `Qwen3EmbeddingTinyModelTest` | Qwen3 Embedding | Q4_0 vocab remainder |

---

## Bug Fixes

### Bug 1: DeBERTa-v2 FP16 overflow in LayerNorm

**File:** `nntrainer/layers/layer_normalization_layer.cpp`

#### Symptom

`DebertaV2DifferentialTest.FP32MatchesHFReference` throws at `embedPrompt()`:

```
std::runtime_error: "ERROR : rms_norm_wrt_width_fp16_intrinsic(float *) is deprecated due to overflow in fp16"
```

#### Root cause

`deberta_v2.cpp` uses nntrainer's built-in `"layer_normalization"` layer.
Inside `layer_normalization_layer.cpp`, the `#else` branch (active when `ENABLE_FP16` is defined)
called `rms_norm_wrt_width_fp16_intrinsic(float*, ...)`:

```cpp
// layer_normalization_layer.cpp (before fix)
#else  // ENABLE_FP16
  if (deviation.getDataType() == FP32 && width_axis_only) {
    nntrainer::rms_norm_wrt_width_fp16_intrinsic(src, dst, row_count, W, epsilon);
  }
```

On ARM with FP16 enabled, the `float*` overload of this function is intentionally deprecated
(`neon_impl_fp16.cpp:2257`) because it accumulates squared values in FP16, which overflows for
typical LayerNorm input ranges. DeBERTa-v2's LayerNorm inputs exceed FP16 max (~65504).

The CausalLM custom `rms_norm.cpp` and `reshaped_rms_norm.cpp` already guard against this:
```cpp
// DO NOT USE rms_norm_wrt_width_fp16_intrinsic. It causes overflow!
```

#### Fix

Switch to the FP32-accumulator variant — the same fix already applied to the CausalLM layers:

```diff
-nntrainer::rms_norm_wrt_width_fp16_intrinsic(src, dst, row_count, W, epsilon);
+// DO NOT USE rms_norm_wrt_width_fp16_intrinsic here — it accumulates
+// squared values in FP16, which overflows for typical LayerNorm inputs
+// (e.g. DeBERTa). Use the FP32-accumulator variant instead.
+nntrainer::rms_norm_wrt_width_fp32_intrinsic(src, dst, row_count, W, epsilon);
```

| | Before | After |
|-|--------|-------|
| `DebertaV2DifferentialTest.FP32MatchesHFReference` | FAILED | **PASSED** |
| All other tests | unchanged | unchanged |

---

### Bug 2: Q4_0 tests fail — `nntr_quantize` cannot open host-absolute tokenizer path

**File:** `Applications/CausalLM/quantize.cpp`

#### Symptom

Q4_0 differential tests fail with `nntr_quantize` printing:

```
[!] FATAL ERROR: Failed to open file: /home/jwon/.../causallm_reference/gemma3_tiny/tokenizer.json
```

Affected: Gemma3, Gemma4, Qwen2, Qwen3 Q4_0 differential tests (FAIL).  
Embedding Q4_0 tests called `GTEST_SKIP()` instead of `ASSERT_TRUE`, so they showed SKIP rather than FAIL.

#### Root cause

Fixture `nntr_config.json` files store `tokenizer_file` as an absolute host path:

```json
"tokenizer_file": "/home/jwon/.../causallm_reference/gemma3_tiny/tokenizer.json"
```

The test framework (`causallm_test_utils.cpp::loadFixtureConfigs()`) overrides this field
before model construction, so FP32 inference tests work correctly:

```cpp
// causallm_test_utils.cpp:465
fc.nntr_cfg["tokenizer_file"] = (dir / "tokenizer.json").string();
```

However, `nntr_quantize` runs as a **separate process** via `std::system()`.
It reads `nntr_config.json` directly and sees the absolute host path.
`quantize.cpp` had path resolution logic only for relative paths:

```cpp
// quantize.cpp (before fix)
std::filesystem::path p = nntr_cfg[key].get<std::string>();
if (p.is_relative())                     // absolute paths pass through unchanged
  nntr_cfg[key] = (model_path / p).string();
```

On Android the host path does not exist, so `transformer.cpp::LoadBytesFromFile()` throws.

#### Fix

Add a fallback for absolute paths that are not accessible in the current environment:
resolve by filename relative to `model_path`.

```diff
 if (p.is_relative()) {
   nntr_cfg[key] = (std::filesystem::path(model_path) / p).string();
+} else if (!std::filesystem::exists(p)) {
+  // Absolute path from another environment (e.g. host path on Android device) —
+  // resolve by filename relative to the model directory.
+  nntr_cfg[key] = (std::filesystem::path(model_path) / p.filename()).string();
 }
```

On the host the absolute path exists, so existing behaviour is preserved.
On Android, `tokenizer.json` lives next to `nntr_config.json` in the fixture directory,
so it is found correctly after the fallback.

| | Before | After |
|-|--------|-------|
| Gemma3/4, Qwen2/3 Q4_0 differential | FAILED (4) | **PASSED** |
| Embedding Q4_0 differential | SKIPPED (5) | **PASSED** |
| All other tests | unchanged | unchanged |
