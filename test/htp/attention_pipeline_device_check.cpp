// SPDX-License-Identifier: Apache-2.0
/**
 * @file attention_pipeline_device_check.cpp
 * @brief Deterministic attention outputs and timings for DSP pipeline A/B.
 */
#include <chrono>
#include <cstdio>
#include <cstring>
#include <htp_backend.h>
#include <nntr_hvx.h>
#include <stdexcept>
#include <vector>

struct SharedInputs {
  nntrainer::HtpBackend &backend;
  unsigned q_count, kv_count;
  float *q, *out;
  uint16_t *k, *v;

  SharedInputs(unsigned rows, unsigned cache_rows, unsigned qh, unsigned kvh,
               unsigned hd) :
    backend(nntrainer::HtpBackend::global()),
    q_count(rows * qh * hd),
    kv_count(cache_rows * kvh * hd) {
    q = static_cast<float *>(backend.alloc_shared(q_count * sizeof(float)));
    out = static_cast<float *>(backend.alloc_shared(q_count * sizeof(float)));
    k = static_cast<uint16_t *>(
      backend.alloc_shared(kv_count * sizeof(uint16_t)));
    v = static_cast<uint16_t *>(
      backend.alloc_shared(kv_count * sizeof(uint16_t)));
    if (!q || !out || !k || !v)
      throw std::runtime_error("shared allocation failed");
    unsigned seed = rows + hd * 71u;
    auto sample = [&]() {
      seed = seed * 1664525u + 1013904223u;
      return (float(seed >> 8) / 16777215.f - 0.5f) * 4.f;
    };
    for (unsigned i = 0; i < q_count; ++i)
      q[i] = (i / hd) % 17u ? sample() : 0.f;
    for (unsigned i = 0; i < kv_count; ++i) {
      const __fp16 a = (i / hd) % 19u ? sample() : 0.f;
      const __fp16 b = (i / hd) % 23u ? sample() : 0.f;
      std::memcpy(k + i, &a, sizeof(a));
      std::memcpy(v + i, &b, sizeof(b));
    }
  }

  ~SharedInputs() {
    backend.free_shared(q);
    backend.free_shared(out);
    backend.free_shared(k);
    backend.free_shared(v);
  }
};

static void check_call(int rc, const char *operation) {
  if (rc) {
    std::fprintf(stderr, "%s: 0x%x\n", operation, rc);
    throw std::runtime_error(operation);
  }
}

static void run_case(unsigned rows, unsigned hd, unsigned prefix,
                     unsigned window, float softcap, bool sinks_on,
                     bool benchmark, FILE *dump) {
  const unsigned qh = hd == 64u ? 32u : 8u, kvh = qh / 4u;
  const unsigned cache_rows = rows + prefix;
  SharedInputs input(rows, cache_rows, qh, kvh, hd);
  const auto session = input.backend.handle();
  uint32_t cache = 0;
  check_call(nntr_hvx_kv_register_q(session, 0u, cache_rows, kvh, hd, &cache),
             "register");
  std::vector<float> sinks(qh);
  for (unsigned h = 0; h < qh; ++h)
    sinks[h] = (int(h % 7u) - 3) / 4.f;
  std::vector<float> reference;
  const unsigned repeats = benchmark ? 12u : 2u;
  for (unsigned repeat = 0; repeat < repeats; ++repeat) {
    uint32_t stats[20] = {};
    const auto start = std::chrono::steady_clock::now();
    check_call(nntr_hvx_attn_q8_npu_step(
                 session, cache, 0u, input.k, input.kv_count, input.v,
                 input.kv_count, rows, 0u, cache_rows, qh, window, softcap,
                 input.q, input.q_count, sinks_on ? sinks.data() : nullptr,
                 sinks_on ? int(qh) : 0, input.out, input.q_count, stats, 20),
               "attention");
    const auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(
                           std::chrono::steady_clock::now() - start)
                           .count();
    if (reference.empty())
      reference.assign(input.out, input.out + input.q_count);
    else if (std::memcmp(reference.data(), input.out,
                         input.q_count * sizeof(float)))
      throw std::runtime_error("repeated output differs");
    if (benchmark && repeat >= 2u)
      std::printf("BENCH rows=%u repeat=%u wall_us=%lld dsp_us=%u "
                  "quant_us=%u append_us=%u attn_us=%u qprep_us=%u "
                  "dma_us=%u qk_us=%u dequant_us=%u softmax_us=%u "
                  "pquant_us=%u pv_us=%u oupd_us=%u store_us=%u\n",
                  rows, repeat, static_cast<long long>(wall_us), stats[3],
                  stats[4], stats[0], stats[1], stats[8], stats[9], stats[10],
                  stats[11], stats[12], stats[13], stats[14], stats[15],
                  stats[16]);
  }
  if (dump && std::fwrite(input.out, sizeof(float), input.q_count, dump) !=
                input.q_count)
    throw std::runtime_error("output write failed");
  check_call(nntr_hvx_kv_release_q(session, cache), "release");
  if (!benchmark)
    std::printf("CASE rows=%u hd=%u prefix=%u window=%u softcap=%.1f "
                "sinks=%u\n",
                rows, hd, prefix, window, softcap, unsigned(sinks_on));
}

int main(int argc, char **argv) {
  try {
    if (!nntrainer::HtpBackend::global().enabled())
      throw std::runtime_error("HTP backend unavailable");
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
      for (unsigned rows : {1024u, 2048u})
        run_case(rows, 64u, 0u, 0u, 0.f, false, true, nullptr);
      return 0;
    }
    if (argc != 3 || std::strcmp(argv[1], "--dump") != 0)
      throw std::runtime_error("use --bench or --dump output.bin");
    FILE *dump = std::fopen(argv[2], "wb");
    if (!dump)
      throw std::runtime_error("cannot open output file");
    unsigned cases = 0;
    for (unsigned hd : {32u, 64u, 96u, 128u, 256u})
      for (unsigned rows : {5u, 65u, 256u, 1024u})
        for (unsigned mode = 0; mode < 4u; ++mode) {
          run_case(rows, hd, mode & 1u ? 37u : 0u, mode & 2u ? 37u : 0u,
                   mode & 2u ? 8.f : 0.f, mode & 1u, false, dump);
          ++cases;
        }
    run_case(2048u, 64u, 0u, 0u, 0.f, false, false, dump);
    if (std::fclose(dump))
      throw std::runtime_error("output close failed");
    std::printf("ALL_ATTENTION_PIPELINE_CASES_PASSED cases=%u\n", cases + 1u);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
