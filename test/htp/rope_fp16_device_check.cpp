// SPDX-License-Identifier: Apache-2.0
/**
 * @file rope_fp16_device_check.cpp
 * @date 12 October 2026
 * @brief Device parity and timing harness for prefill RoPE with fp16 output.
 */

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <htp_backend.h>
#include <nntr_hvx.h>
#include <stdexcept>
#include <vector>

static void run(unsigned m, unsigned n, unsigned hd, bool overlap, bool bench,
                FILE *dump) {
  auto &backend = nntrainer::HtpBackend::global();
  const unsigned count = m * n, cs_count = m * 2u * hd;
  auto *x = static_cast<float *>(backend.alloc_shared(count * sizeof(float)));
  auto *cs =
    static_cast<float *>(backend.alloc_shared(cs_count * sizeof(float)));
  auto *out =
    overlap
      ? reinterpret_cast<uint16_t *>(x)
      : static_cast<uint16_t *>(backend.alloc_shared(count * sizeof(uint16_t)));
  if (!x || !cs || !out)
    throw std::runtime_error("allocation failed");
  std::vector<float> input(count);
  unsigned seed = m + 7u * n;
  for (unsigned i = 0; i < count; ++i) {
    seed = seed * 1664525u + 1013904223u;
    input[i] = (float(seed >> 8) / 16777215.f - .5f) * 20.f;
    if (i % 31u == 0u)
      input[i] = (i % 2u ? -1.f : 1.f) * (1.f + 0x1p-11f);
    if (i % 47u == 0u)
      input[i] *= 8192.f;
    if (i % 71u == 0u)
      input[i] *= 0x1p-24f;
  }
  for (unsigned r = 0; r < m; ++r)
    for (unsigned d = 0; d < hd; ++d) {
      const float angle = float(r * (d % (hd / 2u) + 1u)) / 8192.f;
      cs[r * 2u * hd + d] = std::cos(angle);
      cs[r * 2u * hd + hd + d] = std::sin(angle);
    }
  std::vector<uint16_t> ref;
  for (unsigned repeat = 0; repeat < (bench ? 14u : 2u); ++repeat) {
    std::memcpy(x, input.data(), count * sizeof(float));
    uint32_t elapsed = 0;
    const auto begin = std::chrono::steady_clock::now();
    const int rc = nntr_hvx_inference_math_timed_f32(
      backend.handle(), 5u, m, n, hd, 0.f, 1.f, x, count, nullptr, 0, cs,
      cs_count, reinterpret_cast<uint8_t *>(out), count * sizeof(uint16_t),
      &elapsed, 1);
    if (rc) {
      std::fprintf(stderr, "RPC failed 0x%x\n", rc);
      throw std::runtime_error("RoPE");
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
    if (ref.empty())
      ref.assign(out, out + count);
    else if (std::memcmp(ref.data(), out, count * sizeof(uint16_t)))
      throw std::runtime_error("repeat differs");
    if (bench && repeat >= 2u)
      std::printf("BENCH M=%u N=%u repeat=%u wall_us=%lld dsp_us=%u\n", m, n,
                  repeat, static_cast<long long>(us), elapsed);
  }
  if (dump && std::fwrite(out, sizeof(uint16_t), count, dump) != count)
    throw std::runtime_error("write");
  backend.free_shared(x);
  backend.free_shared(cs);
  if (!overlap)
    backend.free_shared(out);
  if (!bench)
    std::printf("CASE M=%u N=%u hd=%u overlap=%u\n", m, n, hd,
                unsigned(overlap));
}

int main(int argc, char **argv) {
  try {
    if (!nntrainer::HtpBackend::global().enabled())
      throw std::runtime_error("HTP unavailable");
    if (argc == 2 && std::strcmp(argv[1], "--bench") == 0) {
      for (unsigned n : {512u, 2048u})
        run(1024u, n, 64u, false, true, nullptr);
      return 0;
    }
    if (argc != 3 || std::strcmp(argv[1], "--dump"))
      throw std::runtime_error("use --bench or --dump output.bin");
    FILE *file = std::fopen(argv[2], "wb");
    if (!file)
      throw std::runtime_error("open");
    unsigned cases = 0;
    for (unsigned m : {1u, 5u, 65u, 1024u})
      for (unsigned hd : {64u, 128u})
        for (unsigned heads : {1u, 8u, 32u})
          for (bool overlap : {false, true}) {
            run(m, hd * heads, hd, overlap, false, file);
            ++cases;
          }
    if (std::fclose(file))
      throw std::runtime_error("close");
    std::printf("ALL_ROPE_FP16_CASES_PASSED cases=%u\n", cases);
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
  return 0;
}
