// SPDX-License-Identifier: Apache-2.0
/** Exact composite-vs-sequential publications, resident-cache continuation,
 * and bounded timing. Both paths invoke the original numerical kernels. */
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <htp_backend.h>
#include <nntr_hvx.h>
#include <stdexcept>
#include <string>
#include <vector>

static void require(bool yes, const char *what) {
  if (!yes)
    throw std::runtime_error(what);
}
static void call(int rc, const char *what) {
  if (rc)
    throw std::runtime_error(std::string(what) + ": " + std::to_string(rc));
}
template <typename T> struct Shared {
  T *p;
  size_t count;
  explicit Shared(size_t n) :
    p(static_cast<T *>(
      nntrainer::HtpBackend::global().alloc_shared(n * sizeof(T)))),
    count(n) {
    require(p != nullptr, "shared allocation");
  }
  ~Shared() { nntrainer::HtpBackend::global().free_shared(p); }
  Shared(const Shared &) = delete;
  Shared &operator=(const Shared &) = delete;
};
static constexpr unsigned M = 1024, QW = 2048, KW = 512;
static constexpr size_t NQ = size_t(M) * QW, NK = size_t(M) * KW;
struct Path {
  Shared<float> q{NQ}, out{NQ};
  Shared<uint16_t> k{NK}, v{NK};
  uint32_t cache = UINT32_MAX;
  Path() {
    call(nntr_hvx_kv_register_q(nntrainer::HtpBackend::global().handle(), 0,
                                1152, 8, 64, &cache),
         "register Q8");
  }
  ~Path() {
    if (cache != UINT32_MAX)
      nntr_hvx_kv_release_q(nntrainer::HtpBackend::global().handle(), cache);
  }
};

static void parity(const void *a, const void *b, size_t bytes,
                   const char *what) {
  require(std::memcmp(a, b, bytes) == 0, what);
}
static void run(unsigned seed, bool benchmark) {
  auto &backend = nntrainer::HtpBackend::global();
  const auto handle = backend.handle();
  Shared<float> rawk(NK), rawv(NK), cs(size_t(M) * 128);
  std::vector<float> rawq(NQ), savedk(NK), savedv(NK);
  const unsigned case_seed = seed;
  auto sample = [&] {
    seed = seed * 1664525u + 1013904223u;
    return (float(seed >> 8) / 16777215.0f - 0.5f) * 4.0f;
  };
  for (size_t i = 0; i < NQ; ++i)
    rawq[i] = (i / 64) % 17 ? sample() : -0.0f;
  for (size_t i = 0; i < NK; ++i) {
    rawk.p[i] = (i / 64) % 19 ? sample() : 0.0f;
    rawv.p[i] = (i / 64) % 23 ? sample() : -0.0f;
  }
  // Include finite FP16 rounding boundaries, subnormals and both zero signs.
  const uint32_t edges[] = {0,           0x80000000u, 1,          0x80000001u,
                            0x33800000u, 0x3f801000u, 0xbf801000u};
  for (size_t i = 0; i < NK; i += 257) {
    std::memcpy(rawk.p + i, edges + (i / 257) % 7, 4);
    std::memcpy(rawv.p + i, edges + ((i / 257) + 3) % 7, 4);
  }
  savedk.assign(rawk.p, rawk.p + NK);
  savedv.assign(rawv.p, rawv.p + NK);
  for (unsigned r = 0; r < M; ++r)
    for (unsigned d = 0; d < 64; ++d) {
      const float angle = float(r * (d % 32 + 1)) / 8192.0f;
      cs.p[size_t(r) * 128 + d] = std::cos(angle);
      cs.p[size_t(r) * 128 + 64 + d] = std::sin(angle);
    }
  Path separate, composite;
  auto execute = [&](Path &p, bool combined, unsigned repeat) {
    std::memcpy(p.q.p, rawq.data(), NQ * 4);
    std::memset(p.k.p, 0xa5, NK * 2);
    std::memset(p.v.p, 0x5a, NK * 2);
    std::memset(p.out.p, 0xcc, NQ * 4);
    uint32_t status[24] = {};
    const auto begin = std::chrono::steady_clock::now();
    if (combined) {
      call(nntr_hvx_attn_q8_prefill_prepare(
             handle, p.cache, M, cs.p, M * 128, p.q.p, NQ, rawk.p, NK, rawv.p,
             NK, p.k.p, NK, p.v.p, NK, p.out.p, NQ, status, 24),
           "composite outer status");
      require(status[0] == 2 && status[1] == 0 && status[2] == 0,
              "composite completion status");
    } else {
      call(nntr_hvx_rope_qkv_prefill_f32(
             handle, M, QW, KW, 64, cs.p, M * 128, p.q.p, NQ, rawk.p, NK,
             rawv.p, NK, p.q.p, NQ, p.k.p, NK, p.v.p, NK, status + 3, 1),
           "sequential prepare");
      call(nntr_hvx_attn_q8_npu_step(handle, p.cache, 0, p.k.p, NK, p.v.p, NK,
                                     M, 0, M, 32, 0, 0.0f, p.q.p, NQ, nullptr,
                                     0, p.out.p, NQ, status + 4, 20),
           "sequential attention");
    }
    const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                      std::chrono::steady_clock::now() - begin)
                      .count();
    if (benchmark && repeat >= 2)
      std::printf("BENCH seed=%u mode=%s repeat=%u wall_us=%lld prepare_us=%u "
                  "attention_us=%u\n",
                  case_seed, combined ? "composite" : "separate", repeat,
                  static_cast<long long>(us), status[3], status[7]);
  };
  const unsigned repeats = benchmark ? 8 : 1;
  for (unsigned repeat = 0; repeat < repeats; ++repeat) {
    if (repeat & 1) {
      execute(composite, true, repeat);
      execute(separate, false, repeat);
    } else {
      execute(separate, false, repeat);
      execute(composite, true, repeat);
    }
    parity(separate.q.p, composite.q.p, NQ * 4, "Q FP32 bits");
    parity(separate.k.p, composite.k.p, NK * 2, "public K FP16 bits");
    parity(separate.v.p, composite.v.p, NK * 2, "public V FP16 bits");
    parity(separate.out.p, composite.out.p, NQ * 4, "prefill output bits");
    parity(rawk.p, savedk.data(), NK * 4, "raw K preservation");
    parity(rawv.p, savedv.data(), NK * 4, "raw V preservation");
  }
  // Consume the two resident Q8 caches with identical existing entry points.
  // This checks internal append side effects through their normal consumers.
  Shared<float> nextq(17 * QW), nextout0(17 * QW), nextout1(17 * QW);
  Shared<uint16_t> nextk(17 * KW), nextv(17 * KW);
  for (size_t i = 0; i < nextq.count; ++i)
    nextq.p[i] = sample();
  for (size_t i = 0; i < nextk.count; ++i) {
    const __fp16 k = sample(), v = sample();
    std::memcpy(nextk.p + i, &k, 2);
    std::memcpy(nextv.p + i, &v, 2);
  }
  uint32_t stats[20];
  for (unsigned which = 0; which < 2; ++which)
    call(nntr_hvx_attn_q8_npu_step(
           handle, which ? composite.cache : separate.cache, M, nextk.p,
           17 * KW, nextv.p, 17 * KW, 17, M, M + 17, 32, 0, 0.0f, nextq.p,
           17 * QW, nullptr, 0, which ? nextout1.p : nextout0.p, 17 * QW, stats,
           20),
         "resident Q8 continuation");
  parity(nextout0.p, nextout1.p, 17 * QW * 4, "continuation output bits");
  for (unsigned which = 0; which < 2; ++which)
    call(nntr_hvx_attn_q_step(handle, which ? composite.cache : separate.cache,
                              M + 17, nextk.p, KW, nextv.p, KW, 1, M + 17,
                              M + 18, 32, 0, 0.0f, nextq.p, QW, nullptr, 0,
                              which ? nextout1.p : nextout0.p, QW, stats, 20),
         "resident Q8 decode");
  parity(nextout0.p, nextout1.p, QW * 4, "decode output bits");
  std::printf("CASE seed=%u Q_FP32=exact KV_FP16=exact raw_KV=unchanged "
              "prefill=exact continuation17=exact decode1=exact\n",
              case_seed);
}

int main(int argc, char **argv) {
  try {
    const bool bench = argc == 2 && std::strcmp(argv[1], "--bench") == 0;
    require(argc == 1 || bench, "use [--bench]");
    require(nntrainer::HtpBackend::global().enabled(), "HTP unavailable");
    for (unsigned seed : {17u, 29u, 83u})
      run(seed, bench);
    std::puts("ALL_ATTENTION_COMPOSITE_CASES_PASSED cases=3");
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "FAIL: %s\n", e.what());
    return 1;
  }
}
