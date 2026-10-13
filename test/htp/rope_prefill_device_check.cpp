// SPDX-License-Identifier: Apache-2.0
// Compare the fused prepare with the existing three public math calls.
#include <chrono>
#include <cmath>
#include <compute_ops.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <htp_backend.h>
#include <htp_rope_prefill.h>
#include <nntr_hvx.h>
#include <stdexcept>
#include <string>
#include <vector>

struct Buffer {
  void *base;
  bool shared;
  Buffer(size_t bytes, bool ion) : shared(ion) {
    base = shared ? nntrainer::HtpBackend::global().alloc_shared(bytes)
                  : std::malloc(bytes);
    if (!base)
      throw std::runtime_error("allocation");
  }
  ~Buffer() {
    if (shared)
      nntrainer::HtpBackend::global().free_shared(base);
    else
      std::free(base);
  }
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
};

static unsigned cases = 0, invalid = 0;
static void require(bool yes, const char *what) {
  if (!yes)
    throw std::runtime_error(what);
}

static void run(unsigned rows, unsigned qw, unsigned kw, unsigned position,
                unsigned offset, bool shared, const std::string &bench,
                FILE *dump) {
  const size_t nq = static_cast<size_t>(rows) * qw;
  const size_t nk = static_cast<size_t>(rows) * kw;
  Buffer qb((nq + 64) * 4, shared), kb((nk + 64) * 4, shared);
  Buffer vb((nk + 64) * 4, shared), kob((nk + 64) * 2, shared);
  Buffer vob((nk + 64) * 2, shared);
  float *q = static_cast<float *>(qb.base) + offset;
  float *k = static_cast<float *>(kb.base) + offset;
  float *v = static_cast<float *>(vb.base) + offset;
  uint16_t *ko = static_cast<uint16_t *>(kob.base) + offset;
  uint16_t *vo = static_cast<uint16_t *>(vob.base) + offset;
  std::vector<float> original(nq), cs(static_cast<size_t>(rows) * 128u);
  uint32_t seed = 17u + position;
  for (size_t i = 0; i < nq; ++i) {
    seed = seed * 1664525u + 1013904223u;
    original[i] = (static_cast<int32_t>(seed) / 2147483648.f) * 8.f;
  }
  for (size_t i = 0; i < nk; ++i) {
    seed = seed * 1664525u + 1013904223u;
    k[i] = (static_cast<int32_t>(seed) / 2147483648.f) * 8.f;
    v[i] = k[i] * 1.001f;
  }
  const uint32_t bits[] = {0u, 0x80000000u, 0x3f801000u, 0x477fe000u,
                           1u, 0x33800000u, 0x7f800000u, 0x7fc12345u};
  if (bench.empty())
    for (size_t i = 0; i < nk; i += 71u) {
      std::memcpy(k + i, &bits[(i / 71u) % 8u], 4);
      std::memcpy(v + i, &bits[(i / 71u + 3u) % 8u], 4);
    }
  for (unsigned r = 0; r < rows; ++r)
    for (unsigned d = 0; d < 64; ++d) {
      const float angle = float((r + position) * (d % 32u + 1u)) / 8192.f;
      cs[static_cast<size_t>(r) * 128u + d] = std::cos(angle);
      cs[static_cast<size_t>(r) * 128u + 64u + d] = std::sin(angle);
    }
  auto reset = [&] {
    std::memcpy(q, original.data(), nq * 4);
    std::memset(kob.base, 0xa5, (nk + 64) * 2);
    std::memset(vob.base, 0x5a, (nk + 64) * 2);
  };
  auto separate = [&] {
    auto *ops = nntrainer::get_htp_ops();
    using Op = nntrainer::ComputeOps::InferenceMath;
    require(ops->inference_math_fp32(Op::RopeFp16, rows, kw, 64, 0.f, 1.f, k,
                                     nullptr, cs.data(), ko),
            "K RoPE");
    require(ops->inference_math_fp32(Op::CastFp16, rows, kw, 64, 0.f, 1.f, v,
                                     nullptr, nullptr, vo),
            "V cast");
    require(ops->inference_math_fp32(Op::Rope, rows, qw, 64, 0.f, 1.f, q,
                                     nullptr, cs.data(), q),
            "Q RoPE");
  };
  auto fused = [&] {
    require(nntrainer::htp_rope_qkv_prefill(rows, qw, kw, 64, cs.data(), q, k,
                                            v, ko, vo),
            "fused prepare");
  };
  reset();
  separate();
  const std::vector<float> refq(q, q + nq);
  const std::vector<uint16_t> refk(ko, ko + nk), refv(vo, vo + nk);
  auto compare = [&] {
    require(std::memcmp(q, refq.data(), nq * 4) == 0, "query bit parity");
    require(std::memcmp(ko, refk.data(), nk * 2) == 0, "key bit parity");
    require(std::memcmp(vo, refv.data(), nk * 2) == 0, "value bit parity");
    require(ko[nk] == 0xa5a5 && vo[nk] == 0x5a5a, "output tail guard");
    if (offset)
      require(static_cast<uint16_t *>(kob.base)[0] == 0xa5a5 &&
                static_cast<uint16_t *>(vob.base)[0] == 0x5a5a,
              "output leading guard");
  };
  reset();
  fused();
  compare();
  if (!bench.empty()) {
    for (unsigned repeat = 0; repeat < 14; ++repeat) {
      reset();
      const auto begin = std::chrono::steady_clock::now();
      if (bench == "separate")
        separate();
      else
        fused();
      const auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                        std::chrono::steady_clock::now() - begin)
                        .count();
      compare();
      if (repeat >= 2)
        std::printf(
          "BENCH mode=%s M=%u Q=%u KV=%u shared=%u repeat=%u wall_us=%lld\n",
          bench.c_str(), rows, qw, kw, unsigned(shared), repeat,
          static_cast<long long>(us));
    }
  } else {
    for (unsigned bad = 0; bad < 10; ++bad) {
      unsigned m = rows, hd = 64, qwidth = qw;
      const float *key = k, *phase = cs.data(), *value = v;
      uint16_t *keyout = ko, *valueout = vo;
      switch (bad) {
      case 0:
        m = 1;
        break;
      case 1:
        hd = 128;
        break;
      case 2:
        qwidth = qw - 1;
        break;
      case 3:
        key = nullptr;
        break;
      case 4:
        phase = nullptr;
        break;
      case 5:
        key = q;
        break;
      case 6:
        value = q;
        break;
      case 7:
        keyout = vo;
        break;
      case 8:
        keyout = reinterpret_cast<uint16_t *>(k);
        break;
      case 9:
        valueout = reinterpret_cast<uint16_t *>(v);
        break;
      }
      require(!nntrainer::htp_rope_qkv_prefill(m, qwidth, kw, hd, phase, q, key,
                                               value, keyout, valueout),
              "unsupported shape/alias was accepted");
      compare();
      ++invalid;
    }
    if (shared && rows == 2 && qw == 64 && position == 0) {
      for (unsigned bad = 0; bad < 4; ++bad) {
        uint32_t elapsed = 99;
        const int rc = nntr_hvx_rope_qkv_prefill_f32(
          nntrainer::HtpBackend::global().handle(), rows, qw, kw, 64, cs.data(),
          static_cast<int>(cs.size()) - (bad == 0), q,
          static_cast<int>(nq) - (bad == 1), k, static_cast<int>(nk), v,
          static_cast<int>(nk), q, static_cast<int>(nq), ko,
          static_cast<int>(nk) - (bad == 2), vo, static_cast<int>(nk), &elapsed,
          bad == 3 ? 0 : 1);
        require(rc != 0, "malformed RPC lengths were accepted");
        compare();
        ++invalid;
      }
    }
    if (dump) {
      require(std::fwrite(q, 4, nq, dump) == nq, "dump Q");
      require(std::fwrite(ko, 2, nk, dump) == nk, "dump K");
      require(std::fwrite(vo, 2, nk, dump) == nk, "dump V");
    }
    ++cases;
  }
}

int main(int argc, char **argv) {
  try {
    setenv("NNTR_HTP_PREFILL_QKV_PREPARE", "1", 1);
    require(nntrainer::HtpBackend::global().enabled(), "HTP unavailable");
    if (argc == 3 && std::strcmp(argv[1], "--bench") == 0) {
      const std::string mode = argv[2];
      require(mode == "fused" || mode == "separate", "bench mode");
      run(1024, 2048, 512, 0, 0, true, mode, nullptr);
      return 0;
    }
    FILE *dump = nullptr;
    if (argc == 3 && std::strcmp(argv[1], "--dump") == 0) {
      dump = std::fopen(argv[2], "wb");
      require(dump != nullptr, "open dump");
    } else {
      require(argc == 2 && std::strcmp(argv[1], "--check") == 0,
              "use --check, --dump file, or --bench fused|separate");
    }
    for (unsigned rows : {2u, 3u, 5u, 31u, 32u, 33u, 65u, 1024u})
      for (unsigned shape = 0; shape < 4; ++shape)
        for (unsigned position : {0u, 31u, 3072u})
          for (bool shared : {false, true}) {
            const unsigned qw[] = {64, 128, 512, 2048};
            const unsigned kw[] = {64, 64, 512, 512};
            run(rows, qw[shape], kw[shape], position,
                shared ? position % 2u : 0u, shared, "", dump);
          }
    if (dump)
      require(std::fclose(dump) == 0, "close dump");
    std::printf("ROPE PREFILL DEVICE PASS cases=%u invalid_transactions=%u\n",
                cases, invalid);
    return 0;
  } catch (const std::exception &e) {
    std::fprintf(stderr, "ROPE PREFILL DEVICE FAIL: %s\n", e.what());
    return 1;
  }
}
