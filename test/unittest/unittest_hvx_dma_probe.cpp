// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   unittest_hvx_dma_probe.cpp
 * @date   21 Sep 2026
 * @brief  Device probe: arena DMA rate by shape / workers / vote, and DDR
 *         bandwidth with the CPU and the DSP reading at once
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * MoeChunkReplay (#87) replays the MoE layer call's own M=1 descriptor
 * list through dma_replay, next to one real traced call, then the 16 #100
 * cells (DMA_REPLAY_X lines, docs/plans/100-dma-chunk-list.md section 3.2)
 * with a content check that fails on a stale window.
 *
 * Runs on an Android device only. Requires libnntr_hvx_skel.so on
 * ADSP_LIBRARY_PATH; run once with the vote-on skel and once with the
 * vote-off one (test/htp/build.sh, HEX_EXTRA_CFLAGS=-DNNTR_HVX_NO_BUS_VOTE).
 * Reports rather than asserts on bandwidth -- the numbers are the answer
 * (docs/plans/77-first-handoff.md section 3.4 / 3.5, LEDGER items 3, 4).
 * What it does assert: the mapping is live (checksum) and the descriptor
 * count matches the host-side plan.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#include <AEEStdErr.h>
#include <remote.h>

#include "htp_backend/hmx/hexkl_dma_trace.h"
#include "nntr_dma_probe_plan.h"
#include "nntr_hvx.h"
#include "nntr_moe_dma_plan.h"

namespace {

std::string hex(int err) {
  std::ostringstream os;
  os << "0x" << std::hex << std::setw(8) << std::setfill('0')
     << static_cast<unsigned>(err);
  return os.str();
}

/** @brief nntr_dma_pattern: the byte at every 64-aligned offset is 0xA5,
 *  so the DSP's 64-stride coverage checksum over any descriptor payload is
 *  0xA5 per sample whatever the plan put there; 32 bytes on sits a tag of
 *  the 4 KiB page (#100), which the replay's res[12] sums. */
inline uint8_t pattern(size_t i) {
  return nntr_dma_pattern(static_cast<uint32_t>(i));
}

struct Shape {
  const char *name;
  uint32_t row_size, nrows, src_stride;
};

const Shape kShapes[] = {
  {"i", 4096u, 256u, 4096u},
  {"i1", 1048576u, 1u, 1048576u},
  {"ii", 16384u, 64u, 57344u},
  {"iii", 8192u, 64u, 57344u},
};

/** @brief Session + two rpcmem chunks attached as arenas, the mapping path
 *  the model uses (HtpComputeOps::place, kArenaChunkMax = 256 MiB). */
class HvxDmaProbe : public ::testing::Test {
protected:
  using RpcAlloc = void *(*)(int, uint32_t, int);
  using RpcFree = void (*)(void *);
  using RpcToFd = int (*)(void *);
  using FastrpcMmap = int (*)(int, int, void *, int, size_t, int);
  using FastrpcMunmap = int (*)(int, int, void *, size_t);

  void SetUp() override {
    remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
    int err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                     &unsigned_pd, sizeof(unsigned_pd));
    ASSERT_EQ(err, AEE_SUCCESS) << "enabling unsigned PD failed: " << hex(err);
    const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
    err = nntr_hvx_open(uri.c_str(), &handle_);
    ASSERT_EQ(err, AEE_SUCCESS)
      << "nntr_hvx_open failed: " << hex(err)
      << " -- is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?";

    auto init = (void (*)(void))dlsym(RTLD_DEFAULT, "rpcmem_init");
    alloc_ = (RpcAlloc)dlsym(RTLD_DEFAULT, "rpcmem_alloc");
    free_ = (RpcFree)dlsym(RTLD_DEFAULT, "rpcmem_free");
    to_fd_ = (RpcToFd)dlsym(RTLD_DEFAULT, "rpcmem_to_fd");
    mmap_ = (FastrpcMmap)dlsym(RTLD_DEFAULT, "fastrpc_mmap");
    munmap_ = (FastrpcMunmap)dlsym(RTLD_DEFAULT, "fastrpc_munmap");
    if (!alloc_ || !free_ || !to_fd_ || !mmap_) {
      GTEST_SKIP() << "rpcmem/fastrpc_mmap not available";
    }
    if (init) {
      init();
    }
    // 2 x 256 MiB, falling back to 2 x 128 MiB (plan section 5); the
    // bytes= field of every line says which one was in use.
    for (uint32_t bytes : {256u << 20, 128u << 20}) {
      if (TryAttach(bytes)) {
        break;
      }
    }
    ASSERT_EQ(chunks_.size(), 2u) << "could not attach two arena chunks";
  }

  bool TryAttach(uint32_t bytes) {
    Detach();
    for (int k = 0; k < 2; ++k) {
      Chunk c;
      c.bytes = bytes;
      c.buf = alloc_(25 /*RPCMEM_HEAP_ID_SYSTEM*/, 1, static_cast<int>(bytes));
      if (c.buf == nullptr) {
        std::cout << "DMA_PROBE_NOTE rpcmem_alloc(" << bytes << ") failed\n";
        Detach();
        return false;
      }
      auto *p = static_cast<uint8_t *>(c.buf);
      for (size_t i = 0; i < bytes; ++i) {
        p[i] = pattern(i);
      }
      c.fd = to_fd_(c.buf);
      const int rc = mmap_(CDSP_DOMAIN_ID, c.fd, c.buf, 0, bytes,
                           static_cast<int>(FASTRPC_MAP_FD));
      if (rc != 0) {
        std::cout << "DMA_PROBE_NOTE fastrpc_mmap rc=" << hex(rc) << "\n";
        free_(c.buf);
        Detach();
        return false;
      }
      c.mapped = true;
      const int err = nntr_hvx_arena_attach(handle_, c.fd, bytes, &c.arena);
      if (err != AEE_SUCCESS) {
        std::cout << "DMA_PROBE_NOTE arena_attach err=" << hex(err) << "\n";
        chunks_.push_back(c); // so Detach releases the mapping
        Detach();
        return false;
      }
      c.attached = true;
      chunks_.push_back(c);
    }
    return true;
  }

  void Detach() {
    for (auto &c : chunks_) {
      if (c.attached) {
        nntr_hvx_arena_detach(handle_, c.arena);
      }
      if (c.mapped && munmap_) {
        munmap_(CDSP_DOMAIN_ID, c.fd, c.buf, c.bytes);
      }
      if (c.buf) {
        free_(c.buf);
      }
    }
    chunks_.clear();
  }

  void TearDown() override {
    if (handle_) {
      Detach();
      nntr_hvx_close(handle_);
    }
  }

  struct Probe {
    double gbs = 0;
    uint64_t us = 0, bytes = 0;
    bool checksum_ok = false;
    uint32_t workers_used = 0, vote = 0, n_desc = 0;
    int err = AEE_SUCCESS;
  };

  /** @brief Passes so one call moves at least @a min_bytes. */
  uint32_t PassesFor(const Shape &s, uint32_t workers, uint64_t min_bytes,
                     uint32_t *n_desc_out = nullptr) const {
    static nntr_dma_probe_desc plan[NNTR_DMA_PROBE_MAX_DESC];
    const uint32_t vtcm_guess = ((8u << 20) / workers) & ~127u;
    const uint32_t n =
      nntr_dma_probe_plan(chunks_[0].bytes, s.row_size, s.nrows, s.src_stride,
                          workers, vtcm_guess, plan, NNTR_DMA_PROBE_MAX_DESC);
    if (n_desc_out) {
      *n_desc_out = n;
    }
    if (n == 0) {
      return 1;
    }
    const uint64_t per_pass = static_cast<uint64_t>(n) * s.row_size * s.nrows;
    return static_cast<uint32_t>((min_bytes + per_pass - 1) / per_pass);
  }

  Probe Run(uint32_t chunk, const Shape &s, uint32_t workers, uint32_t passes) {
    Probe r;
    std::vector<uint32_t> res(10, 0);
    r.err =
      nntr_hvx_dma_probe(handle_, chunks_[chunk].arena, 0, chunks_[chunk].bytes,
                         s.row_size, s.nrows, s.src_stride, workers, passes,
                         res.data(), static_cast<int>(res.size()));
    if (r.err != AEE_SUCCESS) {
      return r;
    }
    r.us = res[0];
    r.bytes = (static_cast<uint64_t>(res[2]) << 32) | res[1];
    r.workers_used = res[4];
    r.vote = res[5];
    r.n_desc = res[6];
    const uint32_t sum_bytes = std::min(res[7], res[8]);
    const uint32_t want = 0xA5u * ((sum_bytes + 63u) / 64u);
    r.checksum_ok = (res[3] == want);
    r.gbs = r.us ? static_cast<double>(r.bytes) / r.us / 1e3 : 0.0;
    return r;
  }

  /** @brief One grep-able line per cell: 3 runs, the best kept. */
  Probe Cell(const Shape &s, uint32_t workers) {
    uint32_t n_plan = 0;
    const uint32_t passes = PassesFor(s, workers, 512ull << 20, &n_plan);
    Probe best;
    for (int rep = 0; rep < 3; ++rep) {
      Probe r = Run(rep & 1, s, workers, passes);
      if (r.err != AEE_SUCCESS) {
        best = r;
        break;
      }
      if (r.gbs > best.gbs) {
        best = r;
      }
    }
    if (best.err != AEE_SUCCESS) {
      std::cout << "DMA_PROBE shape=" << s.name << " workers=" << workers
                << " skipped err=" << hex(best.err) << "\n";
      return best;
    }
    std::cout << std::fixed << std::setprecision(1)
              << "DMA_PROBE shape=" << s.name << " workers=" << workers
              << " vote=" << best.vote << " gbs=" << best.gbs
              << " us=" << best.us << " bytes=" << best.bytes
              << " checksum_ok=" << (best.checksum_ok ? "y" : "n")
              << " workers_used=" << best.workers_used
              << " n_desc=" << best.n_desc << " passes=" << passes << "\n";
    EXPECT_TRUE(best.checksum_ok) << s.name << " workers=" << workers;
    EXPECT_EQ(best.n_desc, n_plan) << s.name << " workers=" << workers;
    return best;
  }

  struct Chunk {
    void *buf = nullptr;
    uint32_t bytes = 0;
    int fd = -1;
    uint32_t arena = 0;
    bool mapped = false, attached = false;
  };

  remote_handle64 handle_ = 0;
  RpcAlloc alloc_ = nullptr;
  RpcFree free_ = nullptr;
  RpcToFd to_fd_ = nullptr;
  FastrpcMmap mmap_ = nullptr;
  FastrpcMunmap munmap_ = nullptr;
  std::vector<Chunk> chunks_;
};

/** @brief XOR-reads [p, p + n) once; the result is returned so the loads
 *  cannot be dropped. NEON 4 x 16 B per step on arm64, uint64 elsewhere. */
uint64_t stream_xor(const uint8_t *p, size_t n) {
#if defined(__ARM_NEON)
  uint8x16_t a0 = vdupq_n_u8(0), a1 = a0, a2 = a0, a3 = a0;
  size_t i = 0;
  for (; i + 64 <= n; i += 64) {
    a0 = veorq_u8(a0, vld1q_u8(p + i));
    a1 = veorq_u8(a1, vld1q_u8(p + i + 16));
    a2 = veorq_u8(a2, vld1q_u8(p + i + 32));
    a3 = veorq_u8(a3, vld1q_u8(p + i + 48));
  }
  uint8x16_t a = veorq_u8(veorq_u8(a0, a1), veorq_u8(a2, a3));
  uint64_t r = vgetq_lane_u64(vreinterpretq_u64_u8(a), 0) ^
               vgetq_lane_u64(vreinterpretq_u64_u8(a), 1);
  for (; i < n; ++i) {
    r ^= p[i];
  }
  return r;
#else
  uint64_t r = 0;
  size_t i = 0;
  for (; i + 8 <= n; i += 8) {
    uint64_t v;
    std::memcpy(&v, p + i, 8);
    r ^= v;
  }
  for (; i < n; ++i) {
    r ^= p[i];
  }
  return r;
#endif
}

} // namespace

/**
 * @brief Section 3.4: 4 shapes x 4 worker counts, this skel's vote state.
 */
TEST_F(HvxDmaProbe, DmaProbeShapes) {
  for (const Shape &s : kShapes) {
    for (uint32_t w = 1; w <= 4; ++w) {
      Cell(s, w);
    }
  }
}

/**
 * @brief Section 3.5: CPU alone, DSP alone, both at once; aggregate is the
 *        sum of the two concurrent rates.
 */
TEST_F(HvxDmaProbe, TwoReaderDdr) {
  using clock = std::chrono::steady_clock;
  const unsigned kThreads = 8;
  const size_t kCpuBytes = 512ull << 20;
  const size_t kSlice = kCpuBytes / kThreads;
  const double kSeconds = 2.0;

  std::vector<uint8_t> cpu_buf;
  cpu_buf.resize(kCpuBytes);
  for (size_t i = 0; i < kCpuBytes; i += 4096) {
    cpu_buf[i] = pattern(i); // pre-fault every page
  }
  for (size_t i = 0; i < kCpuBytes; ++i) {
    cpu_buf[i] = pattern(i);
  }

  // CPU side: each thread streams its slice until `stop`, counting whole
  // passes; a pass that straddles `stop` is not counted, so the bytes
  // are a lower bound inside the window.
  struct CpuStreamer {
    std::atomic<int> go{0}, stop{0};
    std::atomic<uint64_t> passes{0}, xr{0};
    std::vector<std::thread> th;
    clock::time_point t0;
    void Start(const uint8_t *base, unsigned threads, size_t slice) {
      for (unsigned t = 0; t < threads; ++t) {
        th.emplace_back([this, p = base + t * slice, slice]() {
          uint64_t x = 0, n = 0;
          while (!go.load(std::memory_order_acquire)) {
          }
          while (!stop.load(std::memory_order_relaxed)) {
            x ^= stream_xor(p, slice);
            ++n;
          }
          passes.fetch_add(n);
          xr.fetch_xor(x);
        });
      }
      t0 = clock::now();
      go.store(1, std::memory_order_release);
    }
    double Finish(size_t slice) {
      stop.store(1);
      for (auto &t : th) {
        t.join();
      }
      const double s = std::chrono::duration<double>(clock::now() - t0).count();
      const double bytes = static_cast<double>(passes.load()) * slice;
      const double gbs = bytes / s / 1e9;
      std::cout << std::fixed << std::setprecision(2)
                << "DDR_CPU passes=" << passes.load() << " bytes=" << bytes
                << " s=" << s << " gbs=" << gbs << " xor=" << std::hex
                << xr.load() << std::dec << "\n";
      return gbs;
    }
  };

  // 1. CPU alone.
  double cpu_alone = 0;
  {
    CpuStreamer run;
    run.Start(cpu_buf.data(), kThreads, kSlice);
    std::this_thread::sleep_for(std::chrono::duration<double>(kSeconds));
    cpu_alone = run.Finish(kSlice);
  }

  // 2. DSP alone: shape (i) with the best worker count of a quick sweep,
  //    then passes sized for >= kSeconds over both chunks.
  const Shape &lin = kShapes[0];
  uint32_t best_w = 1;
  double best_gbs = 0;
  for (uint32_t w = 1; w <= 4; ++w) {
    Probe r = Run(0, lin, w, PassesFor(lin, w, 256ull << 20));
    if (r.err == AEE_SUCCESS && r.gbs > best_gbs) {
      best_gbs = r.gbs;
      best_w = w;
    }
  }
  ASSERT_GT(best_gbs, 0.0) << "dma_probe failed on every worker count";
  const uint64_t chunk_bytes = chunks_[0].bytes;
  const uint32_t passes = PassesFor(
    lin, best_w,
    static_cast<uint64_t>(kSeconds / 2 * best_gbs * 1e9) + chunk_bytes);
  auto dsp_stream = [&](double *gbs_out) {
    uint64_t us = 0, bytes = 0;
    for (uint32_t k = 0; k < 2; ++k) {
      Probe r = Run(k, lin, best_w, passes);
      ASSERT_EQ(r.err, AEE_SUCCESS) << hex(r.err);
      EXPECT_TRUE(r.checksum_ok);
      us += r.us;
      bytes += r.bytes;
    }
    *gbs_out = us ? static_cast<double>(bytes) / us / 1e3 : 0.0;
    std::cout << std::fixed << std::setprecision(2)
              << "DDR_DSP workers=" << best_w << " passes=" << passes
              << " bytes=" << bytes << " us=" << us << " gbs=" << *gbs_out
              << "\n";
  };
  double dsp_alone = 0;
  dsp_stream(&dsp_alone);

  // 3. Both: CPU threads released just before the (blocking) DSP calls,
  //    stopped as soon as they return.
  double cpu_with = 0, dsp_with = 0;
  {
    CpuStreamer run;
    run.Start(cpu_buf.data(), kThreads, kSlice);
    dsp_stream(&dsp_with);
    cpu_with = run.Finish(kSlice);
  }

  std::cout << std::fixed << std::setprecision(2)
            << "DDR_TWO_READER cpu_alone=" << cpu_alone
            << " dsp_alone=" << dsp_alone << " cpu_with=" << cpu_with
            << " dsp_with=" << dsp_with
            << " aggregate=" << (cpu_with + dsp_with)
            << " cpu_threads=" << kThreads << " dsp_workers=" << best_w
            << " chunk_bytes=" << chunk_bytes << "\n";
}

/**
 * @brief [#87, plan section 3.3] The MoE layer call's own M=1 descriptor
 *        list, replayed with nothing between the pushes.
 *
 * Step 0 runs one real mm_u8i4_moe_layer_timed at M=1 over four expert
 * regions of the pattern arena (registered as arena weights: the HMX does
 * not care what the nibbles are, so no bake is needed for a timing) and
 * reads its trace back -- the DMA_REPLAY_TRACE lines are the in-situ
 * timeline the replays are held against, and its issue times pace the
 * pace=1 cells. Reports rather than asserts on the numbers; asserts the
 * mapping is live (checksum) and the trace has the planner's shape.
 */
TEST_F(HvxDmaProbe, MoeChunkReplay) {
  const uint32_t K = 2048, I = 1792, N = 2048, E = 4, ACC_TILES = 32;
  const uint32_t region = nntr_moe_dma_region_bytes(K, I, N);
  const uint32_t gu_bytes = nntr_moe_dma_gu_bytes(K, I);
  const uint32_t dn_bytes = nntr_moe_dma_dn_bytes(I, N);
  const uint32_t dn_off = nntr_moe_dma_down_off(K, I);
  const Chunk &ch = chunks_[0];
  ASSERT_GE(ch.bytes, (E + 1) * region) << "chunk too small for 4 regions";

  // VTCM layout for the replay: gate_up, down, activation, copy scratch.
  const uint32_t v_gu = 0, v_dn = gu_bytes, v_act = v_dn + dn_bytes,
                 v_copy = v_act + (K / 32u) * 2048u;
  static nntr_moe_dma_item plan[NNTR_MOE_DMA_PLAN_MAX];
  const uint32_t n_plan =
    nntr_moe_dma_plan_m1(K, I, N, E, ACC_TILES, v_gu, v_dn, v_act, v_copy, plan,
                         NNTR_MOE_DMA_PLAN_MAX);
  ASSERT_NE(n_plan, 0u);
  uint32_t plan_push = 0, plan_wait = 0;
  for (uint32_t k = 0; k < n_plan; ++k) {
    (plan[k].op == NNTR_MOE_DMA_OP_PUSH ? plan_push : plan_wait)++;
  }

  // --- step 0: one real timed call over the same regions -----------------
  std::vector<uint32_t> h_gu(E), h_dn(E);
  {
    std::vector<float> ws_gu(2 * I, 0.01f), bias_gu(2 * I, 0.f);
    std::vector<int32_t> cs_gu(2 * I, 0);
    std::vector<float> ws_dn(N, 0.01f), bias_dn(N, 0.f);
    std::vector<int32_t> cs_dn(N, 0);
    for (uint32_t e = 0; e < E; ++e) {
      ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                  handle_, K, 2 * I, ch.arena, e * region, ws_gu.data(),
                  (int)(2 * I), cs_gu.data(), (int)(2 * I), bias_gu.data(),
                  (int)(2 * I), &h_gu[e]),
                AEE_SUCCESS);
      ASSERT_EQ(nntr_hvx_weight_register_u8i4_arena(
                  handle_, I, N, ch.arena, e * region + dn_off, ws_dn.data(),
                  (int)N, cs_dn.data(), (int)N, bias_dn.data(), (int)N,
                  &h_dn[e]),
                AEE_SUCCESS);
    }
  }
  // One row routed to all four experts: decode's shape.
  const std::vector<uint32_t> row_count(E, 1u), row_index(E, 0u);
  const std::vector<float> row_weight(E, 0.25f);
  std::vector<float> act(K), out(N, 0.f);
  for (uint32_t i = 0; i < K; ++i) {
    act[i] = static_cast<float>((i * 7919u) % 1000u) / 500.f - 1.f;
  }
  // test/htp/nntr_hvx_mm_u8i4.c's MOE_N_STAGES (mirrored as
  // HTP_MOE_N_STAGES in htp_compute_ops.cpp): 19 before #87 + 10 (#87's
  // DMA slots) + 1 (#80's MOE_T_PATH) + 1 (#117's MOE_T_M1_FEED, after
  // them; the slots read below did not move).
  const int kMoeStages = 31;
  std::vector<uint32_t> stage(kMoeStages, 0);
  std::vector<uint32_t> trace;
  uint32_t n_words = 0;
  bool traced = false;
  {
    // Two calls: the first warms the page state, the second is traced.
    int err = AEE_SUCCESS;
    for (int rep = 0; rep < 2 && err == AEE_SUCCESS; ++rep) {
      err = nntr_hvx_mm_u8i4_moe_layer_timed(
        handle_, 1, K, I, N, h_gu.data(), (int)E, h_dn.data(), (int)E,
        row_index.data(), (int)E, row_count.data(), (int)E, row_weight.data(),
        (int)E, act.data(), (int)K, out.data(), (int)N, stage.data(),
        kMoeStages);
    }
    if (err != AEE_SUCCESS) {
      std::cout << "DMA_REPLAY_NOTE moe_layer_timed err=" << hex(err)
                << " -- pace=1 cells fall back to pace=0\n";
    } else {
      trace.resize(HEXKL_DMA_TRACE_MAX_WORDS);
      err = nntr_hvx_moe_dma_trace_read(handle_, trace.data(),
                                        (int)trace.size(), &n_words);
      traced = err == AEE_SUCCESS && n_words >= HEXKL_DMA_TRACE_HDR_WORDS &&
               trace[0] == plan_push && trace[1] == plan_wait;
      std::cout << "DMA_REPLAY_TRACE dsp_us=" << stage[0]
                << " desc=" << stage[19] << " waits=" << stage[20]
                << " blocked=" << stage[21] << " wait_us=" << stage[22]
                << " wait_act_us=" << stage[23] << " busy_us=" << stage[24]
                << ".." << stage[25] << " depth_max=" << stage[26]
                << " first_ready_us=" << stage[27]
                << " last_issue_us=" << stage[28] << " trace_words=" << n_words
                << " plan_shape_ok=" << (traced ? "y" : "n") << "\n";
    }
  }
  const uint32_t pw = HEXKL_DMA_TRACE_PUSH_WORDS,
                 ww = HEXKL_DMA_TRACE_WAIT_WORDS;
  auto us = [](uint32_t ticks) { return ticks / 19.2; };
  if (traced) {
    static const char *const kKind[] = {"act", "gate", "up", "down", "copy"};
    static const char *const kSite[] = {"act", "gu", "dn", "copy_in",
                                        "copy_out"};
    const uint32_t *p = trace.data() + HEXKL_DMA_TRACE_HDR_WORDS;
    for (uint32_t k = 0; k < trace[0]; ++k, p += pw) {
      std::cout << std::fixed << std::setprecision(1)
                << "DMA_REPLAY_TRACE push k=" << k << " t=" << us(p[0])
                << " kind=" << (p[1] < 5 ? kKind[p[1]] : "?") << " e=" << p[2]
                << " c=" << p[3] << " bytes=" << p[4] << " row=" << p[5]
                << " nrows=" << p[6] << " stride=" << p[7] << " depth=" << p[9]
                << " done=" << us(p[10]) << ".." << us(p[11]) << "\n";
    }
    for (uint32_t k = 0; k < trace[1]; ++k, p += ww) {
      std::cout << std::fixed << std::setprecision(1)
                << "DMA_REPLAY_TRACE wait k=" << k
                << " site=" << (p[3] < 5 ? kSite[p[3]] : "?")
                << " t=" << us(p[0]) << ".." << us(p[1])
                << " blocked=" << (p[4] ? "y" : "n") << "\n";
    }
  }

  // --- the schedule: the plan, with the traced call's issue times ---------
  std::vector<uint32_t> sched;
  sched.reserve(n_plan * 8);
  {
    const uint32_t *pp = trace.data() + HEXKL_DMA_TRACE_HDR_WORDS;
    const uint32_t *wp = pp + plan_push * pw;
    uint32_t ip = 0, iw = 0;
    for (uint32_t k = 0; k < n_plan; ++k) {
      const nntr_moe_dma_item &it = plan[k];
      uint32_t t_rel = 0;
      if (traced) {
        t_rel =
          it.op == NNTR_MOE_DMA_OP_PUSH ? pp[(ip++) * pw] : wp[(iw++) * ww];
      }
      const uint32_t words[8] = {nntr_moe_dma_word0(&it),
                                 it.expert,
                                 it.src_off,
                                 it.dst_off,
                                 it.row_size,
                                 it.nrows,
                                 it.src_stride,
                                 t_rel};
      sched.insert(sched.end(), words, words + 8);
    }
  }

  const uint32_t want_sum = 0xA5u * (gu_bytes / 64u);
  auto cell = [&](uint32_t workers, uint32_t load, uint32_t pace,
                  uint32_t fresh, uint32_t gap_us) {
    const uint32_t calls = 20;
    std::vector<uint32_t> res(12, 0);
    if (pace && !traced) {
      pace = 0;
    }
    const int err = nntr_hvx_dma_replay(
      handle_, ch.arena, region, sched.data(), (int)sched.size(), workers, load,
      pace, fresh, gap_us, calls, res.data(), (int)res.size());
    if (err != AEE_SUCCESS) {
      std::cout << "DMA_REPLAY workers=" << workers << " load=" << load
                << " pace=" << pace << " fresh=" << fresh
                << " gap_us=" << gap_us << " skipped err=" << hex(err) << "\n";
      return;
    }
    const double us_per_call = static_cast<double>(res[0]) / calls;
    const double gbs = us_per_call > 0 ? res[2] / us_per_call / 1e3 : 0.0;
    std::cout << std::fixed << std::setprecision(1)
              << "DMA_REPLAY workers=" << workers << " load=" << load
              << " pace=" << pace << " fresh=" << fresh << " gap_us=" << gap_us
              << " calls=" << calls << " us_per_call=" << us_per_call
              << " bytes_per_call=" << res[2] << " gbs=" << gbs
              << " wait_us=" << res[3] / (double)calls << " blocked=" << res[4]
              << "/" << plan_wait * calls << " depth_max=" << res[5]
              << " busy_us=" << res[10] / (double)calls << ".."
              << res[11] / (double)calls << " regions=" << res[7]
              << " workers_used=" << res[8] << " load_units=" << res[9]
              << " checksum_ok=" << (res[6] == want_sum ? "y" : "n") << "\n";
    EXPECT_EQ(res[6], want_sum) << "workers=" << workers << " load=" << load;
  };
  for (uint32_t pace = 0; pace <= 1; ++pace) {
    for (uint32_t w : {1u, 2u, 4u}) {
      cell(w, 0, pace, 0, 0);
    }
  }
  cell(1, 1, 1, 0, 0);
  cell(1, 2, 1, 0, 0);
  cell(1, 0, 1, 1, 0);
  cell(1, 0, 1, 0, 600);
  cell(1, 0, 1, 1, 600);

  // --- [#100] the cells of plan 100 section 3.2, one run, workers 1 ------
  // res[12] is the tag sum over every push's VTCM window; the host
  // simulates the same list (nntr_moe_dma_tag_sum, checked against a byte
  // copy by replay_cells_host_check) with the skel's region count, so a
  // transfer that did not land -- or, for fresh = 1, landed a call late --
  // fails the line. The rates are reported, not asserted. The expectation
  // takes list order as landing order; only the traced* cells (the packed
  // gate/up chunks overlap, #99) and iii_chain (a slot rewritten 8 pushes
  // on) have overlapping destinations in flight at once, every other cell
  // waits or drains before it rewrites a destination.
  {
    static nntr_moe_dma_item items[NNTR_MOE_DMA_PLAN_MAX];
    static uint8_t samples[(8u << 20) / 64u + 1u];
    const uint32_t calls = 20;
    for (uint32_t id = 0; id < NNTR_MOE_DMA_N_CELLS; ++id) {
      const char *name = "?";
      uint32_t fresh = 0, load = 0, n_push = 0, modes = 0;
      const uint32_t n =
        nntr_moe_dma_cell(id, plan, n_plan, K, I, N, E, items,
                          NNTR_MOE_DMA_PLAN_MAX, &name, &fresh, &load);
      ASSERT_NE(n, 0u) << "cell " << id;
      std::vector<uint32_t> s;
      for (uint32_t k = 0; k < n; ++k) {
        const nntr_moe_dma_item &it = items[k];
        const uint32_t words[8] = {nntr_moe_dma_word0(&it),
                                   it.expert,
                                   it.src_off,
                                   it.dst_off,
                                   it.row_size,
                                   it.nrows,
                                   it.src_stride,
                                   0u};
        s.insert(s.end(), words, words + 8);
        if (it.op == NNTR_MOE_DMA_OP_PUSH) {
          ++n_push;
          // flags 0 is the replay's default, packed until #99 lands
          modes |= (it.flags & NNTR_MOE_DMA_DST_STRIDED) ? 2u : 1u;
        }
      }
      std::vector<uint32_t> res(13, 0);
      const int err = nntr_hvx_dma_replay(handle_, ch.arena, region, s.data(),
                                          (int)s.size(), 1, load, 0, fresh, 0,
                                          calls, res.data(), (int)res.size());
      if (err != AEE_SUCCESS) {
        std::cout << "DMA_REPLAY_X name=" << name << " skipped err=" << hex(err)
                  << "\n";
        ADD_FAILURE() << name << ": " << hex(err) << " -- a skel without #100?";
        continue;
      }
      const uint32_t want =
        nntr_moe_dma_tag_sum(items, n, calls, fresh, res[7], region, samples);
      const double us_per_call = static_cast<double>(res[0]) / calls;
      const double gbs = us_per_call > 0 ? res[2] / us_per_call / 1e3 : 0.0;
      static const char *const kMode[] = {"?", "packed", "strided", "mixed"};
      std::cout << std::fixed << std::setprecision(1)
                << "DMA_REPLAY_X name=" << name << " dst=" << kMode[modes]
                << " load=" << load << " fresh=" << fresh << " desc=" << n_push
                << " bytes_per_call=" << res[2]
                << " us_per_call=" << us_per_call << " gbs=" << gbs
                << " wait_us=" << res[3] / (double)calls
                << " busy_us=" << res[10] / (double)calls << ".."
                << res[11] / (double)calls << " depth_max=" << res[5]
                << " regions=" << res[7] << " load_units=" << res[9]
                << " tag=" << res[12] << "/" << want << " checksum_ok="
                << (res[12] == want ? "y" : "n")
                // a skel without #100 leaves res[12] at 0 on the unflagged
                // cells; a live window never sums to 0 (host check)
                << (res[12] == 0u ? " stale_skel_or_nothing_landed" : "")
                << "\n";
      EXPECT_EQ(res[12], want) << name;
    }
  }

  for (uint32_t e = 0; e < E; ++e) {
    nntr_hvx_weight_release_u8i4(handle_, h_gu[e]);
    nntr_hvx_weight_release_u8i4(handle_, h_dn[e]);
  }
}

int main(int argc, char **argv) {
  int result = -1;
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return 0;
  }
  try {
    result = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
  return result;
}
