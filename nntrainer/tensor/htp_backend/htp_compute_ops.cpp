// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_compute_ops.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon/HMX) ComputeOps entry point.
 *
 * Compiled only when ENABLE_HEXKL is defined.
 *
 * HtpComputeOps overrides exactly the ops it accelerates and inherits the
 * CPU implementation of everything else from CpuComputeOps.
 *
 * It deliberately does NOT derive from the abstract ComputeOps base the way
 * ClComputeOps does. A layer's engine covers every tensor in it, not just
 * the ones this backend has a kernel for: Lfm2MoELayer's router gate is
 * FP32 by construction (lfm2_moe_layer.cpp keeps gate and expert_bias
 * unquantized), so with engine=htp its dot() reaches sgemm_fp32, which has
 * no supports_* guard and no fallback of its own in FloatTensor::dot. Off
 * the abstract base that threw "ComputeOps::sgemm_fp32 not implemented by
 * this backend" on the first prefill. CpuComputeOps is stateless, so
 * inheriting it costs nothing and makes every unaccelerated op behave the
 * way it does with engine=cpu.
 *
 * gemm_q4_0_accel_fp32 is the first kernel: one FastRPC call per Q4_0 FC
 * dot(), single weight, single activation -- hexkl_mm_u8i4_layer_run with
 * n_handles=1 underneath. It is Tier 1 of docs/htp_attention/
 * 40_moe_ffn_htp_task.md section 3 and benefits every FC layer under
 * engine=htp, not just the LFM2 MoE FFN that motivated this task.
 *
 * gemm_q4_0_batch_fp32 is the same call with n_handles > 1: several
 * weights sharing one activation (LFM2-MoE decode's selected experts'
 * gate_up projections against the single routed token) go out as one
 * FastRPC call so hexkl_mm_u8i4_layer_run can prefetch the next handle's
 * weight while the current one computes.
 */

#ifdef ENABLE_HEXKL

#include <compute_ops.h>
#include <cpu_backend.h>
#include <cpu_ops_table.h>
#include <htp_act_quant.h>
#include <htp_activation_reuse.h>
#include <htp_attn_q8_quant.h>
#include <htp_backend.h>
#include <htp_dspq_wire.h>
#include <htp_expert_host_staging.h>
#include <htp_graph_desc.h>
#include <htp_kv_cache.h>
#include <htp_moe_opts.h>
#include <htp_prepared_fc.h>
#include <htp_prepared_qs2.h>
#include <htp_q4_0_convert.h>
#include <htp_qs2_mapped_file.h>
#include <htp_qs2_reader.h>
#include <htp_rope_prefill.h>
#include <htp_rope_prefill_shape.h>
#include <htp_rpcmem.h>
#include <htp_wh_layout.h>
#include <m1_ops_det.h>
#include <nntrainer_log.h>
#include <q4_gemv_cpu_det.h>
#include <swiglu_det.h>
#include <thread_manager.h>
#include <weight_direct_file.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sched.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <remote.h>

#include <hmx/hexkl_dma_trace.h>

#include <nntr_hvx.h>

namespace nntrainer {

namespace {

/**
 * @brief Slots mm_u8i4_layer_timed fills.
 *
 * Restated here because the ARM side cannot include the DSP source that
 * declares them (nntr_hvx_mm_u8i4.c). Nine slots include HVX MM time and
 * actual dispatch; the DSP also accepts the legacy seven-slot callers.
 */
enum {
  HTP_T_DSP_TOTAL = 0,
  HTP_T_QUANT,
  HTP_T_DEQUANT,
  HTP_T_ACC_READ,
  HTP_T_ACC_COPY,
  HTP_T_DRAIN,
  HTP_T_ACC_STRIDE,
  HTP_T_MM,
  HTP_T_PATH,
  HTP_N_STAGES
};

/**
 * @brief Slots mm_u8i4_layer_fused_timed fills.
 *
 * Restated for the same reason as HTP_T_* (the ARM side cannot include the
 * DSP source). NOT the HTP_T_* layout: the fused call runs a second QUANT
 * pass (the SwiGLU output's requant) and a SwiGLU stage the unfused call
 * does not have, so it carries its own slot list and its own Bucket
 * counter -- folding SwiGLU into an existing slot would put its time in a
 * column named after something else.
 */
enum {
  HTP_FU_T_DSP_TOTAL = 0,
  HTP_FU_T_QUANT,
  HTP_FU_T_SWIGLU,
  HTP_FU_T_DEQUANT,
  HTP_FU_T_ACC_READ,
  HTP_FU_T_ACC_COPY,
  HTP_FU_T_DRAIN,
  HTP_FU_T_ACC_STRIDE,
  HTP_FU_N_STAGES
};

/**
 * @brief Slots mm_u8i4_gate_up_swiglu_timed fills, in order -- test/htp/
 * nntr_hvx_mm_u8i4.c's GU_T_* enum restated (the ARM side cannot include
 * that DSP source). NOT the same layout as HTP_FU_T_*: this call has no
 * down-side accumulator at all, so there is no ACC_COPY slot -- reusing
 * HTP_FU_T_*'s indices against this array would silently read DRAIN's
 * value out of the slot ACC_STRIDE actually lives in.
 */
enum {
  HTP_GU_T_DSP_TOTAL = 0,
  HTP_GU_T_QUANT,
  HTP_GU_T_SWIGLU,
  HTP_GU_T_DEQUANT,
  HTP_GU_T_ACC_READ,
  HTP_GU_T_DRAIN,
  HTP_GU_T_ACC_STRIDE,
  HTP_GU_N_STAGES
};

/**
 * @brief Slots mm_u8i4_moe_layer_timed fills, in order -- test/htp/
 * nntr_hvx_mm_u8i4.c's MOE_T_* restated, for the same reason HTP_GU_T_* is.
 * One slot none of the others have: the routing multiply and scatter-add,
 * which live on the ARM side until this call takes them.
 */
enum {
  HTP_MOE_T_DSP_TOTAL = 0,
  HTP_MOE_T_QUANT,
  HTP_MOE_T_SWIGLU,
  HTP_MOE_T_DEQUANT,
  HTP_MOE_T_ACC_READ,
  HTP_MOE_T_DRAIN,
  HTP_MOE_T_SCATTER,
  HTP_MOE_T_GATHER,
  HTP_MOE_T_REQUANT,
  HTP_MOE_T_BLOCKS,
  HTP_MOE_T_MM,
  HTP_MOE_T_DMA_KB,
  HTP_MOE_T_DMA_FIRST,
  HTP_MOE_T_ALLOC, /**< the layer call's own malloc and free */
  HTP_MOE_T_DMA_FIRST_KB,
  HTP_MOE_T_DRAIN_DN,
  HTP_MOE_T_PUSH,
  HTP_MOE_T_STAGE,
  HTP_MOE_T_ACC_STRIDE,
  /** [#87] The DMA ring trace's per-call numbers, hexkl_probe.h's
      HEXKL_PROBE_DMA_* in the same order. Counts unless named _US. */
  HTP_MOE_T_DMA_DESC,
  HTP_MOE_T_DMA_WAITS,
  HTP_MOE_T_DMA_WAITS_BLOCKED,
  HTP_MOE_T_DMA_WAIT_US,
  HTP_MOE_T_DMA_WAIT_ACT_US,
  HTP_MOE_T_DMA_BUSY_LO_US,
  HTP_MOE_T_DMA_BUSY_HI_US,
  HTP_MOE_T_DMA_DEPTH_MAX,
  HTP_MOE_T_DMA_FIRST_READY_US,
  HTP_MOE_T_DMA_LAST_ISSUE_US,
  HTP_MOE_T_PATH,    /**< NOT us: 0 = HMX block loop, 1 = M=1 HVX GEMV */
  HTP_MOE_T_M1_FEED, /**< NOT us: 1 = the GEMV read VTCM fed by DMA (#117) */
  HTP_MOE_N_STAGES
};

/**
 * @brief Per-stage timing for the HTP path. Off unless NNTR_HTP_PROFILE is set.
 *
 * NNTR_HTP_PROFILE=1 times the three host-side stages a weight goes through:
 * the Q4_0/QS4CX -> HexKL-registry conversion, the weight_register FastRPC,
 * and the per-dot() layer call. =2 additionally routes the layer call through
 * mm_u8i4_layer_timed, so the DSP reports its own microseconds and the
 * FastRPC transport share becomes host_us - dsp_us rather than a guess
 * (mobile_e2e_run_guide.md section 6, evidence step 2).
 *
 * Registration is accounted separately from the layer call on purpose: it
 * runs once per weight pointer and therefore lands entirely in the first
 * forward pass, so "prefill got slower but decode did not" and "every call
 * got slower" are different findings and have to be separable.
 */
class HtpProfile {
public:
  static HtpProfile &global() {
    static HtpProfile instance;
    return instance;
  }

  int level() const { return level_; }

  static uint64_t nowUs() {
    return static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch())
        .count());
  }

  /** [#85] The per-token entry's own line under the layer rows: how many
   *  forward calls, ops per call, the DSP time of the whole call and the
   *  sum of the per-op pcycle brackets (level >= 2, forward_debug). The
   *  MoE op's stages go to the M==1 bucket through addInvokeMoeLayer, so
   *  that row stays comparable to the per-layer path's. */
  void addInvokeForward(unsigned n_ops, uint64_t dsp_us, uint64_t op_pcyc) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++graph_calls_;
    graph_ops_ += n_ops;
    graph_dsp_us_ += dsp_us;
    graph_op_pcyc_ += op_pcyc;
  }
  /** [#130] One resident op's pcycle bracket, by kind (level >= 2). */
  void addGraphOp(unsigned kind, uint64_t pcyc) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (kind < HTP_OP_KIND_N) {
      ++graph_kind_n_[kind];
      graph_kind_pcyc_[kind] += pcyc;
    }
  }
  void setGraphResident(uint32_t mask) { graph_resident_ = mask; }

  void addRegister(uint64_t total_us, uint64_t convert_us, uint64_t rpc_us,
                   bool ion) {
    std::lock_guard<std::mutex> lock(mutex_);
    ++reg_calls_;
    reg_total_us_ += total_us;
    convert_us_ += convert_us;
    rpc_us_ += rpc_us;
    if (ion)
      ++reg_ion_calls_;
  }

  /** @brief One expert brought in from the model file at forward time, or
   *  one released to make room (doc 52): read_us is the pread into the
   *  arena slot, rpc_us the register/release FastRPC calls. Held as
   *  "pending" and attributed to the next MoE layer call's row, which is
   *  the call that paid for it, so the row reads miss/call directly. */
  void addExpertLoad(uint64_t read_us, uint64_t rpc_us, uint64_t loads) {
    std::lock_guard<std::mutex> lock(mutex_);
    miss_pending_ += loads;
    miss_read_pending_us_ += read_us;
    miss_rpc_pending_us_ += rpc_us;
    miss_total_ += loads;
    miss_read_total_us_ += read_us;
    miss_rpc_total_us_ += rpc_us;
  }

  /** @brief Experts read while a layer call ran (doc 52 section 10.10):
   *  wait_us is only the read time the call did not cover. Kept apart from
   *  the misses so the miss columns keep meaning a synchronous miss. */
  void addPrefetch(uint64_t n, uint64_t done_on_entry, uint64_t wait_us,
                   uint64_t rpc_us, uint32_t caller_cpus,
                   uint32_t reader_cpus) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefetch_n_ += n;
    prefetch_done_on_entry_ += done_on_entry;
    prefetch_wait_us_ += wait_us;
    prefetch_rpc_us_ += rpc_us;
    prefetch_caller_cpus_ |= caller_cpus;
    prefetch_reader_cpus_ |= reader_cpus;
  }

  /** @brief One expert a reader thread read, and how long it took (doc 52
   *  section 10.30). */
  void addPrefetchRead(uint64_t read_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    prefetch_read_us_ += read_us;
    ++prefetch_reads_;
  }

  void addInvoke(unsigned M, unsigned K, unsigned N, uint64_t host_us,
                 const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    // kind 3: a plain FC call. It gets its own row because the attention
    // q/o projections have the MoE layer call's K and N_out (2048 x 2048),
    // and the first all-on run averaged 12 two-millisecond FC calls into
    // the 22 fifteen-millisecond MoE calls of one "M>1" row.
    Bucket &b = buckets_[std::make_tuple(K, N, M == 1, 3)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_T_QUANT];
      b.dequant_us += stage_us[HTP_T_DEQUANT];
      b.acc_us += stage_us[HTP_T_ACC_READ] + stage_us[HTP_T_ACC_COPY];
      b.drain_us += stage_us[HTP_T_DRAIN];
      b.mm_us += stage_us[HTP_T_MM];
      b.m1_calls += stage_us[HTP_T_PATH] != 0u;
      b.m1_feed_calls += stage_us[HTP_T_PATH] != 0u;
    }
  }

  /** Same buckets as addInvoke -- the fused call has its own K/N shape (the
   * down matmul's output width, not gate_up's), so it lands in its own row
   * and the two-dot path's numbers stay directly comparable across builds. */
  void addInvokeFused(unsigned M, unsigned K, unsigned N, uint64_t host_us,
                      const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N, M == 1, 0)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_FU_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_FU_T_QUANT];
      b.swiglu_us += stage_us[HTP_FU_T_SWIGLU];
      b.dequant_us += stage_us[HTP_FU_T_DEQUANT];
      b.acc_us += stage_us[HTP_FU_T_ACC_READ] + stage_us[HTP_FU_T_ACC_COPY];
      b.drain_us += stage_us[HTP_FU_T_DRAIN];
    }
  }

  /** Same buckets again, this call's own (smaller) stage layout -- see
   * HTP_GU_T_*'s doc comment for why it is not HTP_FU_T_*. Bucketed under
   * (K, 2*inter, M==1) -- gate_up's own real shape -- not the down matmul's,
   * so this row is directly comparable to what a plain (unfused) gate_up
   * dot() would have shown for the same layer. */
  void addInvokeGateUpSwiglu(unsigned M, unsigned K, unsigned N_gate_up,
                             uint64_t host_us, const uint32_t *stage_us) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_gate_up, M == 1, 0)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_GU_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_GU_T_QUANT];
      b.swiglu_us += stage_us[HTP_GU_T_SWIGLU];
      b.dequant_us += stage_us[HTP_GU_T_DEQUANT];
      b.acc_us += stage_us[HTP_GU_T_ACC_READ];
      b.drain_us += stage_us[HTP_GU_T_DRAIN];
    }
  }

  /** One call now covers a whole layer, so this bucket's `calls` is 1 where
   *  the split-call path's was 64. Bucketed under (K, N_out) -- the layer's
   *  own shape -- which is deliberately NOT either of the two shapes the
   *  path it replaces reported, so a profile cannot be read as if the two
   *  were the same row. `rows` stays M, the tokens the layer saw, not the
   *  expert-slot count, so it means the same thing it always did.
   *
   *  The scatter and the FastRPC-buffer staging go into their own field,
   *  not into acc_us. Folding them in looked tidy and cost a measurement:
   *  the first run of this call put 104.8 ms in a bucket ACC_READ and the
   *  scatter shared, and the profile could not say which. */
  /** @param kind 0 the MoE layer; 1 the dense FFN routed through the MoE
   *  layer kernel (doc 51 section 1); 2 the conv block (doc 51 section 2).
   *  Each files in its own row: the dense FFN has the MoE layer's shapes
   *  and the conv block the MoE row's K and N_out, so the key alone could
   *  not tell them apart. All three fill the same stage_us columns. */
  void addInvokeMoeLayer(unsigned M, unsigned K, unsigned N_out,
                         uint64_t host_us, const uint32_t *stage_us,
                         const HtpRpcBuffer &act_stage,
                         const HtpRpcBuffer &out_stage, size_t in_arg_bytes,
                         int kind = 0, bool via_dspq = false,
                         bool direct_io = false) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_out, M == 1, kind)];
    ++b.calls;
    b.dspq_calls += via_dspq ? 1 : 0;
    b.direct_io_calls += direct_io ? 1 : 0;
    b.rows += M;
    b.host_us += host_us;
    // [#88] What this call staged through, for the staging: line. The
    // class sizes are the max over the bucket's calls (one class per shape
    // in practice), ion is the AND (one heap fallback voids the number).
    b.stage_act_bytes = std::max(b.stage_act_bytes, act_stage.size());
    b.stage_out_bytes = std::max(b.stage_out_bytes, out_stage.size());
    b.stage_ion = b.stage_ion && act_stage.isIon() && out_stage.isIon();
    b.in_arg_bytes = std::max<uint64_t>(b.in_arg_bytes, in_arg_bytes);
    b.swiglu_hidden = true;
    b.misses += miss_pending_;
    b.miss_read_us += miss_read_pending_us_;
    b.miss_rpc_us += miss_rpc_pending_us_;
    miss_pending_ = miss_read_pending_us_ = miss_rpc_pending_us_ = 0;
    if (stage_us != nullptr) {
      b.dsp_us += stage_us[HTP_MOE_T_DSP_TOTAL];
      b.quant_us += stage_us[HTP_MOE_T_QUANT];
      b.swiglu_us += stage_us[HTP_MOE_T_SWIGLU];
      b.dequant_us += stage_us[HTP_MOE_T_DEQUANT];
      b.acc_us += stage_us[HTP_MOE_T_ACC_READ];
      b.scatter_us += stage_us[HTP_MOE_T_SCATTER];
      b.stage_us += stage_us[HTP_MOE_T_STAGE];
      b.gather_us += stage_us[HTP_MOE_T_GATHER];
      b.requant_us += stage_us[HTP_MOE_T_REQUANT];
      b.blocks += stage_us[HTP_MOE_T_BLOCKS];
      b.mm_us += stage_us[HTP_MOE_T_MM];
      b.dma_kb += stage_us[HTP_MOE_T_DMA_KB];
      b.dma_first_us += stage_us[HTP_MOE_T_DMA_FIRST];
      b.drain_dn_us += stage_us[HTP_MOE_T_DRAIN_DN];
      b.dma_first_kb += stage_us[HTP_MOE_T_DMA_FIRST_KB];
      b.alloc_us += stage_us[HTP_MOE_T_ALLOC];
      b.push_us += stage_us[HTP_MOE_T_PUSH];
      b.drain_us += stage_us[HTP_MOE_T_DRAIN];
      b.dma_desc += stage_us[HTP_MOE_T_DMA_DESC];
      b.dma_waits += stage_us[HTP_MOE_T_DMA_WAITS];
      b.dma_waits_blocked += stage_us[HTP_MOE_T_DMA_WAITS_BLOCKED];
      b.dma_wait_us += stage_us[HTP_MOE_T_DMA_WAIT_US];
      b.dma_wait_act_us += stage_us[HTP_MOE_T_DMA_WAIT_ACT_US];
      b.dma_busy_lo_us += stage_us[HTP_MOE_T_DMA_BUSY_LO_US];
      b.dma_busy_hi_us += stage_us[HTP_MOE_T_DMA_BUSY_HI_US];
      if (stage_us[HTP_MOE_T_DMA_DEPTH_MAX] > b.dma_depth_max) {
        b.dma_depth_max = stage_us[HTP_MOE_T_DMA_DEPTH_MAX];
      }
      b.dma_first_ready_us += stage_us[HTP_MOE_T_DMA_FIRST_READY_US];
      b.dma_last_issue_us += stage_us[HTP_MOE_T_DMA_LAST_ISSUE_US];
      b.m1_calls += (stage_us[HTP_MOE_T_PATH] != 0u) ? 1u : 0u;
      b.m1_feed_calls += (stage_us[HTP_MOE_T_M1_FEED] != 0u) ? 1u : 0u;
    }
  }

  /** @brief [#87] Whether this call's per-descriptor trace should be
   *  dumped: the first NNTR_HTP_DMA_TRACE (default 3) timed calls of each
   *  bucket. Returns the 1-based call ordinal to print, or 0. */
  unsigned dmaTraceOrdinal(unsigned K, unsigned N_out, unsigned M,
                           int kind = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_out, M == 1, kind)];
    if (b.dma_traces >= dma_trace_calls_) {
      return 0;
    }
    return static_cast<unsigned>(++b.dma_traces);
  }

  /**
   * @brief [#87] Prints one traced call, one line per push and per wait,
   *        as read back by moe_dma_trace_read. Ticks are 19.2 MHz.
   *
   * At level 3 the trace is the last of the repeats -- a warm call -- which
   * the header states as rep=; level 2 prints single-shot (cold) calls.
   * Both are wanted (plan 87 hypothesis g).
   */
  static void dumpDmaTrace(unsigned ordinal, unsigned M, int reps,
                           const uint32_t *w, uint32_t n_words) {
    static const char *const kKind[] = {"act", "gate", "up", "down", "copy"};
    static const char *const kSite[] = {"act", "gu", "dn", "copy_in",
                                        "copy_out"};
    auto us = [](uint32_t ticks) { return ticks / 19.2; };
    if (n_words < HEXKL_DMA_TRACE_HDR_WORDS) {
      std::fprintf(stderr, "[HTP-DMA] call=%u M=%u trace unavailable\n",
                   ordinal, M);
      return;
    }
    const uint32_t n_push = w[0], n_wait = w[1], pw = w[6], ww = w[7];
    if (pw != HEXKL_DMA_TRACE_PUSH_WORDS || ww != HEXKL_DMA_TRACE_WAIT_WORDS ||
        n_words < HEXKL_DMA_TRACE_HDR_WORDS + n_push * pw + n_wait * ww) {
      std::fprintf(stderr, "[HTP-DMA] call=%u M=%u trace layout mismatch\n",
                   ordinal, M);
      return;
    }
    std::fprintf(stderr,
                 "[HTP-DMA] call=%u M=%u rep=%d/%d desc=%u waits=%u blocked=%u "
                 "depth_max=%u dropped=%u end=%.1fus\n",
                 ordinal, M, reps, reps, n_push, n_wait, w[2], w[3], w[4],
                 us(w[5]));
    const uint32_t *p = w + HEXKL_DMA_TRACE_HDR_WORDS;
    for (uint32_t k = 0; k < n_push; ++k, p += pw) {
      std::fprintf(stderr,
                   "[HTP-DMA] call=%u push k=%u t=%.1f kind=%s e=%u c=%u "
                   "bytes=%u row=%u nrows=%u stride=%u idx=%u depth=%u "
                   "done=%.1f..%.1f\n",
                   ordinal, k, us(p[0]), p[1] < 5 ? kKind[p[1]] : "?", p[2],
                   p[3], p[4], p[5], p[6], p[7], p[8], p[9], us(p[10]),
                   us(p[11]));
    }
    for (uint32_t k = 0; k < n_wait; ++k, p += ww) {
      std::fprintf(stderr,
                   "[HTP-DMA] call=%u wait k=%u site=%s idx=%u t=%.1f..%.1f "
                   "blocked=%s\n",
                   ordinal, k, p[3] < 5 ? kSite[p[3]] : "?", p[2], us(p[0]),
                   us(p[1]), p[4] ? "y" : "n");
    }
  }

  /** ARM-side staging: the memcpy of the activation into the rpcmem buffer
   *  and of the result back out. It sits OUTSIDE every host_us window
   *  above (those start after the copy, so that transport = host - dsp
   *  stays a FastRPC number), which meant it was invisible -- and at this
   *  model's shapes it is ~0.9 MB per expert, 64 times per layer. Counted
   *  here so "HTP host time" stops understating what the path costs. */
  void addStaging(uint64_t us, uint64_t bytes, int line) {
    std::lock_guard<std::mutex> lock(mutex_);
    staging_us_ += us;
    staging_bytes_ += bytes;
    auto &l = staging_by_line_[line];
    l[0] += us;
    l[1] += bytes;
    l[2] += 1;
  }

  ~HtpProfile() {
    if (level_ != 0)
      dump();
  }

private:
  struct Bucket {
    uint64_t calls = 0;
    uint64_t rows = 0; /**< summed M, so prefill batching is visible */
    /** MoE layer calls that took the M=1 HVX GEMV path (#80). Beside
        blocks it is the per-call proof of which path a row's numbers came
        from: m1_gemv=calls/calls with blocks=0 is the GEMV, 0/calls with
        blocks=4*calls the HMX loop. */
    uint64_t m1_calls = 0;
    /** Of those, the calls whose GEMV read VTCM slabs the DMA engine fed
        (#117): feed=calls/calls with a DMA ring: line is the feed, 0/calls
        the arena read. */
    uint64_t m1_feed_calls = 0;
    /** Expert cache misses paid before this row's calls (doc 52), and
        what they cost outside the call: the file read and the FastRPC
        register/release round trips. Both sit OUTSIDE host_us. */
    uint64_t misses = 0;
    uint64_t miss_read_us = 0;
    uint64_t miss_rpc_us = 0;
    uint64_t host_us = 0;
    uint64_t dsp_us = 0;
    uint64_t quant_us = 0;
    uint64_t swiglu_us = 0; /**< fused calls only; 0 elsewhere */
    /** swiglu_us is WORKER time that ran under the HMX (the MoE layer
        kernel's pool jobs, the conv block's stage units) rather than a
        synchronous pass, so the residual must not subtract it. Set by the
        call type that fills the bucket. */
    bool swiglu_hidden = false;
    /** MoE layer call only: the routing multiply and scatter-add, plus
        copying the FastRPC buffers to and from cached heap. Both are work
        no other call type does, so they get their own column rather than
        being folded into one that means something else. */
    uint64_t scatter_us = 0;
    /** MoE layer call only, and separate from scatter_us for the same
        reason scatter_us is separate from acc_us: these are the copies to
        and from the FastRPC buffers, they are the one thing the DMA change
        touches, and merged into scatter there is no reading of the profile
        that says whether it worked. */
    uint64_t stage_us = 0;
    /** MoE layer call only. quant_us used to hold all three; see
        hexkl_probe.h's HEXKL_PROBE_GATHER for why they split. */
    uint64_t gather_us = 0;
    uint64_t requant_us = 0;
    /** NOT microseconds: 64-row HMX blocks issued, summed over experts and
        calls. Printed as a count beside the times because the matmul scales
        with it -- 1776 routed rows can cost 32 blocks or 48 depending only
        on how the router spread them. */
    uint64_t blocks = 0;
    /** MoE layer call only. mm_us is the HMX issue loop measured rather
        than left as the subtraction residual; what the residual still holds
        after subtracting it is the honest "unnamed" figure. dma_kb and
        dma_first_us together give the DDR-to-VTCM weight rate. */
    uint64_t mm_us = 0;
    uint64_t dma_kb = 0;
    uint64_t dma_first_us = 0;
    /** The down drain apart from gate_up's, and the push itself. One bucket
        held both drains and the push sat unnamed in the residual; see
        hexkl_probe.h for why that left the 4.9 ms unreadable. */
    uint64_t drain_dn_us = 0;
    uint64_t dma_first_kb = 0;
    /** The layer call's own malloc and free -- about 12.8 MB a call. Was
        unnamed in the residual, which read 1019 us on the bake path and
        23888 on the weight-cache path with the kernel unchanged. */
    uint64_t alloc_us = 0;
    uint64_t push_us = 0;
    uint64_t dequant_us = 0;
    uint64_t acc_us = 0;
    uint64_t drain_us = 0;
    /** [#87] The DMA ring trace's per-call numbers, summed over calls
        (depth_max is the max). Printed on the DMA ring: line under weight
        DMA:, never folded into the columns above. */
    uint64_t dma_desc = 0;
    uint64_t dma_waits = 0;
    uint64_t dma_waits_blocked = 0;
    uint64_t dma_wait_us = 0;
    uint64_t dma_wait_act_us = 0;
    uint64_t dma_busy_lo_us = 0;
    uint64_t dma_busy_hi_us = 0;
    uint64_t dma_depth_max = 0;
    uint64_t dma_first_ready_us = 0;
    uint64_t dma_last_issue_us = 0;
    uint64_t dma_traces = 0; /**< per-descriptor dumps printed so far */
    /** [#88] MoE layer call only: the ION staging class the call rode
        (stage() in HtpComputeOps), whether both landed on ION, and the
        bytes of the non-ION in-args the stub hands the driver (the 48-byte
        primitive block plus the five routing/handle sequences). Printed
        as the staging: line so the transport column has its inventory
        beside it instead of in a plan. */
    size_t stage_act_bytes = 0;
    size_t stage_out_bytes = 0;
    bool stage_ion = true;
    uint64_t in_arg_bytes = 0;
    /** [#141] Of calls, how many rode the dspqueue; then in_arg_bytes is
        the request message, not the stub's non-ION in-args. */
    uint64_t dspq_calls = 0;
    uint64_t direct_io_calls =
      0; /**< staging capacities include spare buffers */
  };

  HtpProfile() {
    const char *env = std::getenv("NNTR_HTP_PROFILE");
    level_ = (env != nullptr) ? std::atoi(env) : 0;
    const char *trace_env = std::getenv("NNTR_HTP_DMA_TRACE");
    dma_trace_calls_ = (trace_env != nullptr) ? std::atoi(trace_env) : 3;
    // Captured at construction, not at static-destruction time in dump():
    // HtpProfile::global() is always reached through a HtpBackend::global()
    // call first (every accelerated entry point fetches the session handle
    // before it ever touches the profile), so construction order is safe;
    // destruction order is not something to lean on for a second singleton.
    qos_mode_ = HtpBackend::global().qosMode();
  }

  static double ms(uint64_t us) { return static_cast<double>(us) / 1000.0; }

  void dump() {
    // stderr, not the logger: this runs at static destruction, and the run
    // is driven from an adb shell where stderr is what the operator sees.
    std::fprintf(
      stderr,
      "\n[HTP-PROFILE] level=%d qos_mode=%d (2=poll 1=PM "
      "0=interrupt-driven -- every transport number below is only "
      "comparable to 34_fc_measured.md's 326us/call at qos_mode=2)\n",
      level_, qos_mode_);

    uint64_t invoke_us = 0;
    for (const auto &entry : buckets_)
      invoke_us += entry.second.host_us;

    std::fprintf(stderr,
                 "[HTP-PROFILE] weight registration (once per weight, so all "
                 "of it lands in the first forward)\n");
    std::fprintf(stderr,
                 "[HTP-PROFILE]   weights registered : %llu\n"
                 "[HTP-PROFILE]   convert to registry: %10.1f ms  (%.2f "
                 "ms/weight)\n"
                 "[HTP-PROFILE]   register FastRPC   : %10.1f ms  (%.2f "
                 "ms/weight)\n"
                 "[HTP-PROFILE]   alloc + other      : %10.1f ms\n"
                 "[HTP-PROFILE]   registration total : %10.1f ms\n"
                 "[HTP-PROFILE]   rpcmem/ION buffer  : %llu/%llu weights "
                 "(the rest used plain heap -- pinned+mapped per call)\n",
                 (unsigned long long)reg_calls_, ms(convert_us_),
                 reg_calls_ ? ms(convert_us_) / reg_calls_ : 0.0, ms(rpc_us_),
                 reg_calls_ ? ms(rpc_us_) / reg_calls_ : 0.0,
                 ms(reg_total_us_ - convert_us_ - rpc_us_), ms(reg_total_us_),
                 (unsigned long long)reg_ion_calls_,
                 (unsigned long long)reg_calls_);

    // M==1 is decode's shape, but an expert that drew a single token during
    // prefill lands in the same row -- read the two as shapes, not phases.
    std::fprintf(stderr,
                 "[HTP-PROFILE] layer calls (M==1 is decode's shape)\n");
    for (const auto &entry : buckets_) {
      const unsigned k = std::get<0>(entry.first);
      const unsigned n = std::get<1>(entry.first);
      const bool decode = std::get<2>(entry.first);
      // The kind names the row on both shapes: at decode the attention
      // FC calls and the MoE layer share K=2048 N=2048, so "M==1" alone
      // would print two rows nobody could tell apart. The MoE row stays
      // the bare "M==1" every handoff greps for.
      static const char *const kind_name[] = {"M>1", "M>1 dense", "M>1 conv",
                                              "M>1 FC"};
      static const char *const kind_name_m1[] = {"M==1", "M==1 dense",
                                                 "M==1 conv", "M==1 FC"};
      const int kind = std::get<3>(entry.first);
      const Bucket &b = entry.second;
      std::fprintf(stderr,
                   "[HTP-PROFILE]   K=%-5u N=%-5u %-10s calls=%-7llu "
                   "rows=%-8llu host=%9.1f ms (%7.1f us/call)",
                   k, n, (decode ? kind_name_m1 : kind_name)[kind],
                   (unsigned long long)b.calls, (unsigned long long)b.rows,
                   ms(b.host_us),
                   b.calls ? static_cast<double>(b.host_us) / b.calls : 0.0);
      if (level_ >= 2 && b.calls != 0) {
        const double dsp_per = static_cast<double>(b.dsp_us) / b.calls;
        const double host_per = static_cast<double>(b.host_us) / b.calls;
        const double quant_per = static_cast<double>(b.quant_us) / b.calls;
        const double swiglu_per = static_cast<double>(b.swiglu_us) / b.calls;
        const double scatter_per = static_cast<double>(b.scatter_us) / b.calls;
        const double dequant_per = static_cast<double>(b.dequant_us) / b.calls;
        const double acc_per = static_cast<double>(b.acc_us) / b.calls;
        const double drain_per = static_cast<double>(b.drain_us) / b.calls;
        const double stage_per = static_cast<double>(b.stage_us) / b.calls;
        const double gather_per = static_cast<double>(b.gather_us) / b.calls;
        const double requant_per = static_cast<double>(b.requant_us) / b.calls;
        const double mm_meas_per = static_cast<double>(b.mm_us) / b.calls;
        const double drain_dn_per =
          static_cast<double>(b.drain_dn_us) / b.calls;
        const double push_per = static_cast<double>(b.push_us) / b.calls;
        const double alloc_per = static_cast<double>(b.alloc_us) / b.calls;
        // What the accelerator actually exists for, by subtraction: the DSP
        // clock minus every stage that is a format change or a wait. Nothing
        // on the DSP times the HMX issue loop directly, and adding a probe
        // inside it would perturb what it measures -- this residue is the
        // honest number, and it is the one to compare against the layer's
        // MAC count. It also absorbs whatever the probes do not name, so
        // treat it as an upper bound on the matmul, not an exact figure.
        /** mm is the residual, so every named stage has to be subtracted --
           scatter included, or the MoE layer call's scatter time would be
           reported as matmul. Except SWIGLU on the MoE layer kernel's rows
           (MoE, dense, conv; upstream doc 53 section 8.4) and on the M=1
           GEMV path (#102): there it is WORKER time inside pool jobs
           (hexkl_mm_u8i4_moe.c's moe_worker_probe_add, hexkl_conv_block.c's
           cb_stage_probe_add) or the sum of every lane's wall time inside
           the two GEMV stages -- work that ran under the HMX or beside the
           caller rather than instead of it, not a stage on this clock, so
           subtracting it made the residual negative (3.3 ms of hidden work
           against a 4.0 ms conv call). Left in the sum for the fused layer
           call, where SWIGLU is its own synchronous elementwise pass
           (htp_moe_opts.h). */
        const bool hidden = b.swiglu_hidden || b.m1_calls != 0;
        const double mm_per = htp_moe_row_rest_us(
          dsp_per,
          quant_per + dequant_per + acc_per + drain_per + scatter_per +
            stage_per + gather_per + requant_per + mm_meas_per + drain_dn_per +
            push_per + alloc_per,
          swiglu_per, hidden);
        std::fprintf(
          stderr,
          "  dsp=%7.1f us/call (%4.1f%%) transport=%7.1f us/call"
          "  [quant %.1f gather %.1f requant %.1f swiglu%s %.1f "
          "dequant %.1f acc %.1f drain %.1f+%.1f push %.1f "
          "scatter %.1f alloc %.1f "
          "stage %.1f mm %.1f | rest<=%.1f (%.1f%% of host) "
          "blocks=%llu m1_gemv=%llu/%llu feed=%llu/%llu]",
          dsp_per, host_per > 0.0 ? 100.0 * dsp_per / host_per : 0.0,
          host_per - dsp_per, quant_per, gather_per, requant_per,
          hidden ? "(hidden)" : "", swiglu_per, dequant_per, acc_per, drain_per,
          drain_dn_per, push_per, scatter_per, alloc_per, stage_per,
          mm_meas_per, mm_per, host_per > 0.0 ? 100.0 * mm_per / host_per : 0.0,
          (unsigned long long)b.blocks, (unsigned long long)b.m1_calls,
          (unsigned long long)b.calls, (unsigned long long)b.m1_feed_calls,
          (unsigned long long)b.calls);
      }
      if (level_ >= 2 && b.calls != 0 && b.dma_first_us != 0) {
        // The first weight wait happens with an empty ring, so it times a
        // transfer rather than a pipeline. The size it covers is reported
        // alongside rather than derived from the shapes: since the weight
        // started arriving in chunks that wait covers one chunk, and
        // dividing its time by the whole weight claimed 114.7 GB/s for a
        // link that does about 33.
        const double first_us = static_cast<double>(b.dma_first_us) / b.calls;
        const double first_kb = static_cast<double>(b.dma_first_kb) / b.calls;
        const double kb = static_cast<double>(b.dma_kb) / b.calls;
        const double dsp_us = static_cast<double>(b.dsp_us) / b.calls;
        std::fprintf(stderr,
                     "\n[HTP-PROFILE]     weight DMA: %.0f KB/call, first "
                     "%.0f KB took %.0f us = %.1f GB/s; averaged over the "
                     "call %.1f GB/s",
                     kb, first_kb, first_us,
                     first_us > 0.0 ? first_kb * 1.024 / first_us : 0.0,
                     dsp_us > 0.0 ? kb * 1.024 / dsp_us : 0.0);
      } else if (level_ >= 2 && b.calls != 0 && b.m1_calls == b.calls &&
                 b.m1_feed_calls == 0) {
        // [#102] The GEMV path reads the weights straight from the arena and
        // never waits on the ring, so there is no first-wait rate to print;
        // this line keeps the row block's line count and says what swiglu
        // means on this path (lanes busy = swiglu / mm, 6 lanes on v79).
        // ponytail: no GB/s here -- the bucket knows K, N_out and M but not
        // inter, so the rate stays hand arithmetic (weight bytes / mm).
        // Upgrade: count DMA_KB-style bytes on the GEMV path in the kernel
        // (a skel change), or pass the bytes to addInvokeMoeLayer.
        const double mm_us = static_cast<double>(b.mm_us) / b.calls;
        std::fprintf(stderr,
                     "\n[HTP-PROFILE]     weight DMA: n/a (direct arena read "
                     "inside mm, no ring; swiglu = lane-time, %.2f lanes busy "
                     "over mm)",
                     mm_us > 0.0
                       ? static_cast<double>(b.swiglu_us) / b.calls / mm_us
                       : 0.0);
      }
      if (level_ >= 2 && b.calls != 0 && b.dma_desc != 0) {
        // [#87] How the call used the ring. busy is a bracket, not a point
        // (hexkl_dma_trace.h), so the engine rate is a range: bytes over
        // busy_hi up to bytes over busy_lo. gu / dn restate the drain
        // columns above so the three wait sites read side by side.
        const double n = static_cast<double>(b.calls);
        const double kb = static_cast<double>(b.dma_kb) / n;
        const double busy_lo = static_cast<double>(b.dma_busy_lo_us) / n;
        const double busy_hi = static_cast<double>(b.dma_busy_hi_us) / n;
        std::fprintf(
          stderr,
          "\n[HTP-PROFILE]     DMA ring: desc=%.0f/call waits=%.0f (blocked "
          "%.1f) wait=%.1f us [act %.1f gu %.1f+dn %.1f]",
          static_cast<double>(b.dma_desc) / n,
          static_cast<double>(b.dma_waits) / n,
          static_cast<double>(b.dma_waits_blocked) / n,
          static_cast<double>(b.dma_wait_us) / n,
          static_cast<double>(b.dma_wait_act_us) / n,
          static_cast<double>(b.drain_us) / n,
          static_cast<double>(b.drain_dn_us) / n);
        if (busy_hi > 0.0) {
          std::fprintf(stderr, " busy=%.0f..%.0f us -> engine %.1f..%.1f GB/s",
                       busy_lo, busy_hi, kb * 1.024 / busy_hi,
                       busy_lo > 0.0 ? kb * 1.024 / busy_lo : 0.0);
        } else {
          // The skel refuses the union when a call pushed more than the
          // trace table holds (prefill can); the counts above still stand.
          std::fprintf(stderr, " busy=n/a (trace truncated past %u pushes)",
                       static_cast<unsigned>(HEXKL_DMA_TRACE_MAX_PUSH));
        }
        std::fprintf(stderr,
                     "  depth max=%llu  first expert ready at %.0f us  last "
                     "issue at %.0f us of %.0f",
                     (unsigned long long)b.dma_depth_max,
                     static_cast<double>(b.dma_first_ready_us) / n,
                     static_cast<double>(b.dma_last_issue_us) / n,
                     static_cast<double>(b.dsp_us) / n);
      }
      if (level_ >= 2 && b.stage_act_bytes != 0) {
        if (b.direct_io_calls)
          std::fprintf(stderr,
                       "\n[HTP-PROFILE]     direct model I/O: %llu/%llu calls; "
                       "staging capacities below include spare buffers",
                       (unsigned long long)b.direct_io_calls,
                       (unsigned long long)b.calls);
        // [#88] The MoE call's per-call transport inventory. The driver's
        // cache maintenance covers the whole staging buffer, so the class
        // size is the number that explains a transport figure; rpc allocs
        // is process-wide and must not grow with calls; the in-args are the
        // driver-copied bytes outside ION (prim block + 5 sequences; the
        // sixth sequence and the rout are the two staging buffers).
        std::fprintf(stderr,
                     "\n[HTP-PROFILE]     staging: act %zu B out %zu B ion=%c  "
                     "rpc allocs=%u (session)  ",
                     b.stage_act_bytes, b.stage_out_bytes,
                     b.stage_ion ? 'y' : 'n', HtpRpcBuffer::allocCount());
        if (b.dspq_calls != 0) {
          // [#141] The request message is copied into the queue's shared
          // memory; the act/out classes above are the queue's buffers.
          std::fprintf(stderr, "via=dspq %llu/%llu calls msg=%llu B",
                       (unsigned long long)b.dspq_calls,
                       (unsigned long long)b.calls,
                       (unsigned long long)b.in_arg_bytes);
        } else {
          std::fprintf(stderr, "non-ION in-args=6/%llu B",
                       (unsigned long long)b.in_arg_bytes);
        }
      }
      if (b.misses != 0 || b.miss_rpc_us != 0) {
        std::fprintf(
          stderr,
          "\n[HTP-PROFILE]     expert misses: %llu (%.2f/call), "
          "file read %.1f ms (%.2f ms/call, %.2f ms/miss), "
          "swap rpc %.1f ms (%.2f ms/call) -- outside "
          "host= above",
          (unsigned long long)b.misses, static_cast<double>(b.misses) / b.calls,
          ms(b.miss_read_us), ms(b.miss_read_us) / static_cast<double>(b.calls),
          b.misses ? ms(b.miss_read_us) / b.misses : 0.0, ms(b.miss_rpc_us),
          ms(b.miss_rpc_us) / static_cast<double>(b.calls));
      }
      std::fprintf(stderr, "\n");
    }

    if (graph_calls_ != 0) {
      const double n = static_cast<double>(graph_calls_);
      char names[128];
      std::fprintf(stderr,
                   "[HTP-PROFILE]   graph: calls=%llu ops/call=%.2f "
                   "resident=%s dsp=%.1f us/call op_pcyc=%.0f/call pcyc/op:",
                   (unsigned long long)graph_calls_,
                   static_cast<double>(graph_ops_) / n,
                   htp_graph_kinds_str(graph_resident_, names, sizeof(names)),
                   static_cast<double>(graph_dsp_us_) / n,
                   static_cast<double>(graph_op_pcyc_) / n);
      for (unsigned k = 0; k < HTP_OP_KIND_N; ++k) {
        if (graph_kind_n_[k] != 0)
          std::fprintf(stderr, " %s=%.0f", htp_graph_kind_name(k),
                       static_cast<double>(graph_kind_pcyc_[k]) /
                         static_cast<double>(graph_kind_n_[k]));
      }
      std::fprintf(stderr, "\n");
    }
    std::fprintf(
      stderr,
      "[HTP-PROFILE] layer calls total : %10.1f ms\n"
      "[HTP-PROFILE] arm staging memcpy: %10.1f ms  (%.1f MB in+out, "
      "%.1f GB/s) -- outside every host= above\n"
      "[HTP-PROFILE] HTP host time     : %10.1f ms "
      "(registration + layer calls + staging)\n\n",
      ms(invoke_us), ms(staging_us_),
      static_cast<double>(staging_bytes_) / (1024.0 * 1024.0),
      staging_us_ ? static_cast<double>(staging_bytes_) / staging_us_ / 1000.0
                  : 0.0,
      ms(reg_total_us_ + invoke_us + staging_us_));
    for (const auto &l : staging_by_line_)
      if (l.second[0] >= 1000)
        std::fprintf(stderr,
                     "[HTP-PROFILE]   staging at htp_compute_ops.cpp:%d: "
                     "%8.1f ms  %8.1f MB  %llu copies\n",
                     l.first, ms(l.second[0]),
                     static_cast<double>(l.second[1]) / (1024.0 * 1024.0),
                     (unsigned long long)l.second[2]);
    if (miss_total_ != 0) {
      std::fprintf(stderr,
                   "[HTP-PROFILE] expert cache misses: %llu, file read "
                   "%.1f ms (%.2f ms/miss), swap rpc %.1f ms "
                   "(%.2f ms/miss)\n\n",
                   (unsigned long long)miss_total_, ms(miss_read_total_us_),
                   ms(miss_read_total_us_) / miss_total_,
                   ms(miss_rpc_total_us_),
                   ms(miss_rpc_total_us_) / miss_total_);
    }
    if (prefetch_n_ != 0) {
      std::fprintf(stderr,
                   "[HTP-PROFILE] expert prefetch: %llu experts read under "
                   "the layer calls, exposed wait %.1f ms (%.2f ms/expert), "
                   "register rpc %.1f ms (%.2f ms/expert)\n",
                   (unsigned long long)prefetch_n_, ms(prefetch_wait_us_),
                   ms(prefetch_wait_us_) / prefetch_n_, ms(prefetch_rpc_us_),
                   ms(prefetch_rpc_us_) / prefetch_n_);
      // [doc 52 sections 10.18, 10.20] How far ahead the readers were: the
      // experts already read when their layer asked for them, and the
      // cores each side ran on.
      auto cpus = [](uint32_t m) {
        std::string out;
        for (int c = 0; c < 32; ++c)
          if (m & (1u << c))
            out += (out.empty() ? "" : ",") + std::to_string(c);
        return out.empty() ? std::string("?") : out;
      };
      std::fprintf(stderr,
                   "[HTP-PROFILE] expert prefetch progress: %llu of %llu read "
                   "when their layer asked (%.0f%%), caller cpus {%s}, reader "
                   "cpus {%s}\n\n",
                   (unsigned long long)prefetch_done_on_entry_,
                   (unsigned long long)prefetch_n_,
                   100.0 * prefetch_done_on_entry_ / prefetch_n_,
                   cpus(prefetch_caller_cpus_).c_str(),
                   cpus(prefetch_reader_cpus_).c_str());
      // [doc 52 section 10.30] The per-expert read on a reader thread: page
      // cache 1.4 ms with 2 readers and 2.1 with 4 (they share the write
      // bandwidth), flash several times that.
      if (prefetch_reads_ != 0)
        std::fprintf(stderr,
                     "[HTP-PROFILE] expert prefetch readers: %.2f ms/expert on "
                     "a reader, %llu experts\n\n",
                     ms(prefetch_read_us_) / prefetch_reads_,
                     (unsigned long long)prefetch_reads_);
    }
  }

  int level_ = 0;
  int dma_trace_calls_ = 3;
  int qos_mode_ = 0;
  std::mutex mutex_;
  uint64_t reg_calls_ = 0;
  uint64_t reg_ion_calls_ = 0;
  uint64_t reg_total_us_ = 0;
  uint64_t staging_us_ = 0;
  uint64_t staging_bytes_ = 0;
  /** us, bytes, copies by the caller's line of stagedMemcpy. */
  std::map<int, std::array<uint64_t, 3>> staging_by_line_;
  uint64_t convert_us_ = 0;
  uint64_t rpc_us_ = 0;
  uint64_t graph_calls_ = 0; /**< [#85] addInvokeForward */
  uint64_t graph_ops_ = 0;
  uint64_t graph_dsp_us_ = 0;
  uint64_t graph_op_pcyc_ = 0;
  uint32_t graph_resident_ = 0; /**< [#130] the description's mask */
  uint64_t graph_kind_n_[HTP_OP_KIND_N] = {0};
  uint64_t graph_kind_pcyc_[HTP_OP_KIND_N] = {0};
  uint64_t miss_pending_ = 0, miss_read_pending_us_ = 0,
           miss_rpc_pending_us_ = 0;
  uint64_t miss_total_ = 0, miss_read_total_us_ = 0, miss_rpc_total_us_ = 0;
  uint64_t prefetch_n_ = 0, prefetch_wait_us_ = 0, prefetch_rpc_us_ = 0;
  uint64_t prefetch_done_on_entry_ = 0;
  uint64_t prefetch_read_us_ = 0, prefetch_reads_ = 0;
  uint32_t prefetch_caller_cpus_ = 0, prefetch_reader_cpus_ = 0;
  /** (K, N, M == 1, kind): kind 0 is every layer call, 1 the dense FFN
   *  through the MoE layer kernel (doc 51). */
  std::map<std::tuple<unsigned, unsigned, bool, int>, Bucket> buckets_;
};

/**
 * @brief memcpy that charges itself to HtpProfile's staging line.
 *
 * Every host_us window in this file starts AFTER the activation is copied
 * into the rpcmem buffer and ends BEFORE the result is copied back out, on
 * purpose: that is what keeps `transport = host - dsp` a FastRPC number
 * rather than a FastRPC-plus-memcpy number. The consequence was that the
 * copies appeared nowhere at all, and on this model's MoE they are not
 * small -- ~450 KB in and ~450 KB out per expert call, 64 calls per layer.
 * Same copy, one accumulator, so the profile's bottom line stops
 * understating what the path costs.
 */
/**
 * @brief NNTR_L2_CHECK: does the DSP still hand back non-finite values?
 *
 * The two L2 failures in docs/htp_attention/43_moe_ffn_measured_next_
 * levers.md section 7 were diagnosed as hvx_recip_qf32 returning NaN for
 * any gate at or below hvx_swiglu_row_f32's exp clamp. That diagnosis is
 * only worth what a device can confirm, and the model's own output cannot
 * confirm it: wrong text is equally consistent with a dozen other causes.
 * This is the discriminator. If it counts zero and the text is still
 * wrong, the SwiGLU clamp is NOT the (whole) bug and the search moves
 * elsewhere; if it counts non-zero, whatever skel is on the device does
 * not have the fix in it.
 *
 * Cheap where it matters most: a NaN SwiGLU lane lands in its row's
 * requantization scale (hvx_quant_rows_u8_params scans the whole row for
 * min/max), so scanning m_pad floats -- 64 of them -- catches it before
 * the value has spread anywhere. Off unless the env var is set.
 */
inline bool l2CheckEnabled() {
  static const bool on = std::getenv("NNTR_L2_CHECK") != nullptr;
  return on;
}

/** @brief NNTR_L2_DIFF: run both MoE FFN paths and report their SNR. See
 *         HtpComputeOps::l2Diff for what the number separates. */
inline bool l2DiffEnabled() {
  static const bool on = std::getenv("NNTR_L2_DIFF") != nullptr;
  return on;
}

/**
 * @brief NNTR_L2_SHADOW: run the fused path, then hand the model the
 *        REFERENCE result instead of the fused one.
 *
 * Splits "the fused path computes wrong values" from "the fused path has a
 * side effect" -- the two remaining stories, and no SNR number can tell
 * them apart. Under this flag both paths execute in full, touching every
 * buffer, taking every lock, leaving every piece of state exactly as a
 * normal fused run would; only the floats the layer goes on to use are
 * swapped for the reference's.
 *
 *   text becomes correct -> the values are the fault, and the 67-80 dB
 *                           calls are worth chasing;
 *   text still broken    -> the values are NOT the fault (they match the
 *                           working path at 142 dB on 84% of calls
 *                           anyway), and the bug is a side effect --
 *                           aliasing, a clobbered workspace, retained
 *                           state -- that differencing output values can
 *                           never surface.
 *
 * Implies NNTR_L2_DIFF: the reference it hands over is the one l2Diff
 * already computes.
 */
inline bool l2ShadowEnabled() {
  static const bool on = std::getenv("NNTR_L2_SHADOW") != nullptr;
  return on;
}

/** @brief Reports the first non-finite element of @a v, once per call. */
inline void l2CheckFinite(const char *what, const float *v, size_t n,
                          unsigned M, unsigned K, unsigned N) {
  size_t bad = 0;
  size_t first = 0;
  for (size_t i = 0; i < n; ++i) {
    if (!std::isfinite(v[i])) {
      if (bad == 0)
        first = i;
      ++bad;
    }
  }
  if (bad != 0) {
    std::fprintf(stderr,
                 "[L2-CHECK] %s: %llu/%llu non-finite (first idx %llu = %g) "
                 "M=%u K=%u N=%u\n",
                 what, (unsigned long long)bad, (unsigned long long)n,
                 (unsigned long long)first, static_cast<double>(v[first]), M, K,
                 N);
  }
}

inline void stagedMemcpy(void *dst, const void *src, size_t bytes,
                         int line = __builtin_LINE()) {
  HtpProfile &p = HtpProfile::global();
  if (p.level() == 0) {
    std::memcpy(dst, src, bytes);
    return;
  }
  const uint64_t t0 = HtpProfile::nowUs();
  std::memcpy(dst, src, bytes);
  p.addStaging(HtpProfile::nowUs() - t0, bytes, line);
}

} // namespace

// --- nntrainer/nntrainer#4343: fp16 and quantized KV-cache attention ---
// Not in the in-process host twin: see the ops at the end of HtpComputeOps.
#ifndef NNTR_HTP_INPROC
namespace {

/**
 * @brief NNTR_HTP_ATTN_TRACE=1 logs every attention call with its host-side
 *        wall time and the skel's stats, so a model run shows where a step
 *        spends its time: in the kernel, or around the FastRPC call.
 */
bool attn_trace_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NNTR_HTP_ATTN_TRACE");
    return e && *e && *e != '0';
  }();
  return on;
}

/**
 * @brief Opt in to prefill ION staging with NNTR_HTP_ATTN_F16_ION=1.
 * Profiling reduced attention time, but expert prefetch remained the
 * bottleneck and whole-model prefill showed no clear improvement.
 */
bool attn_f16_ion_enabled() {
  static const bool on = [] {
    const char *e = std::getenv("NNTR_HTP_ATTN_F16_ION");
    return e && std::strcmp(e, "1") == 0;
  }();
  return on;
}

int64_t now_us() {
  return std::chrono::duration_cast<std::chrono::microseconds>(
           std::chrono::steady_clock::now().time_since_epoch())
    .count();
}

/**
 * @brief Few-query FP16 attention uses HVX across the supported head widths.
 *        Prefill continues to use the tiled HMX implementation.
 */
constexpr unsigned int kDecodeMaxRows = 5;
constexpr unsigned int kDecodeMaxHeadDim = 512;

} // namespace
#endif /* NNTR_HTP_INPROC */

class HtpComputeOps : public CpuComputeOps {
  using HostExpertBank = HtpExpertHostStaging<ExpertFileDesc>;

public:
  /** [#130] The per-token entry's call count at close, the one line the
   *  host E2E harness reads (plan 130 section 3.5): tokens are the calls
   *  that started at the list's first resident op. No RPC here: the
   *  session may already be closed. The expert prefetch readers (doc 52)
   *  are stopped first. */
  ~HtpComputeOps() override {
    if (auto bank = hostExpertBank())
      end_qs2_host_prefetch(bank->token());
    for (unsigned i = 0; i < 3; ++i)
      if (shared_io_calls_[i])
        std::fprintf(stderr,
                     "[HTP-SHARED-IO] kind=%u calls=%llu direct_bytes=%llu\n",
                     i, static_cast<unsigned long long>(shared_io_calls_[i]),
                     static_cast<unsigned long long>(shared_io_bytes_[i]));
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      prefetch_stop_ = true;
    }
    prefetch_cv_.notify_all();
    for (std::thread &t : prefetch_readers_)
      t.join();
    if (expert_updates_queued_)
      std::fprintf(stderr,
                   "[HTP-EXPERT-UPDATE] mode=%u queued=%llu fused_batches=%llu "
                   "rpc_batches=%llu rpc_us=%llu dsp_update_us=%llu "
                   "abandoned=%llu\n",
                   expertUpdateMode(),
                   (unsigned long long)expert_updates_queued_,
                   (unsigned long long)expert_update_fused_batches_,
                   (unsigned long long)expert_update_rpc_batches_,
                   (unsigned long long)expert_update_rpc_us_,
                   (unsigned long long)expert_update_dsp_us_,
                   (unsigned long long)expert_updates_abandoned_);
    if (qs2_reads_.load())
      std::fprintf(stderr,
                   "[HTP-QS2-IO] weights=%llu read_us=%llu copy_us=%llu "
                   "metadata_us=%llu copied_bytes=%llu demand_batches=%llu "
                   "parallel_batches=%llu max_misses=%llu\n",
                   (unsigned long long)qs2_reads_.load(),
                   (unsigned long long)qs2_read_us_.load(),
                   (unsigned long long)qs2_copy_us_.load(),
                   (unsigned long long)qs2_metadata_us_.load(),
                   (unsigned long long)qs2_copied_bytes_.load(),
                   (unsigned long long)qs2_demand_batches_,
                   (unsigned long long)qs2_parallel_batches_,
                   (unsigned long long)qs2_max_misses_);
    if (fwd_calls_ != 0) {
      std::fprintf(stderr,
                   "[HTP] graph: forward calls=%llu tokens=%llu "
                   "calls/token=%.2f\n",
                   (unsigned long long)fwd_calls_,
                   (unsigned long long)fwd_tokens_,
                   fwd_tokens_ ? static_cast<double>(fwd_calls_) /
                                   static_cast<double>(fwd_tokens_)
                               : 0.0);
    }
    if (cpu_fc_skipped_ != 0) // [#132 Part B] the harness's second line
      std::fprintf(stderr, "[HTP] graph: cpu fc skipped=%llu\n",
                   (unsigned long long)cpu_fc_skipped_);
    if (math_calls_[0] || sigmoid_calls_)
      std::fprintf(
        stderr,
        "[HTP] model math calls: norm=%llu add=%llu swiglu=%llu "
        "geglu=%llu rope=%llu rope_fp16=%llu cast_fp16=%llu "
        "sigmoid_topk=%llu\n",
        (unsigned long long)math_calls_[0], (unsigned long long)math_calls_[1],
        (unsigned long long)math_calls_[2], (unsigned long long)math_calls_[3],
        (unsigned long long)math_calls_[4], (unsigned long long)math_calls_[5],
        (unsigned long long)math_calls_[6], (unsigned long long)sigmoid_calls_);
    if (queued_fc_calls_ || queued_qkv_calls_)
      std::fprintf(stderr, "[HTP] queued projections: fc=%llu qkv=%llu\n",
                   (unsigned long long)queued_fc_calls_,
                   (unsigned long long)queued_qkv_calls_);
    if (rope_qkv_calls_)
      std::fprintf(stderr, "[HTP] fused QKV RoPE calls=%llu\n",
                   (unsigned long long)rope_qkv_calls_);
    if (consumer_norm_calls_)
      std::fprintf(stderr, "[HTP] norm consumer fusion calls=%llu pending=%d\n",
                   (unsigned long long)consumer_norm_calls_,
                   pending_norm_.norm != nullptr ||
                     pending_prefill_conv_norm_.norm != nullptr ||
                     pending_prefill_router_norm_.norm != nullptr ||
                     pending_prefill_qkv_norm_.norm != nullptr);
    if (fused_router_calls_ || fused_qkv_calls_)
      std::fprintf(stderr, "[HTP] fused RPC calls: router=%llu qkv_norm=%llu\n",
                   (unsigned long long)fused_router_calls_,
                   (unsigned long long)fused_qkv_calls_);
    if (rope_prefill_calls_)
      std::fprintf(stderr,
                   "[HTP-MATH] shape=M>1 op=rope_qkv_prefill calls=%llu "
                   "prepare_us=%llu rpc_us=%llu dsp_us=%llu copyout_us=%llu "
                   "bytes=%llu\n",
                   (unsigned long long)rope_prefill_calls_,
                   (unsigned long long)rope_prefill_prepare_us_,
                   (unsigned long long)rope_prefill_rpc_us_,
                   (unsigned long long)rope_prefill_dsp_us_,
                   (unsigned long long)rope_prefill_copyout_us_,
                   (unsigned long long)rope_prefill_staged_bytes_);
    static const char *const names[] = {"norm",      "add",     "swiglu",
                                        "geglu",     "rope",    "rope_fp16",
                                        "cast_fp16", "add_norm"};
    for (unsigned phase = 0; phase < 2; ++phase)
      for (unsigned op = 0; op < 8; ++op) {
        const auto &t = math_timing_[phase][op];
        if (!t.calls)
          continue;
        std::fprintf(
          stderr,
          "[HTP-MATH] shape=%s op=%s calls=%llu prepare_us=%llu "
          "rpc_us=%llu dsp_us=%llu copyout_us=%llu bytes=%llu\n",
          phase ? "M>1" : "M==1", names[op], (unsigned long long)t.calls,
          (unsigned long long)t.prepare_us, (unsigned long long)t.rpc_us,
          (unsigned long long)t.dsp_us, (unsigned long long)t.copyout_us,
          (unsigned long long)t.bytes);
      }
  }

  void sgemm_fp32(unsigned order, bool ta, bool tb, unsigned M, unsigned N,
                  unsigned K, float alpha, const float *a, unsigned lda,
                  const float *b, unsigned ldb, float beta, float *c,
                  unsigned ldc) override {
    const char *strict = std::getenv("NNTR_HTP_STRICT");
    if (strict && std::atoi(strict))
      throw std::runtime_error("NNTR_HTP_STRICT: CPU sgemm fallback forbidden");
    CpuComputeOps::sgemm_fp32(order, ta, tb, M, N, K, alpha, a, lda, b, ldb,
                              beta, c, ldc);
  }

  uint64_t queued_fc_calls_ = 0, queued_qkv_calls_ = 0;
  // Directly passed bytes, including inputs staging could already reuse.
  uint64_t shared_io_calls_[3] = {}, shared_io_bytes_[3] = {};
  uint64_t rope_qkv_calls_ = 0;
  bool rope_qkv_decode(unsigned qw, unsigned kw, unsigned hd, const float *cs,
                       const float *q, const float *k, const float *v,
                       float *qo, uint16_t *ko, uint16_t *vo) override {
    static const bool enabled = [] {
      const char *e = std::getenv("NNTR_HTP_QKV_ROPE_FUSION");
      return !e || std::atoi(e) != 0;
    }();
    if (!enabled || !qw || !kw || !hd || hd % 64 || qw % hd || kw % hd || !cs ||
        !q || !k || !v || !qo || !ko || !vo)
      return false;
    const size_t input_bytes =
      4u * (static_cast<size_t>(qw) + 2ull * kw + 2ull * hd);
    const size_t output_bytes = 4u * (static_cast<size_t>(qw) + kw);
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    if (!dspqReady(static_cast<remote_handle64>(HtpBackend::global().handle()),
                   input_bytes, output_bytes, sizeof(htp_dspq_rope_qkv_req)))
      return false;
    auto &st = *dspq_;
    float *in = reinterpret_cast<float *>(st.act->data());
    stagedMemcpy(in, q, qw * sizeof(float));
    stagedMemcpy(in + qw, k, kw * sizeof(float));
    stagedMemcpy(in + qw + kw, v, kw * sizeof(float));
    stagedMemcpy(in + qw + 2u * kw, cs, 2u * hd * sizeof(float));
    const htp_dspq_rope_qkv_req req = {
      HTP_DSPQ_OP_ROPE_QKV, ++st.seq, 0, qw, kw, hd};
    const int rc = dspqExchange(&req, sizeof(req), req.seq, input_bytes,
                                output_bytes, nullptr);
    if (rc != AEE_SUCCESS)
      throw std::runtime_error("HTP fused QKV RoPE failed: " +
                               std::to_string(rc));
    stagedMemcpy(qo, st.out->data(), qw * sizeof(float));
    stagedMemcpy(ko, st.out->data() + 4u * qw, kw * sizeof(uint16_t));
    stagedMemcpy(vo, st.out->data() + 4u * qw + 2u * kw, kw * sizeof(uint16_t));
    ++rope_qkv_calls_;
    return true;
  }

  uint64_t rope_prefill_calls_ = 0;
  uint64_t rope_prefill_prepare_us_ = 0, rope_prefill_rpc_us_ = 0;
  uint64_t rope_prefill_dsp_us_ = 0, rope_prefill_copyout_us_ = 0;
  uint64_t rope_prefill_staged_bytes_ = 0;

  bool ropeQkvPrefill(unsigned M, unsigned qw, unsigned kw, unsigned hd,
                      const float *cs, float *q, const float *k, const float *v,
                      uint16_t *ko, uint16_t *vo) {
    static const bool enabled = [] {
      const char *e = std::getenv("NNTR_HTP_PREFILL_QKV_PREPARE");
      return e && std::atoi(e) != 0;
    }();
    if (!enabled ||
        !nntr_rope_prefill_valid(M, qw, kw, hd, cs, q, k, v, ko, vo))
      return false;
    const uint64_t enter = HtpProfile::nowUs();
    const size_t nq = static_cast<size_t>(M) * qw;
    const size_t nk = static_cast<size_t>(M) * kw;
    const size_t nc = static_cast<size_t>(M) * 2u * hd;
    auto &backend = HtpBackend::global();
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    size_t staged_bytes = 0;
    auto input = [&](const float *src, size_t bytes, unsigned slot) {
      if (backend.isSharedRange(src, bytes))
        return src;
      auto *dst = reinterpret_cast<float *>(
        stage(rope_prefill_pool_[slot], bytes).data());
      stagedMemcpy(dst, src, bytes);
      staged_bytes += bytes;
      return static_cast<const float *>(dst);
    };
    float *rq = const_cast<float *>(input(q, nq * sizeof(float), 0));
    const float *rk = input(k, nk * sizeof(float), 1);
    const float *rv = input(v, nk * sizeof(float), 2);
    const float *rcs = input(cs, nc * sizeof(float), 3);
    const bool direct_k = backend.isSharedRange(ko, nk * sizeof(uint16_t));
    const bool direct_v = backend.isSharedRange(vo, nk * sizeof(uint16_t));
    uint16_t *rko =
      direct_k ? ko
               : reinterpret_cast<uint16_t *>(
                   stage(rope_prefill_pool_[4], nk * sizeof(uint16_t)).data());
    uint16_t *rvo =
      direct_v ? vo
               : reinterpret_cast<uint16_t *>(
                   stage(rope_prefill_pool_[5], nk * sizeof(uint16_t)).data());
    uint32_t dsp_us = 0;
    const uint64_t start = HtpProfile::nowUs();
    const int err = nntr_hvx_rope_qkv_prefill_f32(
      static_cast<remote_handle64>(backend.handle()), M, qw, kw, hd, rcs,
      static_cast<int>(nc), rq, static_cast<int>(nq), rk, static_cast<int>(nk),
      rv, static_cast<int>(nk), rq, static_cast<int>(nq), rko,
      static_cast<int>(nk), rvo, static_cast<int>(nk), &dsp_us, 1);
    const uint64_t end = HtpProfile::nowUs();
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP prefill QKV RoPE failed: " +
                               std::to_string(err));
    if (rq != q) {
      stagedMemcpy(q, rq, nq * sizeof(float));
      staged_bytes += nq * sizeof(float);
    }
    if (!direct_k) {
      stagedMemcpy(ko, rko, nk * sizeof(uint16_t));
      staged_bytes += nk * sizeof(uint16_t);
    }
    if (!direct_v) {
      stagedMemcpy(vo, rvo, nk * sizeof(uint16_t));
      staged_bytes += nk * sizeof(uint16_t);
    }
    ++rope_prefill_calls_;
    rope_prefill_prepare_us_ += start - enter;
    rope_prefill_rpc_us_ += end - start;
    rope_prefill_dsp_us_ += dsp_us;
    rope_prefill_copyout_us_ += HtpProfile::nowUs() - end;
    rope_prefill_staged_bytes_ += staged_bytes;
    return true;
  }

  struct PendingNorm {
    std::string consumer;
    unsigned N = 0;
    float eps = 0;
    bool add = false;
    std::vector<float> inputs;
    float *sum = nullptr, *norm = nullptr;
  } pending_norm_;
  uint64_t consumer_norm_calls_ = 0;

  struct PendingPrefillNorm {
    unsigned rows = 0, width = 0;
    float eps = 0;
    const float *x = nullptr, *residual = nullptr;
    std::vector<float> gamma;
    float *sum = nullptr, *norm = nullptr;
  } pending_prefill_conv_norm_, pending_prefill_router_norm_,
    pending_prefill_qkv_norm_;

  enum class PrefillNormConsumer { Conv, Router, Qkv };

  static bool prefillNormOverlap(const void *a, size_t an, const void *b,
                                 size_t bn) {
    if (!a || !b || !an || !bn)
      return false;
    const auto aa = reinterpret_cast<uintptr_t>(a);
    const auto bb = reinterpret_cast<uintptr_t>(b);
    return aa <= bb ? bb - aa < an : aa - bb < bn;
  }

  bool defer_prefill_conv_norm(unsigned rows, unsigned width, float eps,
                               const float *x, const float *residual,
                               const float *gamma, float *sum,
                               float *norm) override {
    return deferPrefillNorm(PrefillNormConsumer::Conv, rows, width, eps, x,
                            residual, gamma, sum, norm);
  }

  bool defer_prefill_router_norm(unsigned rows, unsigned width, float eps,
                                 const float *x, const float *residual,
                                 const float *gamma, float *sum,
                                 float *norm) override {
    return deferPrefillNorm(PrefillNormConsumer::Router, rows, width, eps, x,
                            residual, gamma, sum, norm);
  }

  bool defer_prefill_qkv_norm(unsigned rows, unsigned width, float eps,
                              const float *x, const float *residual,
                              const float *gamma, float *sum,
                              float *norm) override {
    return deferPrefillNorm(PrefillNormConsumer::Qkv, rows, width, eps, x,
                            residual, gamma, sum, norm);
  }

  bool deferPrefillNorm(PrefillNormConsumer consumer, unsigned rows,
                        unsigned width, float eps, const float *x,
                        const float *residual, const float *gamma, float *sum,
                        float *norm) {
    static const bool enabled = [] {
      const char *e = std::getenv("NNTR_HTP_PREFILL_CONV_NORM");
      const char *stats = std::getenv("NNTR_ACT_STATS");
      return e && std::atoi(e) != 0 && (!stats || std::atoi(stats) == 0);
    }();
    static const bool router_enabled = [] {
      const char *e = std::getenv("NNTR_HTP_PREFILL_ROUTER_NORM");
      const char *reuse = std::getenv("NNTR_HTP_ROUTER_Q4_QUANT_REUSE");
      const char *stats = std::getenv("NNTR_ACT_STATS");
      return e && std::atoi(e) != 0 && reuse && std::atoi(reuse) != 0 &&
             (!stats || std::atoi(stats) == 0);
    }();
    static const bool qkv_enabled = [] {
      const char *e = std::getenv("NNTR_HTP_PREFILL_QKV_NORM");
      const char *stats = std::getenv("NNTR_ACT_STATS");
      return e && std::atoi(e) != 0 && (!stats || std::atoi(stats) == 0);
    }();
    const bool router = consumer == PrefillNormConsumer::Router;
    const bool qkv = consumer == PrefillNormConsumer::Qkv;
    const uint64_t count = static_cast<uint64_t>(rows) * width;
    if (!(router ? router_enabled
          : qkv  ? qkv_enabled
                 : enabled) ||
        rows < 2 || !width || width % 32 || count > INT32_MAX / sizeof(float) ||
        !x || !gamma || !norm || (residual && !sum))
      return false;
    if (router &&
        (rows <= 4 || rows > 2048 || width > 4096 || width % 128 ||
         rows > fcMaxRows(width) || HtpProfile::global().level() >= 3 ||
         uint64_t((rows + 63u) & ~63u) * width > (4u << 20)))
      return false;
    if (qkv &&
        (rows > 2048 || width > 4096 || width % 128 || rows > fcMaxRows(width)))
      return false;
    const size_t bytes = static_cast<size_t>(count) * sizeof(float);
    auto &backend = HtpBackend::global();
    if (!backend.isSharedRange(x, bytes) ||
        !backend.isSharedRange(norm, bytes) ||
        (residual && (!backend.isSharedRange(residual, bytes) ||
                      !backend.isSharedRange(sum, bytes))))
      return false;
    for (const float *dst : {sum, norm}) {
      if (!dst)
        continue;
      if (prefillNormOverlap(dst, bytes, gamma, width * sizeof(float)))
        return false;
      for (const float *src : {x, residual})
        if (src != dst && prefillNormOverlap(dst, bytes, src, bytes))
          return false;
    }
    if (prefillNormOverlap(sum, bytes, norm, bytes))
      return false;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    if (pending_prefill_conv_norm_.norm || pending_prefill_router_norm_.norm ||
        pending_prefill_qkv_norm_.norm || pending_norm_.norm)
      throw std::runtime_error("HTP deferred norm was not consumed");
    auto &p = router ? pending_prefill_router_norm_
              : qkv  ? pending_prefill_qkv_norm_
                     : pending_prefill_conv_norm_;
    // The graph can unload the RMSNorm weight before its designated consumer.
    p.gamma.assign(gamma, gamma + width);
    p.rows = rows;
    p.width = width;
    p.eps = eps;
    p.x = x;
    p.residual = residual;
    p.sum = residual ? sum : nullptr;
    p.norm = norm;
    return true;
  }

  void discard_deferred_norm() noexcept override {
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    pending_norm_.norm = nullptr;
    pending_norm_.sum = nullptr;
    pending_prefill_conv_norm_ = {};
    pending_prefill_router_norm_ = {};
    pending_prefill_qkv_norm_ = {};
    prepared_moe_cookie_ = prepared_moe_lease_ = 0;
    moe_input_reuse_.cancel();
  }

  bool defer_decode_norm(const std::string &consumer, unsigned N, float eps,
                         const float *x, const float *residual,
                         const float *gamma, float *sum, float *norm) override {
    static const bool enabled = [] {
      const char *e = std::getenv("NNTR_HTP_NORM_CONSUMER_FUSION");
      return !e || std::atoi(e) != 0;
    }();
    if (!enabled ||
        (consumer != "conv" && consumer != "router" && consumer != "qkv") ||
        !N || N % 32 || N > 4096 || !x || !gamma || !norm || (residual && !sum))
      return false;
    if (consumer == "router") {
      const char *queue = std::getenv("NNTR_HTP_DSPQ");
      const char *router = std::getenv("NNTR_HTP_DSPQ_ROUTER");
      if ((queue && std::atoi(queue) == 0) ||
          (router && std::atoi(router) == 0) ||
          std::getenv("NNTR_HTP_ROUTER_U8I8"))
        return false;
    }
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    if (pending_norm_.norm || pending_prefill_conv_norm_.norm ||
        pending_prefill_router_norm_.norm || pending_prefill_qkv_norm_.norm)
      throw std::runtime_error("HTP deferred norm was not consumed");
    auto &p = pending_norm_;
    p.consumer = consumer;
    p.N = N;
    p.eps = eps;
    p.add = residual != nullptr;
    p.sum = sum;
    p.norm = norm;
    p.inputs.resize(static_cast<size_t>(N) * (p.add ? 3 : 2));
    std::memcpy(p.inputs.data(), x, N * sizeof(float));
    if (p.add)
      std::memcpy(p.inputs.data() + N, residual, N * sizeof(float));
    std::memcpy(p.inputs.data() + N * (p.add ? 2 : 1), gamma,
                N * sizeof(float));
    return true;
  }

  PendingNorm *pendingNorm(const char *consumer, const float *x, unsigned M,
                           unsigned K) {
    auto &p = pending_norm_;
    if (!p.norm)
      return nullptr;
    if (p.consumer != consumer || p.norm != x || M != 1 || p.N != K)
      throw std::runtime_error("HTP deferred norm consumer mismatch");
    return &p;
  }

  void publishNorm(const float *result) {
    auto &p = pending_norm_;
    if (p.sum)
      stagedMemcpy(p.sum, result, p.N * sizeof(float));
    stagedMemcpy(p.norm, result + p.N, p.N * sizeof(float));
    p.norm = nullptr;
    p.sum = nullptr;
    ++consumer_norm_calls_;
  }

  bool inference_math_fp32(InferenceMath op, unsigned M, unsigned N,
                           unsigned chunk, float eps, float scale,
                           const float *x, const float *aux, const float *param,
                           void *out) override {
    return inferenceMath(op, M, N, chunk, eps, scale, x, aux, param, out,
                         nullptr);
  }

  bool add_rmsnorm_fp32(unsigned M, unsigned N, float eps, const float *x,
                        const float *residual, const float *gamma, float *sum,
                        float *norm) override {
    if (!norm)
      return false;
    return inferenceMath(InferenceMath::AddRmsNorm, M, N, N, eps, 1.0f, x,
                         residual, gamma, sum, norm);
  }

  bool inferenceMath(InferenceMath op, unsigned M, unsigned N, unsigned chunk,
                     float eps, float scale, const float *x, const float *aux,
                     const float *param, void *out, float *norm) {
    static const bool detail = [] {
      const char *e = std::getenv("NNTR_HTP_MATH_PROFILE");
      return e && std::atoi(e) != 0;
    }();
    const uint64_t enter = detail ? HtpProfile::nowUs() : 0;
    const unsigned kind = static_cast<unsigned>(op);
    const size_t count = static_cast<size_t>(M) * N;
    const bool half =
      op == InferenceMath::RopeFp16 || op == InferenceMath::CastFp16;
    const size_t plen =
      (op == InferenceMath::RmsNorm || op == InferenceMath::AddRmsNorm)
        ? (param ? chunk : 0u)
      : op == InferenceMath::Rope || op == InferenceMath::RopeFp16
        ? static_cast<size_t>(M) * 2u * chunk
        : 0u;
    const bool binary =
      op == InferenceMath::Add || op == InferenceMath::SwiGLU ||
      op == InferenceMath::GeGLU || op == InferenceMath::AddRmsNorm;
    if (kind > 7 || (op == InferenceMath::AddRmsNorm && !norm) || !M || !N ||
        N % 32 ||
        count > INT32_MAX / (norm ? 2 * sizeof(float) : sizeof(float)) ||
        plen > INT32_MAX / sizeof(float) || !x || !out || (binary && !aux) ||
        (plen && !param))
      return false;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const char *split_env = std::getenv("NNTR_HTP_DSPQ_MATH_SPLIT");
    const bool split_aux = M == 1 && (kind == 2u || kind == 3u) &&
                           (!split_env || std::atoi(split_env) != 0) &&
                           count * 8u > HTP_DSPQ_BUF_BYTES &&
                           count * 4u <= HTP_DSPQ_BUF_BYTES;
    const size_t queue_in =
      (count * (binary && !split_aux ? 2 : 1) + plen) * sizeof(float);
    const size_t queue_out = count * (half   ? sizeof(uint16_t)
                                      : norm ? 2 * sizeof(float)
                                             : sizeof(float));
    static const bool queue_math = [] {
      const char *e = std::getenv("NNTR_HTP_DSPQ_MATH");
      return !e || std::atoi(e) != 0;
    }();
    const auto math_session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    if (M == 1 && queue_math &&
        dspqReady(math_session, queue_in, queue_out,
                  sizeof(htp_dspq_math_req))) {
      auto &st = *dspq_;
      if (split_aux && !st.math_aux) {
        auto buf = std::make_unique<HtpRpcBuffer>(HTP_DSPQ_BUF_BYTES);
        const auto &mem = HtpRpcMemApi::get();
        if (!buf->isIon() || mem.mmap(st.domain, buf->fd(), buf->data(), 0,
                                      buf->size(), FASTRPC_MAP_FD) != 0)
          throw std::runtime_error("HTP queue math input mapping failed");
        st.math_aux = std::move(buf);
      }
      float *packed = reinterpret_cast<float *>(st.act->data());
      stagedMemcpy(packed, x, count * sizeof(float));
      if (split_aux)
        stagedMemcpy(st.math_aux->data(), aux, count * sizeof(float));
      else if (binary)
        stagedMemcpy(packed + count, aux, count * sizeof(float));
      if (plen)
        stagedMemcpy(packed + count * (binary ? 2 : 1), param,
                     plen * sizeof(float));
      const htp_dspq_math_req req = {
        HTP_DSPQ_OP_MATH,
        ++st.seq,
        (detail ? HTP_DSPQ_FLAG_TIMED : 0u) |
          (split_aux ? HTP_DSPQ_MATH_AUX_BUFFER : 0u),
        kind,
        M,
        N,
        chunk,
        eps,
        scale,
        static_cast<uint32_t>(count),
        binary ? static_cast<uint32_t>(count) : 0u,
        static_cast<uint32_t>(plen),
        static_cast<uint32_t>(queue_out)};
      uint32_t stages[HTP_DSPQ_STAGES] = {};
      const uint64_t start = HtpProfile::nowUs();
      const int err = dspqExchange(&req, sizeof(req), req.seq, queue_in,
                                   queue_out, detail ? stages : nullptr,
                                   split_aux ? st.math_aux.get() : nullptr);
      const uint64_t end = HtpProfile::nowUs();
      if (err != AEE_SUCCESS)
        throw std::runtime_error("HTP queued math failed: " +
                                 std::to_string(err));
      stagedMemcpy(out, st.out->data(),
                   norm ? count * sizeof(float) : queue_out);
      if (norm)
        stagedMemcpy(norm, st.out->data() + count * sizeof(float),
                     count * sizeof(float));
      if (detail) {
        auto &t = math_timing_[0][kind];
        ++t.calls;
        t.prepare_us += start - enter;
        t.rpc_us += end - start;
        t.dsp_us += stages[0];
        t.copyout_us += HtpProfile::nowUs() - end;
        t.bytes += queue_in + queue_out + (split_aux ? count * 4u : 0u);
      }
      ++math_calls_[kind];
      return true;
    }
    const size_t bytes = count * sizeof(float);
    auto &backend = HtpBackend::global();
    size_t copied_bytes = 0;
    auto input = [&](const float *src, size_t size, auto &pool) {
      // Small decode RPCs are faster with dedicated, small rpcmem mappings.
      if (M > 1 && backend.isSharedRange(src, size))
        return src;
      auto *dst = reinterpret_cast<float *>(stage(pool, size).data());
      stagedMemcpy(dst, src, size);
      copied_bytes += size;
      return static_cast<const float *>(dst);
    };
    const float *a = input(x, bytes, act_pool_);
    const float *b = binary ? input(aux, bytes, x2_pool_) : nullptr;
    const float *p =
      plen ? input(param, plen * sizeof(float), rope_pool_) : nullptr;
    const size_t out_bytes = count * (half   ? sizeof(uint16_t)
                                      : norm ? 2 * sizeof(float)
                                             : sizeof(float));
    const bool direct_out =
      M > 1 && !norm && backend.isSharedRange(out, out_bytes);
    const bool direct_fused = M > 1 && norm &&
                              backend.isSharedRange(out, bytes) &&
                              backend.isSharedRange(norm, bytes);
    uint8_t *y = (direct_out || direct_fused)
                   ? static_cast<uint8_t *>(out)
                   : stage(out_pool_, out_bytes).data();
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint64_t start = HtpProfile::nowUs();
    uint32_t dsp_us = 0;
    const int err =
      direct_fused ? nntr_hvx_add_rmsnorm_f32(
                       session, M, N, eps, a, static_cast<int>(count), b,
                       static_cast<int>(count), p, static_cast<int>(plen),
                       static_cast<float *>(out), static_cast<int>(count), norm,
                       static_cast<int>(count), &dsp_us, 1)
      : detail
        ? nntr_hvx_inference_math_timed_f32(
            session, kind, M, N, chunk, eps, scale, a, static_cast<int>(count),
            b, binary ? static_cast<int>(count) : 0, p, static_cast<int>(plen),
            y, static_cast<int>(out_bytes), &dsp_us, 1)
        : nntr_hvx_inference_math_f32(
            session, kind, M, N, chunk, eps, scale, a, static_cast<int>(count),
            b, binary ? static_cast<int>(count) : 0, p, static_cast<int>(plen),
            y, static_cast<int>(out_bytes));
    const uint64_t rpc_end = HtpProfile::nowUs();
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP inference math op=" + std::to_string(kind) +
                               " failed: " + std::to_string(err));
    if (norm && !direct_fused) {
      stagedMemcpy(out, y, bytes);
      stagedMemcpy(norm, y + bytes, bytes);
    } else if (!direct_out && !direct_fused) {
      stagedMemcpy(out, y, out_bytes);
    }
    if (detail) {
      auto &t = math_timing_[M > 1][kind];
      ++t.calls;
      t.prepare_us += start - enter;
      t.rpc_us += rpc_end - start;
      t.dsp_us += dsp_us;
      t.copyout_us += HtpProfile::nowUs() - rpc_end;
      t.bytes += copied_bytes + ((direct_out || direct_fused) ? 0 : out_bytes);
    }
    ++math_calls_[kind];
    if (HtpProfile::global().level())
      HtpProfile::global().addInvoke(M, N, N, rpc_end - start, nullptr);
    return true;
  }

  static bool unsupportedRpc(int err) {
    const uint32_t remote = static_cast<uint32_t>(err);
    return err == AEE_EUNSUPPORTED || err == AEE_EUNSUPPORTEDAPI ||
           remote == 0x80000414u || remote == 0x8000046cu;
  }

  PendingPrefillNorm takePrefillRouterNorm(const float *x, unsigned M,
                                           unsigned K) {
    PendingPrefillNorm p;
    {
      std::lock_guard<std::mutex> lock(invoke_mutex_);
      if (pending_prefill_router_norm_.norm) {
        p = std::move(pending_prefill_router_norm_);
        pending_prefill_router_norm_ = {};
        prepared_moe_cookie_ = prepared_moe_lease_ = 0;
      }
    }
    if (p.norm && (p.rows != M || p.width != K || p.norm != x))
      throw std::runtime_error("HTP deferred prefill router norm mismatch");
    return p;
  }

  void materializePrefillNorm(const PendingPrefillNorm &p) {
    if (!p.norm)
      return;
    const bool materialized =
      p.residual
        ? add_rmsnorm_fp32(p.rows, p.width, p.eps, p.x, p.residual,
                           p.gamma.data(), p.sum, p.norm)
        : inferenceMath(InferenceMath::RmsNorm, p.rows, p.width, p.width, p.eps,
                        1.0f, p.x, nullptr, p.gamma.data(), p.norm, nullptr);
    if (!materialized)
      throw std::runtime_error("HTP deferred prefill norm fallback");
  }

  PendingPrefillNorm takePrefillQkvNorm(const float *x, unsigned M,
                                        unsigned K) {
    PendingPrefillNorm p;
    {
      std::lock_guard<std::mutex> lock(invoke_mutex_);
      p = std::move(pending_prefill_qkv_norm_);
      pending_prefill_qkv_norm_ = {};
    }
    if (p.norm && (p.rows != M || p.width != K || p.norm != x))
      throw std::runtime_error("HTP deferred prefill QKV norm mismatch");
    return p;
  }

  void materialize_prefill_qkv_norm(const float *x, unsigned rows,
                                    unsigned width) override {
    materializePrefillNorm(takePrefillQkvNorm(x, rows, width));
  }

  bool router_qs4cx_prefill(void *w, float *scales, const float *x,
                            float *logits, unsigned M, unsigned K, unsigned E,
                            unsigned top_k, unsigned n_sel, const float *bias,
                            unsigned *sel, float *weight) override {
    const PendingPrefillNorm p = takePrefillRouterNorm(x, M, K);
    if (p.norm) {
      if (invokePrefillRouterNorm(p, w, scales, E, top_k, n_sel, bias, sel,
                                  weight))
        return true;
      // The false result guarantees that sum/norm have not been published.
      // Every ordinary router fallback must see a materialized input, even
      // when it returns false and the caller falls through to generic FC.
      materializePrefillNorm(p);
    }
    return routerQs4cxPrefill(w, scales, x, logits, M, K, E, top_k, n_sel, bias,
                              sel, weight);
  }

  bool invokePrefillRouterNorm(const PendingPrefillNorm &p, void *w,
                               float *scales, unsigned E, unsigned top_k,
                               unsigned n_sel, const float *bias, unsigned *sel,
                               float *weight) {
    const unsigned M = p.rows, K = p.width;
    if (!w || !scales || !E || E > 128 || E % 32 || !top_k || top_k > n_sel ||
        n_sel > E || !bias || !sel || !weight || M <= 4 || M > 2048 ||
        K > 4096 || K % 128 || M > fcMaxRows(K) ||
        uint64_t((M + 63u) & ~63u) * K > (4u << 20) ||
        HtpProfile::global().level() >= 3)
      return false;
    {
      std::lock_guard<std::mutex> lock(invoke_mutex_);
      if (!moe_input_reuse_.token(p.norm, M, K))
        return false;
    }
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const FcHandles &fh = get_or_register_fc(w, session, K, E, scales);
    if (fh.handles.size() != 1)
      return false;
    sendMoeOptsOnce(session);
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const uint64_t lease = moe_input_reuse_.token(p.norm, M, K);
    if (!lease)
      return false;
    // The normalized graph output is still uninitialized here. Do not call
    // stageInput or remember a staged activation until it is published.
    const int count = static_cast<int>(M * K);
    uint64 cookie = 0;
    const uint64_t start = HtpProfile::nowUs();
    const int err = nntr_hvx_router_q4_prefill_norm_prepare(
      session, M, K, E, fh.handles[0], top_k, n_sel, p.eps, p.x, count,
      p.residual, p.residual ? count : 0, p.gamma.data(), K, p.sum,
      p.residual ? count : 0, p.norm, count, bias, E, sel, M * n_sel, weight,
      M * top_k, &cookie);
    // EUNSUPPORTED is explicitly pre-publication in this RPC. All other
    // errors can follow writes to operands aliased by sum/norm; never retry.
    if (unsupportedRpc(err))
      return false;
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP prefill router norm failed: " +
                               std::to_string(err));
    if (!cookie)
      throw std::runtime_error("HTP prefill router norm returned empty lease");
    prepared_moe_cookie_ = cookie;
    prepared_moe_lease_ = lease;
    ++consumer_norm_calls_;
    HtpProfile &profile = HtpProfile::global();
    if (profile.level()) {
      const uint64_t elapsed = HtpProfile::nowUs() - start;
      profile.addInvoke(M, K, E, elapsed, nullptr);
      std::fprintf(stderr, "[HTP-ROUTER-NORM] M=%u K=%u rpc_us=%llu\n", M, K,
                   (unsigned long long)elapsed);
    }
    return true;
  }

  bool routerQs4cxPrefill(void *w, float *scales, const float *x, float *logits,
                          unsigned M, unsigned K, unsigned E, unsigned top_k,
                          unsigned n_sel, const float *bias, unsigned *sel,
                          float *weight) {
    if (M == 1u)
      return routerQ4Decode(w, scales, x, K, E, top_k, n_sel, bias, sel,
                            weight);
    static const bool fuse = [] {
      const char *v = std::getenv("NNTR_HTP_ROUTER_Q4_FUSION");
      return v && std::atoi(v) != 0;
    }();
    static const bool reuse = [] {
      const char *v = std::getenv("NNTR_HTP_ROUTER_Q4_REUSE");
      return v && std::atoi(v) != 0;
    }();
    static const bool quant_reuse = [] {
      const char *v = std::getenv("NNTR_HTP_ROUTER_Q4_QUANT_REUSE");
      return v && std::atoi(v) != 0;
    }();
    if ((!fuse && !reuse && !quant_reuse) || M <= 1 || !K || K % 32 || !E ||
        E > 128 || E % 32 || !top_k || top_k > n_sel || n_sel > E || !w ||
        !scales || !x || !bias || !sel || !weight ||
        static_cast<uint64_t>(M) * K > INT32_MAX ||
        static_cast<uint64_t>(M) * E > INT32_MAX)
      return false;
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const FcHandles &fh = get_or_register_fc(w, session, K, E, scales);
    if (fh.handles.size() != 1)
      return false;
    const unsigned step = fcMaxRows(K);
    if (!fuse && !quant_reuse) {
      invokeFc(session, fh, const_cast<float *>(x), logits, M, E, K,
               reuse && M <= step);
      return sigmoid_topk_fp32(M, E, top_k, n_sel, logits, bias, sel, weight);
    }
    sendMoeOptsOnce(session);
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    HtpProfile &profile = HtpProfile::global();
    const uint64_t lease = moe_input_reuse_.token(x, M, K);
    const bool prepare = quant_reuse && lease && profile.level() < 3 && M > 4 &&
                         M <= step && M <= 2048 && K <= 4096 &&
                         K % 128u == 0u &&
                         uint64_t((M + 63u) & ~63u) * K <= (4u << 20);
    prepared_moe_cookie_ = prepared_moe_lease_ = 0;
    for (unsigned m0 = 0; m0 < M; m0 += step) {
      const unsigned rows = std::min(step, M - m0);
      const float *input = x + static_cast<size_t>(m0) * K;
      const size_t input_bytes = static_cast<size_t>(rows) * K * sizeof(float);
      auto &staged = stageInput(act_pool_, input, input_bytes, &out_pool_);
      const float *source = reinterpret_cast<const float *>(staged.data());
      const uint64_t t0 = HtpProfile::nowUs();
      uint64 cookie = 0;
      bool used_prepare = prepare;
      int err =
        prepare
          ? nntr_hvx_router_q4_prefill_prepare(
              session, rows, K, E, fh.handles[0], source, rows * K, top_k,
              n_sel, bias, E, sel, rows * n_sel, weight, rows * top_k, &cookie)
          : nntr_hvx_router_q4_prefill(
              session, rows, K, E, fh.handles[0], source, rows * K, top_k,
              n_sel, bias, E, sel + static_cast<size_t>(m0) * n_sel,
              rows * n_sel, weight + static_cast<size_t>(m0) * top_k,
              rows * top_k);
      // Hexagon's AEE status codes include the remote error offset.
      if (prepare && unsupportedRpc(err)) {
        used_prepare = false;
        err = nntr_hvx_router_q4_prefill(
          session, rows, K, E, fh.handles[0], source, rows * K, top_k, n_sel,
          bias, E, sel, rows * n_sel, weight, rows * top_k);
      }
      if (err != AEE_SUCCESS)
        throw std::runtime_error("HTP Q4 router failed: " +
                                 std::to_string(err));
      if (used_prepare) {
        if (!cookie)
          throw std::runtime_error(
            "HTP prepared router returned an empty lease");
        prepared_moe_cookie_ = cookie;
        prepared_moe_lease_ = lease;
      }
      if (profile.level())
        profile.addInvoke(rows, K, E, HtpProfile::nowUs() - t0, nullptr);
      // A chunk is not the full activation promised by the caller's lease.
      if (reuse && M <= step)
        moe_input_reuse_.remember(x, M, K, &staged);
    }
    return true;
  }

  bool routerQ4Decode(void *w, float *scales, const float *x, unsigned K,
                      unsigned E, unsigned top_k, unsigned n_sel,
                      const float *bias, unsigned *sel, float *weight) {
    const char *enabled = std::getenv("NNTR_HTP_ROUTER_Q4_DECODE");
    const char *fc = std::getenv("NNTR_HTP_FC_M1_HVX");
    const char *queue = std::getenv("NNTR_HTP_DSPQ_ROUTER");
    htp_dspq_router_q4_req req = {
      HTP_DSPQ_OP_ROUTER_Q4, 0, 0, K, E, 0, top_k, n_sel, 0, 0.0f};
    if ((enabled && std::atoi(enabled) == 0) || (fc && std::atoi(fc) == 0) ||
        (queue && std::atoi(queue) == 0) ||
        !htp_dspq_router_q4_shape_valid(&req) || !w || !scales || !x || !bias ||
        !sel || !weight)
      return false;
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const FcHandles &fh = get_or_register_fc(w, session, K, E, scales);
    if (fh.handles.size() != 1)
      return false;
    sendMoeOptsOnce(session);
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    PendingNorm *pending = pendingNorm("router", x, 1, K);
    req.norm_kind = pending ? (pending->add ? 2u : 1u) : 0u;
    req.eps = pending ? pending->eps : 0.0f;
    req.w_handle = fh.handles[0];
    const size_t in_bytes = htp_dspq_router_q4_input_bytes(&req);
    const size_t out_bytes = htp_dspq_router_q4_output_bytes(&req);
    if (!dspqReady(session, in_bytes, out_bytes, sizeof(req))) {
      if (pending)
        throw std::runtime_error("Deferred Q4 router norm requires DSP queue");
      return false;
    }
    auto &st = *dspq_;
    req.seq = ++st.seq;
    float *input = reinterpret_cast<float *>(st.act->data());
    const size_t input_count = static_cast<size_t>(req.norm_kind + 1u) * K;
    stagedMemcpy(input, pending ? pending->inputs.data() : x,
                 input_count * sizeof(float));
    stagedMemcpy(input + input_count, bias, E * sizeof(float));
    const uint64_t start = HtpProfile::nowUs();
    const int err =
      dspqExchange(&req, sizeof(req), req.seq, in_bytes, out_bytes, nullptr);
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP queued Q4 router failed: " +
                               std::to_string(err));
    const float *result = reinterpret_cast<const float *>(st.out->data());
    const size_t norm_off = htp_dspq_router_q4_norm_offset(&req);
    if (pending)
      publishNorm(result + norm_off);
    st.retained_norm = x;
    st.retained_seq = st.seq;
    st.retained_offset = (norm_off + K) * sizeof(float);
    st.retained_bytes = K * sizeof(float);
    stagedMemcpy(sel, result, n_sel * sizeof(unsigned));
    stagedMemcpy(weight, result + n_sel, top_k * sizeof(float));
    ++fused_router_calls_;
    if (HtpProfile::global().level())
      HtpProfile::global().addInvoke(1, K, E, HtpProfile::nowUs() - start,
                                     nullptr);
    return true;
  }

  bool sigmoid_topk_fp32(unsigned M, unsigned E, unsigned top_k, unsigned n_sel,
                         const float *logits, const float *bias, unsigned *sel,
                         float *weight) override {
    if (!M || !E || E > 128 || !top_k || top_k > n_sel || n_sel > E ||
        static_cast<uint64_t>(M) * E > INT32_MAX || !logits || !bias || !sel ||
        !weight)
      return false;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const int err = nntr_hvx_sigmoid_topk_f32(
      session, M, E, top_k, n_sel, logits, static_cast<int>(M * E), bias, E,
      sel, static_cast<int>(M * n_sel), weight, static_cast<int>(M * top_k));
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP sigmoid top-k failed: " +
                               std::to_string(err));
    ++sigmoid_calls_;
    return true;
  }

  bool supports_gemm_q4_0_accel_fp32() const override { return true; }

  // ponytail: decode-shaped (M == 1) calls are declined for the MoE FFN,
  // not accelerated -- reversing 34_fc_measured.md section5.1's own
  // conclusion for the isolated single-FC comparison it was measured on.
  // That comparison is still correct in isolation (M=1 padding tax ~40us
  // against 113us of DSP-only work, so one call is cheap) -- what it does
  // not cover is 22 MoE layers' worth of M==1 calls stacked into one
  // decode token. Device-measured on this branch: gate_up decode's DMA is
  // fully exposed at M=1 (drain 419.2us for a 14.7MB weight group = 35
  // GB/s, no different from one handle -- cross-matmul prefetch has
  // nothing to hide behind at this width, unlike the FC benchmark's
  // narrower weights), so the real per-layer cost is ~2.6ms/token, and
  // 22 layers' worth is a ~57ms/token transport floor alone (17.5 TPS
  // ceiling) against a CPU decode baseline measured at ~25.9 TPS on the
  // same device. HTP cannot win decode until the call is fused across
  // projections or the DMA-exposure problem above is fixed -- ceiling:
  // revisit if 41_moe_ffn_e2e_and_perf_task.md's P1.3 (fused gate_up+
  // swiglu+down, one call per layer) or P2 (u8 in/out, cuts payload 4x)
  // lands, since either changes this arithmetic. Until then declining M=1
  // costs nothing on the FC path this predicate was written for (single
  // dot(), not a MoE stack) and saves the one MoE FFN caller from a
  // documented loss.
  bool accelerates_q4_0_at_m1() const override { return false; }
  // QS4CX weights live on the DSP in their own format and nowhere else
  // once the FCs are quantized to it offline (doc 57 section 5): decode's
  // row goes to the same kernel as prefill, 64-row pad tax and all (one
  // call per FC, a few ms a token), rather than to a CPU kernel on another
  // packing of the same bytes.
  bool accelerates_qs4cx_at_m1() const override { return true; }

  /** @brief The handles one conv block is registered as: in_proj's column
   *  thirds a, b, c (get_or_register_fc's slices at this model's K) and
   *  out_proj. Declared up here because invokeConvBlock takes it by
   *  reference -- a parameter type has to be complete where the function
   *  is declared. */
  struct ConvHandles {
    std::vector<uint32_t> h_in; /**< a, b, c */
    uint32_t h_out = 0;
  };

  // matAdata: Q4_0x4-repacked weight bytes, identity-cached across calls --
  // the same pointer for the lifetime of a loaded model (inference does not
  // move weight tensors), which is what makes registering once and keying
  // the HexKL handle off it safe.
  void gemm_q4_0_accel_fp32(void *matAdata, float *matBdata, float *matCdata,
                            unsigned int M, unsigned int N,
                            unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    // One handle for a weight the layer kernel's VTCM double buffer holds,
    // several column slices for a wider one (doc 50 section 3): the kernel
    // takes them as one call either way, prefetching each slice behind the
    // previous one's matmul, and copyOut puts the slices' output blocks
    // back into the caller's [M x N].
    const FcHandles &fh = get_or_register_fc(matAdata, session, K, N);

    // ponytail: NOT invokeLayerU8In. Quantizing the activation on ARM/NEON
    // moved that work off HVX (already vectorized, already writes straight
    // into VTCM for HMX to read -- no DRAM detour ever existed for this
    // step) onto ARM scalar/NEON code paying its own cost outside every
    // number this file's profiler measures. Device-measured: DSP time did
    // drop 19-26%, but wall-clock barely moved because the ARM-side
    // quantize (300-420us unvectorized, ~200us with a magic-number round)
    // ate the transport savings it was supposed to create. Reverted to
    // the HVX-quantizes-into-VTCM path. invokeLayerU8In/htp_act_quant.*
    // stay in the tree, unused, in case a caller that is not this one
    // ever legitimately arrives with pre-quantized bytes already in hand.
    //
    // In row chunks when the activation would not fit VTCM beside the
    // slice double buffer (fcMaxRows): the dense FFN's down projection is
    // K = 7168, 3.1 MiB of u8 rows at M = 444 and 7 at M = 1024. Each
    // chunk is a whole call, so a chunk costs the FastRPC fixed ~0.4 ms;
    // one call covers every K = 2048 weight up to 1,920 rows.
    invokeFc(session, fh, matBdata, matCdata, M, N, K);
  }

  /** @brief Rows one layer call may carry at this K.
   *
   * hexkl_mm_u8i4_layer_run holds the whole activation (m_pad x K u8) in
   * VTCM next to the widest handle's double buffer -- 2 x 2 MiB once
   * fcSliceCols has sized the handles -- a result tile and the HMX config.
   * Budgeted at 7.75 MiB of the 8 MiB VTCM, so 3.75 MiB of rows: 1,920 at
   * K = 2048, 512 at K = 7168. Whole 64-row blocks, at least one.
   * ponytail: 7.75 MiB stands in for the session's real vtcm_size less
   * hexkl_micro_hmx_config_size(), which the host does not see; a kernel
   * that walked 64-row blocks like hexkl_mm_u8i4_moe.c would need no cap. */
  static unsigned int fcMaxRows(unsigned int K) {
    constexpr size_t kVtcmBudget = (size_t(8) << 20) - (size_t(256) << 10);
    constexpr size_t kSliceDouble = size_t(2) * (size_t(2) << 20);
    unsigned int rows =
      static_cast<unsigned int>((kVtcmBudget - kSliceDouble) / K);
    rows -= rows % 64u;
    return rows < 64u ? 64u : rows;
  }

  // Several Q4_0 weights that share ONE activation -- LFM2-MoE decode's
  // top-K selected experts' gate_up projections, all against the single
  // token just routed -- go out as ONE FastRPC call across their handles
  // instead of one call per weight. hexkl_mm_u8i4_layer_run's own doc
  // (test/htp/nntr_hvx.idl mm_u8i4_layer) says what that call buys beyond
  // saving round trips: it double-buffers each handle's weight into VTCM
  // with the next handle prefetched while the current one computes, which
  // only happens with more than one handle in the call -- measured 1.7-2x
  // over one-call-per-weight (docs/htp_attention/34_fc_measured.md
  // section4 items C and E). Nothing here changes CPU: FloatTensor::dot's
  // vector overload falls back to the same per-weight loop this replaces
  // when supports_gemm_q4_0_batch_fp32() is false, so this is additive.
  //
  // NOTE: this override was written and syntax-checked in an earlier
  // session but never actually landed in a commit -- Lfm2MoELayer's
  // caller-side grouping shipped without it, so every "grouped" decode
  // call silently fell through Tensor::dot's un-accelerated per-weight
  // loop (ComputeOps::gemm_q4_0_fp32, plain CPU dequant+GEMM) instead of
  // reaching HTP at all. Confirmed by NNTR_HTP_PROFILE=2 output showing
  // zero calls of any shape for gate_up at M==1 after the caller-side
  // change landed. This commit is that missing override.
  bool supports_gemm_q4_0_batch_fp32() const override { return true; }

  void gemm_q4_0_batch_fp32(std::vector<void *> matAdata, float *matBdata,
                            std::vector<float *> matCdata, unsigned int M,
                            std::vector<unsigned int> N,
                            unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    std::vector<uint32_t> handles(n);
    unsigned int n_total = 0;
    for (size_t i = 0; i < n; ++i) {
      handles[i] = get_or_register(matAdata[i], session, K, N[i]);
      n_total += N[i];
    }

    // mm_u8i4_layer writes one contiguous [M x N_i] block per handle, in
    // call order, and copyOut hands each weight its block straight out of
    // the staging buffer. Two callers: decode's grouped expert gate_ups at
    // M == 1, and the attention q/k/v projections at prefill (qkv_layer,
    // doc 51 section 2.21) -- 444 rows, where a heap copy in between
    // would be 5 MB a call. Row chunks as gemm_q4_0_accel_fp32 makes them,
    // for the same VTCM reason.
    const unsigned int step = fcMaxRows(K);
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      const unsigned int m = std::min(step, M - m0);
      std::vector<float *> dsts(n);
      for (size_t i = 0; i < n; ++i)
        dsts[i] = matCdata[i] + static_cast<size_t>(m0) * N[i];
      invokeLayer(session, handles.data(), static_cast<int>(n),
                  matBdata + static_cast<size_t>(m0) * K, nullptr, m, n_total,
                  K, &N, &dsts);
    }
  }

  bool supports_gemm_q4_0_batch_norm_fp32() const override { return true; }

  bool invokePrefillQkvNorm(
    remote_handle64 session, const PendingPrefillNorm &p,
    const std::vector<uint32_t> &handles, const std::vector<unsigned> &widths,
    const std::vector<float *> &outputs, const std::vector<unsigned> &chunks,
    const std::vector<float> &gammas, float post_eps, const float *rope_cs,
    unsigned rope_hd, unsigned rope_handles) {
    if (!p.norm || handles.size() != 3 || widths.size() != 3 ||
        outputs.size() != 3 || chunks.size() != 3)
      return false;
    const unsigned M = p.rows, K = p.width;
    const size_t bytes = static_cast<size_t>(M) * K * sizeof(float);
    auto &backend = HtpBackend::global();
    size_t out_bytes[3];
    for (unsigned i = 0; i < 3; ++i) {
      const uint64_t count = static_cast<uint64_t>(M) * widths[i];
      if (!widths[i] || widths[i] > 4096 || widths[i] % 32 ||
          count > INT32_MAX / sizeof(float))
        return false;
      out_bytes[i] = static_cast<size_t>(count) * sizeof(float);
      if (!backend.isSharedRange(outputs[i], out_bytes[i]) ||
          prefillNormOverlap(outputs[i], out_bytes[i], p.norm, bytes) ||
          prefillNormOverlap(outputs[i], out_bytes[i], p.sum, bytes))
        return false;
      for (unsigned j = 0; j < i; ++j)
        if (prefillNormOverlap(outputs[i], out_bytes[i], outputs[j],
                               out_bytes[j]))
          return false;
    }
    const uint64_t cs_count = static_cast<uint64_t>(M) * 2u * rope_hd;
    if (cs_count > INT32_MAX / sizeof(float))
      return false;
    sendMoeOptsOnce(session);
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const float *cs = nullptr;
    if (rope_hd) {
      float *staged = reinterpret_cast<float *>(
        stage(rope_pool_, static_cast<size_t>(cs_count) * sizeof(float))
          .data());
      stagedMemcpy(staged, rope_cs,
                   static_cast<size_t>(cs_count) * sizeof(float));
      cs = staged;
    }
    // Direct graph writes must not leave old staged copies eligible for reuse.
    for (StagingPool *pool : {&act_pool_, &out_pool_, &x2_pool_}) {
      pool->holds.clear();
      pool->copied_out.clear();
    }
    moe_input_reuse_.invalidate();
    const int count = static_cast<int>(static_cast<size_t>(M) * K);
    const uint64_t start = HtpProfile::nowUs();
    const int err = nntr_hvx_mm_u8i4_qkv_prefill_norm(
      session, M, K, widths[0], widths[1], widths[2], handles.data(), 3,
      post_eps, chunks.data(), 3, gammas.data(), gammas.size(), rope_hd,
      rope_handles, cs, static_cast<int>(cs_count), p.eps, p.x, count,
      p.residual, p.residual ? count : 0, p.gamma.data(), K, p.sum,
      p.residual ? count : 0, p.norm, count, outputs[0], M * widths[0],
      outputs[1], M * widths[1], outputs[2], M * widths[2]);
    // The RPC reports unsupported only before it can publish aliased outputs.
    if (unsupportedRpc(err))
      return false;
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP prefill QKV norm failed: " +
                               std::to_string(err));
    ++consumer_norm_calls_;
    ++shared_io_calls_[1];
    shared_io_bytes_[1] += bytes * (p.residual ? 4u : 2u) + out_bytes[0] +
                           out_bytes[1] + out_bytes[2];
    HtpProfile &profile = HtpProfile::global();
    if (profile.level()) {
      const uint64_t elapsed = HtpProfile::nowUs() - start;
      profile.addInvoke(M, K, widths[0] + widths[1] + widths[2], elapsed,
                        nullptr);
      std::fprintf(stderr, "[HTP-QKV-NORM] M=%u K=%u rpc_us=%llu\n", M, K,
                   (unsigned long long)elapsed);
    }
    return true;
  }

  void gemm_q4_0_batch_norm_fp32(
    std::vector<void *> matAdata, std::vector<float *> matAscale,
    float *matBdata, std::vector<float *> matCdata, unsigned int M,
    std::vector<unsigned int> N, unsigned int K, const float *pre_gamma,
    const std::vector<unsigned int> &post_chunk, const float *post_gamma,
    float eps, const float *rope_cs, unsigned int rope_hd,
    unsigned int rope_weights, bool weights_wh) override {
    const PendingPrefillNorm pending = takePrefillQkvNorm(matBdata, M, K);
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    if (N.size() != n || matCdata.size() != n ||
        (!post_chunk.empty() && post_chunk.size() != n) ||
        (!matAscale.empty() && matAscale.size() != n) ||
        (weights_wh && matAscale.size() != n) ||
        (rope_hd != 0 && (rope_weights > n || rope_cs == nullptr))) {
      throw std::invalid_argument(
        "gemm_q4_0_batch_norm_fp32: one post_chunk / scale per weight, "
        "rope on a prefix of them");
    }
    // Every weight as the column slices gemm_q4_0_accel_fp32 would make
    // of it (a 2816 x 4096 weight does not fit the VTCM double buffer
    // whole), all of them one call: handle by handle, the slice's chunk
    // and its weight's gamma again, and where the slice's rows land.
    std::vector<uint32_t> handles;
    std::vector<unsigned int> chunks;
    std::vector<float> gammas;
    std::vector<OutSlice> slices;
    unsigned int n_total = 0;
    unsigned int rope_slices = 0;
    const float *g = post_gamma;
    for (size_t i = 0; i < n; ++i) {
      const FcHandles &fh = get_or_register_fc(
        matAdata[i], session, K, N[i],
        matAscale.empty() ? nullptr : matAscale[i], weights_wh);
      const unsigned int chunk = post_chunk.empty() ? 0u : post_chunk[i];
      unsigned int c0 = 0;
      for (size_t j = 0; j < fh.handles.size(); ++j) {
        const unsigned int cols = fh.cols[j];
        if ((chunk != 0u && cols % chunk != 0u) ||
            (rope_hd != 0u && i < rope_weights && cols % rope_hd != 0u)) {
          throw std::invalid_argument(
            "gemm_q4_0_batch_norm_fp32: a slice of " + std::to_string(cols) +
            " columns does not hold whole chunks of " + std::to_string(chunk) +
            " / heads of " + std::to_string(rope_hd));
        }
        if (rope_hd != 0u && i < rope_weights)
          ++rope_slices;
        handles.push_back(fh.handles[j]);
        chunks.push_back(chunk);
        if (chunk != 0u)
          gammas.insert(gammas.end(), g, g + chunk);
        slices.push_back({matCdata[i], N[i], c0, cols});
        c0 += cols;
        n_total += cols;
      }
      if (chunk != 0u)
        g += chunk;
    }
    FusedNorms norms;
    norms.pre_gamma = pre_gamma;
    norms.post_chunk = post_chunk.empty() ? nullptr : &chunks;
    norms.post_gamma = gammas.data();
    norms.eps = eps;
    norms.rope_hd = rope_hd;
    norms.rope_handles = rope_slices;
    // Both norms and the RoPE are per row, so the row chunks
    // gemm_q4_0_batch_fp32 makes for VTCM carry them unchanged (the table
    // rows advance with the activation rows).
    const unsigned int step = fcMaxRows(K);
    if (pending.norm) {
      if (!pre_gamma && n == 3 && handles.size() == 3 && M <= step &&
          invokePrefillQkvNorm(session, pending, handles, N, matCdata, chunks,
                               gammas, eps, rope_cs, rope_hd, rope_slices)) {
        if (weights_wh)
          ++fused_qkv_calls_;
        return;
      }
      materializePrefillNorm(pending);
    }
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      const unsigned int m = std::min(step, M - m0);
      std::vector<OutSlice> rows = slices;
      for (OutSlice &o : rows)
        o.dst += static_cast<size_t>(m0) * o.stride;
      // The table rows through ION like the activation: a plain host
      // pointer is pinned and mapped by the driver every call (~155 MB/s,
      // doc 34 section 4 item F), 1-2 MB a prefill layer here.
      norms.rope_cs = nullptr;
      if (rope_hd) {
        const size_t bytes =
          static_cast<size_t>(m) * 2 * rope_hd * sizeof(float);
        float *cs = reinterpret_cast<float *>(stage(rope_pool_, bytes).data());
        stagedMemcpy(cs, rope_cs + static_cast<size_t>(m0) * 2 * rope_hd,
                     bytes);
        norms.rope_cs = cs;
      }
      invokeLayer(session, handles.data(), static_cast<int>(handles.size()),
                  matBdata + static_cast<size_t>(m0) * K, nullptr, m, n_total,
                  K, nullptr, nullptr, &norms, &rows);
      if (weights_wh)
        ++fused_qkv_calls_;
    }
  }

  // The block epilogue (see the base declaration): the residual rides the
  // out staging buffer in and the result out, the addends the act buffer
  // (x then x2), gamma as a plain host pointer like the fused norms'.
  bool supports_rmsnorm_add_fp32() const override { return true; }

  void rmsnorm_add_fp32(unsigned int M, unsigned int N, const float *resid,
                        const float *x, const float *x2, const float *gamma,
                        float eps, float scale, float *out) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t rows = static_cast<size_t>(M) * N;
    const size_t x_len = x2 ? 2 * rows : rows;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act =
      reinterpret_cast<float *>(stage(act_pool_, x_len * sizeof(float)).data());
    float *res =
      reinterpret_cast<float *>(stage(out_pool_, rows * sizeof(float)).data());
    stagedMemcpy(act, x, rows * sizeof(float));
    if (x2)
      stagedMemcpy(act + rows, x2, rows * sizeof(float));
    stagedMemcpy(res, resid, rows * sizeof(float));
    const uint64_t t0 = HtpProfile::nowUs();
    const int err = nntr_hvx_rmsnorm_add_f32(
      session, M, N, eps, scale, gamma, gamma ? static_cast<int>(N) : 0, act,
      static_cast<int>(x_len), res, static_cast<int>(rows));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        "nntr_hvx_rmsnorm_add_f32 failed: err=" + std::to_string(err) +
        " (M=" + std::to_string(M) + " N=" + std::to_string(N) + ")");
    }
    stagedMemcpy(out, res, rows * sizeof(float));
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvoke(M, x2 ? 2 * N : N, N, elapsed, nullptr);
  }

  // The router logits (see the base declaration): the rows through the
  // act pool, the logits through the out pool, gamma as a plain host
  // pointer, and the K x E f32 gate weight (1.4 MB at 2816 x 128) from an
  // ION copy made on its first call and kept by the weight's address --
  // as a plain pointer it was pinned and mapped every call (doc 57
  // section 9.10: 9.7 ms a call against the CPU dot's 6).

  uint64_t begin_moe_input_reuse(const float *input, unsigned int M,
                                 unsigned int K) override {
    static const bool enabled = [] {
      const char *e = std::getenv("NNTR_HTP_MOE_INPUT_REUSE");
      return !e || std::strcmp(e, "0") != 0;
    }();
    if (!enabled || M <= 1)
      return 0;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    prepared_moe_cookie_ = prepared_moe_lease_ = 0;
    return moe_input_reuse_.begin(input, M, K);
  }

  void end_moe_input_reuse(uint64_t token) override {
    if (!token)
      return;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    moe_input_reuse_.end(token);
    if (token == prepared_moe_lease_)
      prepared_moe_cookie_ = prepared_moe_lease_ = 0;
  }

  bool supports_router_logits_fp32() const override { return true; }

  const float *routerWeightShared(const float *w, size_t bytes) {
    auto it = router_w_.find(w);
    if (it == router_w_.end()) {
      auto buf = std::make_unique<HtpRpcBuffer>(bytes);
      std::memcpy(buf->data(), w, bytes);
      it = router_w_.emplace(w, std::move(buf)).first;
    }
    return reinterpret_cast<const float *>(it->second->data());
  }

  /** @brief The gate weight (K x E f32) as a u8 x i8 HMX weight, built and
   *  registered on first use and kept by the weight's address: symmetric
   *  int8 per expert column (scale = max |w| / 127), its column sums, no
   *  bias. UINT32_MAX when the backend refuses (the f32 kernel then runs). */
  uint32_t routerWeightU8i8(remote_handle64 session, const float *w,
                            unsigned int K, unsigned int E) {
    auto it = router_u8i8_.find(w);
    if (it != router_u8i8_.end())
      return it->second;
    std::vector<int8_t> q(static_cast<size_t>(K) * E);
    std::vector<float> sc(E), bias(E, 0.0f);
    std::vector<int32_t> colsum(E, 0);
    for (unsigned int e = 0; e < E; ++e) {
      float amax = 0.0f;
      for (unsigned int k = 0; k < K; ++k)
        amax = std::max(amax, std::fabs(w[static_cast<size_t>(k) * E + e]));
      sc[e] = amax > 0.0f ? amax / 127.0f : 1.0f;
      const float inv = 1.0f / sc[e];
      for (unsigned int k = 0; k < K; ++k) {
        const long v = std::lround(w[static_cast<size_t>(k) * E + e] * inv);
        const int8_t qi =
          static_cast<int8_t>(std::max(-127L, std::min(127L, v)));
        q[static_cast<size_t>(k) * E + e] = qi;
        colsum[e] += qi;
      }
    }
    uint32_t h = UINT32_MAX;
    const int err = nntr_hvx_weight_register_u8i8(
      session, K, E, q.data(), static_cast<int>(q.size()), sc.data(),
      static_cast<int>(E), colsum.data(), static_cast<int>(E), bias.data(),
      static_cast<int>(E), &h);
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr,
                   "[HTP] router stays on the f32 kernel: "
                   "weight_register_u8i8 err=%d\n",
                   err);
      h = UINT32_MAX;
    }
    router_u8i8_[w] = h;
    return h;
  }

  void router_logits_fp32(unsigned int M, unsigned int K, unsigned int E,
                          const float *x, const float *gamma, float eps,
                          const float *w, float *logits, unsigned int top_k,
                          unsigned int n_sel, const float *scale,
                          unsigned int *sel, float *weight,
                          bool sigmoid) override {
    // This graph consumer can also have FP32 weights. It has no prepared
    // Q4 input path, so publish the preceding norm before staging its input.
    materializePrefillNorm(takePrefillRouterNorm(x, M, K));
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t x_len = static_cast<size_t>(M) * K;
    const size_t out_len = static_cast<size_t>(M) * E;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    PendingNorm *pending = pendingNorm("router", x, M, K);
    HtpRpcBuffer &act_stage =
      pending ? stage(act_pool_, x_len * sizeof(float))
              : stageInput(act_pool_, x, x_len * sizeof(float));
    float *act = reinterpret_cast<float *>(act_stage.data());
    float *out = reinterpret_cast<float *>(
      stage(out_pool_, out_len * sizeof(float)).data());
    // NNTR_HTP_ROUTER_U8I8=1: the logits on the HMX (u8 x i8). Off by
    // default: on device its logits were up to 0.17-0.39 off the CPU's
    // (the f32 kernel: 2e-5), the per-row u8 step being coarse beside
    // Gemma-4's outlier activations, and the routing that noise changes
    // is not a model's own (doc 57 section 9.33).
    static const bool router_u8i8 =
      std::getenv("NNTR_HTP_ROUTER_U8I8") != nullptr;
    const uint32_t h8 =
      router_u8i8 ? routerWeightU8i8(session, w, K, E) : UINT32_MAX;
    const float *w_shared =
      h8 != UINT32_MAX
        ? nullptr
        : routerWeightShared(w, static_cast<size_t>(K) * E * sizeof(float));
    static const bool queue_router = [] {
      const char *e = std::getenv("NNTR_HTP_DSPQ_ROUTER");
      return !e || std::atoi(e) != 0;
    }();
    const size_t norm_off =
      (static_cast<size_t>(E) + n_sel + top_k + 31u) & ~size_t(31u);
    const size_t input_count = pending ? pending->inputs.size() : K;
    const size_t qin = (input_count + (gamma ? K : 0u) + E) * 4;
    const size_t qout =
      (pending ? norm_off + 2u * K : static_cast<size_t>(E) + n_sel + top_k) *
      4;
    if (M == 1 && sigmoid && h8 == UINT32_MAX && queue_router &&
        dspqReady(session, qin, qout, sizeof(htp_dspq_router_req))) {
      auto &st = *dspq_;
      auto it = st.router_weights.find(w);
      const bool first_use = it == st.router_weights.end();
      if (first_use) {
        auto buf =
          std::make_unique<HtpRpcBuffer>(static_cast<size_t>(K) * E * 4);
        std::memcpy(buf->data(), w, buf->size());
        const auto &mem = HtpRpcMemApi::get();
        if (!buf->isIon() || mem.mmap(st.domain, buf->fd(), buf->data(), 0,
                                      buf->size(), FASTRPC_MAP_FD) != 0)
          throw std::runtime_error("HTP queue router weight mapping failed");
        it = st.router_weights.emplace(w, std::move(buf)).first;
      }
      float *packed = reinterpret_cast<float *>(st.act->data());
      stagedMemcpy(packed, pending ? pending->inputs.data() : x,
                   input_count * sizeof(float));
      if (gamma)
        stagedMemcpy(packed + input_count, gamma, K * sizeof(float));
      stagedMemcpy(packed + input_count + (gamma ? K : 0u), scale,
                   E * sizeof(float));
      const htp_dspq_router_req req = {
        HTP_DSPQ_OP_ROUTER, ++st.seq, 0, K, E, top_k, n_sel,
        gamma ? K : 0u,     eps};
      const uint64_t start = HtpProfile::nowUs();
      htp_dspq_router_norm_req fused = {req, pending && pending->add ? 1u : 0u,
                                        pending ? pending->eps : 0.0f};
      fused.router.op = HTP_DSPQ_OP_ROUTER_NORM;
      const int err =
        dspqExchange(pending ? (const void *)&fused : &req,
                     pending ? sizeof(fused) : sizeof(req), req.seq, qin, qout,
                     nullptr, it->second.get(), first_use);
      if (err != AEE_SUCCESS)
        throw std::runtime_error("HTP queued router failed: " +
                                 std::to_string(err));
      const float *result = reinterpret_cast<const float *>(st.out->data());
      if (pending) {
        const float *norm_key = pending->norm;
        publishNorm(result + norm_off);
        st.retained_norm = norm_key;
        st.retained_seq = st.seq;
        st.retained_offset = (norm_off + K) * sizeof(float);
        st.retained_bytes = K * sizeof(float);
      }
      stagedMemcpy(logits, result, E * sizeof(float));
      stagedMemcpy(sel, result + E, n_sel * sizeof(uint32_t));
      stagedMemcpy(weight, result + E + n_sel, top_k * sizeof(float));
      ++fused_router_calls_;
      if (HtpProfile::global().level())
        HtpProfile::global().addInvoke(M, K, E, HtpProfile::nowUs() - start,
                                       nullptr);
      return;
    }
    if (pending)
      throw std::runtime_error(
        "Deferred router norm requires FP32 sigmoid DSP queue");
    const uint64_t t0 = HtpProfile::nowUs();
    // the selection is small (M x 13 + M x 8 at the softmax router's
    // width) and goes straight to the caller's arrays
    static_assert(sizeof(unsigned int) == sizeof(uint32_t), "sel is u32");
    const int err =
      sigmoid
        ? nntr_hvx_router_sigmoid_f32(
            session, M, K, E, eps, gamma, gamma ? static_cast<int>(K) : 0, h8,
            w_shared,
            w_shared ? static_cast<int>(static_cast<size_t>(K) * E) : 0, act,
            static_cast<int>(x_len), top_k, n_sel, scale, static_cast<int>(E),
            out, static_cast<int>(out_len), reinterpret_cast<uint32_t *>(sel),
            static_cast<int>(M * n_sel), weight, static_cast<int>(M * top_k))
      : h8 != UINT32_MAX
        ? nntr_hvx_router_logits_u8i8_f32(
            session, M, K, E, eps, gamma, gamma ? static_cast<int>(K) : 0, h8,
            act, static_cast<int>(x_len), top_k, top_k ? n_sel : 0u, scale,
            top_k ? static_cast<int>(E) : 0, out, static_cast<int>(out_len),
            reinterpret_cast<uint32_t *>(sel),
            top_k ? static_cast<int>(M * n_sel) : 0, weight,
            top_k ? static_cast<int>(M * top_k) : 0)
        : nntr_hvx_router_logits_f32(
            session, M, K, E, eps, gamma, gamma ? static_cast<int>(K) : 0,
            w_shared, static_cast<int>(static_cast<size_t>(K) * E), act,
            static_cast<int>(x_len), top_k, top_k ? n_sel : 0u, scale,
            top_k ? static_cast<int>(E) : 0, out, static_cast<int>(out_len),
            reinterpret_cast<uint32_t *>(sel),
            top_k ? static_cast<int>(M * n_sel) : 0, weight,
            top_k ? static_cast<int>(M * top_k) : 0);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        "nntr_hvx_router_logits_f32 failed: err=" + std::to_string(err) +
        " (M=" + std::to_string(M) + " K=" + std::to_string(K) +
        " E=" + std::to_string(E) + ")");
    }
    stagedMemcpy(logits, out, out_len * sizeof(float));
    if (sigmoid)
      ++fused_router_calls_;
    moe_input_reuse_.remember(x, M, K, &act_stage);
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvoke(M, K, E, elapsed, nullptr);
  }

  // The lm_head (see the base declaration): the canonical Q4_0 rows as
  // Q4M1 slices of kLmHeadSliceRows -- the same nibbles and scales,
  // reordered (q4m1_from_q4_0), no requantization -- placed in the arena
  // on the first call and attached by offset, then one call per row runs
  // the norm, every slice's GEMV and the softcap. A tied weight is ~415 MB
  // at a 262144 vocabulary: past the DSP heap (it refused at ~100 MiB), so
  // the arena or nothing. A placement or attach refused leaves the
  // lm_head on the CPU for the run, said once on stderr.
  // ponytail: it is placed after every other weight, in the address space
  // they leave; a second session's arena (the decode graph's S2) is the
  // upgrade if the first one has no room on a device.
  bool lm_head_q4_0_fp32(const void *w, unsigned int K, unsigned int N,
                         const float *x, const float *gamma, float eps,
                         float softcap, float *y) override {
    if (w == nullptr || K == 0 || K % 64 != 0 || K > 8192 || N == 0 ||
        N % 32 != 0)
      return false;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const std::vector<uint32_t> *h = lmHeadHandles(w, session, K, N);
    if (h == nullptr)
      return false;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act =
      reinterpret_cast<float *>(stage(act_pool_, K * sizeof(float)).data());
    float *out = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(N) * sizeof(float)).data());
    stagedMemcpy(act, x, K * sizeof(float));
    const uint64_t t0 = HtpProfile::nowUs();
    const int err = nntr_hvx_lm_head_q4m1_f32(
      session, eps, gamma, gamma ? static_cast<int>(K) : 0, softcap, h->data(),
      static_cast<int>(h->size()), act, static_cast<int>(K), out,
      static_cast<int>(N));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        "nntr_hvx_lm_head_q4m1_f32 failed: err=" + std::to_string(err) +
        " (K=" + std::to_string(K) + " N=" + std::to_string(N) + ")");
    }
    stagedMemcpy(y, out, static_cast<size_t>(N) * sizeof(float));
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvoke(1, K, N, elapsed, nullptr);
    return true;
  }

  bool lm_head_q4_0_prepare(const void *w, unsigned int K,
                            unsigned int N) override {
    if (w == nullptr || K == 0 || K % 64 != 0 || K > 8192 || N == 0 ||
        N % 32 != 0)
      return false;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    return lmHeadHandles(w, session, K, N) != nullptr;
  }

  /** @brief The lm_head weight's Q4M1 slice handles, placed on first use;
   *  nullptr when they could not be (then for good, said once). */
  const std::vector<uint32_t> *lmHeadHandles(const void *w,
                                             remote_handle64 session,
                                             unsigned int K, unsigned int N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = lm_head_.find(w);
    if (it != lm_head_.end())
      return it->second.ok ? &it->second.h : nullptr;
    LmHeadSlices lh;
    const uint8_t *canonical = static_cast<const uint8_t *>(w);
    const size_t row_bytes = static_cast<size_t>(K / 32u) * Q4_CPU_BLOCK_BYTES;
    size_t left = 0;
    for (uint32_t r0 = 0; r0 < N; r0 += kLmHeadSliceRows)
      left +=
        q4m1_bytes(K, std::min<uint32_t>(kLmHeadSliceRows, N - r0)) + 4096u;
    const size_t total = left;
    const uint64_t t0 = HtpProfile::nowUs();
    std::string why;
    for (uint32_t r0 = 0; r0 < N; r0 += kLmHeadSliceRows) {
      const uint32_t rows = std::min<uint32_t>(kLmHeadSliceRows, N - r0);
      const uint32_t bytes = static_cast<uint32_t>(q4m1_bytes(K, rows));
      std::vector<uint8_t> q4m1(bytes);
      q4m1_from_q4_0(canonical + r0 * row_bytes, K, rows, q4m1.data());
      uint32_t handle = 0, chunk = 0, off = 0;
      int err;
      if (ensureArena(session) && place(session, bytes, left, &chunk, &off)) {
        std::memcpy(arena_chunks_[chunk].buf->data() + off, q4m1.data(), bytes);
        err = nntr_hvx_q4m1_attach(session, arena_chunks_[chunk].dsp_id, off, K,
                                   rows, &handle);
        if (err != AEE_SUCCESS)
          why = "q4m1_attach err=" + std::to_string(err);
      } else {
        err = nntr_hvx_q4m1_register(session, K, rows, q4m1.data(),
                                     static_cast<int>(bytes), &handle);
        if (err != AEE_SUCCESS)
          why = "no arena room (" + arena_fail_ +
                "), q4m1_register err=" + std::to_string(err);
      }
      if (err != AEE_SUCCESS)
        break;
      lh.h.push_back(handle);
      left = left > bytes + 4096u ? left - bytes - 4096u : 0u;
    }
    if (!why.empty()) {
      for (uint32_t handle : lh.h)
        nntr_hvx_q4m1_release(session, handle);
      std::fprintf(stderr,
                   "[HTP] lm_head stays on the CPU: slice %zu of %u rows "
                   "refused (%s)\n",
                   lh.h.size(), N, why.c_str());
      lm_head_[w] = LmHeadSlices{};
      return nullptr;
    }
    lh.ok = true;
    std::fprintf(stderr,
                 "[HTP] lm_head on the NPU: %u x %u Q4_0 as %zu Q4M1 slices, "
                 "%.1f MiB, placed in %.0f ms\n",
                 N, K, lh.h.size(), static_cast<double>(total) / (1u << 20),
                 static_cast<double>(HtpProfile::nowUs() - t0) / 1000.0);
    return &(lm_head_[w] = std::move(lh)).h;
  }

  // A QS4CX weight was quantized once, straight from FP32, and already
  // holds the int4 values this registry wants -- so the seam is
  // htp_qs4cx_from_packed's bit rearrangement plus a colsum, not a second
  // quantization. That is both more accurate (measured on this branch:
  // mean_abs_err 0.0333 vs 0.0451 through Q4_0, a 26% cut) and cheaper to
  // load (95.9 -> 59.9 ms for a 2048x3584 gate_up, 77.2 -> 22.2 ms for a
  // 1792x2048 down, x86 host). Prefer feeding the FFN QS4CX; the Q4_0
  // entry above stays for weights shared with the CPU path.
  bool supports_gemm_qs4cx_accel_fp32() const override { return true; }

  bool supports_gemm_qs4cx_wh_accel_fp32() const override { return true; }

  void gemm_qs4cx_wh_accel_fp32(void *weights, float *scales, float *input,
                                float *output, unsigned int M, unsigned int N,
                                unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    invokeFc(session, get_or_register_fc(weights, session, K, N, scales, true),
             input, output, M, N, K);
  }

  void gemm_qs4cx_accel_fp32(void *matAdata, float *matAscale, float *matBdata,
                             float *matCdata, unsigned int M, unsigned int N,
                             unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    // Column slices and row chunks as gemm_q4_0_accel_fp32 takes them: one
    // handle over the whole weight overflows VTCM once it is wider than
    // fcSliceCols (2816 x 4096 at M = 512 answered AEE_ENOMEMORY). A weight
    // that fits one slice keeps its handle_cache_ entry, so an expert that
    // register_qs4cx_weight registered is not registered twice.
    invokeFc(session, get_or_register_fc(matAdata, session, K, N, matAscale),
             matBdata, matCdata, M, N, K);
  }

  // The o-proj with the post-attention epilogue (see the base declaration):
  // the FC's row chunks as invokeFc takes them, each one call that also
  // norms, adds the residual and scales, so the FC output never comes back.
  bool gemm_qs4cx_res_add_fp32(void *w, float *w_scale, const float *act,
                               unsigned int M, unsigned int K, unsigned int N,
                               const float *res, const float *gamma, float eps,
                               float scale, float *out) override {
    if (M <= 1 || w == nullptr || act == nullptr || res == nullptr)
      return false;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const FcHandles &fh = get_or_register_fc(w, session, K, N, w_scale);
    const unsigned int step = fcMaxRows(K);
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      const unsigned int m = std::min(step, M - m0);
      const size_t a_bytes = static_cast<size_t>(m) * K * sizeof(float);
      const size_t o_bytes = static_cast<size_t>(m) * N * sizeof(float);
      std::lock_guard<std::mutex> lock(invoke_mutex_);
      float *act_f32 = reinterpret_cast<float *>(
        stageInput(act_pool_, act + static_cast<size_t>(m0) * K, a_bytes)
          .data());
      float *res_f32 =
        reinterpret_cast<float *>(stage(x2_pool_, o_bytes).data());
      float *out_f32 =
        reinterpret_cast<float *>(stage(out_pool_, o_bytes).data());
      stagedMemcpy(res_f32, res + static_cast<size_t>(m0) * N, o_bytes);
      const uint64_t t0 = HtpProfile::nowUs();
      const int err = nntr_hvx_mm_u8i4_layer_res_add(
        session, m, K, eps, scale, gamma, gamma ? static_cast<int>(N) : 0,
        fh.handles.data(), static_cast<int>(fh.handles.size()), act_f32,
        static_cast<int>(m * K), res_f32, static_cast<int>(m * N), out_f32,
        static_cast<int>(m * N));
      const uint64_t elapsed = HtpProfile::nowUs() - t0;
      if (err != AEE_SUCCESS)
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_res_add failed: err=" +
                                 std::to_string(err));
      stagedMemcpy(out + static_cast<size_t>(m0) * N, out_f32, o_bytes);
      HtpProfile &profile = HtpProfile::global();
      if (profile.level())
        profile.addInvoke(m, K, N, elapsed, nullptr);
    }
    return true;
  }

  // Same grouping as gemm_q4_0_batch_fp32, for QS4CX weights -- see the
  // comment on the base declaration (compute_ops.h). Without this,
  // FloatTensor::dot's vector overload had no accelerated path for QS4CX at
  // all: every "grouped" decode call under --moe_dtype QS4CX silently fell
  // through to one dotQs4cx() -- one FastRPC call -- per expert, the same
  // shape of miss gemm_q4_0_batch_fp32's own commit fixed for Q4_0.
  bool supports_gemm_qs4cx_batch_fp32() const override { return true; }

  void gemm_qs4cx_batch_fp32(std::vector<void *> matAdata,
                             std::vector<float *> matAscale, float *matBdata,
                             std::vector<float *> matCdata, unsigned int M,
                             std::vector<unsigned int> N,
                             unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const size_t n = matAdata.size();
    std::vector<uint32_t> handles(n);
    unsigned int n_total = 0;
    for (size_t i = 0; i < n; ++i) {
      handles[i] =
        get_or_register_qs4cx(matAdata[i], matAscale[i], session, K, N[i]);
      n_total += N[i];
    }

    // Same stage-then-hand-out shape as gemm_q4_0_batch_fp32 -- see that
    // function's comment for why: mm_u8i4_layer returns one contiguous
    // block per handle, matCdata is one pointer per weight.
    std::vector<float> out_cat(static_cast<size_t>(M) * n_total);
    invokeLayer(session, handles.data(), static_cast<int>(n), matBdata,
                out_cat.data(), M, n_total, K);

    size_t off = 0;
    for (size_t i = 0; i < n; ++i) {
      const size_t block = static_cast<size_t>(M) * N[i];
      std::memcpy(matCdata[i], out_cat.data() + off, block * sizeof(float));
      off += block;
    }
  }

  // The fused expert FFN (doc 43 §[L2]): one FastRPC call per expert per
  // layer instead of two, with the M x 2I + M x I intermediates never
  // leaving the DSP. The caller (Lfm2MoELayer) gates on M > 1 itself --
  // decode's single token cannot amortize the fused call's 64-row pad
  // tax, the same reasoning as accelerates_q4_0_at_m1() being false.
  // [doc 46] The whole layer in one call. supports_* stays behind the
  // caller's own M > 1 gate for the same reason the fused one does: a
  // single decode token cannot amortize the 64-row pad tax.
  bool supports_gemm_qs4cx_moe_layer_fp32() const override { return true; }

  void gemm_qs4cx_moe_layer_fp32(const std::vector<void *> &gate_up_data,
                                 const std::vector<float *> &gate_up_scale,
                                 const std::vector<void *> &down_data,
                                 const std::vector<float *> &down_scale,
                                 const std::vector<unsigned int> &row_index,
                                 const std::vector<unsigned int> &row_count,
                                 const std::vector<float> &row_weight,
                                 const float *act, float *out, unsigned int M,
                                 unsigned int K, unsigned int inter,
                                 unsigned int N_out, bool weights_wh, bool gelu,
                                 const float *pre_gamma,
                                 const float *post_gamma, float eps,
                                 const float *add_x2, const float *add_gamma,
                                 float add_scale) override {
    const size_t n_experts = gate_up_data.size();
    if (n_experts == 0 || gate_up_scale.size() != n_experts ||
        down_data.size() != n_experts || down_scale.size() != n_experts ||
        row_count.size() != n_experts ||
        row_weight.size() != row_index.size()) {
      throw std::invalid_argument(
        "gemm_qs4cx_moe_layer_fp32: per-expert arrays disagree");
    }

    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    sendMoeOptsOnce(session);
    std::vector<uint32_t> h_gu(n_experts), h_dn(n_experts);
    for (size_t e = 0; e < n_experts; ++e) {
      if (weights_wh) {
        h_gu[e] = get_or_register_wh(gate_up_data[e], gate_up_scale[e], session,
                                     K, 2 * inter);
        h_dn[e] = get_or_register_wh(down_data[e], down_scale[e], session,
                                     inter, N_out);
      } else {
        h_gu[e] = get_or_register_qs4cx(gate_up_data[e], gate_up_scale[e],
                                        session, K, 2 * inter);
        h_dn[e] = get_or_register_qs4cx(down_data[e], down_scale[e], session,
                                        inter, N_out);
      }
    }
    // [#85] With NNTR_HTP_FORWARD=1 and a description from the model, every
    // MoE call binds its handles to the next MoE op in list order (the
    // first pass runs the layers in order), and once all are bound the
    // M == 1 calls go through the per-token entry instead. Errors throw
    // (contract section 2): a fallback to mm_u8i4_moe_layer would report
    // per-layer numbers as the graph's.
    if (forwardSwitch() && !graph_words_.empty()) {
      const uint32_t op = bindMoeOp(h_gu, h_dn, K, inter, N_out);
      // [#132] A sole MOE stretch runs here; with ADD resident the stretch
      // is [MOE ADD RMSNORM] and the next layer's norm hook runs it, so a
      // row the hooks do not drive (row_bound_ false: a one-token prompt
      // binding as it goes) keeps the per-layer path.
      const bool sole = stretch_end_[op] == op + 1u;
      if (M == 1 && moe_bound_ == moe_ops_.size() && graphOp(op)->resident &&
          (sole || row_bound_)) {
        ensureGraphInit(session);
        // pos: no MoE stretch reads the position; the hooks' row does
        runStretchOp(session, op, sole ? 0u : cur_pos_, act, K, out, N_out,
                     &row_index, &row_count, &row_weight);
        return;
      }
      if (M == 1 && !graph_short_warned_ && moe_bound_ != moe_ops_.size()) {
        // moe_htp_layers naming a subset: the list's MoE ops can never all
        // be bound, so say so once instead of silently taking the
        // per-layer path under a measurement switch. The small-op hooks
        // read the same condition and return 0 (the CPU path). All bound
        // but not taken above -- a row the hooks do not drive, or MOE not
        // resident -- is not this case (#132).
        graph_short_warned_ = true;
        std::fprintf(stderr,
                     "[HTP] graph: %zu of %zu MoE ops bound at the first "
                     "M==1 call; the per-token entry is NOT used (MoE ops "
                     "and the RMSNORM / QK_NORM / ROPE / CONV1D_GATE / "
                     "ATTN_M1 hooks alike)\n",
                     moe_bound_, moe_ops_.size());
      }
    }
    invokeMoeLayer(session, h_gu, h_dn, row_index, row_count, row_weight, act,
                   out, M, K, inter, N_out, 0, gelu ? 1u : 0u, pre_gamma,
                   post_gamma, eps, add_x2, add_gamma, add_scale);
  }

  /** [#85] NNTR_HTP_FORWARD=1 routes decode's MoE calls through the
   *  per-token entry. Off by default: with only MOE resident it is 22
   *  calls per token either way (plan 85 section 0), so the default path
   *  stays byte-for-byte today's. */
  static bool forwardSwitch() {
    static const bool on = [] {
      const char *env = std::getenv("NNTR_HTP_FORWARD");
      // [#132 Part B E3] NNTR_HTP_E2E=1 is the per-token entry with every
      // kind resident, on two sessions
      return (env != nullptr && std::atoi(env) != 0) ||
             HtpBackend::e2eRequested();
    }();
    return on;
  }

  /** [#132 Part B E3] The session that runs @a kind: with NNTR_HTP_E2E=1
   *  S1 (the default session) keeps ROUTER_TOPK and MOE and S2 runs every
   *  other kind (plan section 3.2); otherwise the one session. */
  remote_handle64 sessionFor(uint32_t kind) const {
    const HtpBackend &b = HtpBackend::global();
    return static_cast<remote_handle64>(
      e2e_ && (HTP_GRAPH_KINDS_S1 & HTP_GRAPH_KIND_BIT(kind)) == 0u
        ? b.handle2()
        : b.handle());
  }

  /** The graph entries' error, named: 0x8000040E from them means the
   *  skel predates the per-token entry (rule 3), not a list the validator
   *  above accepted. */
  static std::string graphErr(int err) {
    char hex[16];
    std::snprintf(hex, sizeof(hex), "%08x", static_cast<unsigned>(err));
    return std::string(htp_graph_err_name(static_cast<uint32_t>(err))) +
           " (0x" + hex + ")" +
           (static_cast<unsigned>(err) == 0x8000040Eu
              ? " -- AEE_EBADPARM: rebuild the DSP skel (test/htp/build.sh)"
              : "");
  }

  /** [#130] The kinds this ARM side can drive: the MoE call and the
   *  layer hooks (htp_decode_hook.h), with #132's residual add and router.
   *  The skel's own table is checked at graph_init (AEE_ECLASSNOTSUPPORT
   *  names a kind it lacks). */
  static constexpr uint32_t kArmKinds =
    HTP_GRAPH_KIND_BIT(HTP_OP_MOE) | HTP_GRAPH_KIND_BIT(HTP_OP_RMSNORM) |
    HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) | HTP_GRAPH_KIND_BIT(HTP_OP_ROPE) |
    HTP_GRAPH_KIND_BIT(HTP_OP_CONV1D_GATE) |
    HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1) | HTP_GRAPH_KIND_BIT(HTP_OP_ADD) |
    HTP_GRAPH_KIND_BIT(HTP_OP_ROUTER_TOPK) | HTP_GRAPH_KINDS_Q4M1;

  /** [#132 Part B] Rows per lm_head slice (plan 132 section 3.2: 8 slices
   *  of 16384 rows at LFM2.5, 18 MiB each). */
  static constexpr uint32_t kLmHeadSliceRows = 16384u;

  bool set_decode_graph_desc(const std::vector<uint32_t> &words) override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    uint32_t n_ops = 0;
    // The same validator the skel runs, so a bad list fails here with the
    // same code and name before any FastRPC call is made.
    const uint32_t rc = htp_graph_validate(
      words.data(), static_cast<uint32_t>(words.size()), kArmKinds, &n_ops);
    if (rc != 0u) {
      throw std::invalid_argument(std::string("set_decode_graph_desc: ") +
                                  htp_graph_err_name(rc));
    }
    uint32_t mask = 0, present = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      present |= HTP_GRAPH_KIND_BIT(op->kind);
      if (op->resident)
        mask |= HTP_GRAPH_KIND_BIT(op->kind);
    }
    // The ARM's own rule: a resident QK_NORM or ROPE needs ATTN_M1 --
    // no CPU layer consumes their DSP output (the qkv hook writes
    // nothing; ROPE has no layer), so the stretch must end in the
    // attention hook. The DSP enforces only ATTN_M1 -> ROPE.
    if ((mask & (HTP_GRAPH_KIND_BIT(HTP_OP_QK_NORM) |
                 HTP_GRAPH_KIND_BIT(HTP_OP_ROPE))) != 0u &&
        (mask & HTP_GRAPH_KIND_BIT(HTP_OP_ATTN_M1)) == 0u) {
      throw std::invalid_argument("set_decode_graph_desc: QK_NORM / ROPE "
                                  "resident without ATTN_M1 (no CPU layer "
                                  "consumes their output)");
    }
    // [#132] A resident ROUTER_TOPK needs ADD resident: then the ffn norm
    // before it and the ADD after its MOE are resident too, so the MoE
    // layer's one hook is always mid-stretch and the layer skips its
    // router, top-k and experts. Without ADD the MoE call would start the
    // stretch, and it has no row of its own to start it with.
    if ((mask & HTP_GRAPH_KIND_BIT(HTP_OP_ROUTER_TOPK)) != 0u &&
        (mask & HTP_GRAPH_KIND_BIT(HTP_OP_ADD)) == 0u) {
      throw std::invalid_argument("set_decode_graph_desc: ROUTER_TOPK "
                                  "resident without ADD");
    }
    // [#132 Part B, E1] No CPU layer hooks an FC, the dense FFN or a
    // projection, so a Q4M1 kind is resident only when every kind the
    // list has is: the list is then one stretch per token, op 0's norm
    // hook keeps the row and the lm_head hook runs it (one call per
    // token). Its weights come from the model at load
    // (add_decode_graph_q4_0).
    if ((mask & HTP_GRAPH_KINDS_Q4M1) != 0u && mask != present) {
      throw std::invalid_argument(
        "set_decode_graph_desc: FC / DENSE_FFN / LM_HEAD resident without "
        "every other kind (one session runs them only as the whole token)");
    }
    // [#132 Part B] NNTR_HTP_FC_FEED=l2 reads the Q4M1 kinds' weights
    // through the L2 scratch even where VTCM would fit (the second
    // session's feed, #178); unset or vtcm: VTCM when it fits
    const char *feed_env = std::getenv("NNTR_HTP_FC_FEED");
    const std::string feed_s = feed_env ? feed_env : "vtcm";
    if (feed_s != "vtcm" && feed_s != "l2") {
      throw std::invalid_argument("set_decode_graph_desc: NNTR_HTP_FC_FEED=" +
                                  feed_s + " (want vtcm or l2)");
    }
    // [#132 Part B E3] NNTR_HTP_E2E=1: the same description on two
    // sessions, S1 resident for ROUTER_TOPK | MOE and S2 for the rest; each
    // session's copy must pass the validator on its own (plan section 3.2).
    const bool e2e = HtpBackend::e2eRequested();
    if (e2e) {
      if (mask != present) {
        throw std::invalid_argument(
          "set_decode_graph_desc: NNTR_HTP_E2E=1 needs every kind of the "
          "list resident");
      }
      if (graph_inited_) {
        // ponytail: one description per process on the two-session path
        // (the model sets it once); a reload would re-map S2's arena
        throw std::runtime_error("set_decode_graph_desc: a second "
                                 "description with NNTR_HTP_E2E=1");
      }
      for (const uint32_t m : {HTP_GRAPH_KINDS_S1, HTP_GRAPH_KINDS_S2}) {
        std::vector<uint32_t> w = words;
        htp_graph_set_resident(w.data(), m);
        uint32_t n = 0;
        const uint32_t rc2 = htp_graph_validate(
          w.data(), static_cast<uint32_t>(w.size()), kArmKinds, &n);
        if (rc2 != 0u) {
          throw std::invalid_argument(
            std::string("set_decode_graph_desc: the ") +
            (m == HTP_GRAPH_KINDS_S1 ? "S1" : "S2") +
            " mask: " + htp_graph_err_name(rc2));
        }
      }
    }
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    if (graph_inited_) {
      // A second description in one process (a reload): the skel still
      // holds the first graph and would refuse the next init.
      nntr_hvx_graph_release(session);
      if (attn_registered_)
        nntr_hvx_attn_m1_release(session);
    }
    releaseQ4m1(session);
    q4_pending_.clear();
    q4m1_bound_ = false;
    graph_words_ = words;
    e2e_ = e2e;
    if (e2e && !e2e_st_) {
      e2e_st_ = std::make_shared<E2eState>();
      e2e_st_->h1 = session; // S2's handle and domain once it opens
      std::shared_ptr<E2eState> st = e2e_st_;
      HtpBackend::global().atClose([st] { e2eTeardown(*st); });
    }
    // [#132 Part B E5f] NNTR_HTP_FC_LANES=<small>,<large>: the L2 feed's
    // lanes for K <= 2048 / larger K (1..8 each; unset: the DSP's 6 / 3)
    uint32_t fc_lanes = 0;
    if (const char *l = std::getenv("NNTR_HTP_FC_LANES")) {
      unsigned a = 0, b = 0;
      if (std::sscanf(l, "%u,%u", &a, &b) != 2 || a < 1 || a > 8 || b < 1 ||
          b > 8)
        throw std::invalid_argument("NNTR_HTP_FC_LANES=" + std::string(l) +
                                    " (want <small>,<large>, 1..8 each)");
      fc_lanes = (a << 8) | (b << 12);
    }
    // [#194, htp_moe_ppl] NNTR_HTP_PPL_LEVERS=<mask>: the non-bit-exact
    // levers of plan 194 section 3.1, bit n = lever Ln (0x2: L1, the native
    // FC / DENSE_FFN / LM_HEAD kernels). Unset or 0 = the CPU-exact kernels
    // (E0, bit-identical to htp_moe); a bit no lever owns yet is refused.
    uint32_t levers = 0;
    if (const char *l = std::getenv("NNTR_HTP_PPL_LEVERS")) {
      char *end = nullptr;
      const unsigned long v = std::strtoul(l, &end, 0);
      if (end == l || *end != '\0' || (v & ~0x2ul) != 0)
        throw std::invalid_argument("NNTR_HTP_PPL_LEVERS=" + std::string(l) +
                                    " (known: 0x2 = L1 native FC)");
      levers = static_cast<uint32_t>(v);
    }
    std::fprintf(stderr, "[HTP] ppl levers=0x%x L1=%s\n", levers,
                 (levers & 0x2u) ? "native_fc" : "exact");
    for (uint32_t i = 0; i < n_ops; ++i) {
      htp_graph_op *op = htp_graph_op_at(graph_words_.data(), i);
      if ((HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) != 0u)
        op->feed = (feed_s == "l2" ? HTP_GRAPH_FEED_L2 : 0u) | fc_lanes |
                   ((levers & 0x2u) ? HTP_GRAPH_FEED_NATIVE : 0u);
    }
    resident_mask_ = mask;
    moe_ops_.clear();
    for (auto &v : kind_ops_)
      v.clear();
    stretch_start_.assign(n_ops, 0);
    stretch_end_.assign(n_ops, 0);
    attn_ordinal_.assign(n_ops, 0);
    param_bound_.assign(n_ops, 0);
    conv_next_pos_.assign(n_ops, HTP_GRAPH_NO_OP);
    first_resident_op_ = HTP_GRAPH_NO_OP;
    uint32_t n_attn = 0;
    for (uint32_t i = 0; i < n_ops; ++i) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      if (op->kind == HTP_OP_MOE)
        moe_ops_.push_back(i);
      if (op->kind == HTP_OP_ATTN_M1)
        attn_ordinal_[i] = n_attn++;
      if (!op->resident)
        continue;
      kind_ops_[op->kind].push_back(i);
      if (first_resident_op_ == HTP_GRAPH_NO_OP)
        first_resident_op_ = i;
      // a stretch is a maximal run of resident ops [s, e)
      stretch_start_[i] =
        (i > 0 && htp_graph_op_cat(words.data(), i - 1)->resident)
          ? stretch_start_[i - 1]
          : i;
    }
    for (uint32_t i = n_ops; i-- > 0;) {
      const htp_graph_op *op = htp_graph_op_cat(words.data(), i);
      if (!op->resident)
        continue;
      stretch_end_[i] =
        (i + 1 < n_ops && htp_graph_op_cat(words.data(), i + 1)->resident)
          ? stretch_end_[i + 1]
          : i + 1;
    }
    kv_len_.assign(n_attn, 0);
    std::fill(std::begin(kind_next_), std::end(kind_next_), 0u);
    cur_pos_ = HTP_GRAPH_NO_OP;
    clearPending();
    seed_ordinal_ = HTP_GRAPH_NO_OP;
    rope_bound_ = false;
    attn_registered_ = false;
    moe_bound_ = 0;
    moe_op_by_handle_.clear();
    graph_inited_ = false;
    HtpProfile::global().setGraphResident(mask);
    char names[128];
    std::fprintf(stderr,
                 "[HTP] graph: description n_ops=%u resident=%s "
                 "moe_ops=%zu\n",
                 n_ops, htp_graph_kinds_str(mask, names, sizeof(names)),
                 moe_ops_.size());
    return true;
  }

  /** [#132 Part B] The model's Q4_0 weights of the decode list's FC,
   *  DENSE_FFN and LM_HEAD ops, in list order (lfm2_moe_causallm.cpp at
   *  load): kept as pointers here, converted and registered at graph init
   *  (bindQ4m1). False when no such kind is resident. */
  bool add_decode_graph_q4_0(const void *data, unsigned K, unsigned N,
                             bool canonical) override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (graph_words_.empty() || (resident_mask_ & HTP_GRAPH_KINDS_Q4M1) == 0u)
      return false;
    if (graph_inited_) {
      throw std::runtime_error(
        "add_decode_graph_q4_0: the graph is already initialised");
    }
    q4_pending_.push_back({data, K, N, canonical});
    return true;
  }

  /** [#132 Part B] One weight's rows [r0, r0 + rows) as a Q4M1 handle:
   *  canonical block_q4_0 (unpacked from the CPU's repack unless the
   *  model says it is canonical already, as the tied lm_head is), the
   *  Q4M1 reorder, q4m1_register. The same nibbles and scales the CPU
   *  reads (plan 132 section 2), so no format tag moves. */
  uint32_t registerQ4m1(remote_handle64 session, const uint8_t *canonical,
                        uint32_t K, uint32_t r0, uint32_t rows) {
    const size_t row_bytes = static_cast<size_t>(K / 32u) * Q4_CPU_BLOCK_BYTES;
    std::vector<uint8_t> q4m1(q4m1_bytes(K, rows));
    q4m1_from_q4_0(canonical + r0 * row_bytes, K, rows, q4m1.data());
    uint32_t h = 0;
    if (e2e_) {
      // [#132 Part B E3] into S2's arena (the S1 path's place / tryChunk /
      // arena_attach on S2's handle and effective domain), written before
      // the attach; S2's heap holds no weight (the #178 heap rule)
      E2eState &e = *e2e_st_;
      const uint32_t bytes = static_cast<uint32_t>(q4m1.size());
      uint32_t chunk = 0, off = 0;
      if (!placeOn(e.h2, e.dom2, e.arena, e.arena_cap, bytes, q4m1_left_,
                   &chunk, &off)) {
        throw std::runtime_error("S2 FC arena: " + arena_fail_);
      }
      std::memcpy(e.arena[chunk].buf->data() + off, q4m1.data(), bytes);
      q4m1_left_ = q4m1_left_ > bytes ? q4m1_left_ - bytes : 0u;
      e.attach_bytes += bytes;
      const int err =
        nntr_hvx_q4m1_attach(e.h2, e.arena[chunk].dsp_id, off, K, rows, &h);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_q4m1_attach(K=" + std::to_string(K) +
                                 " N=" + std::to_string(rows) +
                                 " off=" + std::to_string(off) +
                                 ") failed: " + graphErr(err));
      }
      e.q4m1.push_back(h);
      return h;
    }
    const int err = nntr_hvx_q4m1_register(session, K, rows, q4m1.data(),
                                           static_cast<int>(q4m1.size()), &h);
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_q4m1_register(K=" + std::to_string(K) +
                               " N=" + std::to_string(rows) +
                               ") failed: " + graphErr(err));
    }
    q4m1_handles_.push_back(h);
    return h;
  }

  /** [#132 Part B] Frees every Q4M1 handle registered for the graph (after
   *  graph_release, or when bindQ4m1 / graph_init failed half-way). */
  void releaseQ4m1(remote_handle64 session) {
    for (uint32_t h : q4m1_handles_)
      nntr_hvx_q4m1_release(session, h);
    q4m1_handles_.clear();
    if (e2e_st_) { // [#132 Part B E3] S2's slots (the arena stays mapped)
      for (uint32_t h : e2e_st_->q4m1)
        nntr_hvx_q4m1_release(e2e_st_->h2, h);
      e2e_st_->q4m1.clear();
    }
  }

  /** [#132 Part B] Binds q4_pending_ to the resident Q4M1 ops in list
   *  order before graph_init checks them: an FC takes weights of its K
   *  until their widths make its N (q | k | v are three), a DENSE_FFN up,
   *  gate and down, the LM_HEAD the tied table in kLmHeadSliceRows slices.
   *  Throws when the model's weights and the list disagree. */
  void bindQ4m1(remote_handle64 session) {
    size_t next = 0;
    std::vector<uint8_t> canon;
    q4m1_left_ = 0; // [#132 Part B E3] what S2's arena chunks are sized for
    for (const Q4Pending &p : q4_pending_)
      q4m1_left_ += q4m1_bytes(p.K, p.N) + 4096u;
    auto take = [&](uint32_t op, uint32_t K, uint32_t N) -> const uint8_t * {
      if (next >= q4_pending_.size())
        throw std::runtime_error("set_decode_graph_desc: op " +
                                 std::to_string(op) + " (" +
                                 htp_graph_kind_name(graphOp(op)->kind) +
                                 ") has no Q4_0 weight left "
                                 "(the model handed " +
                                 std::to_string(q4_pending_.size()) + ")");
      const Q4Pending &p = q4_pending_[next++];
      if (p.K != K || (N != 0u && p.N != N))
        throw std::runtime_error(
          "set_decode_graph_desc: op " + std::to_string(op) + " wants a " +
          std::to_string(K) + " x " + (N ? std::to_string(N) : "*") +
          " Q4_0 weight, the model's next is " + std::to_string(p.K) + " x " +
          std::to_string(p.N));
      const size_t bytes =
        static_cast<size_t>(p.N) * (p.K / 32u) * Q4_CPU_BLOCK_BYTES;
      if (p.canonical)
        return static_cast<const uint8_t *>(p.data);
      canon.resize(bytes);
      nntrainer::unpack_q4_0(p.data, canon.data(), bytes, p.N, p.K);
      return canon.data();
    };
    const uint32_t n_ops = static_cast<uint32_t>(stretch_start_.size());
    for (uint32_t i = 0; i < n_ops; ++i) {
      htp_graph_op *op = htp_graph_op_at(graph_words_.data(), i);
      if (!op->resident ||
          (HTP_GRAPH_KINDS_Q4M1 & HTP_GRAPH_KIND_BIT(op->kind)) == 0u)
        continue;
      uint32_t parts = 0;
      if (op->kind == HTP_OP_DENSE_FFN) {
        op->h_gu[0] = registerQ4m1(session, take(i, op->K, op->N), op->K, 0u,
                                   op->N); // up
        op->h_gu[1] = registerQ4m1(session, take(i, op->K, op->N), op->K, 0u,
                                   op->N); // gate
        op->h_dn[0] = registerQ4m1(session, take(i, op->N, op->N_out), op->N,
                                   0u, op->N_out);
        parts = 3u;
      } else if (op->kind == HTP_OP_LM_HEAD) {
        const uint8_t *w = take(i, op->K, op->N);
        for (uint32_t r0 = 0; r0 < op->N; r0 += kLmHeadSliceRows)
          op->h_gu[parts++] = registerQ4m1(
            session, w, op->K, r0, std::min(kLmHeadSliceRows, op->N - r0));
      } else {
        uint32_t sum = 0;
        while (sum < op->N && parts < HTP_GRAPH_MAX_EXPERTS) {
          const uint8_t *w = take(i, op->K, 0u);
          const uint32_t n = q4_pending_[next - 1].N;
          op->h_gu[parts++] = registerQ4m1(session, w, op->K, 0u, n);
          sum += n;
        }
        if (sum != op->N)
          throw std::runtime_error(
            "set_decode_graph_desc: FC op " + std::to_string(i) +
            " is N=" + std::to_string(op->N) + ", its weights sum to " +
            std::to_string(sum));
      }
      op->n_experts = parts;
    }
    if (next != q4_pending_.size())
      throw std::runtime_error(
        "set_decode_graph_desc: the model handed " +
        std::to_string(q4_pending_.size()) + " Q4_0 weights, the list's " +
        "FC / DENSE_FFN / LM_HEAD ops took " + std::to_string(next));
    std::fprintf(stderr, "[HTP] graph: q4m1 weights=%zu handles=%zu feed=%s\n",
                 q4_pending_.size(),
                 e2e_ ? e2e_st_->q4m1.size() : q4m1_handles_.size(),
                 q4m1FeedName());
  }

  /** [#130] The op record of @a op in the description (the words are
   *  the model's; the resident bits and shapes do not change after
   *  set_decode_graph_desc). */
  const htp_graph_op *graphOp(uint32_t op) const {
    return htp_graph_op_cat(graph_words_.data(), op);
  }

  /** [#130] graph_set_param once per (op, which); throws on any error
   *  (contract section 2: no silent fallback). */
  void setParam(remote_handle64 session, uint32_t op, uint32_t which,
                const float *data, unsigned n, unsigned want,
                const char *what) {
    if (data == nullptr || n != want) {
      throw std::runtime_error(std::string("decode_op_fp32: ") + what +
                               " of op " + std::to_string(op) + ": " +
                               std::to_string(n) + " floats, want " +
                               std::to_string(want));
    }
    const int err =
      nntr_hvx_graph_set_param(session, op, which, data, static_cast<int>(n));
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(std::string("nntr_hvx_graph_set_param(") + what +
                               ") failed at op " + std::to_string(op) + ": " +
                               graphErr(err));
    }
  }

  /** [#130] The per-token hook (compute_ops.h): the counter of its kind
   *  names the op, the stretch tables say what to run. Hooks fire only at
   *  one row, from the model's thread, in list order within a token; the
   *  counters reset when pos changes, so every token re-synchronises and
   *  a prefill (rows > 1, no hooks) never disturbs them. */
  int decode_op_fp32(unsigned kind, unsigned pos, const float *in,
                     unsigned in_len, float *out, unsigned out_len,
                     const float *param, unsigned param_len, const float *state,
                     unsigned state_len, float eps) override {
    if (!forwardSwitch() || graph_words_.empty() || kind >= HTP_OP_KIND_N ||
        (resident_mask_ & HTP_GRAPH_KIND_BIT(kind)) == 0u)
      return 0;
    if (pos != cur_pos_) {
      if (pending_op_ != HTP_GRAPH_NO_OP) {
        throw std::runtime_error(
          "decode_op_fp32: pos moved to " + std::to_string(pos) +
          " with the stretch input of op " + std::to_string(pending_op_) +
          " pending (its last hook never came)");
      }
      std::fill(std::begin(kind_next_), std::end(kind_next_), 0u);
      cur_pos_ = pos;
      // Decided once per row, so a row that starts before the MoE handles
      // are all bound (a one-token prompt binds them as it goes) runs on
      // the CPU end to end: a hook after the last MoE call would otherwise
      // resolve to the first op of its kind. The subset case
      // (moe_htp_layers) stays 0 for good; gemm_qs4cx_moe_layer_fp32
      // warned once.
      row_bound_ = moe_bound_ == moe_ops_.size();
    }
    if (!row_bound_)
      return 0;
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    ensureGraphInit(session);
    // [#132 Part B E3] a kind's parameters go to the session that runs it
    const remote_handle64 psess = sessionFor(kind);
    const std::vector<uint32_t> &ops = kind_ops_[kind];
    if (kind_next_[kind] >= ops.size()) {
      throw std::runtime_error(std::string("decode_op_fp32: more ") +
                               htp_graph_kind_name(kind) + " hooks at pos " +
                               std::to_string(pos) + " than the list's " +
                               std::to_string(ops.size()) + " ops");
    }
    const uint32_t op = ops[kind_next_[kind]++];
    const htp_graph_op *rec = graphOp(op);
    if (rec->kind == HTP_OP_RMSNORM || rec->kind == HTP_OP_QK_NORM) {
      uint32_t bits;
      std::memcpy(&bits, &eps, sizeof(bits));
      if (bits != rec->eps_bits) {
        throw std::runtime_error("decode_op_fp32: op " + std::to_string(op) +
                                 " epsilon differs from the description's");
      }
    }
    switch (kind) {
    case HTP_OP_RMSNORM:
      if (!param_bound_[op]) {
        setParam(psess, op, HTP_GRAPH_PARAM_GAMMA, param, param_len, rec->K,
                 "gamma");
        param_bound_[op] = 1;
      }
      break;
    case HTP_OP_CONV1D_GATE:
      if (!param_bound_[op]) {
        setParam(psess, op, HTP_GRAPH_PARAM_CONV_W, param, param_len,
                 3u * rec->N, "conv_w");
        param_bound_[op] = 1;
      }
      // The DSP advances the state itself; re-seed from the CPU's copy
      // after any jump (the first token after a prefill, which refreshed
      // it; plan 130 section 3.3).
      if (conv_next_pos_[op] != pos) {
        setParam(psess, op, HTP_GRAPH_PARAM_CONV_STATE, state, state_len,
                 2u * rec->N, "conv state");
        if (dumpAllDir() != nullptr)
          dumpAllFile(std::string(dumpAllDir()) + "/convstate_" +
                        std::to_string(op) + ".f32",
                      state, state_len);
      }
      conv_next_pos_[op] = pos + 1u;
      break;
    case HTP_OP_QK_NORM:
      if (!param_bound_[op]) {
        setParam(psess, op, HTP_GRAPH_PARAM_GAMMA, param, param_len,
                 2u * rec->head_dim, "q | k gamma");
        param_bound_[op] = 1;
      }
      if (in == nullptr || in_len != rec->K) {
        throw std::runtime_error("decode_op_fp32: QK_NORM row of " +
                                 std::to_string(in_len) + " floats, want " +
                                 std::to_string(rec->K));
      }
      break;
    case HTP_OP_ATTN_M1: {
      const uint32_t ord = attn_ordinal_[op];
      if (!rope_bound_) {
        setParam(psess, HTP_GRAPH_NO_OP, HTP_GRAPH_PARAM_ROPE_TABLE, param,
                 param_len, graph_words_[6] * 64u, "RoPE table");
        rope_bound_ = true;
      }
      // The DSP cache must hold rows [0, pos): seed it from the layer's
      // copy after any jump (the first token after a prefill, a second
      // prompt). pos 0 needs nothing; the kernel rewinds.
      if (pos != 0 && kv_len_[ord] != pos) {
        seed_ordinal_ = ord;
        --kind_next_[kind]; // the layer calls again for this same op
        return 2;
      }
      kv_len_[ord] = pos + 1u;
      break;
    }
    case HTP_OP_ADD:
    case HTP_OP_LM_HEAD: // [#132 Part B] weights bound at init
      break;
    case HTP_OP_ROUTER_TOPK:
      // [#132] the gate weight [K][E] and the expert bias, once
      if (!param_bound_[op]) {
        setParam(psess, op, HTP_GRAPH_PARAM_ROUTER_W, param, param_len,
                 rec->K * rec->n_experts, "router weight");
        setParam(psess, op, HTP_GRAPH_PARAM_ROUTER_BIAS, state, state_len,
                 rec->n_experts, "router bias");
        param_bound_[op] = 1;
      }
      break;
    default:
      return 0;
    }
    runStretchOp(session, op, pos, in, in_len, out, out_len, nullptr, nullptr,
                 nullptr);
    return 1;
  }

  /** [#132] Every hook's one dispatch over its op's stretch [s, e) (plan
   *  132 section 3.4): the first op keeps its input row (and, for a MOE
   *  op, the ARM's routing) pending and returns; a mid op checks that the
   *  row is there and returns; the last op -- or a sole one -- runs the
   *  stretch on the pending row, or on its own when it is also the first.
   *  The layer skips its kernel either way: only the last op's @a out is
   *  written, and the CPU tensors between are stale by design (the DSP
   *  holds the stream: slot 0 the residual, the routing in the router's
   *  record). */
  void runStretchOp(remote_handle64 session, uint32_t op, uint32_t pos,
                    const float *in, unsigned in_len, float *out,
                    unsigned out_len, const std::vector<unsigned int> *ri,
                    const std::vector<unsigned int> *rc,
                    const std::vector<float> *rw) {
    static const std::vector<unsigned int> no_u;
    static const std::vector<float> no_f;
    const uint32_t s = stretch_start_[op], e = stretch_end_[op];
    // The level-3 repeat, the MoE dump and the M==1 MoE profile row are
    // for the sole MOE stretch only: an ADD in a stretch moves slot 0, so
    // a repeat would add twice, and C (KINDS=MOE) stays comparable to A.
    const bool sole_moe = graphOp(s)->kind == HTP_OP_MOE && e == s + 1u;
    // ROPE has no layer hook: a [ROPE ATTN_M1] stretch starts at the
    // attention hook, whose row is ROPE's input
    uint32_t first = s;
    while (graphOp(first)->kind == HTP_OP_ROPE)
      ++first;
    if (first == op) {
      if (pending_op_ != HTP_GRAPH_NO_OP) {
        throw std::runtime_error(
          "decode_op_fp32: op " + std::to_string(op) +
          " starts a stretch while op " + std::to_string(pending_op_) +
          "'s row is pending (its last hook never came)");
      }
      if (e == op + 1u) {
        invokeForward(session, s, e, pos, ri ? *ri : no_u, rc ? *rc : no_u,
                      rw ? *rw : no_f, in, in_len, out, out_len, sole_moe);
        return;
      }
      if (in == nullptr) {
        throw std::runtime_error("decode_op_fp32: op " + std::to_string(op) +
                                 " starts a stretch with no row");
      }
      pending_in_.assign(in, in + in_len);
      pending_ri_ = ri ? *ri : no_u;
      pending_rc_ = rc ? *rc : no_u;
      pending_rw_ = rw ? *rw : no_f;
      pending_op_ = s;
      return;
    }
    if (pending_op_ != s) {
      throw std::runtime_error(
        "decode_op_fp32: " +
        std::string(htp_graph_kind_name(graphOp(op)->kind)) + " op " +
        std::to_string(op) + " without the row of its stretch's first op " +
        std::to_string(s) + " (its hook never came)");
    }
    if (e == op + 1u) {
      invokeForward(session, s, e, pos, pending_ri_, pending_rc_, pending_rw_,
                    pending_in_.data(),
                    static_cast<unsigned>(pending_in_.size()), out, out_len,
                    sole_moe);
      clearPending();
    }
  }

  void clearPending() {
    pending_op_ = HTP_GRAPH_NO_OP;
    pending_in_.clear();
    pending_ri_.clear();
    pending_rc_.clear();
    pending_rw_.clear();
  }

  /** [#132 Part B] compute_ops.h: with a Q4M1 kind resident every kind
   *  is (set_decode_graph_desc's rule), so the token is one stretch; once
   *  op 0's hook at @a pos has kept the row, the DSP owns the whole row
   *  and the CPU's FC outputs until the lm_head hook are never read. */
  bool decode_row_resident(unsigned pos) override {
    const bool on = (resident_mask_ & HTP_GRAPH_KINDS_Q4M1) != 0u &&
                    row_bound_ && pos == cur_pos_ &&
                    pending_op_ != HTP_GRAPH_NO_OP;
    cpu_fc_skipped_ += on;
    return on;
  }

  bool decode_kv_seed_fp32(unsigned n_rows, const float *k_rows,
                           const float *v_rows) override {
    if (seed_ordinal_ == HTP_GRAPH_NO_OP) {
      throw std::runtime_error(
        "decode_kv_seed_fp32: no attention hook asked for a seed");
    }
    const remote_handle64 session = sessionFor(HTP_OP_ATTN_M1);
    const htp_graph_op *rec = graphOp(kind_ops_[HTP_OP_ATTN_M1][0]);
    const int n = static_cast<int>(n_rows * rec->n_kv * rec->head_dim);
    const int err = nntr_hvx_attn_m1_kv_append(session, seed_ordinal_, 0u,
                                               n_rows, k_rows, n, v_rows, n);
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_attn_m1_kv_append failed at layer " +
                               std::to_string(seed_ordinal_) + ", " +
                               std::to_string(n_rows) +
                               " rows: " + graphErr(err));
    }
    if (dumpAllDir() != nullptr) {
      const std::string base =
        std::string(dumpAllDir()) + "/kvseed_" + std::to_string(seed_ordinal_);
      dumpAllFile(base + "_k.f32", k_rows, static_cast<size_t>(n));
      dumpAllFile(base + "_v.f32", v_rows, static_cast<size_t>(n));
    }
    kv_len_[seed_ordinal_] = n_rows;
    seed_ordinal_ = HTP_GRAPH_NO_OP;
    return true;
  }

  /** Binds a layer's handles to the next unbound MoE op (call order =
   *  list order on the first pass) and returns the op that owns them. */
  uint32_t bindMoeOp(const std::vector<uint32_t> &h_gu,
                     const std::vector<uint32_t> &h_dn, unsigned K,
                     unsigned inter, unsigned N_out) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    auto it = moe_op_by_handle_.find(h_gu[0]);
    if (it != moe_op_by_handle_.end())
      return it->second;
    if (moe_bound_ >= moe_ops_.size() || graph_inited_) {
      throw std::runtime_error(
        "set_decode_graph_desc: more MoE layers than the list's " +
        std::to_string(moe_ops_.size()) + " MoE ops");
    }
    const uint32_t idx = moe_ops_[moe_bound_];
    htp_graph_op *op = htp_graph_op_at(graph_words_.data(), idx);
    if (op->n_experts != h_gu.size() || op->K != K || op->N != inter ||
        op->N_out != N_out) {
      throw std::runtime_error(
        "set_decode_graph_desc: MoE op " + std::to_string(idx) +
        " expects experts=" + std::to_string(op->n_experts) +
        " K=" + std::to_string(op->K) + " inter=" + std::to_string(op->N) +
        " N_out=" + std::to_string(op->N_out) + ", the layer has " +
        std::to_string(h_gu.size()) + "/" + std::to_string(K) + "/" +
        std::to_string(inter) + "/" + std::to_string(N_out));
    }
    std::copy(h_gu.begin(), h_gu.end(), op->h_gu);
    std::copy(h_dn.begin(), h_dn.end(), op->h_dn);
    moe_op_by_handle_.emplace(h_gu[0], idx);
    ++moe_bound_;
    return idx;
  }

  void ensureGraphInit(remote_handle64 session) {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (graph_inited_)
      return;
    if (e2e_ && !q4m1_bound_)
      e2ePlaceFc(); // a caller that never ran finish_decode_graph_q4_0
    const remote_handle64 fc_session = sessionFor(HTP_OP_FC);
    if ((resident_mask_ & HTP_GRAPH_KINDS_Q4M1) != 0u && !q4m1_bound_) {
      try {
        bindQ4m1(fc_session);
      } catch (...) {
        releaseQ4m1(fc_session); // a retry registers them all again
        throw;
      }
      // [#132 Part B E3] S2's arena is placed once: a retry keeps it
      q4m1_bound_ = e2e_;
    }
    uint32_t n_ops = 0;
    if (e2e_) {
      // [#132 Part B E3] each session its copy of the list: S1 resident for
      // ROUTER_TOPK | MOE (the MoE handles bound at prefill), S2 for the
      // rest (the Q4M1 handles on its arena)
      for (int k = 0; k < 2; ++k) {
        std::vector<uint32_t> w = graph_words_;
        htp_graph_set_resident(w.data(), k == 0 ? HTP_GRAPH_KINDS_S1
                                                : HTP_GRAPH_KINDS_S2);
        const remote_handle64 h = k == 0 ? e2e_st_->h1 : e2e_st_->h2;
        const int err =
          nntr_hvx_graph_init(h, w.data(), static_cast<int>(w.size()), &n_ops);
        if (err != AEE_SUCCESS) {
          if (k == 1)
            nntr_hvx_graph_release(e2e_st_->h1); // a retry inits both again
          throw std::runtime_error(std::string("nntr_hvx_graph_init[") +
                                   (k == 0 ? "S1" : "S2") +
                                   "] failed: " + graphErr(err));
        }
        if (k == 1)
          e2e_st_->graph2 = true;
      }
    } else {
      const int err =
        nntr_hvx_graph_init(session, graph_words_.data(),
                            static_cast<int>(graph_words_.size()), &n_ops);
      if (err != AEE_SUCCESS) {
        releaseQ4m1(session);
        throw std::runtime_error("nntr_hvx_graph_init failed: " +
                                 graphErr(err));
      }
    }
    graph_inited_ = true;
    char names[128];
    std::fprintf(stderr, "[HTP] graph: init n_ops=%u resident=%s moe_ops=%zu\n",
                 n_ops,
                 htp_graph_kinds_str(resident_mask_, names, sizeof(names)),
                 moe_ops_.size());
    // [#130] the session's KV cache for the resident ATTN_M1 ops: one
    // ordinal per attention layer, the shape from the first record
    if (!kind_ops_[HTP_OP_ATTN_M1].empty()) {
      const htp_graph_op *rec = graphOp(kind_ops_[HTP_OP_ATTN_M1][0]);
      const uint32_t n_attn =
        static_cast<uint32_t>(kind_ops_[HTP_OP_ATTN_M1].size());
      const uint32_t max_seq = graph_words_[6];
      const int rc =
        nntr_hvx_attn_m1_register(sessionFor(HTP_OP_ATTN_M1), n_attn, rec->n_kv,
                                  rec->gqa, rec->head_dim, max_seq);
      if (rc != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_attn_m1_register failed: " +
                                 graphErr(rc));
      }
      attn_registered_ = true;
      std::fprintf(stderr,
                   "[HTP] attn_m1: registered layers=%u kv=%u gqa=%u "
                   "head_dim=%u max_seq=%u cache=%llu KiB\n",
                   n_attn, rec->n_kv, rec->gqa, rec->head_dim, max_seq,
                   (unsigned long long)n_attn * rec->n_kv * rec->head_dim *
                     ((max_seq + 63u) / 64u * 64u) * 2u * sizeof(uint16_t) /
                     1024u);
    }
    if (e2e_)
      e2eStart();
  }

  /** [#80] The M=1 GEMV switch, read once from NNTR_MOE_HTP_M1_GEMV and
   *  sent to the DSP once per session through moe_set_opts (the DSP decides
   *  per call on M). On by default since #101; NNTR_MOE_HTP_M1_GEMV=0 is the
   *  opt-out (htp_moe_opts_flags). With the switch on, an error or an echo
   *  that differs from what was sent throws rather than falling back: a
   *  silent fallback would let a run report the HMX loop's numbers as the
   *  GEMV's -- so a skel older than moe_set_opts now fails the first MoE call
   *  unless the opt-out is set. With it off, such a skel is exactly the HMX
   *  loop, so that case only logs. The stderr line (source=default|env) is
   *  the proof of which path a run took when no profile is on;
   *  [HTP-PROFILE]'s m1_gemv= and blocks= are the per-call proof. */
  void sendMoeOptsOnce(remote_handle64 session) {
    std::call_once(moe_opts_once_, [this, session]() {
      const char *env = std::getenv("NNTR_MOE_HTP_M1_GEMV");
      const char *lead_env = std::getenv("NNTR_MOE_HTP_GEMV_LEAD_KB");
      const char *rows1_env = std::getenv("NNTR_MOE_HTP_GEMV_ROWS1");
      const char *feed_env = std::getenv("NNTR_MOE_HTP_GEMV_FEED");
      const uint32_t flags =
        htp_moe_opts_flags(env, lead_env, rows1_env, feed_env) |
        htp_moe_opts_dma_bypass(std::getenv("NNTR_MOE_DMA_BYPASS")) |
        htp_conv_opts_flags(std::getenv("NNTR_HTP_CONV_M1_HVX"),
                            std::getenv("NNTR_HTP_CONV_CHUNK_TILES"),
                            std::getenv("NNTR_HTP_CONV_OUT_CHUNK_TILES")) |
        htp_fc_opts_flags(std::getenv("NNTR_HTP_FC_M1_HVX"),
                          std::getenv("NNTR_HTP_FC_CHUNK_TILES")) |
        htp_qs2_gemv_flags(std::getenv("NNTR_HTP_QS2_GEMV"));
      const char *source = env != nullptr ? "env" : "default";
      uint32_t applied = 0;
      const int err = nntr_hvx_moe_set_opts(session, flags, &applied);
      if (htp_moe_opts_must_match(flags) == HTP_MOE_FLAG_M1_GEMV &&
          err != AEE_SUCCESS) {
        // The opt-out was asked for, and a skel that predates moe_set_opts
        // runs the HMX loop, which is what "off" means: say so and go on.
        std::fprintf(stderr,
                     "[HTP] moe m1 gemv: off (moe_set_opts err=0x%08x; the "
                     "skel predates it) source=%s\n",
                     static_cast<unsigned>(err), source);
        return;
      }
      // Off: only bit 0 (and #158's bypass bit, if asked for) has to be
      // echoed. The tune bits are always sent (htp_moe_opts_flags) but mean
      // nothing on the HMX loop, so a skel that predates them must not fail
      // an opt-out run.
      const uint32_t must_match = htp_moe_opts_must_match(flags);
      if (err != AEE_SUCCESS || ((applied ^ flags) & must_match) != 0u) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "nntr_hvx_moe_set_opts failed: err=0x%08x sent=0x%x "
                      "applied=0x%x",
                      static_cast<unsigned>(err), flags, applied);
        throw std::runtime_error(
          std::string(buf) +
          " (libnntr_hvx_skel.so on the device predates moe_set_opts; "
          "rebuild it: test/htp/build.sh, then push libnntr_hvx_skel.so)");
      }
      applied_moe_flags_ = applied;
      std::fprintf(stderr,
                   "[HTP] conv M1: %s chunk_tiles=%u out_chunk_tiles=%u "
                   "(applied=0x%x)\n",
                   (applied & HTP_CONV_FLAG_M1_HVX) ? "HVX" : "HMX",
                   htp_conv_chunk_tiles(applied),
                   htp_conv_out_chunk_tiles(applied), applied);
      std::fprintf(stderr, "[HTP] FC M1: %s chunk_tiles=%u (applied=0x%x)\n",
                   (applied & HTP_FC_FLAG_M1_HVX) ? "HVX" : "HMX",
                   htp_fc_chunk_tiles(applied), applied);
      // The measured (loop, lead, feed) cell, in the same line. All three
      // knobs are always sent (htp_moe_opts_flags: the env's value or the
      // default), and the echo above confirmed them, so an unset run reads
      // lead=192KB rows1=1 feed=vtcm and applied=0x303e1 (#113's D192
      // under #117's VTCM feed); NNTR_MOE_HTP_GEMV_FEED=0 reads feed=arena
      // and 0x103e1. The level-2 M==1 row's feed=n/calls is the per-call
      // proof. source= still refers to the M1 GEMV switch alone (LEDGER 16).
      std::fprintf(
        stderr,
        "[HTP] moe m1 gemv: %s (applied=0x%x) lead=%uKB rows1=%u "
        "feed=%s dma_bypass=%u source=%s\n",
        (flags & HTP_MOE_FLAG_M1_GEMV) != 0u ? "on" : "off", applied,
        ((flags >> HTP_MOE_GEMV_LEAD_SHIFT) & HTP_MOE_GEMV_LEAD_BITS) *
          HTP_MOE_GEMV_LEAD_KB_UNIT,
        (flags & HTP_MOE_FLAG_GEMV_ROWS1) != 0u ? 1u : 0u,
        htp_moe_opts_feed_name(flags),
        (applied & HTP_MOE_FLAG_DMA_BYPASS) != 0u ? 1u : 0u, source);
    });
  }

  /** Same registration the layer call above does on first use, keyed by
   *  the same data pointer, so the forward-time call is a cache hit. */
  bool register_qs4cx_weight(void *data, const float *scale, unsigned int K,
                             unsigned int N, bool weights_wh) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    if (weights_wh) {
      get_or_register_wh(data, scale, session, K, N);
    } else {
      get_or_register_qs4cx(data, scale, session, K, N);
    }
    return true;
  }

  /** Same, for a Q4_0 FC weight: the conversion and registration that
   *  gemm_q4_0_accel_fp32 would otherwise do on its first call. */
  bool register_q4_0_weight(void *data, unsigned int K,
                            unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_fc(data, session, K, N);
    return true;
  }

  bool register_prepared_qs4cx_wh_weight(
    void *data, const float *scales, unsigned int K, unsigned int N,
    uint64_t source_offset, const std::string &cache_directory) override {
    const uint64_t begin = HtpProfile::nowUs();
    const uint64_t source_hash = preparedFcSourceHash(data, scales, K, N);
    const std::string path =
      cache_directory + "/" + std::to_string(source_offset) + "_" +
      std::to_string(K) + "_" + std::to_string(N) + ".nntrwh";
    // Validate source identity and the complete payload before allocating
    // DSP memory or publishing handles. The reader bounds all allocations.
    const PreparedFcFile prepared =
      readPreparedFc(path, source_offset, K, N, source_hash);
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    std::lock_guard<std::mutex> lock(handle_mutex_);
    if (fc_cache_.count(data)) {
      auto found = prepared_fc_hashes_.find(data);
      if (found != prepared_fc_hashes_.end() && found->second == source_hash)
        return true;
      throw std::runtime_error(
        "prepared FC must be registered before runtime-converted weights: " +
        path);
    }
    if (!ensureArena(session))
      throw std::runtime_error("prepared FC requires a mapped DSP arena: " +
                               path);
    FcHandles handles;
    for (const PreparedFcSlice &slice : prepared.slices) {
      const uint32_t columns = slice.columns;
      const uint32_t bytes = static_cast<uint32_t>(uint64_t(K) * columns / 2);
      ArenaEntry entry;
      entry.K = K;
      entry.N = columns;
      // These allocations follow the MoE arena reservation. Prefer its
      // remaining room, then a bounded 64 MiB chunk rather than converting
      // to a DSP-heap registration when prepared storage is exhausted.
      if (!place(session, bytes, size_t(64) << 20, &entry.chunk, &entry.off))
        throw std::runtime_error("prepared FC arena placement failed: " + path +
                                 ": " + arena_fail_);
      const uint8_t *payload = prepared.payload.data() + slice.offset;
      std::memcpy(arena_chunks_[entry.chunk].buf->data() + entry.off, payload,
                  bytes);
      entry.w_scale.resize(columns);
      entry.colsum_w.resize(columns);
      entry.bias.assign(columns, 0.0f);
      std::memcpy(entry.w_scale.data(), payload + bytes,
                  size_t(columns) * sizeof(float));
      std::memcpy(entry.colsum_w.data(),
                  payload + bytes + size_t(columns) * sizeof(float),
                  size_t(columns) * sizeof(int32_t));
      const uint32_t handle =
        registerFromArena(session, entry, K, columns, HtpProfile::nowUs());
      if (handle == kNoHandle)
        throw std::runtime_error("prepared FC DSP registration failed: " +
                                 path);
      handles.handles.push_back(handle);
      handles.cols.push_back(columns);
      handle_cache_.emplace(static_cast<char *>(data) + slice.column_start,
                            handle);
    }
    fc_cache_.emplace(data, std::move(handles));
    prepared_fc_hashes_.emplace(data, source_hash);
    if (HtpProfile::global().level())
      std::fprintf(
        stderr,
        "[HTP] prepared FC offset=%llu K=%u N=%u slices=%zu "
        "load_us=%llu\n",
        static_cast<unsigned long long>(source_offset), K, N,
        prepared.slices.size(),
        static_cast<unsigned long long>(HtpProfile::nowUs() - begin));
    return true;
  }

  uint64_t
  begin_qs2_host_prefetch(const std::vector<ExpertFileDesc> &descriptors,
                          const void *owner, unsigned int count) override {
    if (!count || count > 160 || !owner || !compactExperts() ||
        preparedQs2Dir() || !weightDirectIO())
      return 0;
    std::lock_guard<std::mutex> scope_lock(host_scope_mutex_);
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      if (host_expert_bank_ || !prefetch_jobs_.empty() ||
          !prefetch_batches_.empty() || !prefetch_retired_.empty())
        return 0;
    }
    std::vector<ExpertFileDesc> selected;
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      if (!expert_slots_wanted_ || expert_slots_placed_ != expert_slots_wanted_)
        return 0;
      // A complete model catalog is supplied in traversal order. The current
      // residency, not a hardcoded layer index, selects the first future
      // misses.
      for (const auto &d : descriptors) {
        if (d.weight_bits != 2)
          return 0;
        if (!experts_.count(d.key_gu) && selected.size() < count)
          selected.push_back(d);
      }
    }
    if (selected.empty())
      return 0;
    uint64_t token = ++host_expert_epoch_;
    if (!token)
      token = ++host_expert_epoch_;
    auto bank = std::make_shared<HostExpertBank>(selected, token, owner);
    startPrefetchReaders();
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      host_expert_wait_us_ = 0;
      host_expert_bank_ = bank;
    }
    prefetch_cv_.notify_all();
    const auto stats = bank->stats();
    std::fprintf(
      stderr,
      "[HTP-HOST-STAGING-ACTIVE] token=%llu count=%zu payload_bytes=%zu\n",
      (unsigned long long)token, stats.count, stats.bytes);
    return token;
  }

  void end_qs2_host_prefetch(uint64_t token) noexcept override {
    std::lock_guard<std::mutex> scope_lock(host_scope_mutex_);
    std::shared_ptr<HostExpertBank> bank;
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      if (!host_expert_bank_ || host_expert_bank_->token() != token)
        return;
      bank = std::move(host_expert_bank_);
    }
    // No queue or handle lock is held while running readers/copies finish.
    bank->cancelAndWait();
    const auto stats = bank->stats();
    std::fprintf(
      stderr,
      "[HTP-HOST-STAGING-DONE] token=%llu count=%zu read=%zu consumed=%zu\n",
      (unsigned long long)token, stats.count, stats.read, stats.taken);
    if (hostExpertProfile()) {
      std::fprintf(
        stderr,
        "[HTP-HOST-STAGING] count=%zu payload_bytes=%zu read=%zu "
        "consumed=%zu cancelled=%zu failed=%zu transition_wait_us=%llu\n",
        stats.count, stats.bytes, stats.read, stats.taken, stats.cancelled,
        stats.failed, (unsigned long long)host_expert_wait_us_.load());
    }
  }

  /**
   * @brief [doc 52] One expert's two WH weights, read from the model file
   *        straight into an arena slot and registered from there.
   *
   * The slot is the unit the LRU above this trades in: gate_up then down,
   * each 4 KB aligned, so a release hands back exactly what the next miss
   * needs and the pool never fragments. pread into the arena rather than
   * activate()'s mmap + memcpy: the arena's CPU mapping is uncached, so the
   * copy the kernel does into it IS the write to DDR the DSP will read,
   * and there is no second copy and no page fault storm to pay. The scales
   * and column sums follow the nibbles in the file (QS4CX_WH_Tensor's
   * layout, written by nntr_quantize_stream), so nothing is computed here.
   *
   * Slot reuse means the DSP reads a slot, the host overwrites it, the DSP
   * reads it again -- a sequence the arena tests never covered.
   * HmxArenaSlotReuse.Uncached (unittest_hvx_mm_u8i4) is the gate for it.
   */
  bool register_qs4cx_wh_expert_file(const ExpertFileDesc &d,
                                     bool at_load) override {
    if (!at_load)
      waitHostExpertReads();
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    std::lock_guard<std::mutex> lock(handle_mutex_);
    if (experts_.count(d.key_gu) != 0)
      return true;
    const uint64_t t0 = HtpProfile::nowUs();
    StagedExpert st = stageExpert(session, d);
    st.rc = readExpert(st, /*use_pool=*/true, /*demand=*/!at_load);
    const uint64_t t_read = HtpProfile::nowUs();
    registerStaged(session, st); // throws, and frees the slot, on failure
    const uint64_t t_rpc = HtpProfile::nowUs();

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0) {
      if (at_load) // the file read stands where the convert used to
        profile.addRegister(t_rpc - t0, t_read - t0, t_rpc - t_read, true);
      else
        profile.addExpertLoad(t_read - t0, t_rpc - t_read, 1);
    }
    return true;
  }

  /** One call's misses reserve disjoint slots before reading. QS2 demand
   *  reads may run in parallel; registration remains one serial round trip. */
  bool register_qs4cx_wh_expert_files(
    const std::vector<ExpertFileDesc> &ds) override {
    waitHostExpertReads();
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    std::lock_guard<std::mutex> lock(handle_mutex_);
    const uint64_t t0 = HtpProfile::nowUs();
    std::vector<StagedExpert> sts;
    try {
      sts.reserve(ds.size());
      for (const ExpertFileDesc &d : ds) {
        if (experts_.count(d.key_gu) != 0)
          continue;
        sts.push_back(stageExpert(session, d));
      }
      static const size_t readers = [] {
        const char *v = std::getenv("NNTR_MOE_MISS_READERS");
        return v ? static_cast<size_t>(std::max(1, std::min(4, std::atoi(v))))
                 : 1u;
      }();
      const bool qs2 = !sts.empty() && compactExperts() && !preparedQs2Dir() &&
                       std::all_of(sts.begin(), sts.end(), [](const auto &st) {
                         return st.d.weight_bits == 2u;
                       });
      const size_t lanes =
        qs2 && readers > 1
          ? std::min<size_t>(
              readers,
              std::min<size_t>(2u * sts.size(),
                               ThreadManager::Global().getComputeThreadCount()))
          : 1u;
      if (qs2) {
        ++qs2_demand_batches_;
        qs2_max_misses_ = std::max<uint64_t>(qs2_max_misses_, sts.size());
      }
      if (lanes > 1) {
        ++qs2_parallel_batches_;
        const size_t jobs = 2u * sts.size();
        std::vector<int> errors(jobs);
        std::vector<std::exception_ptr> exceptions(jobs);
        std::atomic<size_t> next{0};
        ThreadManager::Global().parallel_for(0, lanes, [&](size_t) {
          for (size_t j = next.fetch_add(1); j < jobs; j = next.fetch_add(1)) {
            try {
              // No nested ThreadManager dispatch. Each task owns one weight.
              errors[j] = readExpertPart(sts[j / 2u], j % 2u != 0, false, true);
            } catch (...) {
              exceptions[j] = std::current_exception();
            }
          }
        });
        // Join before propagating exceptions or returning any slot to the pool.
        for (const auto &error : exceptions)
          if (error)
            std::rethrow_exception(error);
        for (size_t i = 0; i < sts.size(); ++i) {
          sts[i].rc = errors[2u * i] ? errors[2u * i] : errors[2u * i + 1u];
          sts[i].host.reset();
        }
      } else {
        for (auto &st : sts)
          st.rc = readExpert(st, /*use_pool=*/true, /*demand=*/true);
      }
    } catch (...) {
      for (const StagedExpert &st : sts)
        free_expert_slots_.push_back(st.slot);
      throw;
    }
    const uint64_t t_read = HtpProfile::nowUs();
    registerStagedBatch(session, sts, true); // decode may commit at MoE
    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0 && !sts.empty())
      profile.addExpertLoad(t_read - t0, HtpProfile::nowUs() - t_read,
                            sts.size());
    return true;
  }

  /**
   * @brief [doc 52 sections 10.10, 10.20] Queues one batch of experts --
   *        one layer's -- to be read into fresh slots by the prefetch
   *        readers, and returns at once. Batches are read and handed back
   *        in the order they were queued, so a caller can keep several
   *        layers in flight and collect the oldest with
   *        prefetch_qs4cx_wh_experts_end.
   *
   * What overlaps is the file read only. The register calls go through the
   * same FastRPC session as the layer calls, and the DSP's weight table is
   * not written while a kernel reads it, so they wait for _end, which the
   * caller makes between calls. The readers are this backend's own
   * threads, not ThreadManager's: they have to keep reading while the
   * caller blocks in FastRPC and while ThreadManager's workers run the ARM
   * side between calls. They stay off the caller's core (section 10.19:
   * sharing it slowed each layer call by up to 7 ms).
   */
  bool prefetch_qs4cx_wh_experts_begin(
    const std::vector<ExpertFileDesc> &ds) override {
    return beginExpertPrefetch(ds, false);
  }

  bool prefetch_qs4cx_wh_experts_decode(
    const std::vector<ExpertFileDesc> &ds) override {
    if (ds.size() > 64)
      throw std::invalid_argument(
        "Decode prefetch supports at most 64 experts");
    return beginExpertPrefetch(ds, true);
  }

  bool beginExpertPrefetch(const std::vector<ExpertFileDesc> &ds, bool decode) {
    waitHostExpertReads();
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    auto batch = std::make_unique<PrefetchBatch>();
    batch->experts.reserve(ds.size());
    batch->decode = decode;
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      // Make failure cleanup allocation-free before reserving any slots.
      free_expert_slots_.reserve(free_expert_slots_.size() + ds.size());
      try {
        for (const ExpertFileDesc &d : ds) {
          if (experts_.count(d.key_gu) == 0)
            batch->experts.push_back(stageExpert(session, d));
        }
      } catch (...) {
        for (const StagedExpert &st : batch->experts)
          free_expert_slots_.push_back(st.slot);
        throw;
      }
    }
    if (batch->experts.empty())
      return false;
    const size_t count = batch->experts.size();
    try {
      startPrefetchReaders();
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      htpPublishExpertBatch(batch, prefetch_batches_, prefetch_jobs_);
    } catch (...) {
      // No jobs from this batch are visible. Return slots before rethrowing;
      // the local batch then releases every host lease during unwinding.
      std::lock_guard<std::mutex> lock(handle_mutex_);
      for (const auto &st : batch->experts)
        free_expert_slots_.push_back(st.slot);
      throw;
    }
    if (decode && count < prefetch_readers_.size()) {
      for (size_t i = 0; i < count; ++i)
        prefetch_cv_.notify_one();
    } else {
      prefetch_cv_.notify_all();
    }
    return true;
  }

  /** @brief Waits for the oldest queued batch, registers it, and returns
   *  its keys. Empty when nothing is queued. */
  std::vector<const void *> prefetch_qs4cx_wh_experts_end() override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint64_t t0 = HtpProfile::nowUs();
    std::unique_ptr<PrefetchBatch> batch;
    size_t done_on_entry = 0;
    uint64_t ready_mask = 0;
    {
      std::unique_lock<std::mutex> lock(prefetch_mutex_);
      if (prefetch_batches_.empty())
        return {};
      PrefetchBatch *b = prefetch_batches_.front().get();
      done_on_entry = b->done;
      ready_mask = b->done_mask;
      prefetch_done_cv_.wait(lock,
                             [b] { return b->done == b->experts.size(); });
      batch = std::move(prefetch_batches_.front());
      prefetch_batches_.pop_front();
    }
    const uint64_t t_wait = HtpProfile::nowUs();
    std::lock_guard<std::mutex> lock(handle_mutex_);
    std::exception_ptr first;
    try {
      registerStagedBatch(session, batch->experts);
    } catch (...) {
      first = std::current_exception();
    }
    std::vector<const void *> keys;
    for (const StagedExpert &st : batch->experts)
      if (experts_.count(st.d.key_gu) != 0)
        keys.push_back(st.d.key_gu);
    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0) // the read time is only what the calls left
      profile.addPrefetch(batch->experts.size(), done_on_entry, t_wait - t0,
                          HtpProfile::nowUs() - t_wait, prefetch_caller_cpus_,
                          prefetch_reader_cpus_.load());
    static const bool batch_trace =
      std::getenv("NNTR_MOE_PREFETCH_BATCH_TRACE") != nullptr;
    if (batch_trace)
      std::fprintf(stderr,
                   "[MOE-PREFETCH-BATCH] n=%zu ready=%zu ready_mask=%llu "
                   "wait_us=%llu register_us=%llu\n",
                   batch->experts.size(), done_on_entry,
                   (unsigned long long)ready_mask,
                   (unsigned long long)(t_wait - t0),
                   (unsigned long long)(HtpProfile::nowUs() - t_wait));
    if (first)
      std::rethrow_exception(first);
    return keys;
  }

  ExpertPrefetchCompletion prefetch_qs4cx_wh_experts_select(
    const std::vector<const void *> &needed) override {
    const uint64_t start = HtpProfile::nowUs();
    ExpertPrefetchCompletion result;
    std::vector<StagedExpert> selected;
    std::vector<ExpertSlot> discarded;
    uint64_t ready_mask = 0, useful_mask = 0, cancelled_mask = 0;
    static const bool keep_ready = [] {
      const char *v = std::getenv("NNTR_MOE_DECODE_PREFETCH_KEEP_READY");
      return !v || std::atoi(v) != 0;
    }();
    size_t count = 0, ready = 0;
    {
      std::unique_lock<std::mutex> lock(prefetch_mutex_);
      if (prefetch_batches_.empty())
        return result;
      PrefetchBatch *batch = prefetch_batches_.front().get();
      if (!batch->decode || batch->experts.size() > 64)
        throw std::logic_error("Selective collect requires a decode batch");
      count = batch->experts.size();
      ready = batch->done;
      ready_mask = batch->done_mask;
      selected.reserve(count);
      discarded.reserve(count);
      for (size_t i = 0; i < count; ++i)
        if (std::find(needed.begin(), needed.end(),
                      batch->experts[i].d.key_gu) != needed.end())
          useful_mask |= uint64_t(1) << i;
      // Cancel unstarted wrong guesses. Running readers keep their slots until
      // they publish completion; no live buffer is recycled or overwritten.
      for (auto it = prefetch_jobs_.begin(); it != prefetch_jobs_.end();) {
        const uint64_t bit = it->first == batch ? uint64_t(1) << it->second : 0;
        if (bit && !(useful_mask & bit)) {
          batch->done_mask |= bit;
          ++batch->done;
          ++result.cancelled;
          cancelled_mask |= bit;
          batch->experts[it->second].host.reset();
          it = prefetch_jobs_.erase(it);
        } else {
          ++it;
        }
      }
      prefetch_done_cv_.wait(lock, [batch, useful_mask] {
        return (batch->done_mask & useful_mask) == useful_mask;
      });
      for (size_t i = 0; i < count; ++i) {
        const uint64_t bit = uint64_t(1) << i;
        // Finished guesses can be reused by future tokens at no extra wait.
        // Cancelled or failed guesses have no valid payload to register.
        if ((useful_mask & bit) ||
            (keep_ready && (batch->done_mask & bit) &&
             !(cancelled_mask & bit) && batch->experts[i].rc == 0))
          selected.push_back(std::move(batch->experts[i]));
        else if (batch->done_mask & bit)
          discarded.push_back(batch->experts[i].slot);
        else {
          batch->recycle_mask |= bit;
          ++result.pending;
        }
      }
      if (result.pending)
        prefetch_retired_.push_back(std::move(prefetch_batches_.front()));
      prefetch_batches_.pop_front();
    }
    const uint64_t read_done = HtpProfile::nowUs();
    const auto session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      free_expert_slots_.insert(free_expert_slots_.end(), discarded.begin(),
                                discarded.end());
      registerStagedBatch(session, selected, true);
      for (const auto &st : selected)
        result.keys.push_back(st.d.key_gu);
    }
    const uint64_t registered = HtpProfile::nowUs();
    auto &profile = HtpProfile::global();
    if (profile.level())
      profile.addPrefetch(count, ready, read_done - start,
                          registered - read_done, prefetch_caller_cpus_,
                          prefetch_reader_cpus_.load());
    if (std::getenv("NNTR_MOE_PREFETCH_BATCH_TRACE"))
      std::fprintf(
        stderr,
        "[MOE-PREFETCH-BATCH] n=%zu ready=%zu ready_mask=%llu wait_us=%llu "
        "register_us=%llu selected=%zu cancelled=%zu pending=%zu\n",
        count, ready, (unsigned long long)ready_mask,
        (unsigned long long)(read_done - start),
        (unsigned long long)(registered - read_done), result.keys.size(),
        result.cancelled, result.pending);
    return result;
  }

  size_t prefetch_qs4cx_wh_experts_reap() override {
    std::vector<ExpertSlot> freed;
    {
      std::lock_guard<std::mutex> lock(prefetch_mutex_);
      for (auto it = prefetch_retired_.begin();
           it != prefetch_retired_.end();) {
        auto &batch = **it;
        const uint64_t done = batch.done_mask & batch.recycle_mask;
        for (size_t i = 0; i < batch.experts.size(); ++i)
          if (done & (uint64_t(1) << i))
            freed.push_back(batch.experts[i].slot);
        batch.recycle_mask &= ~done;
        if (!batch.recycle_mask)
          it = prefetch_retired_.erase(it);
        else
          ++it;
      }
    }
    std::lock_guard<std::mutex> lock(handle_mutex_);
    free_expert_slots_.insert(free_expert_slots_.end(), freed.begin(),
                              freed.end());
    return freed.size();
  }

  void reserve_qs4cx_wh_expert_slots(size_t n) override {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    expert_slots_wanted_ = n;
  }

  /**
   * @brief Retires one expert without a round trip (doc 52 section 10.12).
   *
   * The expert stops being a hit at once -- its keys leave handle_cache_ --
   * but its two DSP handles stay registered and travel with the slot into
   * the free list. The next expert read into that slot releases them in
   * the same weight_swap_u8i4_arena call that registers it. That is safe
   * because nothing can pass a retired handle to the DSP any more, and the
   * slot's bytes are only overwritten between calls, never under one that
   * reads them. Handles held this way count against the DSP's table, but
   * only up to the number of free slots, which a steady state keeps at
   * zero: every eviction is followed by the load that wanted its slot.
   */
  bool release_qs4cx_wh_expert(const void *key_gu) override {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = experts_.find(key_gu);
    if (it == experts_.end())
      return false;
    ExpertSlot slot = it->second.slot;
    slot.h_gu = it->second.h_gu;
    slot.h_dn = it->second.h_dn;
    // A completed wrong prediction may be evicted before its first use.
    // Its old DSP handles still belong to this slot; no RPC is necessary.
    auto pending = std::find_if(
      pending_expert_updates_.begin(), pending_expert_updates_.end(),
      [key_gu](const StagedExpert &st) { return st.d.key_gu == key_gu; });
    if (pending != pending_expert_updates_.end()) {
      pending_expert_updates_.erase(pending);
      ++expert_updates_abandoned_;
    }
    handle_cache_.erase(key_gu);
    handle_cache_.erase(it->second.key_dn);
    free_expert_slots_.push_back(slot);
    experts_.erase(it);
    return true;
  }

  bool supports_gemm_qs4cx_fused_swiglu_fp32() const override { return true; }

  void gemm_qs4cx_fused_swiglu_fp32(std::vector<void *> matAdata,
                                    std::vector<float *> matAscale,
                                    float *matBdata, float *matCdata,
                                    unsigned int M, std::vector<unsigned int> N,
                                    unsigned int K) override {
    if (matAdata.size() != 2 || matAscale.size() != 2 || N.size() != 2) {
      throw std::invalid_argument(
        "gemm_qs4cx_fused_swiglu_fp32 needs exactly [gate_up, down]");
    }
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const unsigned int inter = N[0] / 2;
    const uint32_t h_gu =
      get_or_register_qs4cx(matAdata[0], matAscale[0], session, K, N[0]);
    const uint32_t h_dn =
      get_or_register_qs4cx(matAdata[1], matAscale[1], session, inter, N[1]);
    // ponytail: split-call path (invokeGateUpSwiglu + invokeLayerU8In), NOT
    // invokeFused. The one-call fused kernel (still in the tree, dormant --
    // see invokeFused's own comment) produced wrong output on this model's
    // real weights; root cause never found despite passing its own
    // synthetic-data unit test. This is the smaller-surface alternative
    // doc 43 §7 describes: gate_up+SwiGLU+requant in one call (one weight,
    // one VTCM region set), feeding the requantized u8 intermediate into
    // the ALREADY-device-verified u8in path for down instead of
    // reimplementing the down matmul. Verified stage-by-stage on device
    // (unittest_hvx_mm_u8i4's GateUpSwiglu* tests) before being wired here.
    invokeGateUpSwiglu(session, h_gu, matBdata, M, K, inter);
    invokeLayerU8InRaw(session, &h_dn, 1, htp_act_m_pad(M), matCdata, M, N[1],
                       inter);
    if (l2DiffEnabled() || l2ShadowEnabled())
      l2Diff(session, h_gu, h_dn, matBdata, matCdata, M, K, inter, N[1]);
  }

  /**
   * @brief NNTR_L2_DIFF: the one control neither L2 attempt ever applied.
   *
   * Both attempts were gated on unittest_hvx_mm_u8i4's SNR against
   * fill_deterministic weights and activations, passed at 138-141 dB, and
   * still broke the real model -- so the variable nobody has held fixed is
   * the DATA, not the kernel. This runs the unit test's exact reference
   * (mm_u8i4_layer on gate_up, SwiGLU on the host with expf, mm_u8i4_layer
   * on down) against the split-call path, on THIS model's registered weight
   * bytes and THIS forward's real activation, and reports the SNR per call.
   *
   * It bisects the search in one run:
   *   high SNR -> the kernels agree on real data too, so the fault is on the
   *               ARM side of the call (what is fed in, what is done with
   *               what comes out), not inside the DSP;
   *   low SNR  -> the kernels genuinely disagree on real data, and the
   *               synthetic distribution was hiding it -- the dB number to
   *               chase, with the offending M printed next to it.
   *
   * Debug path: plain heap buffers, three extra FastRPC round trips per
   * expert. Never on unless the env var is.
   */
  void l2Diff(remote_handle64 session, uint32_t h_gu, uint32_t h_dn,
              const float *matBdata, float *got, unsigned int M, unsigned int K,
              unsigned int inter, unsigned int N_out) {
    std::vector<float> act(static_cast<size_t>(M) * K);
    std::memcpy(act.data(), matBdata, act.size() * sizeof(float));

    std::vector<float> gu_out(static_cast<size_t>(M) * 2 * inter, 0.0f);
    int err = nntr_hvx_mm_u8i4_layer(
      session, M, K, &h_gu, 1, act.data(), static_cast<int>(act.size()),
      gu_out.data(), static_cast<int>(gu_out.size()));
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr, "[L2-DIFF] reference gate_up failed: %d\n", err);
      return;
    }

    // [A1] swiglu_det, the same specification Lfm2MoELayer's two-dot path
    // and the DSP's hvx_swiglu_det.h both run. This used std::exp, which
    // made total_flips measure the wrong thing: a nonzero count could mean
    // either that the fused kernel disagreed with the host path the model
    // actually uses, or merely that both disagreed with libm. Against this
    // reference, total_flips == 0 is exactly the property the fused path
    // needs, and anything else is a real divergence.
    std::vector<float> mid(static_cast<size_t>(M) * inter);
    for (unsigned int m = 0; m < M; ++m) {
      for (unsigned int j = 0; j < inter; ++j) {
        const float g = gu_out[static_cast<size_t>(m) * 2 * inter + j];
        const float u = gu_out[static_cast<size_t>(m) * 2 * inter + inter + j];
        mid[static_cast<size_t>(m) * inter + j] = swiglu_det_one(g, u);
      }
    }

    std::vector<float> ref(static_cast<size_t>(M) * N_out, 0.0f);
    err = nntr_hvx_mm_u8i4_layer(session, M, inter, &h_dn, 1, mid.data(),
                                 static_cast<int>(mid.size()), ref.data(),
                                 static_cast<int>(ref.size()));
    if (err != AEE_SUCCESS) {
      std::fprintf(stderr, "[L2-DIFF] reference down failed: %d\n", err);
      return;
    }

    double sig = 0.0, noise = 0.0, max_abs_err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
      const double r = ref[i], d = static_cast<double>(got[i]) - r;
      sig += r * r;
      noise += d * d;
      if (std::fabs(d) > max_abs_err)
        max_abs_err = std::fabs(d);
    }
    const double snr = (noise == 0.0) ? 999.0 : 10.0 * std::log10(sig / noise);

    // Outlier-row hypothesis: hvx_quant_rows_u8_params sets ONE scale per
    // row from that row's own min/max (the same scan the NaN bug poisoned
    // -- doc 43 section 5). A real trained model's SwiGLU intermediate can
    // have an outlier value fill_deterministic's bounded uniform fill never
    // produces; if one lane dominates a row's range, the rest of that row's
    // 255 levels spread thinner and the row's requant noise rises without
    // ever going non-finite. `mid` is exactly that intermediate (pre-
    // quantization, host f32), already computed above for the reference --
    // this reuses it rather than adding a DSP round trip.
    //
    // Printed for EVERY call, not just suspect ones: the first round only
    // ever showed the bad calls' own spans (0.05-0.24), with nothing to
    // compare them against, so the hypothesis was untestable -- a narrow
    // span could mean "outlier-free" (hypothesis holds) or "this model's
    // rows are always this narrow" (hypothesis is irrelevant). This line
    // is the missing baseline.
    float call_max_span = 0.0f;
    size_t call_max_span_row = 0;
    for (unsigned int m = 0; m < M; ++m) {
      float row_min = mid[static_cast<size_t>(m) * inter];
      float row_max = row_min;
      for (unsigned int j = 0; j < inter; ++j) {
        const float v = mid[static_cast<size_t>(m) * inter + j];
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }
      if (row_max - row_min > call_max_span) {
        call_max_span = row_max - row_min;
        call_max_span_row = m;
      }
    }

    if (snr < 100.0) {
      size_t worst_row = 0;
      double worst_row_err = -1.0;
      for (unsigned int m = 0; m < M; ++m) {
        double row_err = 0.0;
        for (unsigned int n = 0; n < N_out; ++n) {
          const double d =
            static_cast<double>(got[static_cast<size_t>(m) * N_out + n]) -
            ref[static_cast<size_t>(m) * N_out + n];
          row_err += d * d;
        }
        if (row_err > worst_row_err) {
          worst_row_err = row_err;
          worst_row = m;
        }
      }
      float row_min = mid[worst_row * inter], row_max = row_min;
      for (unsigned int j = 0; j < inter; ++j) {
        const float v = mid[static_cast<size_t>(worst_row) * inter + j];
        row_min = std::min(row_min, v);
        row_max = std::max(row_max, v);
      }

      // Bin-flip test, take 1 (dev_scale held fixed): is this a rounding-
      // boundary sensitivity given a SHARED scale? hvx_quant_rows_u8_params/
      // hvx_quant_pack_u8_ah are the SAME function on both paths -- verified
      // by inspection, not assumed -- so a difference cannot come from the
      // quantizer having two implementations. What CAN differ is the float
      // value handed to it: the reference quantizes the host's exact
      // expf(); the device quantized its own hvx_exp_sf/hvx_recip_qf32
      // approximation (~1e-6 relative error, well within spec -- doc 43's
      // L2 fix candidate). act_ah_buf_/act_scale_scratch_/act_zp_scratch_
      // still hold the device's own gate_up_swiglu output for this exact
      // forward's worst row -- invokeLayerU8InRaw only reads them, never
      // clears them -- so this first pass re-quantizes `mid` with the
      // DEVICE's OWN scale/zp and counts per-element disagreements.
      //
      // Take 2 (below, scale_diff/total_flips): the first round's result
      // (0 flips on the two WORST calls, 1 flip on the three milder ones)
      // means take 1 tests the wrong thing when it comes back clean --
      // hvx_quant_rows_u8_params derives scale/zp from THIS row's own
      // min/max, so if the approximation nudges the row's extreme value
      // even slightly, the REFERENCE path (which quantizes `mid` with a
      // scale/zp it computes fresh from `mid`'s own min/max, independent of
      // the device's) uses a DIFFERENT scale than the device did -- not a
      // sparse per-element flip but a systematic per-row bias that shifts
      // every one of the row's `inter` dequantized values in the same
      // direction, which a down matmul's summation does not cancel out.
      // Reimplements hvx_quant_rows_u8_params' exact formula (source read,
      // not guessed) on `mid` alone to get that independent scale/zp, then
      // diffs against the device's actual bytes the same way take 1 did.
      constexpr uint32_t kTileRow = 64, kTileInner = 32, kActTileBytes = 2048;
      const uint32_t n_ktiles = inter / kTileInner;
      const uint32_t rb = static_cast<uint32_t>(worst_row) / kTileRow;
      const uint32_t r = static_cast<uint32_t>(worst_row) % kTileRow;
      const float dev_scale = act_scale_scratch_[worst_row];
      const int32_t dev_zp = act_zp_scratch_[worst_row];
      const uint8_t *ah = act_ah_buf_->data();

      const float rmin = std::min(row_min, 0.0f);
      const float rmax = std::max(row_max, 0.0f);
      const float host_scale = (rmax > rmin) ? (rmax - rmin) / 255.0f : 1.0f;
      const int32_t host_zp =
        (rmax > rmin)
          ? std::max(0, std::min(255, static_cast<int32_t>(
                                        std::nearbyint(-rmin / host_scale))))
          : 0;

      // Flip count over the WHOLE block, not just worst_row. The previous
      // round counted only the worst row and its result ("1 flip") was
      // reported as if it described the call -- it did not. This is the
      // number that can honestly be held against row_sq_err and the SNR.
      uint32_t block_flips = 0;
      for (uint32_t m = 0; m < M; ++m) {
        const float s_m = act_scale_scratch_[m];
        const int32_t z_m = act_zp_scratch_[m];
        if (s_m == 0.0f)
          continue;
        const uint32_t rb_m = m / kTileRow, r_m = m % kTileRow;
        for (uint32_t j = 0; j < inter; ++j) {
          const uint32_t kt = j / kTileInner, c = j % kTileInner;
          const size_t idx =
            (static_cast<size_t>(rb_m) * n_ktiles + kt) * kActTileBytes +
            r_m * kTileInner + c;
          int32_t hb = static_cast<int32_t>(std::nearbyint(
                         mid[static_cast<size_t>(m) * inter + j] / s_m)) +
                       z_m;
          hb = std::max(0, std::min(255, hb));
          if (static_cast<int32_t>(ah[idx]) != hb)
            ++block_flips;
        }
      }

      uint32_t flips = 0, total_flips = 0;
      int32_t max_flip = 0, max_total_flip = 0;
      for (uint32_t j = 0; j < inter; ++j) {
        const uint32_t kt = j / kTileInner, c = j % kTileInner;
        const size_t idx =
          (static_cast<size_t>(rb) * n_ktiles + kt) * kActTileBytes +
          r * kTileInner + c;
        const int32_t dev_byte = ah[idx];
        const float v = mid[static_cast<size_t>(worst_row) * inter + j];

        int32_t byte_dev_scale =
          dev_scale != 0.0f
            ? static_cast<int32_t>(std::nearbyint(v / dev_scale)) + dev_zp
            : dev_zp;
        byte_dev_scale = std::max(0, std::min(255, byte_dev_scale));
        const int32_t d1 = dev_byte - byte_dev_scale;
        if (d1 != 0) {
          ++flips;
          if (std::abs(d1) > std::abs(max_flip))
            max_flip = d1;
        }

        int32_t byte_host_scale =
          static_cast<int32_t>(std::nearbyint(v / host_scale)) + host_zp;
        byte_host_scale = std::max(0, std::min(255, byte_host_scale));
        const int32_t d2 = dev_byte - byte_host_scale;
        if (d2 != 0) {
          ++total_flips;
          if (std::abs(d2) > std::abs(max_total_flip))
            max_total_flip = d2;
        }
      }
      const double scale_diff_pct =
        host_scale != 0.0f
          ? 100.0 * (static_cast<double>(dev_scale) - host_scale) / host_scale
          : 0.0;

      std::fprintf(
        stderr,
        "[L2-DIFF] M=%-4u K=%u inter=%u N=%u  snr=%8.2f dB  "
        "max_abs_err=%g  worst_row=%zu row_sq_err=%g "
        "mid_range=[%.4f, %.4f] mid_span=%.4f  call_max_span=%.4f@row%zu  "
        "bin_flips=%u/%u max_flip=%d  dev_scale=%.9g host_scale=%.9g "
        "scale_diff=%.6f%%  total_flips=%u/%u max_total_flip=%d  "
        "block_flips=%u/%u\n",
        M, K, inter, N_out, snr, max_abs_err, worst_row, worst_row_err, row_min,
        row_max, row_max - row_min, call_max_span, call_max_span_row, flips,
        inter, max_flip, dev_scale, host_scale, scale_diff_pct, total_flips,
        inter, max_total_flip, block_flips, static_cast<uint32_t>(M) * inter);
      if (l2ShadowEnabled())
        std::memcpy(got, ref.data(), ref.size() * sizeof(float));
      return;
    }
    std::fprintf(stderr,
                 "[L2-DIFF] M=%-4u K=%u inter=%u N=%u  snr=%8.2f dB  "
                 "max_abs_err=%g  call_max_span=%.4f@row%zu\n",
                 M, K, inter, N_out, snr, max_abs_err, call_max_span,
                 call_max_span_row);
    if (l2ShadowEnabled())
      std::memcpy(got, ref.data(), ref.size() * sizeof(float));
  }

private:
  /** Direct prefill I/O for tensors already allocated in rpcmem. Bit 0:
   * MoE, bit 1: FC, bit 2: convolution. Decode keeps its small queue buffers.
   * The caller's host tensor remains fully valid after the synchronous RPC.
   * Set NNTR_HTP_PREFILL_SHARED_IO=0 to compare with the staging path. */
  static bool sharedPrefillIo(unsigned bit) {
    static const unsigned mask = [] {
      const char *e = std::getenv("NNTR_HTP_PREFILL_SHARED_IO");
      return e ? static_cast<unsigned>(std::strtoul(e, nullptr, 0)) : 7u;
    }();
    return (mask & bit) != 0;
  }

  /** @brief Grows @a buf to at least @a bytes, reusing it otherwise.
   *
   *  One buffer for activation, one for output, reused across every call
   *  instead of one rpcmem alloc/free per matmul -- the buffers this
   *  replaces (the tensor pool's plain heap pointers, passed straight
   *  through) are pinned and mapped by the FastRPC driver on EVERY call
   *  (htp_rpcmem.h's own doc comment; measured at ~155 MB/s vs ION's
   *  44.6 GB/s, docs/htp_attention/34_fc_measured.md section4 item F). The
   *  memcpy this adds is the trade: cheap relative to a per-call pin+map,
   *  but that trade is exactly what NNTR_HTP_PROFILE's transport column is
   *  for confirming on device -- do not assume the win without it. */
  static void ensureCapacity(std::unique_ptr<HtpRpcBuffer> &buf, size_t bytes) {
    if (!buf || buf->size() < bytes) {
      buf = std::make_unique<HtpRpcBuffer>(bytes);
    }
  }

  /** @brief ION scratch by size class, one buffer per power of two from
   *  64 KiB up, so a call stages through a buffer near its own size.
   *
   * One pair grown to the largest shape (ensureCapacity) made every call
   * pay for that shape: the driver's cache maintenance on a cached ION
   * buffer covers the whole dma-buf, not the bytes passed, so the MoE
   * decode call's 8 KB rode a 3.6 MB pair at 553 us of transport, a
   * 10.9 MB out buffer at 814 and a 12.7 MB pair at 1633 (doc 50 section
   * 3.6: 36-59 us/MB, cache-flush speed), and prefill's calls paid the
   * same. With classes, decode stays on the 64 KiB pair. */
  struct StagingPool {
    std::map<size_t, std::unique_ptr<HtpRpcBuffer>> by_class;
    /** What stageInput last copied into a class's buffer, cleared by any
     *  other stage() of it (doc 57 section 9.31). */
    std::map<size_t, std::pair<const void *, size_t>> holds;
    /** The host tensor a class's buffer was last copied out to, whole
     *  (copyOutOf), cleared by any other stage() of it. */
    std::map<size_t, std::pair<const void *, size_t>> copied_out;
  };
  static size_t stageClass(size_t bytes) {
    size_t cls = size_t(64) << 10;
    while (cls < bytes)
      cls <<= 1;
    return cls;
  }
  HtpRpcBuffer &stage(StagingPool &pool, size_t bytes,
                      uint32_t flags = HTP_RPC_FLAGS_DEFAULT) {
    if (&pool == &act_pool_)
      moe_input_reuse_.invalidate();
    const size_t cls = stageClass(bytes);
    pool.holds.erase(cls);
    pool.copied_out.erase(cls);
    auto &slot = pool.by_class[cls];
    if (!slot)
      slot = std::make_unique<HtpRpcBuffer>(cls, flags);
    return *slot;
  }
  /** @brief Whether @a b still holds the copy of @a a: 32 spread 64-byte
   *  windows and the tail compared. ponytail: a rewrite that keeps every
   *  sampled window is missed; a full compare costs about what the copy
   *  does. */
  static bool sampledSame(const void *a_, const void *b_, size_t bytes) {
    const uint8_t *a = static_cast<const uint8_t *>(a_);
    const uint8_t *b = static_cast<const uint8_t *>(b_);
    bool same =
      bytes < 64 || std::memcmp(a + bytes - 64, b + bytes - 64, 64) == 0;
    for (size_t i = 0; same && i < 32 && bytes >= 64; ++i) {
      const size_t off = (bytes - 64) / 32 * i;
      same = std::memcmp(a + off, b + off, 64) == 0;
    }
    return same;
  }
  /** @brief stage() plus the copy of @a bytes from @a src, skipped when the
   *  buffer still holds exactly that copy: the router, dense and MoE calls
   *  of a Gemma-4 layer each stage the same 11.5 MB post-attention rows.
   *  Any other stage() of the buffer forgets the copy, and before skipping
   *  the copy is compared (sampledSame) with @a src, so a host tensor
   *  rewritten (or another one placed at the same address) is copied again.
   *
   *  With @a from, a buffer of that pool that @a src was copied out of
   *  whole (copyOutOf) and still matches is swapped in instead of copied:
   *  the dense output is the MoE call's x2 and the MoE output the next
   *  qkv call's input (doc 57 section 9.34). Stage this before the call's
   *  own output buffer, which may be the one swapped in. */
  HtpRpcBuffer &stageInput(StagingPool &pool, const void *src, size_t bytes,
                           StagingPool *from = nullptr,
                           int line = __builtin_LINE()) {
    if (&pool == &act_pool_ || from == &act_pool_)
      moe_input_reuse_.invalidate();
    const size_t cls = stageClass(bytes);
    auto &slot = pool.by_class[cls];
    if (!slot)
      slot = std::make_unique<HtpRpcBuffer>(cls);
    auto it = pool.holds.find(cls);
    if (it != pool.holds.end() && it->second.first == src &&
        it->second.second == bytes && sampledSame(src, slot->data(), bytes))
      return *slot;
    if (from) {
      auto c = from->copied_out.find(cls);
      auto &fslot = from->by_class[cls];
      if (c != from->copied_out.end() && c->second.first == src &&
          c->second.second == bytes && fslot &&
          sampledSame(src, fslot->data(), bytes)) {
        std::swap(slot, fslot);
        from->holds.erase(cls);
        from->copied_out.erase(cls);
        pool.copied_out.erase(cls);
        pool.holds[cls] = {src, bytes};
        return *slot;
      }
    }
    stagedMemcpy(slot->data(), src, bytes, line);
    pool.holds[cls] = {src, bytes};
    return *slot;
  }
  /** @brief The copy of a call's whole output out of @a pool's buffer for
   *  @a bytes, remembered for stageInput's @a from. */
  static void copyOutOf(StagingPool &pool, void *dst, size_t bytes,
                        int line = __builtin_LINE()) {
    const size_t cls = stageClass(bytes);
    stagedMemcpy(dst, pool.by_class[cls]->data(), bytes, line);
    pool.copied_out[cls] = {dst, bytes};
  }

  /** @brief The one FastRPC layer call both accelerated entries make.
   *
   *  Under NNTR_HTP_PROFILE >= 2 it goes through mm_u8i4_layer_timed so the
   *  DSP's own microseconds come back alongside the host wall clock; the
   *  difference between the two is the FastRPC transport. The production
   *  path (profile off) still calls the untimed entry, which is why the
   *  DSP probes cost nothing when nobody is measuring.
   *
   *  Not static any more: it now owns act_pool_/out_pool_, rpcmem-backed
   *  scratch buffers by size class reused across calls (stage above).
   *  invoke_mutex_ guards them -- required once they are shared
   *  mutable state, and consistent with the single-owner assumption
   *  test/htp/nntr_hvx_session.h already documents for the one HTP session
   *  a process opens (one VTCM arena, one HMX lock).
   */
  /** @brief Copies the kernel's output back into the caller's [M x N].
   *
   * hexkl_mm_u8i4_layer_run writes one contiguous [M x N_i] block per
   * handle, in call order (its out_off += M * h->N), not a row-major
   * [M x sum(N_i)]. One handle is therefore the whole [M x N] and copies
   * straight through; the slices of one weight (get_or_register_fc) are
   * interleaved back, row by row, into their column ranges; separate
   * weights (@a dsts, the batch call) each get their block whole. */
  static void copyOut(float *dst, const float *out_cat, unsigned int M,
                      unsigned int N, const std::vector<unsigned int> *blocks,
                      const std::vector<float *> *dsts = nullptr,
                      int line = __builtin_LINE()) {
    if (dsts) {
      size_t off = 0;
      for (size_t i = 0; i < dsts->size(); ++i) {
        const size_t block = static_cast<size_t>(M) * (*blocks)[i];
        stagedMemcpy((*dsts)[i], out_cat + off, block * sizeof(float), line);
        off += block;
      }
      return;
    }
    if (!blocks || blocks->size() < 2) {
      stagedMemcpy(dst, out_cat, static_cast<size_t>(M) * N * sizeof(float),
                   line);
      return;
    }
    size_t off = 0;
    unsigned int c0 = 0;
    for (unsigned int n : *blocks) {
      for (unsigned int r = 0; r < M; ++r) {
        stagedMemcpy(dst + static_cast<size_t>(r) * N + c0,
                     out_cat + off + static_cast<size_t>(r) * n,
                     static_cast<size_t>(n) * sizeof(float), line);
      }
      off += static_cast<size_t>(M) * n;
      c0 += n;
    }
  }

  /** @param blocks N of each handle in call order, when @a handles are the
   *  slices of one weight and matCdata is its whole [M x N], or separate
   *  weights whose [M x N_i] blocks go to @a dsts[i]; NULL when each
   *  handle's output stays a block of its own (or there is only one). */
  /** @brief Where one handle's [M x cols] output block lands: columns
   *  [c0, c0 + cols) of a [M x stride] destination. */
  struct OutSlice {
    float *dst;
    unsigned int stride, c0, cols;
  };

  /** @brief The RMSNorms folded into one layer call (mm_u8i4_layer_norm):
   *  nothing set = the plain call. */
  struct FusedNorms {
    const float *pre_gamma = nullptr;                      /**< K floats */
    const std::vector<unsigned int> *post_chunk = nullptr; /**< per handle */
    const float *post_gamma = nullptr; /**< the chunks, concatenated */
    float eps = 0.0f;
    const float *rope_cs = nullptr; /**< M rows of 2*rope_hd, or nullptr */
    unsigned int rope_hd = 0;       /**< 0: no RoPE */
    unsigned int rope_handles = 0;  /**< the handles rotated, a prefix */
    bool any() const { return pre_gamma || post_chunk || rope_hd; }
  };

  /** Queue the existing HVX FC entry, preserving its quantization and
   * fused-norm boundaries. Return false only before submitting a packet. */
  bool invokeLayerQueue(remote_handle64 session, const uint32_t *handles,
                        int num_handles, float *input, float *output,
                        unsigned M, unsigned N, unsigned K,
                        const std::vector<unsigned int> *blocks,
                        const std::vector<float *> *dsts,
                        const FusedNorms *norms,
                        const std::vector<OutSlice> *slices,
                        PendingNorm *pending) {
    const bool fused = norms && norms->any();
    const char *env =
      std::getenv(fused ? "NNTR_HTP_DSPQ_QKV" : "NNTR_HTP_DSPQ_FC");
    if (M != 1 || (env && std::atoi(env) == 0) ||
        !(applied_moe_flags_ & HTP_FC_FLAG_M1_HVX) || num_handles <= 0 ||
        num_handles > static_cast<int>(HTP_DSPQ_FC_HANDLES) ||
        (fused &&
         (!slices || slices->size() != static_cast<size_t>(num_handles))))
      return false;
    HtpProfile &profile = HtpProfile::global();
    const bool timed = profile.level() >= 2;
    htp_dspq_fc_req req = {};
    req.op = HTP_DSPQ_OP_FC;
    req.flags = timed ? HTP_DSPQ_FLAG_TIMED : 0u;
    req.K = K;
    req.N = N;
    req.count = num_handles;
    req.fused = fused;
    req.norm_kind = pending ? (pending->add ? 2u : 1u) : 0u;
    req.norm_eps = pending ? pending->eps : 0.0f;
    std::copy(handles, handles + num_handles, req.handles);
    if (fused) {
      req.eps = norms->eps;
      req.pre_len = norms->pre_gamma ? K : 0;
      req.rope_hd = norms->rope_hd;
      req.rope_handles = norms->rope_handles;
      if (norms->post_chunk) {
        if (norms->post_chunk->size() != static_cast<size_t>(num_handles))
          return false;
        for (int i = 0; i < num_handles; ++i) {
          req.chunks[i] = (*norms->post_chunk)[i];
          if (req.chunks[i] > 16384u - req.post_len)
            return false;
          req.post_len += req.chunks[i];
        }
      }
      size_t base = 0, next = 0;
      for (int i = 0; i < num_handles; ++i) {
        const OutSlice &o = (*slices)[i];
        if (o.c0 == 0) {
          base = next;
          next += o.stride;
        }
        req.offsets[i] = base + o.c0;
        req.strides[i] = o.stride;
      }
    }
    if (!htp_dspq_fc_shape_valid(&req))
      return false;
    const size_t input_bytes = htp_dspq_fc_input_bytes(&req);
    const size_t output_bytes = htp_dspq_fc_output_bytes(&req);
    if (!dspqReady(session, input_bytes, output_bytes, sizeof(req)))
      return false;
    auto &st = *dspq_;
    req.seq = ++st.seq;
    st.retained_norm = nullptr;
    float *packed = reinterpret_cast<float *>(st.act->data());
    const size_t act_count = static_cast<size_t>(req.norm_kind + 1u) * K;
    stagedMemcpy(packed, pending ? pending->inputs.data() : input,
                 act_count * 4);
    if (req.pre_len)
      stagedMemcpy(packed + act_count, norms->pre_gamma, req.pre_len * 4);
    if (req.post_len)
      stagedMemcpy(packed + act_count + req.pre_len, norms->post_gamma,
                   req.post_len * 4);
    if (req.rope_hd)
      stagedMemcpy(packed + act_count + req.pre_len + req.post_len,
                   norms->rope_cs, 2u * req.rope_hd * 4);
    uint32_t stages[HTP_DSPQ_STAGES] = {};
    const uint64_t start = HtpProfile::nowUs();
    const int err = dspqExchange(&req, sizeof(req), req.seq, input_bytes,
                                 output_bytes, timed ? stages : nullptr);
    const uint64_t elapsed = HtpProfile::nowUs() - start;
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP queued FC failed: " + std::to_string(err));
    const float *result = reinterpret_cast<const float *>(st.out->data());
    if (pending)
      publishNorm(result + N);
    if (fused) {
      for (int i = 0; i < num_handles; ++i) {
        const OutSlice &o = (*slices)[i];
        if (o.c0 == 0)
          stagedMemcpy(o.dst, result + req.offsets[i],
                       o.stride * sizeof(float));
      }
      ++queued_qkv_calls_;
    } else {
      copyOut(output, result, M, N, blocks, dsts);
      ++queued_fc_calls_;
    }
    if (profile.level())
      profile.addInvoke(M, K, N, elapsed, timed ? stages : nullptr);
    return true;
  }

  void invokeLayer(remote_handle64 session, const uint32_t *handles,
                   int num_handles, float *matBdata, float *matCdata,
                   unsigned int M, unsigned int N, unsigned int K,
                   const std::vector<unsigned int> *blocks = nullptr,
                   const std::vector<float *> *dsts = nullptr,
                   const FusedNorms *norms = nullptr,
                   const std::vector<OutSlice> *slices = nullptr,
                   bool retain_moe_input = false,
                   bool allow_strided_output = false) {
    sendMoeOptsOnce(session);
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);
    const auto shape = [&]() {
      return " (M=" + std::to_string(M) + " K=" + std::to_string(K) +
             " N=" + std::to_string(N) +
             " handles=" + std::to_string(num_handles) + ")";
    };

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    PendingNorm *pending = pendingNorm("qkv", matBdata, M, K);
    if (pending && (!norms || !norms->any() || !slices))
      throw std::runtime_error(
        "Deferred QKV norm requires the fused projection path");
    if (invokeLayerQueue(session, handles, num_handles, matBdata, matCdata, M,
                         N, K, blocks, dsts, norms, slices, pending))
      return;
    const bool shared_io = M > 1 && sharedPrefillIo(2u);
    const bool direct_in =
      shared_io && !pending &&
      HtpBackend::global().isSharedRange(
        matBdata, static_cast<size_t>(act_len) * sizeof(float));
    // Column slices can write directly into the caller's row-major tensor
    // using the existing ld entry. Keep batches and fused norms on their
    // original paths; invokeFc also checks overlap across ALL row chunks.
    std::vector<uint32_t> fc_out_off, fc_out_ld;
    bool strided_out = false;
    if (shared_io && allow_strided_output && !pending && !slices && !dsts &&
        (!norms || !norms->any()) && blocks && num_handles > 1 &&
        blocks->size() == static_cast<size_t>(num_handles) && act_len > 0 &&
        out_len > 0 &&
        HtpBackend::global().isSharedRange(
          matCdata, static_cast<size_t>(out_len) * sizeof(float))) {
      uint64_t column = 0;
      for (unsigned int width : *blocks) {
        if (width == 0 || width % 32u || column + width > N ||
            column + static_cast<uint64_t>(M - 1u) * N + width >
              static_cast<uint64_t>(out_len))
          break;
        fc_out_off.push_back(static_cast<uint32_t>(column));
        fc_out_ld.push_back(N);
        column += width;
      }
      strided_out = column == N && fc_out_off.size() == blocks->size();
    }
    const bool direct_out =
      strided_out ||
      (shared_io && !pending && !slices && !dsts &&
       (!blocks || blocks->size() == 1) &&
       HtpBackend::global().isSharedRange(
         matCdata, static_cast<size_t>(out_len) * sizeof(float)));
    if (direct_in || direct_out) {
      ++shared_io_calls_[1];
      shared_io_bytes_[1] +=
        (direct_in ? static_cast<size_t>(act_len) * sizeof(float) : 0u) +
        (direct_out ? static_cast<size_t>(out_len) * sizeof(float) : 0u);
    }
    float *act_f32;
    HtpRpcBuffer *act_stage = nullptr;
    if (pending) {
      act_f32 = reinterpret_cast<float *>(
        stage(act_pool_, pending->inputs.size() * 4).data());
      stagedMemcpy(act_f32, pending->inputs.data(), pending->inputs.size() * 4);
    } else if (direct_in) {
      act_f32 = matBdata;
    } else {
      act_stage =
        &stageInput(act_pool_, matBdata,
                    static_cast<size_t>(act_len) * sizeof(float), &out_pool_);
      act_f32 = reinterpret_cast<float *>(act_stage->data());
    }
    const size_t out_bytes =
      (static_cast<size_t>(out_len) + (pending ? 2u * K : 0u)) * sizeof(float);
    float *out_cat;
    if (direct_out) {
      out_cat = matCdata;
      // Preserve stage()'s invalidation without allocating unused scratch.
      const size_t cls = stageClass(out_bytes);
      out_pool_.holds.erase(cls);
      out_pool_.copied_out.erase(cls);
    } else {
      out_cat = reinterpret_cast<float *>(stage(out_pool_, out_bytes).data());
    }

    HtpProfile &profile = HtpProfile::global();
    if (strided_out) {
      const uint64_t t0 = HtpProfile::nowUs();
      const int err = nntr_hvx_mm_u8i4_layer_norm_ld(
        session, M, K, 0.0f, nullptr, 0, nullptr, 0, nullptr, 0, 0, 0, nullptr,
        0, handles, num_handles, fc_out_off.data(), num_handles,
        fc_out_ld.data(), num_handles, act_f32, act_len, out_cat, out_len);
      if (err != AEE_SUCCESS)
        throw std::runtime_error(
          "HTP strided FC failed: err=" + std::to_string(err) + shape());
      if (retain_moe_input && act_stage)
        moe_input_reuse_.remember(matBdata, M, K, act_stage);
      // The existing ld entry has no timed twin. As with fused norms,
      // report host wall time without inventing a DSP stage measurement.
      if (profile.level())
        profile.addInvoke(M, K, N, HtpProfile::nowUs() - t0, nullptr);
      return;
    }
    if (norms && norms->any()) {
      // The norms ride the same call; the timed twin has no norm variant,
      // so this path counts as an untimed call in the profile.
      int post_gamma_len = 0;
      if (norms->post_chunk) {
        for (unsigned int c : *norms->post_chunk)
          post_gamma_len += static_cast<int>(c);
      }
      // With slices, each lands in its weight's row-major [M x N] on the
      // DSP (out_off / out_ld), so a weight comes out as one copy; the
      // slice-row copies this replaces were 132.8 ms of 4 KB memcpys a
      // 1024-token prefill (doc 57 section 9.34). A weight's slices come
      // in column order from c0 = 0 (gemm_q4_0_batch_norm_fp32).
      std::vector<uint32_t> out_off, out_ld;
      if (slices) {
        size_t base = 0, next = 0;
        for (const OutSlice &o : *slices) {
          if (o.c0 == 0) {
            base = next;
            next += static_cast<size_t>(M) * o.stride;
          }
          out_off.push_back(static_cast<uint32_t>(base + o.c0));
          out_ld.push_back(o.stride);
        }
      }
      const uint64_t t0 = HtpProfile::nowUs();
      const int pre_len = norms->pre_gamma ? static_cast<int>(K) : 0;
      const uint32_t *chunk =
        norms->post_chunk ? norms->post_chunk->data() : nullptr;
      const int chunk_len =
        norms->post_chunk ? static_cast<int>(norms->post_chunk->size()) : 0;
      const int cs_len =
        norms->rope_hd ? static_cast<int>(M * 2 * norms->rope_hd) : 0;
      const int err =
        pending ? nntr_hvx_mm_u8i4_layer_norm_ld_input(
                    session, K, norms->eps, norms->pre_gamma, pre_len, chunk,
                    chunk_len, norms->post_gamma, post_gamma_len,
                    norms->rope_hd, norms->rope_handles, norms->rope_cs, cs_len,
                    handles, num_handles, out_off.data(), num_handles,
                    out_ld.data(), num_handles, pending->add, pending->eps,
                    act_f32, pending->inputs.size(), out_cat, out_len + 2u * K)
        : slices
          ? nntr_hvx_mm_u8i4_layer_norm_ld(
              session, M, K, norms->eps, norms->pre_gamma, pre_len, chunk,
              chunk_len, norms->post_gamma, post_gamma_len, norms->rope_hd,
              norms->rope_handles, norms->rope_cs, cs_len, handles, num_handles,
              out_off.data(), num_handles, out_ld.data(), num_handles, act_f32,
              act_len, out_cat, out_len)
          : nntr_hvx_mm_u8i4_layer_norm(
              session, M, K, norms->eps, norms->pre_gamma, pre_len, chunk,
              chunk_len, norms->post_gamma, post_gamma_len, norms->rope_hd,
              norms->rope_handles, norms->rope_cs, cs_len, handles, num_handles,
              act_f32, act_len, out_cat, out_len);
      const uint64_t elapsed = HtpProfile::nowUs() - t0;
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_norm failed: err=" +
                                 std::to_string(err) + shape());
      }
      if (pending)
        publishNorm(out_cat + out_len);
      if (slices) {
        for (size_t i = 0; i < slices->size(); ++i) {
          const OutSlice &o = (*slices)[i];
          if (o.c0 == 0)
            stagedMemcpy(o.dst, out_cat + out_off[i],
                         static_cast<size_t>(M) * o.stride * sizeof(float));
        }
      } else {
        if (!direct_out)
          copyOut(matCdata, out_cat, M, N, blocks, dsts);
      }
      if (profile.level())
        profile.addInvoke(M, K, N, elapsed, nullptr);
      return;
    }
    if (profile.level() == 0) {
      const int err =
        nntr_hvx_mm_u8i4_layer(session, M, K, handles, num_handles, act_f32,
                               act_len, out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer failed: err=" +
                                 std::to_string(err) + shape());
      }
      if (!direct_out)
        copyOut(matCdata, out_cat, M, N, blocks, dsts);
      if (retain_moe_input && act_stage)
        moe_input_reuse_.remember(matBdata, M, K, act_stage);
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_layer_timed(session, M, K, handles, num_handles,
                                           act_f32, act_len, out_cat, out_len,
                                           stage_us, HTP_N_STAGES)
            : nntr_hvx_mm_u8i4_layer(session, M, K, handles, num_handles,
                                     act_f32, act_len, out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(std::string(timed
                                             ? "nntr_hvx_mm_u8i4_layer_timed"
                                             : "nntr_hvx_mm_u8i4_layer") +
                               " failed: err=" + std::to_string(err) + shape());
    }
    if (!direct_out)
      copyOut(matCdata, out_cat, M, N, blocks, dsts);
    if (retain_moe_input && act_stage)
      moe_input_reuse_.remember(matBdata, M, K, act_stage);
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief Same call as invokeLayer, but the activation is quantized and
   *  AH-tile-packed on the ARM side first (htp_act_quant.h) and sent as u8
   *  instead of f32 -- see mm_u8i4_layer_u8in's IDL doc for what this buys.
   *  Used by both accel entries, which the M > 1 gate means only ever run
   *  at prefill shapes (accelerates_q4_0_at_m1() is false), matching this
   *  path's scope: 41_moe_ffn_e2e_and_perf_task.md's P2. */
  void invokeLayerU8In(remote_handle64 session, const uint32_t *handles,
                       int num_handles, const float *matBdata, float *matCdata,
                       unsigned int M, unsigned int N, unsigned int K) {
    const uint32_t m_pad = htp_act_m_pad(M);
    const size_t act_ah_bytes = static_cast<size_t>(m_pad) * K;
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_ah_buf_, act_ah_bytes);
    uint8_t *act_ah = act_ah_buf_->data();
    float *out_cat = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());

    if (act_scale_scratch_.size() < m_pad) {
      act_scale_scratch_.resize(m_pad);
      act_zp_scratch_.resize(m_pad);
    }
    htp_quant_pack_u8_ah(matBdata, M, K, act_ah, act_scale_scratch_.data(),
                         act_zp_scratch_.data());

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_u8in(
        session, M, K, handles, num_handles, act_ah,
        static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad), act_zp_scratch_.data(),
        static_cast<int>(m_pad), out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_u8in failed: err=" +
                                 std::to_string(err));
      }
      stagedMemcpy(matCdata, out_cat,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed
        ? nntr_hvx_mm_u8i4_layer_u8in_timed(
            session, M, K, handles, num_handles, act_ah,
            static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
            static_cast<int>(m_pad), act_zp_scratch_.data(),
            static_cast<int>(m_pad), out_cat, out_len, stage_us, HTP_N_STAGES)
        : nntr_hvx_mm_u8i4_layer_u8in(
            session, M, K, handles, num_handles, act_ah,
            static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
            static_cast<int>(m_pad), act_zp_scratch_.data(),
            static_cast<int>(m_pad), out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_u8in_timed"
                          : "nntr_hvx_mm_u8i4_layer_u8in") +
        " failed: err=" + std::to_string(err));
    }
    stagedMemcpy(matCdata, out_cat,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief The fused MoE expert FFN call: gate_up -> SwiGLU -> down in one
   *  FastRPC round trip (doc 43 §[L2]). Same scratch-buffer reuse and
   *  timed-entry split as invokeLayer; the SwiGLU stage gets its own
   *  profile bucket via addInvokeFused. */
  void invokeFused(remote_handle64 session, const uint32_t *handles,
                   float *matBdata, float *matCdata, unsigned int M,
                   unsigned int N, unsigned int K) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act_f32 = reinterpret_cast<float *>(
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float)).data());
    float *out_f32 = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_fused(
        session, M, K, handles, 2, act_f32, act_len, out_f32, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_fused failed: err=" +
                                 std::to_string(err));
      }
      stagedMemcpy(matCdata, out_f32,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_FU_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed
        ? nntr_hvx_mm_u8i4_layer_fused_timed(session, M, K, handles, 2, act_f32,
                                             act_len, out_f32, out_len,
                                             stage_us, HTP_FU_N_STAGES)
        : nntr_hvx_mm_u8i4_layer_fused(session, M, K, handles, 2, act_f32,
                                       act_len, out_f32, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_fused_timed"
                          : "nntr_hvx_mm_u8i4_layer_fused") +
        " failed: err=" + std::to_string(err));
    }
    stagedMemcpy(matCdata, out_f32,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvokeFused(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  /** @brief [#84] NNTR_HTP_DUMP=<dir>: every MoE call's input (M x K f32)
   *  and output (M x N_out f32) as <dir>/moe_<call>_in.f32 / _out.f32, in
   *  call order, plus one line per call in <dir>/manifest.txt
   *  (name entry M K inter N_out kind r=<row_count per expert>, the
   *  routing the ARM handed the call -- [#136] so the comparator can
   *  tell a top-k flip at a near-tie from a fault). Read once, off unless the
   *  variable is set: with it unset this is one static load and a branch
   *  per call. tools/htp/htp_dump_eval.py compares two such directories;
   *  the first file that differs names the call, which is where a _det is
   *  missing (plan 84 section 3.1). Never in a tok/s run: 16 KiB of file
   *  I/O per call. Callers hold invoke_mutex_, so the counter is plain. */
  static void dumpMoeCall(const char *entry, const float *act, size_t act_n,
                          const float *out, size_t out_n, unsigned int M,
                          unsigned int K, unsigned int inter,
                          unsigned int N_out, int kind,
                          const std::vector<unsigned int> &row_count) {
    static const char *dir = std::getenv("NNTR_HTP_DUMP");
    if (dir == nullptr)
      return;
    static unsigned int call = 0;
    char name[32];
    std::snprintf(name, sizeof(name), "moe_%05u", call);
    const std::string base = std::string(dir) + "/" + name;
    const auto write = [&](const std::string &path, const float *p, size_t n) {
      FILE *f = std::fopen(path.c_str(), "wb");
      if (f == nullptr || std::fwrite(p, sizeof(float), n, f) != n)
        throw std::runtime_error("NNTR_HTP_DUMP: cannot write " + path);
      std::fclose(f);
    };
    write(base + "_in.f32", act, act_n);
    write(base + "_out.f32", out, out_n);
    // Truncated by this process's first call: a reused directory would
    // otherwise list a previous run's calls, and its stale files with them.
    FILE *m = std::fopen((std::string(dir) + "/manifest.txt").c_str(),
                         call == 0 ? "w" : "a");
    if (m == nullptr)
      throw std::runtime_error("NNTR_HTP_DUMP: cannot write the manifest");
    ++call;
    std::fprintf(m, "%s %s %u %u %u %u %d r=", name, entry, M, K, inter, N_out,
                 kind);
    for (size_t e = 0; e < row_count.size(); ++e)
      std::fprintf(m, "%s%u", e ? "," : "", row_count[e]);
    std::fprintf(m, "\n");
    std::fclose(m);
  }

  /** @brief [#136] NNTR_HTP_DUMP_ALL=1 (next to NNTR_HTP_DUMP=<dir>): what
   *  the ARM side hands the DSP between and inside the per-token
   *  stretches, all of it -- every stretch's input / output as
   *  <dir>/fwd_<call>_<kinds>_{in,out}.f32 with one line per call in
   *  <dir>/forward_manifest.txt (name op resume pos K N_out kinds), the KV
   *  seed rows as kvseed_<ordinal>_{k,v}.f32 and the conv state seed as
   *  convstate_<op>.f32. Its own manifest, so the MoE goldens
   *  (manifest.txt) stay byte-equal. The device handoff of #136 turns it
   *  on for one run; the workstation replays the stretch inputs through
   *  graph_host_check's emulation and names the first stretch whose
   *  device output leaves it (plan 136 section 3.2). Never in a tok/s run. */
  static const char *dumpAllDir() {
    static const char *dir = std::getenv("NNTR_HTP_DUMP_ALL") != nullptr
                               ? std::getenv("NNTR_HTP_DUMP")
                               : nullptr;
    return dir;
  }
  static void dumpAllFile(const std::string &path, const float *p, size_t n) {
    FILE *f = std::fopen(path.c_str(), "wb");
    if (f == nullptr || std::fwrite(p, sizeof(float), n, f) != n)
      throw std::runtime_error("NNTR_HTP_DUMP_ALL: cannot write " + path);
    std::fclose(f);
  }
  void dumpForwardCall(uint32_t op, uint32_t resume, uint32_t pos,
                       const float *act, size_t K, const float *out,
                       size_t N_out) {
    const char *dir = dumpAllDir();
    if (dir == nullptr)
      return;
    static unsigned int call = 0;
    std::string kinds;
    for (uint32_t i = op; i < resume; ++i)
      kinds +=
        std::string(i == op ? "" : "-") + htp_graph_kind_name(graphOp(i)->kind);
    char name[64];
    std::snprintf(name, sizeof(name), "fwd_%05u_%s", call, kinds.c_str());
    const std::string base = std::string(dir) + "/" + name;
    dumpAllFile(base + "_in.f32", act, K);
    dumpAllFile(base + "_out.f32", out, N_out);
    FILE *m = std::fopen((std::string(dir) + "/forward_manifest.txt").c_str(),
                         call == 0 ? "w" : "a");
    if (m == nullptr)
      throw std::runtime_error(
        "NNTR_HTP_DUMP_ALL: cannot write the forward manifest");
    ++call;
    std::fprintf(m, "%s %u %u %u %zu %zu %s\n", name, op, resume, pos, K, N_out,
                 kinds.c_str());
    std::fclose(m);
  }

  /** @brief [#85] The per-token entry over one resident stretch [op,
   *  expected_resume): the same staging pools as invokeMoeLayer, and for
   *  the MoE stretch (@a moe) the same profile bucket, so the M==1 row
   *  reads the same across the two paths. The DSP must stop exactly at
   *  expected_resume (the ARM's stretch table, plan 130 section 3.2);
   *  anything else throws. */
  void invokeForward(remote_handle64 session, uint32_t op,
                     uint32_t expected_resume, uint32_t pos,
                     const std::vector<unsigned int> &row_index,
                     const std::vector<unsigned int> &row_count,
                     const std::vector<float> &row_weight, const float *act,
                     unsigned int K, float *out, unsigned int N_out, bool moe) {
    if (e2e_) {
      // [#132 Part B E3] with every kind resident the token is the one
      // stretch [0, n_ops), run by the token driver on the two sessions
      if (op != 0u || expected_resume != stretch_start_.size() ||
          act == nullptr || out == nullptr)
        throw std::runtime_error(
          "token driver: stretch [" + std::to_string(op) + ", " +
          std::to_string(expected_resume) + ") is not the whole token");
      tokenForward(pos, act, K, out, N_out);
      return;
    }
    const int act_len = static_cast<int>(K);
    const int out_len = static_cast<int>(N_out);
    const uint32_t n_limit = expected_resume - op;
    if (act == nullptr || out == nullptr || expected_resume <= op) {
      throw std::runtime_error("nntr_hvx_forward: bad stretch [" +
                               std::to_string(op) + ", " +
                               std::to_string(expected_resume) + ")");
    }

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    HtpRpcBuffer &act_stage =
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float));
    HtpRpcBuffer &out_stage =
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float));
    float *act_f32 = reinterpret_cast<float *>(act_stage.data());
    float *out_f32 = reinterpret_cast<float *>(out_stage.data());
    stagedMemcpy(act_f32, act, static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    std::vector<uint32_t> op_pcyc(n_limit, 0u);
    const bool timed = profile.level() >= 2;
    // Level 3's 5x repeat is for the stateless MoE stretch only: a
    // CONV1D_GATE op advances the DSP conv state on every call, an
    // ATTN_M1 op appends to the cache (a rewind, harmless, but not free).
    const int reps = (profile.level() >= 3 && moe) ? 5 : 1;
    uint64_t best_elapsed = UINT64_MAX;
    uint32_t resume_at = 0;
    int err = AEE_SUCCESS;
    for (int rep = 0; rep < reps && err == AEE_SUCCESS; ++rep) {
      uint32_t rep_stage[HTP_MOE_N_STAGES] = {0};
      std::vector<uint32_t> rep_pcyc(n_limit, 0u);
      uint32_t rep_resume = 0;
      const uint64_t t0 = profile.level() ? HtpProfile::nowUs() : 0;
      err = timed ? nntr_hvx_forward_debug(
                      session, op, n_limit, pos, row_index.data(),
                      static_cast<int>(row_index.size()), row_count.data(),
                      static_cast<int>(row_count.size()), row_weight.data(),
                      static_cast<int>(row_weight.size()), act_f32, act_len,
                      out_f32, out_len, &rep_resume, rep_pcyc.data(),
                      static_cast<int>(n_limit), rep_stage, HTP_MOE_N_STAGES)
                  : nntr_hvx_forward(
                      session, op, pos, row_index.data(),
                      static_cast<int>(row_index.size()), row_count.data(),
                      static_cast<int>(row_count.size()), row_weight.data(),
                      static_cast<int>(row_weight.size()), act_f32, act_len,
                      out_f32, out_len, &rep_resume);
      const uint64_t elapsed = profile.level() ? HtpProfile::nowUs() - t0 : 0;
      if (err == AEE_SUCCESS && elapsed < best_elapsed) {
        best_elapsed = elapsed;
        std::memcpy(stage_us, rep_stage, sizeof(stage_us));
        op_pcyc.swap(rep_pcyc);
        resume_at = rep_resume;
      }
    }
    const uint64_t elapsed = (best_elapsed == UINT64_MAX) ? 0 : best_elapsed;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_forward_debug" : "nntr_hvx_forward") +
        " failed at op " + std::to_string(op) + " pos " + std::to_string(pos) +
        ": " + graphErr(err));
    }
    if (resume_at != expected_resume) {
      throw std::runtime_error("nntr_hvx_forward: resume_at " +
                               std::to_string(resume_at) + " from op " +
                               std::to_string(op) + ", the stretch ends at " +
                               std::to_string(expected_resume));
    }
    stagedMemcpy(out, out_f32, static_cast<size_t>(out_len) * sizeof(float));
    ++fwd_calls_;
    if (op == first_resident_op_)
      ++fwd_tokens_;
    if (moe)
      dumpMoeCall("forward", act, static_cast<size_t>(act_len), out,
                  static_cast<size_t>(out_len), 1u, K, 0u, N_out, 0, row_count);
    dumpForwardCall(op, expected_resume, pos, act, K, out, N_out);
    if (profile.level()) {
      uint64_t pcyc_sum = 0;
      for (uint32_t i = 0; i < n_limit; ++i) {
        pcyc_sum += op_pcyc[i];
        if (timed)
          profile.addGraphOp(graphOp(op + i)->kind, op_pcyc[i]);
      }
      if (moe) {
        // The prim block (start_op, pos, six lengths) and the three routing
        // sequences; the 64 handles no longer travel (plan 83 section 2).
        const size_t in_arg_bytes =
          40 + sizeof(uint32_t) *
                 (row_index.size() + row_count.size() + row_weight.size());
        profile.addInvokeMoeLayer(1, K, N_out, elapsed,
                                  timed ? stage_us : nullptr, act_stage,
                                  out_stage, in_arg_bytes);
      }
      profile.addInvokeForward(
        resume_at - op, timed ? stage_us[HTP_MOE_T_DSP_TOTAL] : 0, pcyc_sum);
    }
  }

  /** @brief [#141] The M==1 MoE call's dspqueue (plan 141-dspq-moe.md
   *  section 3.4): the queue, its two ION buffers mapped once, and the
   *  counters. Shared with the HtpBackend close hook, which tears it down
   *  before the session closes: the two singletons' destruction order is
   *  not fixed, so the hook must own what it touches. */
  struct DspqMoe {
    enum State { UNTRIED, ON, OFF } state = UNTRIED;
    bool broken = false; /**< a transport failure after ON: every later
                              call throws, never FastRPC (section 3.5) */
    const HtpDspqApi *api = nullptr;
    dspqueue_t q = nullptr;
    remote_handle64 session = 0;
    int domain = CDSP_DOMAIN_ID; /**< [#132 Part B E3] S2's effective one */
    const char *tag = "dspq";    /**< log prefix: "dspq" (S1), "dspq[S2]" */
    std::unique_ptr<HtpRpcBuffer> act, out, math_aux;
    // Owned until the close hook unmaps them, even if ComputeOps dies first.
    std::unordered_map<const float *, std::unique_ptr<HtpRpcBuffer>>
      router_weights;
    struct ConvInputCache {
      htp_dspq_conv_req binding = {};
      uint32_t slot = 0;
      bool valid = false;
      std::vector<float> conv, gamma, state;
    };
    // Content snapshots preserve arbitrary caller edits, sequence resets and
    // intervening FastRPC/prefill calls. Owned by this exact queue lifetime.
    std::unordered_map<const float *, ConvInputCache> conv_inputs;
    uint64_t conv_cached_calls = 0, conv_saved_bytes = 0;
    uint64_t conv_w_updates = 0, conv_gamma_updates = 0, conv_state_updates = 0;
    bool act_mapped = false, out_mapped = false;
    uint32_t seq = 0;
    const float *retained_norm = nullptr;
    uint32_t retained_seq = 0;
    size_t retained_offset = 0, retained_bytes = 0;
    uint64_t retained_calls = 0;
    uint64_t calls = 0;
    uint32_t arm_spin_us = 0;
    std::atomic<int> cb_err{0}; /**< the queue's error callback */
    std::vector<uint32_t> msg;  /**< the request message, reused */
  };

  static constexpr uint32_t kDspqTimeoutUs = 5000000; /**< 5 s, never a hang */

  static void dspqErrorCb(dspqueue_t, AEEResult err, void *ctx) {
    static_cast<DspqMoe *>(ctx)->cb_err.store(err == 0 ? -1 : err);
  }

  /** @brief Undoes whatever creation got through, in reverse. */
  static uint32_t dspqRelease(DspqMoe &st) {
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    uint32_t fail = 0; // [#132 Part B] fastrpc_munmap refusals
    for (auto &entry : st.router_weights)
      fail += mem.munmap(st.domain, entry.second->fd(), entry.second->data(),
                         entry.second->size()) != 0;
    st.router_weights.clear();
    st.conv_inputs.clear();
    if (st.math_aux) {
      fail += mem.munmap(st.domain, st.math_aux->fd(), st.math_aux->data(),
                         st.math_aux->size()) != 0;
      st.math_aux.reset();
    }
    if (st.act_mapped)
      fail += mem.munmap(st.domain, st.act->fd(), st.act->data(),
                         st.act->size()) != 0;
    if (st.out_mapped)
      fail += mem.munmap(st.domain, st.out->fd(), st.out->data(),
                         st.out->size()) != 0;
    st.act_mapped = st.out_mapped = false;
    if (st.q != nullptr)
      st.api->close(st.q);
    st.q = nullptr;
    return fail;
  }

  /** @brief The close hook: QUIT, stop the DSP thread, close, unmap. */
  static void dspqTeardown(DspqMoe &st) {
    if (dspqStop(st))
      dspqRelease(st);
  }

  /** @brief QUIT and the DSP thread joined, nothing unmapped yet (the
   *  two-session teardown unmaps in its own order). @return whether the
   *  queue was on. */
  static bool dspqStop(DspqMoe &st) {
    if (st.state != DspqMoe::ON)
      return false;
    st.state = DspqMoe::OFF;
    if (st.conv_cached_calls)
      std::fprintf(stderr,
                   "[HTP] conv input cache: calls=%llu slots=%zu "
                   "saved_bytes=%llu w_updates=%llu gamma_updates=%llu "
                   "state_updates=%llu\n",
                   (unsigned long long)st.conv_cached_calls,
                   st.conv_inputs.size(),
                   (unsigned long long)st.conv_saved_bytes,
                   (unsigned long long)st.conv_w_updates,
                   (unsigned long long)st.conv_gamma_updates,
                   (unsigned long long)st.conv_state_updates);
    if (st.retained_calls)
      std::fprintf(stderr,
                   "[HTP] retained router input calls=%llu bytes=%llu\n",
                   (unsigned long long)st.retained_calls,
                   (unsigned long long)(st.retained_calls * st.retained_bytes));
    const uint32_t quit[2] = {HTP_DSPQ_OP_QUIT, 0};
    st.api->write(st.q, 0, 0, nullptr, sizeof(quit),
                  reinterpret_cast<const uint8_t *>(quit), kDspqTimeoutUs);
    uint32_t res[4] = {0, 0, 0, 0};
    const int err = nntr_hvx_dspq_stop(st.session, res, 4);
    std::fprintf(stderr,
                 "[HTP] %s: close calls=%llu served=%u bad=%u "
                 "empty_polls=%u dsp_spin_us=%u stop_err=0x%x\n",
                 st.tag, (unsigned long long)st.calls, res[0], res[1], res[2],
                 res[3], static_cast<unsigned>(err));
    return true;
  }

  /** @brief Creation, once, at the first M==1 MoE call (section 3.4); any
   *  failure prints one "dspq: off" line and is never retried. */
  void dspqCreate(remote_handle64 session) {
    const char *spin_env = std::getenv("NNTR_HTP_DSPQ_SPIN_US");
    dspq_ = dspqMake(
      session, CDSP_DOMAIN_ID, "dspq", HTP_DSPQ_BUF_BYTES,
      spin_env ? static_cast<uint32_t>(std::strtoul(spin_env, nullptr, 10))
               : 1000u);
  }

  /** [#132 Part B E3] NNTR_HTP_E2E_SPIN_US (default 0): how long a
   *  waiting session spins (with a pause) before it sleeps -- at each hop
   *  and on its dspqueue after a token. E5b's 1000 us kept one of the six
   *  hardware threads busy through the other session's compute; E5d read
   *  20.8 / 20.2 / 16.5 tok/s at 0 / 20 / 1000. */
  static uint32_t e2eSpinUs() {
    static const uint32_t us = [] {
      const char *e = std::getenv("NNTR_HTP_E2E_SPIN_US");
      return e ? static_cast<uint32_t>(std::strtoul(e, nullptr, 10)) : 0u;
    }();
    return us;
  }

  /** @brief A queue on @a session in @a domain with an out buffer of
   *  @a out_bytes (S1's: HTP_DSPQ_BUF_BYTES; [#132 Part B E3] S2's holds
   *  the logits too). Its teardown is an HtpBackend close hook. */
  std::shared_ptr<DspqMoe> dspqMake(remote_handle64 session, int domain,
                                    const char *tag, size_t out_bytes,
                                    uint32_t dsp_spin_us) {
    auto st = std::make_shared<DspqMoe>();
    st->state = DspqMoe::OFF;
    st->session = session;
    st->domain = domain;
    st->tag = tag;
    char why[96] = "";
    int err = 0;
    auto fail = [&](const char *step, int e) {
      std::snprintf(why, sizeof(why), "%s 0x%x", step,
                    static_cast<unsigned>(e));
      err = e != 0 ? e : -1;
    };
    const HtpDspqApi &api = HtpDspqApi::get();
    st->api = &api;
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    if (api.missing != nullptr) {
      std::snprintf(why, sizeof(why), "dlsym %s", api.missing);
      err = -1;
    }
    if (err == 0) {
      const int e = api.create(domain, 0, HTP_DSPQ_REQ_QUEUE_BYTES,
                               HTP_DSPQ_RESP_QUEUE_BYTES, nullptr, dspqErrorCb,
                               st.get(), &st->q);
      if (e != AEE_SUCCESS) {
        st->q = nullptr;
        fail("create", e);
      }
    }
    uint64_t id = 0;
    if (err == 0) {
      const int e = api.export_(st->q, &id);
      if (e != AEE_SUCCESS)
        fail("export", e);
    }
    if (err == 0) {
      st->act = std::make_unique<HtpRpcBuffer>(HTP_DSPQ_BUF_BYTES);
      st->out = std::make_unique<HtpRpcBuffer>(out_bytes);
      if (!st->act->isIon() || !st->out->isIon() || st->act->fd() < 0 ||
          st->out->fd() < 0 || mem.mmap == nullptr) {
        fail("ion", 0);
      }
    }
    if (err == 0) {
      int e = mem.mmap(domain, st->act->fd(), st->act->data(), 0,
                       st->act->size(), FASTRPC_MAP_FD);
      st->act_mapped = (e == 0);
      if (e == 0) {
        e = mem.mmap(domain, st->out->fd(), st->out->data(), 0, st->out->size(),
                     FASTRPC_MAP_FD);
        st->out_mapped = (e == 0);
      }
      if (e != 0)
        fail("fastrpc_mmap", e);
    }
    if (err == 0) {
      const int e = nntr_hvx_dspq_start(session, id, dsp_spin_us);
      if (e != AEE_SUCCESS)
        fail(static_cast<unsigned>(e) == 0x8000040Eu
               ? "dspq_start (stale skel: rebuild test/htp/build.sh)"
               : "dspq_start",
             e);
    }
    if (err != 0) {
      dspqRelease(*st);
      std::fprintf(stderr, "[HTP] %s: off (%s) -- MoE calls stay on FastRPC\n",
                   tag, why);
      return st;
    }
    st->state = DspqMoe::ON;
    st->arm_spin_us = HtpBackend::global().pollUs();
    if (out_bytes == HTP_DSPQ_BUF_BYTES)
      std::fprintf(stderr,
                   "[HTP] %s: on queue=0x%llx dsp_spin_us=%u arm_spin_us=%u "
                   "buffers=2x%u ion=y\n",
                   tag, (unsigned long long)id, dsp_spin_us, st->arm_spin_us,
                   static_cast<unsigned>(HTP_DSPQ_BUF_BYTES));
    else
      std::fprintf(stderr,
                   "[HTP] %s: on queue=0x%llx dsp_spin_us=%u arm_spin_us=%u "
                   "buffers=%u+%zu ion=y domain=%d\n",
                   tag, (unsigned long long)id, dsp_spin_us, st->arm_spin_us,
                   static_cast<unsigned>(HTP_DSPQ_BUF_BYTES), out_bytes,
                   domain);
    HtpBackend::global().atClose([st] { dspqTeardown(*st); });
    return st;
  }

  /** @brief Whether this call can ride the queue; creates it on first use.
   *  Caller holds invoke_mutex_. */
  bool dspqReady(remote_handle64 session, size_t act_bytes, size_t out_bytes,
                 uint64_t msg_bytes) {
    // On by default since the #141 sitting (user, 2026-09-28): bit-identical
    // MoE dumps, decode +7.1 / +5.0 %. NNTR_HTP_DSPQ=0 keeps FastRPC.
    static const bool enabled = [] {
      const char *e = std::getenv("NNTR_HTP_DSPQ");
      return e == nullptr || std::atoi(e) != 0;
    }();
    if (!enabled)
      return false;
    if (!dspq_)
      dspqCreate(session);
    // The DSP may still hold the failed packet and its buffers: neither a
    // retry on the queue nor a FastRPC fallback is safe.
    if (dspq_->broken)
      throw std::runtime_error("dspq: the queue failed on an earlier call");
    // ponytail: a call larger than the packet or the 64 KiB buffers takes
    // FastRPC; every MoE layer of this model has one shape that fits.
    return dspq_->state == DspqMoe::ON && dspq_->session == session &&
           act_bytes <= HTP_DSPQ_BUF_BYTES && out_bytes <= HTP_DSPQ_BUF_BYTES &&
           msg_bytes <= HTP_DSPQ_MAX_MSG;
  }

  /** @brief One request/response pair (section 3.1); the activation is
   *  already in the queue's act buffer and the output lands in its out
   *  buffer. Transport failures throw: the DSP may still hold the packet,
   *  so a FastRPC retry could run on the same buffers at once. Caller
   *  holds invoke_mutex_.
   *  @return the kernel's rc */
  int dspqCall(unsigned M, unsigned K, unsigned inter, unsigned N_out,
               const std::vector<uint32_t> &h_gu,
               const std::vector<uint32_t> &h_dn,
               const std::vector<unsigned int> &row_index,
               const std::vector<unsigned int> &row_count,
               const std::vector<float> &row_weight, size_t act_bytes,
               size_t out_bytes, uint32_t glu, uint32_t *stage_us,
               size_t act_offset = 0, bool act_sync = true) {
    static_assert(HTP_MOE_N_STAGES == HTP_DSPQ_STAGES,
                  "htp_dspq_wire.h's stage count is the timed call's");
    DspqMoe &st = *dspq_;
    const uint32_t ne = static_cast<uint32_t>(h_gu.size());
    const uint32_t nr = static_cast<uint32_t>(row_index.size());
    const auto updates = expertUpdatePacket(K, inter, N_out);
    const uint32_t seq = ++st.seq;
    const htp_dspq_req_hdr hdr = {
      HTP_DSPQ_OP_MOE,
      seq,
      (stage_us ? HTP_DSPQ_FLAG_TIMED : 0u) | (glu ? HTP_DSPQ_FLAG_GELU : 0u) |
        (updates.empty() ? 0u : HTP_DSPQ_FLAG_EXPERT_UPDATES) |
        (!updates.empty() && expertUpdateMode() == 3
           ? HTP_DSPQ_FLAG_EXPERT_BATCH_DMA
           : 0u),
      M,
      K,
      inter,
      N_out,
      ne,
      nr};
    const size_t base_words = htp_dspq_req_bytes(ne, nr) / 4;
    const size_t words =
      base_words +
      (updates.empty()
         ? 0u
         : 1u + updates.size() * sizeof(htp_dspq_expert_update) / 4u);
    st.msg.resize(words);
    uint32_t *w = st.msg.data();
    std::memcpy(w, &hdr, sizeof(hdr));
    w += sizeof(hdr) / 4;
    std::memcpy(w, h_gu.data(), 4u * ne);
    std::memcpy(w + ne, h_dn.data(), 4u * ne);
    std::memcpy(w + 2 * ne, row_count.data(), 4u * ne);
    std::memcpy(w + 3 * ne, row_index.data(), 4u * nr);
    std::memcpy(w + 3 * ne + nr, row_weight.data(), 4u * nr);
    if (!updates.empty()) {
      st.msg[base_words] = static_cast<uint32_t>(updates.size());
      std::memcpy(st.msg.data() + base_words + 1u, updates.data(),
                  updates.size() * sizeof(htp_dspq_expert_update));
    }
    uint32_t update_us = 0;
    int rc;
    try {
      rc = dspqExchange(st.msg.data(), words * 4, seq, act_bytes, out_bytes,
                        stage_us, nullptr, true, act_offset, act_sync,
                        !updates.empty() && stage_us ? &update_us : nullptr);
    } catch (...) {
      expert_updates_failed_ |= !updates.empty();
      throw;
    }
    if (!updates.empty()) {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      if (rc != AEE_SUCCESS)
        expert_updates_failed_ = true;
      else {
        pending_expert_updates_.clear();
        ++expert_update_fused_batches_;
        expert_update_dsp_us_ += update_us;
        if (stage_us)
          stage_us[HTP_MOE_T_DSP_TOTAL] += update_us;
      }
    }
    return rc;
  }

  /** @brief Caller holds invoke_mutex_; a failed transport is never retried. */
  int dspqExchange(const void *message, size_t message_bytes, uint32_t seq,
                   size_t act_bytes, size_t out_bytes, uint32_t *stage_us,
                   HtpRpcBuffer *extra = nullptr, bool extra_sync = true,
                   size_t act_offset = 0, bool act_sync = true,
                   uint32_t *expert_update_us = nullptr) {
    DspqMoe &st = *dspq_;
    struct dspqueue_buffer bufs[3] = {};
    const uint32_t nb = extra ? 3u : 2u;
    bufs[0].fd = static_cast<uint32_t>(st.act->fd());
    bufs[0].size = static_cast<uint32_t>(act_bytes);
    bufs[0].offset = static_cast<uint32_t>(act_offset);
    bufs[0].flags = DSPQUEUE_BUFFER_FLAG_REF;
    if (act_sync)
      bufs[0].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                       DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    bufs[0].ptr = st.act->data() + act_offset;
    bufs[1].fd = static_cast<uint32_t>(st.out->fd());
    bufs[1].size = static_cast<uint32_t>(out_bytes);
    bufs[1].flags = DSPQUEUE_BUFFER_FLAG_REF;
    bufs[1].ptr = st.out->data();

    if (extra) {
      bufs[2].fd = static_cast<uint32_t>(extra->fd());
      bufs[2].size = static_cast<uint32_t>(extra->size());
      bufs[2].ptr = extra->data();
      // Router weights are immutable. Flush the host copy on first use;
      // subsequent packets retain the same read-only DSP mapping.
      bufs[2].flags = DSPQUEUE_BUFFER_FLAG_REF;
      if (extra_sync)
        bufs[2].flags |= DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                         DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    }
    const HtpDspqApi &api = *st.api;
    int err = api.write(st.q, 0, nb, bufs, static_cast<uint32_t>(message_bytes),
                        static_cast<const uint8_t *>(message), kDspqTimeoutUs);
    htp_dspq_resp resp;
    uint32_t flags = 0, rnb = 0, len = 0;
    struct dspqueue_buffer rbufs[3] = {};
    if (err == AEE_SUCCESS) {
      // Spin for the poll-QoS window, as the FastRPC call does, then block.
      const uint64_t t0 = HtpProfile::nowUs();
      for (uint32_t spins = 1;; ++spins) {
        err = api.read_noblock(st.q, &flags, 3, &rnb, rbufs, sizeof(resp), &len,
                               reinterpret_cast<uint8_t *>(&resp));
        if (err != AEE_EWOULDBLOCK)
          break;
        if ((spins & 255u) == 0 && HtpProfile::nowUs() - t0 >= st.arm_spin_us) {
          err = api.read(st.q, &flags, 3, &rnb, rbufs, sizeof(resp), &len,
                         reinterpret_cast<uint8_t *>(&resp), kDspqTimeoutUs);
          break;
        }
      }
    }
    const int cb = st.cb_err.load();
    if (err != AEE_SUCCESS || cb != 0 || len < HTP_DSPQ_RESP_BASE_BYTES ||
        resp.seq != seq || rnb != nb) {
      st.broken = true;
      char msg[160];
      std::snprintf(msg, sizeof(msg),
                    "dspq: call %u failed: err=0x%x cb_err=0x%x len=%u "
                    "seq=%u nb=%u",
                    seq, static_cast<unsigned>(err), static_cast<unsigned>(cb),
                    len, len >= 4 ? resp.seq : 0u, rnb);
      throw std::runtime_error(msg);
    }
    ++st.calls;
    if (stage_us != nullptr && resp.rc == AEE_SUCCESS) {
      if (len != HTP_DSPQ_RESP_BASE_BYTES + 4u * HTP_DSPQ_STAGES +
                   (expert_update_us ? 4u : 0u))
        throw std::runtime_error("dspq: timed response without stage slots");
      std::memcpy(stage_us, resp.stage_us, sizeof(resp.stage_us));
      if (expert_update_us)
        *expert_update_us = resp.expert_update_us;
    }
    return resp.rc;
  }

  struct E2eState; // [#132 Part B E3] defined with the members below

  /** @brief [#132 Part B E3] The two-session decode's ARM side (plan
   *  section 3.2): the token driver on both sessions over one mailbox page,
   *  S1's token packets on its MoE dspqueue, S2's on its own queue. Created
   *  at graph init; the HtpBackend close hook e2eTeardown undoes it. */
  void e2eStart() {
    E2eState &e = *e2e_st_;
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    if (!dspq_) // S1's queue: token packets only at decode, so the E2E spin
      dspq_ =
        dspqMake(e.h1, CDSP_DOMAIN_ID, "dspq", HTP_DSPQ_BUF_BYTES, e2eSpinUs());
    if (dspq_->state != DspqMoe::ON || dspq_->session != e.h1) {
      throw std::runtime_error("NNTR_HTP_E2E=1: S1's dspqueue is off (see "
                               "the dspq line; NNTR_HTP_DSPQ=0?): the token "
                               "packets ride it");
    }
    e.q1 = dspq_;
    const size_t logits_bytes = static_cast<size_t>(graph_words_[5]) * 4u;
    e.q2 =
      dspqMake(e.h2, e.dom2, "dspq[S2]",
               std::max<size_t>(HTP_DSPQ_BUF_BYTES, logits_bytes), e2eSpinUs());
    if (e.q2->state != DspqMoe::ON) {
      throw std::runtime_error("NNTR_HTP_E2E=1: S2's dspqueue is off (see "
                               "the dspq[S2] line)");
    }
    e.mbox = std::make_unique<HtpRpcBuffer>(kMboxBytes, HTP_RPC_FLAGS_UNCACHED);
    if (!e.mbox->isIon() || e.mbox->fd() < 0 || mem.mmap == nullptr) {
      throw std::runtime_error("NNTR_HTP_E2E=1: no ION page for the mailbox");
    }
    // zeroed before either side maps it, as the #178 probe's page: no
    // sequence word or header a first read could take for a post
    std::memset(e.mbox->data(), 0, e.mbox->size());
    const int fd = e.mbox->fd();
    int rc = mem.mmap(CDSP_DOMAIN_ID, fd, e.mbox->data(), 0, e.mbox->size(),
                      FASTRPC_MAP_FD);
    e.mbox1 = rc == 0;
    if (rc == 0) {
      rc =
        mem.mmap(e.dom2, fd, e.mbox->data(), 0, e.mbox->size(), FASTRPC_MAP_FD);
      e.mbox2 = rc == 0;
    }
    if (rc != 0) {
      throw std::runtime_error("NNTR_HTP_E2E=1: fastrpc_mmap of the mailbox "
                               "page failed: " +
                               std::to_string(rc));
    }
    const uint32_t spin_us = e2eSpinUs();
    rc = nntr_hvx_token_driver_start(
      e.h1, fd, static_cast<uint32_t>(e.mbox->size()), 1u, spin_us);
    e.drv1 = rc == AEE_SUCCESS;
    if (rc == AEE_SUCCESS)
      rc = nntr_hvx_token_driver_start(
        e.h2, fd, static_cast<uint32_t>(e.mbox->size()), 0u, spin_us);
    e.drv2 = e.drv1 && rc == AEE_SUCCESS;
    if (rc != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_token_driver_start[" +
                               std::string(e.drv1 ? "S2" : "S1") +
                               "] failed: " + graphErr(rc));
    }
    uint32_t rounds = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(stretch_start_.size()); ++i)
      rounds += graphOp(i)->kind == HTP_OP_ROUTER_TOPK;
    e.rounds = rounds;
    e.spin_us = spin_us;
    std::fprintf(stderr,
                 "[HTP] token driver: on s1_effdom=%d s2_effdom=%d mbox=%zu "
                 "spin_us=%u rounds=%u hops/token=%u logits_buf=%zu\n",
                 static_cast<int>(CDSP_DOMAIN_ID), e.dom2, e.mbox->size(),
                 spin_us, rounds, 2u * rounds, logits_bytes);
  }

  /** @brief [#132 Part B E3] The close hook, before the sessions close:
   *  S2's queue, both drivers, the mailbox, then S2's graph, slots and
   *  arena, each buffer fastrpc_munmap'd before nntr_hvx_close (#178's
   *  rule). S1's arena and graph stay as on the one-session path. */
  static void e2eTeardown(E2eState &e) {
    static_assert(HTP_OP_KIND_N == HTP_DSPQ_TOKEN_KINDS,
                  "htp_dspq_wire.h's kind count is the graph's");
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    // [#132 Part B E5g] every DSP thread stopped before any munmap: both
    // queues (QUIT, joined), then both token drivers (their HAP_mmap_put);
    // then the mappings in the reverse of their order (S2 arena at load; S1
    // queue, S2 queue, the page into S1 then S2 at graph init). The close
    // line counts what was mapped, what fastrpc_munmap / arena_detach
    // refused, and both sessions' DSP heap (E5f: one E run in ~12 left the
    // next process 256 MiB short of S1's 3840).
    if (e.q2)
      dspqStop(*e.q2);
    if (e.q1)
      dspqStop(*e.q1);
    uint32_t r1[5] = {0, 0, 0, 0, 0}, r2[5] = {0, 0, 0, 0, 0};
    const int s2 = e.drv2 ? nntr_hvx_token_driver_stop(e.h2, r2, 5) : 0;
    const int s1 = e.drv1 ? nntr_hvx_token_driver_stop(e.h1, r1, 5) : 0;
    if (e.drv1 || e.drv2) {
      const double n = e.tokens ? static_cast<double>(e.tokens) : 1.0;
      std::fprintf(stderr,
                   "[HTP] graph[S1]: tokens=%llu pcyc/token=%.0f "
                   "wait_us/token=%.1f\n",
                   (unsigned long long)e.tokens,
                   static_cast<double>(e.pcyc1) / n,
                   static_cast<double>(e.wait1_us) / n);
      std::fprintf(stderr,
                   "[HTP] graph[S2]: tokens=%llu pcyc/token=%.0f "
                   "wait_us/token=%.1f\n",
                   (unsigned long long)e.tokens,
                   static_cast<double>(e.pcyc2) / n,
                   static_cast<double>(e.wait2_us) / n);
      // per side: the clock (pcycles over the token's wall time on that
      // side) and the op pcycles per kind per token, with the ms they are
      // at that clock; S1's MOE also per round
      for (int k = 0; k < 2; ++k) {
        const uint64_t *kp = k == 0 ? e.kind1 : e.kind2;
        const double wus =
          static_cast<double>(k == 0 ? e.wall1_us : e.wall2_us);
        const double mhz =
          wus > 0
            ? static_cast<double>(k == 0 ? e.wall1_pcyc : e.wall2_pcyc) / wus
            : 0.0;
        std::string line;
        char buf[96];
        for (uint32_t kd = 0; kd < HTP_OP_KIND_N; ++kd) {
          if (kp[kd] == 0)
            continue;
          const double pc = static_cast<double>(kp[kd]) / n;
          std::snprintf(buf, sizeof(buf), " %s=%.0f(%.3fms)",
                        htp_graph_kind_name(kd), pc,
                        mhz > 0 ? pc / mhz / 1000.0 : 0.0);
          line += buf;
        }
        std::fprintf(
          stderr,
          "[HTP] graph[S%d] per-kind pcyc/token:%s | wall_ms/token=%.3f "
          "mhz=%.0f spin_us=%u\n",
          k + 1, line.c_str(), wus / n / 1000.0, mhz, e.spin_us);
      }
      if (e.rounds != 0 && e.kind1[HTP_OP_MOE] != 0) {
        const double mhz1 =
          e.wall1_us ? static_cast<double>(e.wall1_pcyc) / e.wall1_us : 0.0;
        const double moe = static_cast<double>(e.kind1[HTP_OP_MOE]) / n /
                           static_cast<double>(e.rounds);
        std::fprintf(
          stderr,
          "[HTP] graph[S1] moe pcyc/round=%.0f (%.3f ms) "
          "router pcyc/round=%.0f; s2 fc+dense_ffn+lm_head "
          "ms/token=%.3f (isolated #178: 8.06); arm token_ms=%.3f\n",
          moe, mhz1 > 0 ? moe / mhz1 / 1000.0 : 0.0,
          static_cast<double>(e.kind1[HTP_OP_ROUTER_TOPK]) / n /
            static_cast<double>(e.rounds),
          e.wall2_us
            ? static_cast<double>(e.kind2[HTP_OP_FC] +
                                  e.kind2[HTP_OP_DENSE_FFN] +
                                  e.kind2[HTP_OP_LM_HEAD]) /
                n / (static_cast<double>(e.wall2_pcyc) / e.wall2_us) / 1000.0
            : 0.0,
          static_cast<double>(e.token_us) / n / 1000.0);
      }
      // [#194 L0] where the token's time goes outside the kernels: the
      // ARM's round trip (rt) against S2's own wall (wake = the dspqueue
      // transport and the ARM read's wake-up), each side's hops from post
      // to wake-up (hop_us, inside its wall), tokenForward on the ARM
      // (arm_fwd, rt plus its copies) and the ARM between two tokenForward
      // calls (arm_us: the layer walk, sampler, tokenizer, print)
      // wake split: disp = post -> the DSP thread's read, ret = S2's write
      // -> the ARM's read, s2_pkt = S2's handling outside its token wall;
      // rt = disp_s2 + s2_pkt + ret_s2 when the two clocks are one counter
      // (else these three read nonsense and clk_resid says by how much)
      const double an = e.arm_n ? static_cast<double>(e.arm_n) : 1.0;
      std::fprintf(
        stderr,
        "[HTP] token driver: L0 wake us/token disp s1=%.1f s2=%.1f "
        "s2_pkt=%.1f ret s2=%.1f clk_resid=%.1f\n",
        static_cast<double>(e.disp1_us) / n,
        static_cast<double>(e.disp2_us) / n,
        (static_cast<double>(e.inout2_us) - static_cast<double>(e.wall2_us)) /
          n,
        static_cast<double>(e.ret2_us) / n,
        (static_cast<double>(e.token_us) -
         static_cast<double>(e.disp2_us + e.inout2_us + e.ret2_us)) /
          n);
      std::fprintf(
        stderr,
        "[HTP] token driver: L0 us/token rt=%.1f s2_wall=%.1f s1_wall=%.1f "
        "wake=%.1f hop_us s1=%.1f s2=%.1f arm_fwd=%.1f arm_us=%.1f "
        "arm_n=%llu\n",
        static_cast<double>(e.token_us) / n,
        static_cast<double>(e.wall2_us) / n,
        static_cast<double>(e.wall1_us) / n,
        (static_cast<double>(e.token_us) - static_cast<double>(e.wall2_us)) / n,
        static_cast<double>(e.hop1_us) / n, static_cast<double>(e.hop2_us) / n,
        static_cast<double>(e.fwd_us) / n, static_cast<double>(e.arm_us) / an,
        (unsigned long long)e.arm_n);
      std::fprintf(stderr,
                   "[HTP] token driver: close tokens=%llu hops/token=%.2f "
                   "s1_served=%u s2_served=%u timeouts=%u/%u stale=%u/%u "
                   "id_checked=%llu id_mismatch=%llu stop_err=0x%x/0x%x\n",
                   (unsigned long long)e.tokens,
                   static_cast<double>(e.hops) / n, r1[0], r2[0], r1[2], r2[2],
                   r1[3], r2[3], (unsigned long long)e.id_checked,
                   (unsigned long long)e.id_mismatch, static_cast<unsigned>(s1),
                   static_cast<unsigned>(s2));
    }
    e.drv1 = e.drv2 = false;
    uint32_t info1[7] = {0}, info2[7] = {0};
    const int i1 = e.h1 ? nntr_hvx_session_info(e.h1, info1, 7) : -1;
    const int i2 = e.h2 ? nntr_hvx_session_info(e.h2, info2, 7) : -1;
    size_t mapped = 0;
    uint32_t unmap_fail = 0, detach_fail = 0;
    if (e.mbox) {
      if (e.mbox2) {
        unmap_fail +=
          mem.munmap(e.dom2, e.mbox->fd(), e.mbox->data(), e.mbox->size()) != 0;
        mapped += e.mbox->size();
      }
      if (e.mbox1) {
        unmap_fail += mem.munmap(CDSP_DOMAIN_ID, e.mbox->fd(), e.mbox->data(),
                                 e.mbox->size()) != 0;
        mapped += e.mbox->size();
      }
      e.mbox1 = e.mbox2 = false;
    }
    for (const std::shared_ptr<DspqMoe> &q : {e.q2, e.q1}) {
      if (q && q->q != nullptr) {
        mapped += (q->act_mapped ? q->act->size() : 0u) +
                  (q->out_mapped ? q->out->size() : 0u);
        unmap_fail += dspqRelease(*q);
      }
    }
    if (e.graph2)
      nntr_hvx_graph_release(e.h2);
    e.graph2 = false;
    for (uint32_t h : e.q4m1)
      nntr_hvx_q4m1_release(e.h2, h);
    e.q4m1.clear();
    for (auto c = e.arena.rbegin(); c != e.arena.rend(); ++c) {
      detach_fail += nntr_hvx_arena_detach(e.h2, c->dsp_id) != AEE_SUCCESS;
      if (mem.munmap != nullptr)
        unmap_fail +=
          mem.munmap(e.dom2, c->buf->fd(), c->buf->data(), c->buf->size()) != 0;
      mapped += c->buf->size();
    }
    e.arena.clear();
    std::fprintf(stderr,
                 "[HTP] s2: close mapped_mib=%.2f unmap_fail=%u "
                 "detach_fail=%u heap_used_kib s1=%u s2=%u (info rc "
                 "0x%x/0x%x)\n",
                 static_cast<double>(mapped) / (1024.0 * 1024.0), unmap_fail,
                 detach_fail, info1[4], info2[4], static_cast<unsigned>(i1),
                 static_cast<unsigned>(i2));
  }

  /** @brief [#194 L0] The system counter in us (low 32 bits): the ARM's
   *  cntvct_el0, the counter the DSP's QTimer reads; off aarch64 (the
   *  in-process build, whose DSP clock is CLOCK_MONOTONIC) steady_clock. */
  static uint32_t sysCounterUs() {
#if defined(__aarch64__)
    uint64_t c, f;
    asm volatile("mrs %0, cntvct_el0" : "=r"(c));
    asm volatile("mrs %0, cntfrq_el0" : "=r"(f));
    return f ? static_cast<uint32_t>(static_cast<unsigned __int128>(c) *
                                     1000000u / f)
             : 0u;
#else
    return static_cast<uint32_t>(HtpProfile::nowUs());
#endif
  }

  /** @brief [#132 Part B E3] One decode token on the two sessions: the
   *  embedding row into S2's act buffer, OP_TOKEN to S1 (serve every MoE
   *  round) and to S2 (op 0 to the argmax), both responses. Only the id
   *  comes back unless set_decode_logits asked for the logits (the
   *  NNTR_PPL_DECODE / sampling / bad-words runs), which S2 writes into its
   *  out buffer. Any failure throws; a transport failure marks both
   *  queues broken (the DSP may still hold a packet). */
  void tokenForward(uint32_t pos, const float *act, unsigned K, float *out,
                    unsigned N_out) {
    E2eState &e = *e2e_st_;
    const uint64_t entry_us = HtpProfile::nowUs();
    if (e.last_exit_us != 0) {
      e.arm_us += entry_us - e.last_exit_us;
      ++e.arm_n;
    }
    if (!e.drv1 || !e.drv2 || !e.q2 || !dspq_)
      throw std::runtime_error("token driver: not started (its start threw "
                               "at graph init)");
    DspqMoe &q1 = *dspq_, &q2 = *e.q2;
    if (q1.broken || q2.broken)
      throw std::runtime_error("token driver: a queue failed on an earlier "
                               "token");
    const size_t act_bytes = static_cast<size_t>(K) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(N_out) * sizeof(float);
    if (act_bytes > q2.act->size() || out_bytes > q2.out->size())
      throw std::runtime_error("token driver: row " + std::to_string(K) +
                               " / logits " + std::to_string(N_out) +
                               " do not fit S2's buffers");
    if (have_id_)
      throw std::runtime_error("token driver: the previous token's id was "
                               "never taken (take_decode_token_id)");
    // [#132 Part B E3] the bad-word ids S2's argmax skips (LM_BAN), sent
    // when they change; more than the DSP holds, or none after some (an
    // empty sequence may reach the skel as NULL, the stale-skel code) ->
    // the logits come back and the CPU picks
    const bool logits = want_logits_ || ban_.size() > HTP_GRAPH_MAX_BAN ||
                        (ban_.empty() && !ban_sent_.empty());
    if (!logits && ban_sent_ != ban_) {
      std::vector<float> words(ban_.size());
      std::memcpy(words.data(), ban_.data(), ban_.size() * sizeof(uint32_t));
      setParam(e.h2, kind_ops_[HTP_OP_LM_HEAD][0], HTP_GRAPH_PARAM_LM_BAN,
               words.data(), static_cast<unsigned>(ban_.size()),
               static_cast<unsigned>(ban_.size()), "LM_BAN");
      ban_sent_ = ban_;
    }
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    std::memcpy(q2.act->data(), act, act_bytes);
    const uint32_t tok = e.tok++;
    const htp_dspq_token_req r1 = {HTP_DSPQ_OP_TOKEN, tok, 0u, pos};
    const htp_dspq_token_req r2 = {HTP_DSPQ_OP_TOKEN, tok,
                                   logits ? HTP_DSPQ_TOKEN_LOGITS : 0u, pos};
    struct dspqueue_buffer b[2] = {};
    b[0].fd = static_cast<uint32_t>(q2.act->fd());
    b[0].size = static_cast<uint32_t>(act_bytes);
    b[0].flags = DSPQUEUE_BUFFER_FLAG_REF | DSPQUEUE_BUFFER_FLAG_FLUSH_SENDER |
                 DSPQUEUE_BUFFER_FLAG_INVALIDATE_RECIPIENT;
    b[0].ptr = q2.act->data();
    b[1].fd = static_cast<uint32_t>(q2.out->fd());
    b[1].size = static_cast<uint32_t>(out_bytes);
    b[1].flags = DSPQUEUE_BUFFER_FLAG_REF;
    b[1].ptr = q2.out->data();
    const uint32_t nb2 = logits ? 2u : 1u;
    const uint32_t c0 = sysCounterUs();
    const uint64_t t0 = HtpProfile::nowUs();
    int err =
      q1.api->write(q1.q, 0, 0, nullptr, sizeof(r1),
                    reinterpret_cast<const uint8_t *>(&r1), kDspqTimeoutUs);
    if (err == AEE_SUCCESS)
      err =
        q2.api->write(q2.q, 0, nb2, b, sizeof(r2),
                      reinterpret_cast<const uint8_t *>(&r2), kDspqTimeoutUs);
    htp_dspq_token_resp s1r = {}, s2r = {};
    uint32_t len2 = 0, len1 = 0, rnb2 = 0, rnb1 = 0;
    if (err == AEE_SUCCESS) {
      // blocking reads (the token is ~20 ms; S1 answers before S2 ends)
      uint32_t flags = 0;
      struct dspqueue_buffer rb[2] = {};
      err = q2.api->read(q2.q, &flags, 2, &rnb2, rb, sizeof(s2r), &len2,
                         reinterpret_cast<uint8_t *>(&s2r), kDspqTimeoutUs);
      if (err == AEE_SUCCESS)
        err = q1.api->read(q1.q, &flags, 2, &rnb1, rb, sizeof(s1r), &len1,
                           reinterpret_cast<uint8_t *>(&s1r), kDspqTimeoutUs);
    }
    const uint64_t us = HtpProfile::nowUs() - t0;
    const uint32_t c2 = sysCounterUs(); // after both reads: S1 answered first
    const int cb = q1.cb_err.load() != 0 ? q1.cb_err.load() : q2.cb_err.load();
    if (err != AEE_SUCCESS || cb != 0 || len2 != sizeof(s2r) ||
        len1 != sizeof(s1r) || s2r.seq != tok || s1r.seq != tok ||
        rnb2 != nb2 || rnb1 != 0u) {
      q1.broken = q2.broken = true;
      char msg[200];
      std::snprintf(msg, sizeof(msg),
                    "token driver: token %u pos %u transport failed: "
                    "err=0x%x cb_err=0x%x len=%u/%u seq=%u/%u nb=%u/%u",
                    tok, pos, static_cast<unsigned>(err),
                    static_cast<unsigned>(cb), len1, len2, s1r.seq, s2r.seq,
                    rnb1, rnb2);
      throw std::runtime_error(msg);
    }
    ++q1.calls;
    ++q2.calls;
    if (s1r.rc != AEE_SUCCESS || s2r.rc != AEE_SUCCESS) {
      throw std::runtime_error(
        "token driver: token " + std::to_string(tok) + " pos " +
        std::to_string(pos) + " failed: S1 " + graphErr(s1r.rc) + ", S2 " +
        graphErr(s2r.rc) +
        " (AEE_EEXPIRED: a hop timed out; the FARF names "
        "the side)");
    }
    if (logits) {
      std::memcpy(out, q2.out->data(), out_bytes);
      // S2's argmax against the logits it returned, the ids LM_BAN held at
      // the time masked as S2 masked them
      std::vector<float> keep(ban_sent_.size());
      for (size_t i = 0; i < ban_sent_.size(); ++i) {
        keep[i] = out[ban_sent_[i]];
        out[ban_sent_[i]] = -INFINITY;
      }
      const uint32_t pick = m1_argmax_first(out, N_out);
      for (size_t i = ban_sent_.size(); i-- > 0;)
        out[ban_sent_[i]] = keep[i];
      ++e.id_checked;
      e.id_mismatch += pick != s2r.id;
    } else {
      pending_id_ = s2r.id;
      have_id_ = true;
    }
    ++e.tokens;
    e.hops += s2r.hops;
    e.wait1_us += s1r.wait_us;
    e.wait2_us += s2r.wait_us;
    e.pcyc1 += s1r.pcycles;
    e.pcyc2 += s2r.pcycles;
    e.wall1_us += s1r.wall_us;
    e.wall2_us += s2r.wall_us;
    e.wall1_pcyc += s1r.wall_pcyc;
    e.wall2_pcyc += s2r.wall_pcyc;
    for (uint32_t k = 0; k < HTP_OP_KIND_N; ++k) {
      e.kind1[k] += s1r.kind_pcyc[k];
      e.kind2[k] += s2r.kind_pcyc[k];
    }
    e.token_us += us;
    e.hop1_us += s1r.hop_us;
    e.hop2_us += s2r.hop_us;
    e.disp1_us += static_cast<int32_t>(s1r.t_in_us - c0);
    e.disp2_us += static_cast<int32_t>(s2r.t_in_us - c0);
    e.ret2_us += static_cast<int32_t>(c2 - s2r.t_out_us);
    e.inout2_us += static_cast<int32_t>(s2r.t_out_us - s2r.t_in_us);
    ++fwd_calls_; // one ARM -> S2 packet per token: calls/token = 1.00
    ++fwd_tokens_;
    if (e.tokens == 1)
      std::fprintf(stderr,
                   "[HTP] token driver: first token pos=%u id=%u hops=%u "
                   "us=%llu logits=%d\n",
                   pos, s2r.id, s2r.hops, (unsigned long long)us, logits);
    HtpProfile &profile = HtpProfile::global();
    if (profile.level())
      profile.addInvokeForward(static_cast<unsigned>(stretch_start_.size()), 0,
                               uint64_t(s1r.pcycles) + s2r.pcycles);
    e.last_exit_us = HtpProfile::nowUs();
    e.fwd_us += e.last_exit_us - entry_us;
  }

  /** [#132 Part B E3] compute_ops.h: whether the next decode tokens must
   *  bring their logits back (else only the id travels). */
  void set_decode_logits(bool want) override { want_logits_ = want; }

  /** [#132 Part B E3] compute_ops.h: the bad-word ids the caller's pick
   *  sets to -inf; S2's argmax skips them (LM_BAN) on the id-only path. */
  void set_decode_ban(const unsigned *ids, unsigned n) override {
    ban_.assign(ids, ids + n);
  }

  /** [#132 Part B E3] compute_ops.h: the id S2's argmax picked for the
   *  last decode token, once, when its logits did not come back. */
  bool take_decode_token_id(unsigned *id) override {
    if (!have_id_)
      return false;
    *id = pending_id_;
    have_id_ = false;
    return true;
  }

  /** [#132 Part B E3] compute_ops.h: the model handed its last Q4_0
   *  weight. On the two-session path the FC set goes into S2's arena now,
   *  at load (the startup cell), not at the first decode token. */
  bool finish_decode_graph_q4_0() override {
    std::lock_guard<std::mutex> lock(graph_mutex_);
    if (!e2e_ || q4_pending_.empty() || q4m1_bound_ || graph_inited_)
      return false;
    e2ePlaceFc();
    return true;
  }

  /** [#132 Part B E3] Opens S2 and places the Q4M1 set in its arena: at
   *  load after S1's MoE arena (the model calls finish_decode_graph_q4_0
   *  after repack_weight), else at graph init. Caller holds graph_mutex_.
   *  The banner's s1_arena_mib is what S1 had mapped when S2 opened. */
  void e2ePlaceFc() {
    E2eState &e = *e2e_st_;
    HtpBackend &b = HtpBackend::global();
    const size_t s1_mib = arenaBytes() >> 20;
    uint32_t info1[7] = {0}; // res[4]: S1's heap in use, KiB
    nntr_hvx_session_info(e.h1, info1, 7);
    if (!b.openSecond())
      throw std::runtime_error("NNTR_HTP_E2E=1: the second session did not "
                               "open (" +
                               b.s2Error() + ")");
    e.h2 = static_cast<remote_handle64>(b.handle2());
    e.dom2 = b.effDomain2();
    const uint64_t t0 = HtpProfile::nowUs();
    try {
      bindQ4m1(e.h2);
    } catch (...) {
      releaseQ4m1(e.h2);
      throw;
    }
    q4m1_bound_ = true;
    const double ms = static_cast<double>(HtpProfile::nowUs() - t0) / 1000.0;
    size_t mapped = 0;
    for (const ArenaChunk &c : e.arena)
      mapped += c.buf->size();
    std::fprintf(stderr,
                 "[HTP] s2: fc arena weights=%zu handles=%zu attach_mib=%.1f "
                 "chunks=%zu mapped_mib=%zu feed=%s load_ms=%.1f "
                 "lanes=%s s1_arena_mib=%zu s1_heap_kib=%u\n",
                 q4_pending_.size(), e.q4m1.size(),
                 static_cast<double>(e.attach_bytes) / (1024.0 * 1024.0),
                 e.arena.size(), mapped >> 20, q4m1FeedName(), ms,
                 std::getenv("NNTR_HTP_FC_LANES")
                   ? std::getenv("NNTR_HTP_FC_LANES")
                   : "6,3",
                 s1_mib, info1[4]);
  }

  /** [#132 Part B E3] The feed the FC kinds take: the op's l2, or VTCM only
   *  where their session has some (S2 opens lite with none on the S25). */
  const char *q4m1FeedName() const {
    if (kind_ops_[HTP_OP_FC].empty() ||
        (graphOp(kind_ops_[HTP_OP_FC][0])->feed & HTP_GRAPH_FEED_L2) != 0u)
      return "l2";
    return e2e_ && HtpBackend::global().vtcm2Bytes() == 0u ? "l2" : "vtcm";
  }

  /** @brief [doc 46] One call for the whole layer.
   *
   *  The activation goes over once instead of once per expert, and the
   *  output comes back once: with 32 experts at top-4 the old path shipped
   *  the same token rows four times each in both directions. The routing
   *  table rides along as three small arrays -- 1776 uint32 + 1776 float +
   *  32 uint32 for this model, about 14 KB against the 3.6 MB activation.
   */
  void invokeMoeLayer(remote_handle64 session,
                      const std::vector<uint32_t> &h_gu,
                      const std::vector<uint32_t> &h_dn,
                      const std::vector<unsigned int> &row_index,
                      const std::vector<unsigned int> &row_count,
                      const std::vector<float> &row_weight, const float *act,
                      float *out, unsigned int M, unsigned int K,
                      unsigned int inter, unsigned int N_out, int kind = 0,
                      uint32_t glu = 0, const float *pre_gamma = nullptr,
                      const float *post_gamma = nullptr, float eps = 0.0f,
                      const float *add_x2 = nullptr,
                      const float *add_gamma = nullptr,
                      float add_scale = 1.0f) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N_out);
    const bool with_norms = pre_gamma != nullptr || post_gamma != nullptr;

    const size_t act_bytes = static_cast<size_t>(act_len) * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_len) * sizeof(float);
    uint64_t msg_bytes =
      htp_dspq_req_bytes(static_cast<uint32_t>(h_gu.size()),
                         static_cast<uint32_t>(row_index.size()));

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    // [#141] The M==1 MoE call rides the dspqueue (default since the #141
    // sitting; NNTR_HTP_DSPQ=0 keeps FastRPC): the same DSP function, the same
    // argument bytes, only the transport differs. Anything the packet cannot
    // carry stays on FastRPC.
    const bool via_dspq = M == 1 && kind == 0 && !with_norms &&
                          h_dn.size() == h_gu.size() &&
                          row_count.size() == h_gu.size() &&
                          row_weight.size() == row_index.size() &&
                          dspqReady(session, act_bytes, out_bytes, msg_bytes);
    if (expertUpdateMode()) {
      std::lock_guard<std::mutex> handles(handle_mutex_);
      if (expert_updates_failed_)
        throw std::runtime_error("expert update failed; restart the session");
      const size_t extra = pending_expert_updates_.empty()
                             ? 0u
                             : 4u + pending_expert_updates_.size() *
                                      sizeof(htp_dspq_expert_update);
      if (expertUpdateMode() < 2 || !via_dspq ||
          !expertUpdatesMatch(K, inter, N_out) ||
          msg_bytes + extra > HTP_DSPQ_MAX_MSG)
        flushExpertUpdates(session);
      else
        msg_bytes += extra;
      // Prefill bootstraps physical slot handles. A later prefill commits any
      // remaining decode updates through the ordinary registration fallback.
      decode_expert_updates_ = M == 1;
    }
    // A fused router just produced this expert input on the DSP. Reuse
    // that mapped subsection and alternate the queue buffers. A complete
    // comparison rejects host writes; the sequence rejects queue reuse.
    static const bool retain_decode = [] {
      const char *e = std::getenv("NNTR_HTP_RETAIN_ROUTER_INPUT");
      return !e || std::atoi(e) != 0;
    }();
    size_t act_offset = 0;
    bool retained = false;
    if (via_dspq && retain_decode) {
      auto &q = *dspq_;
      if (q.retained_norm == act && q.retained_seq == q.seq &&
          q.retained_bytes == act_bytes &&
          q.retained_offset + act_bytes <= q.out->size() &&
          std::memcmp(act, q.out->data() + q.retained_offset, act_bytes) == 0) {
        act_offset = q.retained_offset;
        std::swap(q.act, q.out);
        retained = true;
        ++q.retained_calls;
      }
      q.retained_norm = nullptr;
    }
    const uint64_t lease = moe_input_reuse_.token(act, M, K);
    uint64_t prepared_cookie = lease && lease == prepared_moe_lease_ &&
                                   kind == 0 && !with_norms && !add_x2 &&
                                   M > 4 && HtpProfile::global().level() < 3
                                 ? prepared_moe_cookie_
                                 : 0;
    prepared_moe_cookie_ = prepared_moe_lease_ = 0;
    void *saved = moe_input_reuse_.consume(act, M, K);
    static const bool trace_reuse =
      std::getenv("NNTR_HTP_MOE_INPUT_REUSE_TRACE") != nullptr;
    const uint64_t stage_t0 = trace_reuse ? HtpProfile::nowUs() : 0;
    const bool host_prenorm = HtpProfile::global().level() >= 2 && with_norms &&
                              !via_dspq && pre_gamma != nullptr;
    const bool reused = !via_dspq && !host_prenorm && saved != nullptr;
    const bool shared_io = M > 1 && sharedPrefillIo(1u);
    const bool direct_in = shared_io && !host_prenorm &&
                           HtpBackend::global().isSharedRange(act, act_bytes);
    const bool direct_out =
      shared_io && HtpBackend::global().isSharedRange(out, out_bytes);
    HtpRpcBuffer &act_stage = reused     ? *static_cast<HtpRpcBuffer *>(saved)
                              : via_dspq ? *dspq_->act
                              : (prepared_cookie || host_prenorm || direct_in)
                                ? stage(act_pool_, act_bytes)
                                : stageInput(act_pool_, act, act_bytes);
    // The post-FFN epilogue's second addend rides its own staging buffer,
    // staged before out_stage: it may be swapped in from out_pool_.
    float *x2_f32 = nullptr;
    if (add_x2 != nullptr) {
      if (pre_gamma == nullptr || K != N_out || via_dspq)
        throw std::invalid_argument(
          "gemm_qs4cx_moe_layer_fp32: the epilogue needs the raw rows "
          "(pre_gamma) and K == N_out");
      x2_f32 = reinterpret_cast<float *>(
        stageInput(x2_pool_, add_x2, out_bytes, &out_pool_).data());
    }
    if (direct_in || direct_out) {
      ++shared_io_calls_[0];
      shared_io_bytes_[0] +=
        (direct_in ? act_bytes : 0u) + (direct_out ? out_bytes : 0u);
    }
    float *act_f32 =
      direct_in ? const_cast<float *>(act)
                : reinterpret_cast<float *>(act_stage.data() + act_offset);
    HtpRpcBuffer &out_stage =
      via_dspq ? *dspq_->out : stage(out_pool_, out_bytes);
    float *out_f32 =
      direct_out ? out : reinterpret_cast<float *>(out_stage.data());
    // NNTR_HTP_PROFILE>=2 on a call with folded norms: the timed entry
    // (stage breakdown) and the norms on the host around it, so the prefill
    // MoE and dense calls, which always carry them, have stages too (doc 57
    // section 9.26), the post-FFN epilogue too (section 9.34). Measurement
    // only; the norms' and the epilogue's time is outside the call.
    const bool host_norms =
      HtpProfile::global().level() >= 2 && with_norms && !via_dspq;
    if (host_prenorm) {
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(act, act_f32, M, K, eps);
      for (unsigned int r = 0; r < M; ++r)
        for (unsigned int k = 0; k < K; ++k)
          act_f32[static_cast<size_t>(r) * K + k] *= pre_gamma[k];
    } else if (via_dspq && !retained) {
      stagedMemcpy(act_f32, act, act_bytes);
    }
    if (trace_reuse)
      std::fprintf(
        stderr, "[HTP-MOE-INPUT] M=%u K=%u reused=%u bytes=%zu stage_us=%llu\n",
        M, K, reused ? 1u : 0u, act_bytes,
        static_cast<unsigned long long>(HtpProfile::nowUs() - stage_t0));

    // The five small sequences (handles, routing) go from the heap. Tried
    // from one rpcmem buffer (doc 51 section 2.26): transport 627 -> 605
    // us, noise. The gap to the FC calls' ~400 is the call's length, not
    // its arguments -- a call longer than the poll window (htp_backend.cpp,
    // 5 ms) ends in the interrupt-driven wait and pays its wake-up.

    HtpProfile &profile = HtpProfile::global();
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const bool host_post_norm = host_norms && post_gamma != nullptr;
    const bool norm_entry = with_norms && !host_norms;
    // NNTR_HTP_PROFILE=3 runs the call several times on the same input and
    // keeps the fastest. Two runs with no functional change between them
    // differed by 4.5 ms of DSP time (doc 46 section 23.2) -- the
    // compute-bound stage identical, the DDR-touching ones all moving
    // together, so the device's memory system, not the code. Several of the
    // changes on the list are worth 2 ms, which that noise floor swallows.
    // Repeating inside one call puts the comparison at the same temperature
    // and the same contention, and the output is unchanged because the
    // input is.
    const int reps =
      (profile.level() >= 3 && pending_expert_updates_.empty()) ? 5 : 1;
    uint64_t best_elapsed = UINT64_MAX;
    int err = AEE_SUCCESS;
    for (int rep = 0; rep < reps && err == AEE_SUCCESS; ++rep) {
      uint32_t rep_stage[HTP_MOE_N_STAGES] = {0};
      const uint64_t t0 = profile.level() ? HtpProfile::nowUs() : 0;
      const bool attempted_prepared = prepared_cookie != 0;
      if (prepared_cookie) {
        err = nntr_hvx_mm_u8i4_moe_layer_prepared(
          session, prepared_cookie, M, K, inter, N_out, glu, h_gu.data(),
          static_cast<int>(h_gu.size()), h_dn.data(),
          static_cast<int>(h_dn.size()), row_index.data(),
          static_cast<int>(row_index.size()), row_count.data(),
          static_cast<int>(row_count.size()), row_weight.data(),
          static_cast<int>(row_weight.size()), out_f32, out_len,
          timed ? rep_stage : nullptr, timed ? HTP_MOE_N_STAGES : 0);
        prepared_cookie = 0;
      } else {
        err = AEE_EUNSUPPORTED;
      }
      // The prepared kernel rejects unsupported weight kinds before writing
      // output; all such shapes retain the ordinary NPU path below.
      if (err == AEE_EUNSUPPORTED ||
          static_cast<uint32_t>(err) == 0x80000414u) {
        if (attempted_prepared && !direct_in && !reused)
          stagedMemcpy(act_f32, act, act_bytes);
        err = via_dspq
                ? dspqCall(M, K, inter, N_out, h_gu, h_dn, row_index, row_count,
                           row_weight, act_bytes, out_bytes, glu,
                           timed ? rep_stage : nullptr, act_offset, !retained)
              : add_x2 != nullptr && !host_norms
                ? nntr_hvx_mm_u8i4_moe_layer_norm_add(
                    session, M, K, inter, N_out, glu, eps, add_scale, pre_gamma,
                    static_cast<int>(K), post_gamma,
                    post_gamma ? static_cast<int>(N_out) : 0, add_gamma,
                    add_gamma ? static_cast<int>(N_out) : 0, h_gu.data(),
                    static_cast<int>(h_gu.size()), h_dn.data(),
                    static_cast<int>(h_dn.size()), row_index.data(),
                    static_cast<int>(row_index.size()), row_count.data(),
                    static_cast<int>(row_count.size()), row_weight.data(),
                    static_cast<int>(row_weight.size()), act_f32, act_len,
                    x2_f32, out_len, out_f32, out_len)
              : norm_entry
                ? nntr_hvx_mm_u8i4_moe_layer_norm(
                    session, M, K, inter, N_out, glu, eps, pre_gamma,
                    pre_gamma ? static_cast<int>(K) : 0, post_gamma,
                    post_gamma ? static_cast<int>(N_out) : 0, h_gu.data(),
                    static_cast<int>(h_gu.size()), h_dn.data(),
                    static_cast<int>(h_dn.size()), row_index.data(),
                    static_cast<int>(row_index.size()), row_count.data(),
                    static_cast<int>(row_count.size()), row_weight.data(),
                    static_cast<int>(row_weight.size()), act_f32, act_len,
                    out_f32, out_len)
              : timed ? nntr_hvx_mm_u8i4_moe_layer_timed(
                          session, M, K, inter, N_out, glu, h_gu.data(),
                          static_cast<int>(h_gu.size()), h_dn.data(),
                          static_cast<int>(h_dn.size()), row_index.data(),
                          static_cast<int>(row_index.size()), row_count.data(),
                          static_cast<int>(row_count.size()), row_weight.data(),
                          static_cast<int>(row_weight.size()), act_f32, act_len,
                          out_f32, out_len, rep_stage, HTP_MOE_N_STAGES)
                      : nntr_hvx_mm_u8i4_moe_layer(
                          session, M, K, inter, N_out, glu, h_gu.data(),
                          static_cast<int>(h_gu.size()), h_dn.data(),
                          static_cast<int>(h_dn.size()), row_index.data(),
                          static_cast<int>(row_index.size()), row_count.data(),
                          static_cast<int>(row_count.size()), row_weight.data(),
                          static_cast<int>(row_weight.size()), act_f32, act_len,
                          out_f32, out_len);
      }
      const uint64_t elapsed = profile.level() ? HtpProfile::nowUs() - t0 : 0;
      // Fastest wins, stages and all, so the breakdown describes one real call
      // rather than a mix of a fast one and a slow one.
      if (err == AEE_SUCCESS && elapsed < best_elapsed) {
        best_elapsed = elapsed;
        std::memcpy(stage_us, rep_stage, sizeof(stage_us));
      }
    }
    const uint64_t elapsed = (best_elapsed == UINT64_MAX) ? 0 : best_elapsed;
    if (err != AEE_SUCCESS) {
      // 0x8000040E is AEE_EBADPARM, and every length this call passes is
      // derived from the same shapes the kernel checks against -- so by far
      // the likeliest cause is that libnntr_hvx_skel.so on the device is
      // older than this binary and the two disagree on how many stage_us
      // slots there are. build_android.sh does not rebuild the skel; only
      // test/htp/build.sh does, and forgetting that has cost three
      // measurement cycles.
      std::string hint;
      if (static_cast<unsigned>(err) == 0x8000040Eu) {
        hint =
          " (AEE_EBADPARM -- if the shapes are right, rebuild the DSP skel: "
          "test/htp/build.sh, then push libnntr_hvx_skel.so. This binary "
          "expects " +
          std::to_string(static_cast<int>(HTP_MOE_N_STAGES)) +
          " stage_us slots.)";
      }
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_moe_layer_timed"
                          : "nntr_hvx_mm_u8i4_moe_layer") +
        " failed: err=" + std::to_string(err) + hint +
        (via_dspq ? " (via dspq)" : ""));
    }
    if (via_dspq)
      stagedMemcpy(out, out_f32, out_bytes);
    else if (!direct_out)
      copyOutOf(out_pool_, out, out_bytes);
    if (host_post_norm) {
      std::vector<float> raw(out, out + static_cast<size_t>(out_len));
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(raw.data(), out, M, N_out,
                                                   eps);
      for (unsigned int r = 0; r < M; ++r)
        for (unsigned int n = 0; n < N_out; ++n)
          out[static_cast<size_t>(r) * N_out + n] *= post_gamma[n];
    }
    if (host_norms && add_x2) {
      // out = scale * (act + rmsnorm(out + x2) * gamma), as the epilogue.
      for (unsigned int r = 0; r < M; ++r) {
        float *o = out + static_cast<size_t>(r) * N_out;
        const float *x2 = add_x2 + static_cast<size_t>(r) * N_out;
        const float *res = act + static_cast<size_t>(r) * K;
        float ss = 0.0f;
        for (unsigned int n = 0; n < N_out; ++n)
          ss += (o[n] + x2[n]) * (o[n] + x2[n]);
        const float rs = 1.0f / std::sqrt(ss / N_out + eps);
        for (unsigned int n = 0; n < N_out; ++n)
          o[n] = add_scale * (res[n] + (o[n] + x2[n]) * rs *
                                         (add_gamma ? add_gamma[n] : 1.0f));
      }
    }
    dumpMoeCall("moe_layer", act, static_cast<size_t>(act_len), out,
                static_cast<size_t>(out_len), M, K, inter, N_out, kind,
                row_count);
    if (profile.level()) {
      // [#88] The bytes the stub hands the driver outside ION: the 48-byte
      // primitive block (_primIn[12] in generated/nntr_hvx_stub.c) and the
      // five uint32/float sequences that are not the staged activation.
      const size_t in_arg_bytes =
        via_dspq ? static_cast<size_t>(msg_bytes)
                 : 48 + sizeof(uint32_t) *
                          (h_gu.size() + h_dn.size() + row_index.size() +
                           row_count.size() + row_weight.size());
      profile.addInvokeMoeLayer(
        M, K, N_out, elapsed, timed ? stage_us : nullptr, act_stage, out_stage,
        in_arg_bytes, kind, via_dspq, direct_in || direct_out);
    }
    if (timed) {
      // [#87] The per-descriptor trace of the last repeat, for the first
      // NNTR_HTP_DMA_TRACE calls of this bucket. Read now, while the skel's
      // static tables still hold this call; printed now, so the lines sit
      // next to the token they came from.
      const unsigned ordinal = profile.dmaTraceOrdinal(K, N_out, M, kind);
      if (ordinal != 0) {
        std::vector<uint32_t> words(HEXKL_DMA_TRACE_MAX_WORDS);
        uint32_t n_words = 0;
        const int terr = nntr_hvx_moe_dma_trace_read(
          session, words.data(), static_cast<int>(words.size()), &n_words);
        if (terr == AEE_SUCCESS) {
          HtpProfile::dumpDmaTrace(ordinal, M, reps, words.data(), n_words);
        } else {
          std::fprintf(stderr,
                       "[HTP-DMA] call=%u M=%u moe_dma_trace_read err=%d "
                       "(older skel?)\n",
                       ordinal, M, terr);
        }
      }
    }
  }

  bool invokePrefillConvNorm(remote_handle64 session, const ConvHandles &ch,
                             const float *conv_w, const float *act, float *out,
                             float *state, unsigned M, unsigned K, unsigned C,
                             unsigned N_out) {
    PendingPrefillNorm p;
    {
      std::lock_guard<std::mutex> lock(invoke_mutex_);
      if (!pending_prefill_conv_norm_.norm)
        return false;
      p = std::move(pending_prefill_conv_norm_);
      pending_prefill_conv_norm_ = {};
    }
    if (p.rows != M || p.width != K || p.norm != act)
      throw std::runtime_error("HTP deferred prefill Conv norm mismatch");
    const uint64_t out_count = static_cast<uint64_t>(M) * N_out;
    const uint64_t conv_count64 = static_cast<uint64_t>(3u) * C;
    const uint64_t small_count = static_cast<uint64_t>(5u) * C + K;
    if (!conv_w || !out || !state || !N_out || !C ||
        out_count > INT32_MAX / sizeof(float) ||
        small_count > INT32_MAX / sizeof(float))
      throw std::invalid_argument("HTP deferred prefill Conv shape");
    const size_t bytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t out_bytes = static_cast<size_t>(out_count) * sizeof(float);
    const size_t state_count = static_cast<size_t>(2u) * C;
    const size_t state_bytes = state_count * sizeof(float);
    if (prefillNormOverlap(state, state_bytes, p.sum, bytes) ||
        prefillNormOverlap(state, state_bytes, p.norm, bytes) ||
        prefillNormOverlap(state, state_bytes, out, out_bytes))
      throw std::invalid_argument("HTP deferred prefill Conv state alias");
    auto &backend = HtpBackend::global();
    if (!backend.isSharedRange(out, out_bytes) ||
        prefillNormOverlap(out, out_bytes, p.sum, bytes) ||
        prefillNormOverlap(out, out_bytes, p.norm, bytes)) {
      // A graph may use the normalized output in place. Materialize the norm
      // with the existing call before that graph's ordinary Conv path.
      const bool materialized =
        p.residual
          ? add_rmsnorm_fp32(M, K, p.eps, p.x, p.residual, p.gamma.data(),
                             p.sum, p.norm)
          : inferenceMath(InferenceMath::RmsNorm, M, K, K, p.eps, 1.0f, p.x,
                          nullptr, p.gamma.data(), p.norm, nullptr);
      if (!materialized)
        throw std::runtime_error("HTP deferred prefill Conv norm fallback");
      return false;
    }
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const size_t conv_count = static_cast<size_t>(conv_count64);
    const size_t small_bytes = static_cast<size_t>(small_count) * sizeof(float);
    if (!conv_buf_ || conv_buf_->size() < small_bytes)
      conv_buf_ = std::make_unique<HtpRpcBuffer>(small_bytes);
    if (!conv_buf_->data())
      throw std::runtime_error("HTP prefill Conv norm staging allocation");
    float *cw = reinterpret_cast<float *>(conv_buf_->data());
    float *st = cw + conv_count, *gamma = st + state_count;
    std::memcpy(cw, conv_w, conv_count * sizeof(float));
    std::memcpy(st, state, state_count * sizeof(float));
    std::memcpy(gamma, p.gamma.data(), K * sizeof(float));
    HtpProfile &profile = HtpProfile::global();
    const bool timed = profile.level() >= 2;
    uint32_t stages[HTP_MOE_N_STAGES] = {};
    const uint64_t start = HtpProfile::nowUs();
    const int count = static_cast<int>(M * K);
    const int err = nntr_hvx_mm_u8i4_conv_block_prefill_norm(
      session, M, K, C, N_out, ch.h_in.data(), ch.h_in.size(), ch.h_out, cw,
      conv_count, p.eps, p.x, count, p.residual, p.residual ? count : 0, gamma,
      K, p.sum, p.residual ? count : 0, p.norm, count, out,
      static_cast<int>(out_count), st, state_count, timed ? stages : nullptr,
      timed ? HTP_MOE_N_STAGES : 0);
    const uint64_t elapsed = HtpProfile::nowUs() - start;
    if (err != AEE_SUCCESS)
      throw std::runtime_error("HTP prefill Conv norm failed: " +
                               std::to_string(err));
    std::memcpy(state, st, state_count * sizeof(float));
    ++consumer_norm_calls_;
    ++shared_io_calls_[2];
    shared_io_bytes_[2] += bytes * (p.residual ? 4u : 2u) + out_bytes;
    if (profile.level()) {
      profile.addInvokeMoeLayer(M, K, N_out, elapsed, timed ? stages : nullptr,
                                *conv_buf_, *conv_buf_, 64, 2, false, true);
      std::fprintf(stderr,
                   "[HTP-CONV-NORM] M=%u K=%u rpc_us=%llu dsp_us=%u "
                   "shared_bytes=%zu\n",
                   M, K, (unsigned long long)elapsed, timed ? stages[0] : 0u,
                   bytes * (p.residual ? 4u : 2u) + out_bytes);
    }
    return true;
  }

  /** @brief [doc 51 section 2] One call for a whole conv block.
   *
   *  The activation goes over once and the output comes back once; the
   *  four M x C intermediates the block has stay on the DSP. HVX decode
   *  uses the persistent DSP queue by default (NNTR_HTP_DSPQ_CONV=0 opts
   *  out); prefill and HMX stay on FastRPC. The queue packs conv_w and the
   *  incoming state after the activation, and returns state after output
   *  and optional norm results. NNTR_HTP_CONV_INPUT_CACHE=1 sends only changed
   *  conv_w, gamma and state; content comparisons preserve the in/out API.
   *  FastRPC uses a separate small state buffer
   *  so output and state cannot alias the same out_pool_ size class. */
  void invokeConvBlock(remote_handle64 session, const ConvHandles &ch,
                       const float *conv_w, const float *act, float *out,
                       float *state, unsigned int M, unsigned int K,
                       unsigned int C, unsigned int N_out) {
    sendMoeOptsOnce(session);
    if (invokePrefillConvNorm(session, ch, conv_w, act, out, state, M, K, C,
                              N_out))
      return;
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N_out);
    const int conv_len = 3 * static_cast<int>(C);
    const int state_len = 2 * static_cast<int>(C);

    // Separate pools keep cache-policy experiments local to ConvBlock:
    // changing the shared FC/MoE pools would confound the comparison.
    static const bool uncached = [] {
      const char *e = std::getenv("NNTR_CONV_RPC_UNCACHED");
      return e && *e && *e != '0';
    }();
    static const bool trace = [] {
      const char *e = std::getenv("NNTR_CONV_BLOCK_TRACE");
      return e && *e && *e != '0';
    }();
    const uint32_t flags =
      uncached ? HTP_RPC_FLAGS_UNCACHED : HTP_RPC_FLAGS_DEFAULT;
    HtpProfile &profile = HtpProfile::global();
    const bool measure = trace || profile.level() != 0;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    PendingNorm *pending = pendingNorm("conv", act, M, K);
    const size_t input_bytes = pending
                                 ? pending->inputs.size() * sizeof(float)
                                 : static_cast<size_t>(act_len) * sizeof(float);

    const size_t small_bytes =
      static_cast<size_t>(conv_len + state_len) * sizeof(float);
    const bool timed = profile.level() >= 2;
    htp_dspq_conv_req req = {HTP_DSPQ_OP_CONV,
                             0u,
                             timed ? HTP_DSPQ_FLAG_TIMED : 0u,
                             K,
                             C,
                             N_out,
                             {ch.h_in[0], ch.h_in[1], ch.h_in[2]},
                             ch.h_out,
                             pending ? (pending->add ? 2u : 1u) : 0u,
                             pending ? pending->eps : 0.0f};
    const size_t full_queue_in = htp_dspq_conv_input_bytes(&req);
    size_t queue_in = full_queue_in;
    const size_t queue_out = htp_dspq_conv_output_bytes(&req);
    // Read per call so diagnostics can compare the two transports with the
    // same session and weights. The kernel options themselves are immutable.
    const char *queue_env = std::getenv("NNTR_HTP_DSPQ_CONV");
    const bool via_dspq = M == 1 && !uncached &&
                          (!queue_env || std::atoi(queue_env) != 0) &&
                          (applied_moe_flags_ & HTP_CONV_FLAG_M1_HVX) &&
                          htp_dspq_conv_shape_valid(&req) &&
                          dspqReady(session, queue_in, queue_out, sizeof(req));
    htp_dspq_conv_cached_req cached_req = {req, 0u, 0u};
    DspqMoe::ConvInputCache *cached = nullptr;
    const char *cache_env = std::getenv("NNTR_HTP_CONV_INPUT_CACHE");
    if (via_dspq && cache_env && std::atoi(cache_env) != 0) {
      auto &entries = dspq_->conv_inputs;
      auto it = entries.find(state);
      if (it == entries.end() && entries.size() < HTP_DSPQ_CONV_CACHE_SLOTS) {
        const uint32_t slot = static_cast<uint32_t>(entries.size());
        it = entries.emplace(state, DspqMoe::ConvInputCache{}).first;
        it->second.slot = slot;
      }
      if (it != entries.end()) {
        cached = &it->second;
        if (!cached->valid ||
            !htp_dspq_conv_binding_equal(&cached->binding, &req)) {
          cached->valid = false;
          cached->gamma.clear();
        }
        cached_req.conv.op = HTP_DSPQ_OP_CONV_CACHED;
        cached_req.slot = cached->slot;
        const auto differs = [&](const std::vector<float> &snapshot,
                                 const float *data, size_t count) {
          return !cached->valid || snapshot.size() != count ||
                 std::memcmp(snapshot.data(), data, count * sizeof(float)) != 0;
        };
        if (differs(cached->conv, conv_w, conv_len))
          cached_req.updates |= HTP_DSPQ_CONV_UPDATE_W;
        if (pending &&
            differs(cached->gamma,
                    pending->inputs.data() + input_bytes / 4u - K, K))
          cached_req.updates |= HTP_DSPQ_CONV_UPDATE_GAMMA;
        if (differs(cached->state, state, state_len))
          cached_req.updates |= HTP_DSPQ_CONV_UPDATE_STATE;
        queue_in = htp_dspq_conv_cached_input_bytes(&cached_req);
      }
    }
    HtpRpcBuffer &act_stage =
      via_dspq
        ? *dspq_->act
        : stage(uncached ? conv_act_pool_ : act_pool_, input_bytes, flags);
    HtpRpcBuffer &out_stage =
      via_dspq
        ? *dspq_->out
        : stage(uncached ? conv_out_pool_ : out_pool_,
                (static_cast<size_t>(out_len) + (pending ? 2u * K : 0u)) *
                  sizeof(float),
                flags);
    const bool shared_io = M > 1 && sharedPrefillIo(4u) && !uncached;
    const bool direct_in = shared_io && !pending &&
                           HtpBackend::global().isSharedRange(act, input_bytes);
    const bool direct_out =
      shared_io && !pending &&
      HtpBackend::global().isSharedRange(out, static_cast<size_t>(out_len) *
                                                sizeof(float));
    if (direct_in || direct_out) {
      ++shared_io_calls_[2];
      shared_io_bytes_[2] +=
        (direct_in ? input_bytes : 0u) +
        (direct_out ? static_cast<size_t>(out_len) * sizeof(float) : 0u);
    }
    float *act_f32 = direct_in ? const_cast<float *>(act)
                               : reinterpret_cast<float *>(act_stage.data());
    float *out_f32 =
      direct_out ? out : reinterpret_cast<float *>(out_stage.data());
    float *conv_f32 = nullptr, *state_in = nullptr, *state_f32;
    if (via_dspq) {
      req.seq = ++dspq_->seq;
      cached_req.conv.seq = req.seq;
      dspq_->retained_norm = nullptr;
      conv_f32 = act_f32 + input_bytes / sizeof(float);
      state_in = conv_f32 + conv_len;
      state_f32 = out_f32 + out_len + (pending ? 2u * K : 0u);
    } else {
      if (!conv_buf_ || conv_buf_->size() < small_bytes)
        conv_buf_ = std::make_unique<HtpRpcBuffer>(small_bytes, flags);
      conv_f32 = reinterpret_cast<float *>(conv_buf_->data());
      state_in = state_f32 = conv_f32 + conv_len;
    }
    const uint64_t copy_start = trace ? HtpProfile::nowUs() : 0;
    if (cached) {
      const size_t dynamic_len = (req.norm_kind == 2u ? 2u : 1u) * K;
      stagedMemcpy(act_f32, pending ? pending->inputs.data() : act,
                   dynamic_len * sizeof(float));
      float *dst = act_f32 + dynamic_len;
      if (cached_req.updates & HTP_DSPQ_CONV_UPDATE_W) {
        std::memcpy(dst, conv_w, conv_len * sizeof(float));
        dst += conv_len;
      }
      if (cached_req.updates & HTP_DSPQ_CONV_UPDATE_GAMMA) {
        std::memcpy(dst, pending->inputs.data() + dynamic_len,
                    K * sizeof(float));
        dst += K;
      }
      if (cached_req.updates & HTP_DSPQ_CONV_UPDATE_STATE)
        std::memcpy(dst, state, state_len * sizeof(float));
    } else {
      if (!direct_in)
        stagedMemcpy(act_f32, pending ? pending->inputs.data() : act,
                     input_bytes);
      std::memcpy(conv_f32, conv_w,
                  static_cast<size_t>(conv_len) * sizeof(float));
      std::memcpy(state_in, state,
                  static_cast<size_t>(state_len) * sizeof(float));
    }

    const uint64_t input_us = trace ? HtpProfile::nowUs() - copy_start : 0;
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    const uint64_t t0 = measure ? HtpProfile::nowUs() : 0;
    const int err =
      cached ? dspqExchange(&cached_req, sizeof(cached_req), req.seq, queue_in,
                            queue_out, timed ? stage_us : nullptr)
      : via_dspq ? dspqExchange(&req, sizeof(req), req.seq, queue_in, queue_out,
                                timed ? stage_us : nullptr)
      : pending
        ? nntr_hvx_mm_u8i4_conv_block_norm(
            session, K, C, N_out, ch.h_in.data(),
            static_cast<int>(ch.h_in.size()), ch.h_out, conv_f32, conv_len,
            pending->add, pending->eps, act_f32, pending->inputs.size(),
            out_f32 + out_len, 2u * K, out_f32, out_len, state_f32, state_len,
            timed ? stage_us : nullptr, timed ? HTP_MOE_N_STAGES : 0)
      : timed
        ? nntr_hvx_mm_u8i4_conv_block_timed(
            session, M, K, C, N_out, ch.h_in.data(),
            static_cast<int>(ch.h_in.size()), ch.h_out, conv_f32, conv_len,
            act_f32, act_len, out_f32, out_len, state_f32, state_len, stage_us,
            HTP_MOE_N_STAGES)
        : nntr_hvx_mm_u8i4_conv_block(
            session, M, K, C, N_out, ch.h_in.data(),
            static_cast<int>(ch.h_in.size()), ch.h_out, conv_f32, conv_len,
            act_f32, act_len, out_f32, out_len, state_f32, state_len);
    const uint64_t elapsed = measure ? HtpProfile::nowUs() - t0 : 0;
    if (err != AEE_SUCCESS) {
      if (cached)
        cached->valid = false;
      std::string hint;
      if (static_cast<unsigned>(err) == 0x8000040Eu) {
        hint = " (AEE_EBADPARM -- if the shapes are right, the DSP skel on "
               "the device predates mm_u8i4_conv_block: rebuild it with "
               "test/htp/build.sh and push libnntr_hvx_skel.so)";
      }
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_conv_block_timed"
                          : "nntr_hvx_mm_u8i4_conv_block") +
        " failed: err=" + std::to_string(err) + " M=" + std::to_string(M) +
        " K=" + std::to_string(K) + " C=" + std::to_string(C) +
        " N=" + std::to_string(N_out) + hint);
    }
    const uint64_t output_start = trace ? HtpProfile::nowUs() : 0;
    if (cached) {
      if (cached_req.updates & HTP_DSPQ_CONV_UPDATE_W)
        cached->conv.assign(conv_w, conv_w + conv_len);
      if (cached_req.updates & HTP_DSPQ_CONV_UPDATE_GAMMA)
        cached->gamma.assign(pending->inputs.end() - K, pending->inputs.end());
      cached->state.assign(state_f32, state_f32 + state_len);
      cached->binding = req;
      cached->valid = true;
      ++dspq_->conv_cached_calls;
      dspq_->conv_saved_bytes += full_queue_in - queue_in;
      dspq_->conv_w_updates += !!(cached_req.updates & HTP_DSPQ_CONV_UPDATE_W);
      dspq_->conv_gamma_updates +=
        !!(cached_req.updates & HTP_DSPQ_CONV_UPDATE_GAMMA);
      dspq_->conv_state_updates +=
        !!(cached_req.updates & HTP_DSPQ_CONV_UPDATE_STATE);
    }
    if (pending)
      publishNorm(out_f32 + out_len);
    if (!direct_out)
      stagedMemcpy(out, out_f32, static_cast<size_t>(out_len) * sizeof(float));
    std::memcpy(state, state_f32,
                static_cast<size_t>(state_len) * sizeof(float));
    if (trace) {
      std::fprintf(stderr,
                   "[HTP-CONV] M=%u K=%u C=%u N=%u uncached=%d ion=%d "
                   "input_us=%llu rpc_us=%llu dsp_us=%u output_us=%llu via=%s "
                   "cache=%d updates=%u in_bytes=%zu out_bytes=%zu\n",
                   M, K, C, N_out, uncached,
                   act_stage.isIon() && out_stage.isIon() &&
                     (via_dspq || conv_buf_->isIon()),
                   (unsigned long long)input_us, (unsigned long long)elapsed,
                   timed ? stage_us[0] : 0,
                   (unsigned long long)(HtpProfile::nowUs() - output_start),
                   via_dspq ? "dspq" : "fastrpc", cached != nullptr,
                   cached ? cached_req.updates : 0u,
                   via_dspq ? queue_in : input_bytes + small_bytes,
                   via_dspq ? queue_out
                            : (out_len + (pending ? 2u * K : 0u) + state_len) *
                                sizeof(float));
    }
    if (profile.level()) {
      // [#88] Outside ION: the primitive block and the three handles;
      // conv_w and the state ride in conv_buf_ (ION).
      const size_t in_arg_bytes =
        cached     ? sizeof(cached_req)
        : via_dspq ? sizeof(req)
                   : (timed ? 52 : 48) + sizeof(uint32_t) * ch.h_in.size();
      profile.addInvokeMoeLayer(
        M, K, N_out, elapsed, timed ? stage_us : nullptr, act_stage, out_stage,
        in_arg_bytes, /*kind=*/2, via_dspq, direct_in || direct_out);
    }
  }

  /** @brief [L2, split-call variant] gate_up matmul -> SwiGLU -> requantize
   *  to u8 AH, ONE weight -- doc 43 §7's smaller-surface alternative to
   *  invokeFused. Leaves its result in act_ah_buf_/act_scale_scratch_/
   *  act_zp_scratch_ for invokeLayerU8InRaw (below) to consume immediately
   *  after -- both run under gemm_qs4cx_fused_swiglu_fp32's call, so there
   *  is no other caller between them to race with. */
  void invokeGateUpSwiglu(remote_handle64 session, uint32_t handle_gate_up,
                          const float *matBdata, unsigned int M, unsigned int K,
                          unsigned int inter) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const uint32_t m_pad = htp_act_m_pad(M);
    const size_t out_ah_bytes = static_cast<size_t>(m_pad) * inter;

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    ensureCapacity(act_ah_buf_, out_ah_bytes);
    if (act_scale_scratch_.size() < m_pad) {
      act_scale_scratch_.resize(m_pad);
      act_zp_scratch_.resize(m_pad);
    }
    float *act_f32 = reinterpret_cast<float *>(
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float)).data());
    uint8_t *out_ah = act_ah_buf_->data();
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_gate_up_swiglu(
        session, M, K, handle_gate_up, act_f32, act_len, out_ah,
        static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad), act_zp_scratch_.data(),
        static_cast<int>(m_pad));
      if (err != AEE_SUCCESS) {
        throw std::runtime_error(
          "nntr_hvx_mm_u8i4_gate_up_swiglu failed: err=" + std::to_string(err));
      }
      if (l2CheckEnabled())
        l2CheckFinite("gate_up_swiglu out_scale", act_scale_scratch_.data(),
                      m_pad, M, K, 2 * inter);
      return;
    }

    uint32_t stage_us[HTP_GU_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_gate_up_swiglu_timed(
                session, M, K, handle_gate_up, act_f32, act_len, out_ah,
                static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad), act_zp_scratch_.data(),
                static_cast<int>(m_pad), stage_us, HTP_GU_N_STAGES)
            : nntr_hvx_mm_u8i4_gate_up_swiglu(
                session, M, K, handle_gate_up, act_f32, act_len, out_ah,
                static_cast<int>(out_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad), act_zp_scratch_.data(),
                static_cast<int>(m_pad));
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_gate_up_swiglu_timed"
                          : "nntr_hvx_mm_u8i4_gate_up_swiglu") +
        " failed: err=" + std::to_string(err));
    }
    if (l2CheckEnabled())
      l2CheckFinite("gate_up_swiglu out_scale", act_scale_scratch_.data(),
                    m_pad, M, K, 2 * inter);
    profile.addInvokeGateUpSwiglu(M, K, 2 * inter, elapsed,
                                  timed ? stage_us : nullptr);
  }

  /** @brief Down matmul via the u8in path, fed act_ah/scale/zp that are
   *  ALREADY quantized (by invokeGateUpSwiglu, just above) -- unlike
   *  invokeLayerU8In, this does not call htp_quant_pack_u8_ah itself; there
   *  is nothing f32 left to quantize, the DSP produced the bytes directly. */
  void invokeLayerU8InRaw(remote_handle64 session, const uint32_t *handles,
                          int num_handles, unsigned int m_pad_in,
                          float *matCdata, unsigned int M, unsigned int N,
                          unsigned int K) {
    const size_t act_ah_bytes = static_cast<size_t>(m_pad_in) * K;
    const int out_len = static_cast<int>(M) * static_cast<int>(N);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const uint8_t *act_ah = act_ah_buf_->data();
    float *out_cat = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err = nntr_hvx_mm_u8i4_layer_u8in(
        session, M, K, handles, num_handles, act_ah,
        static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
        static_cast<int>(m_pad_in), act_zp_scratch_.data(),
        static_cast<int>(m_pad_in), out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer_u8in failed: err=" +
                                 std::to_string(err));
      }
      if (l2CheckEnabled())
        l2CheckFinite("down out", out_cat, static_cast<size_t>(out_len), M, K,
                      N);
      stagedMemcpy(matCdata, out_cat,
                   static_cast<size_t>(out_len) * sizeof(float));
      return;
    }

    uint32_t stage_us[HTP_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = HtpProfile::nowUs();
    const int err =
      timed ? nntr_hvx_mm_u8i4_layer_u8in_timed(
                session, M, K, handles, num_handles, act_ah,
                static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad_in), act_zp_scratch_.data(),
                static_cast<int>(m_pad_in), out_cat, out_len, stage_us,
                HTP_N_STAGES)
            : nntr_hvx_mm_u8i4_layer_u8in(
                session, M, K, handles, num_handles, act_ah,
                static_cast<int>(act_ah_bytes), act_scale_scratch_.data(),
                static_cast<int>(m_pad_in), act_zp_scratch_.data(),
                static_cast<int>(m_pad_in), out_cat, out_len);
    const uint64_t elapsed = HtpProfile::nowUs() - t0;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error(
        std::string(timed ? "nntr_hvx_mm_u8i4_layer_u8in_timed"
                          : "nntr_hvx_mm_u8i4_layer_u8in") +
        " failed: err=" + std::to_string(err));
    }
    if (l2CheckEnabled())
      l2CheckFinite("down out", out_cat, static_cast<size_t>(out_len), M, K, N);
    stagedMemcpy(matCdata, out_cat,
                 static_cast<size_t>(out_len) * sizeof(float));
    profile.addInvoke(M, K, N, elapsed, timed ? stage_us : nullptr);
  }

  uint32_t get_or_register(void *matAdata, remote_handle64 session, uint32_t K,
                           uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    return get_or_register_unlocked(matAdata, session, K, N);
  }

  /** @note Call with handle_mutex_ already held. */
  uint32_t get_or_register_unlocked(void *matAdata, remote_handle64 session,
                                    uint32_t K, uint32_t N) {
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

    const uint64_t t_begin = HtpProfile::nowUs();
    HtpRpcBuffer q_w4_i8(static_cast<size_t>(K) * N);
    std::vector<float> w_scale(N);
    std::vector<int32_t> colsum_w(N);
    const uint64_t t_convert = HtpProfile::nowUs();
    htp_qs4cx_from_q4_0x4(matAdata, K, N,
                          reinterpret_cast<int8_t *>(q_w4_i8.data()),
                          w_scale.data(), colsum_w.data());
    const uint64_t convert_us = HtpProfile::nowUs() - t_convert;

    return register_locked(matAdata, session, K, N, q_w4_i8, w_scale, colsum_w,
                           t_begin, convert_us);
  }

  /** @brief The handles one FC weight is registered as, in column order. */
  struct FcHandles {
    std::vector<uint32_t> handles;
    std::vector<unsigned int> cols; /**< N of each handle */
  };

  /** Direct FC output must not overwrite a later row chunk's input. */
  static bool fcDisjointIo(const float *input, const float *output, unsigned M,
                           unsigned N, unsigned K) {
    const size_t max_floats =
      std::numeric_limits<size_t>::max() / sizeof(float);
    if (!input || !output || !M || !N || !K || M > max_floats / K ||
        M > max_floats / N)
      return false;
    const size_t input_bytes = static_cast<size_t>(M) * K * sizeof(float);
    const size_t output_bytes = static_cast<size_t>(M) * N * sizeof(float);
    const uintptr_t a = reinterpret_cast<uintptr_t>(input);
    const uintptr_t b = reinterpret_cast<uintptr_t>(output);
    const uintptr_t limit = std::numeric_limits<uintptr_t>::max();
    if (a > limit - input_bytes || b > limit - output_bytes)
      return false;
    return a <= b ? b - a >= input_bytes : a - b >= output_bytes;
  }

  /** @brief One FC weight's layer calls, in fcMaxRows row chunks. */
  void invokeFc(remote_handle64 session, const FcHandles &fh, float *matBdata,
                float *matCdata, unsigned int M, unsigned int N, unsigned int K,
                bool retain_moe_input = false) {
    const bool allow_strided_output =
      M > 1 && sharedPrefillIo(2u) && fcDisjointIo(matBdata, matCdata, M, N, K);
    const unsigned int step = fcMaxRows(K);
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      const unsigned int m = std::min(step, M - m0);
      invokeLayer(
        session, fh.handles.data(), static_cast<int>(fh.handles.size()),
        matBdata + static_cast<size_t>(m0) * K,
        matCdata + static_cast<size_t>(m0) * N, m, N, K, &fh.cols, nullptr,
        nullptr, nullptr, retain_moe_input && M <= step, allow_strided_output);
    }
  }

  /** @brief Columns of a K-deep weight per registered handle.
   *
   * hexkl_mm_u8i4_layer_run lays VTCM out as the whole activation
   * (m_pad x K) | the WIDEST handle's WH bytes twice (its double buffer) |
   * a result tile, and returns AEE_ENOMEMORY when that exceeds the 8 MiB
   * (doc 50 section 3: conv in_proj, 2048 x 6144 = 6 MiB, wanted 12). A
   * 2 MiB slice keeps the double buffer at 4 MiB, which leaves the
   * activation room up to m_pad ~ 1,900 rows at K = 2048.
   * ponytail: the cap is the activation's ceiling, not the kernel's; a
   * prompt past it needs the kernel to walk 64-row blocks the way
   * hexkl_mm_u8i4_moe.c does. */
  static unsigned int fcSliceCols(unsigned int K) {
    // A slice holds whole heads, so a per-head norm folded into the call
    // (mm_u8i4_layer_norm's post chunks) never straddles two handles: 512
    // columns is the widest head dim here and divides every smaller one.
    // ponytail: a head dim that does not divide 512 needs its own
    // alignment passed down, not this constant.
    return preparedFcSliceCols(K);
  }

  /** @brief get_or_register for an FC weight of any width: one handle when
   *  it fits fcSliceCols, else one per column slice, converted once and
   *  registered slice by slice under keys inside the weight (its data
   *  pointer plus the slice's first column -- distinct, and valid as long
   *  as the weight is). Cached by the weight pointer like every handle.
   *  @param qs4cx_scale non-null: @a matAdata is a QS4CX weight with these
   *         per-channel scales (quantized offline, no Q4_0 detour); null:
   *         Q4_0x4 bytes. */
  const FcHandles &get_or_register_fc(void *matAdata, remote_handle64 session,
                                      uint32_t K, uint32_t N,
                                      const float *qs4cx_scale = nullptr,
                                      bool weights_wh = false) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = fc_cache_.find(matAdata);
    if (it != fc_cache_.end())
      return it->second;

    FcHandles fh;
    const unsigned int cap = fcSliceCols(K);
    if (N <= cap && !weights_wh) {
      fh.handles.push_back(
        qs4cx_scale
          ? get_or_register_qs4cx_unlocked(matAdata, qs4cx_scale, session, K, N)
          : get_or_register_unlocked(matAdata, session, K, N));
      fh.cols.push_back(N);
    } else {
      std::vector<int8_t> full(static_cast<size_t>(K) * N);
      std::vector<float> w_scale(N);
      std::vector<int32_t> colsum_w(N);
      // The first slice's profile entry carries the conversion, so its
      // clock starts before it; the later slices' start with their copy.
      uint64_t t_begin = HtpProfile::nowUs();
      if (weights_wh) {
        // Preserve the quantized values while slicing wide FC weights to
        // fit VTCM. WH column sums follow the scales in the model file.
        whUnpack(static_cast<const uint8_t *>(matAdata), K, N, full.data());
        for (uint32_t n = 0; n < N; ++n) {
          w_scale[n] = qs4cx_scale[n];
          colsum_w[n] = static_cast<int32_t>(qs4cx_scale[N + n]);
        }
      } else if (qs4cx_scale)
        htp_qs4cx_from_packed(matAdata, qs4cx_scale, K, N, full.data(),
                              w_scale.data(), colsum_w.data());
      else
        htp_qs4cx_from_q4_0x4(matAdata, K, N, full.data(), w_scale.data(),
                              colsum_w.data());
      uint64_t convert_us = HtpProfile::nowUs() - t_begin;
      for (uint32_t c0 = 0; c0 < N; c0 += cap) {
        const uint32_t n = std::min<uint32_t>(cap, N - c0);
        if (c0 != 0)
          t_begin = HtpProfile::nowUs();
        std::vector<int8_t> rm(static_cast<size_t>(K) * n);
        for (uint32_t k = 0; k < K; ++k) {
          std::memcpy(rm.data() + static_cast<size_t>(k) * n,
                      full.data() + static_cast<size_t>(k) * N + c0, n);
        }
        std::vector<float> ws(w_scale.begin() + c0, w_scale.begin() + c0 + n);
        std::vector<int32_t> cs(colsum_w.begin() + c0,
                                colsum_w.begin() + c0 + n);
        fh.handles.push_back(registerRm(static_cast<char *>(matAdata) + c0,
                                        session, rm.data(), K, n, ws, cs,
                                        t_begin, convert_us));
        fh.cols.push_back(n);
        convert_us = 0; // counted once, on the first slice
      }
    }
    return fc_cache_.emplace(matAdata, std::move(fh)).first->second;
  }

  /** @brief Registers one row-major int8 [K x N] weight (values in [-8, 7],
   *  the form htp_qs4cx_from_* produce) with its scales and column sums.
   *
   * [doc 50 section 3.3] Into a mapped arena chunk's free room first: the
   * 3840 MiB mapped hold 3696 of MoE weights, and the DSP heap gave the
   * loaded app ~100 MiB before AEE_ENOMEMORY. Only room that is already
   * mapped -- a new chunk would take the address space the heap
   * registrations after this one need. The arena wants WH bytes: packed
   * on the host into a cached buffer and copied in whole, since whPack's
   * read-modify-write into the uncached chunk would crawl. When no mapped
   * room is left, the DSP heap, as a whole weight would have gone. A
   * refusal by the DSP leaves the arena bytes unused, once.
   * @param key what handle_cache_ files the handle under: an address inside
   *            the weight this is a slice of, distinct per slice
   * @note  Call with handle_mutex_ already held. */
  uint32_t registerRm(void *key, remote_handle64 session, const int8_t *rm,
                      uint32_t K, uint32_t N, std::vector<float> &ws,
                      std::vector<int32_t> &cs, uint64_t t_begin,
                      uint64_t convert_us) {
    const uint32_t wh_len = static_cast<uint32_t>(whBytes(K, N));
    uint32_t chunk = 0, off = 0;
    if (ensureArena(session) && placeExisting(wh_len, &chunk, &off)) {
      std::vector<uint8_t> wh(wh_len);
      whPack(rm, K, N, wh.data());
      std::memcpy(arena_chunks_[chunk].buf->data() + off, wh.data(), wh_len);
      ArenaEntry e;
      e.chunk = chunk;
      e.off = off;
      e.K = K;
      e.N = N;
      e.w_scale = ws;
      e.colsum_w = cs;
      e.bias.assign(N, 0.0f);
      const uint32_t handle = registerFromArena(session, e, K, N, t_begin);
      if (handle != kNoHandle) {
        handle_cache_.emplace(key, handle);
        return handle;
      }
    }
    HtpRpcBuffer buf(static_cast<size_t>(K) * N);
    std::memcpy(buf.data(), rm, static_cast<size_t>(K) * N);
    return register_locked(key, session, K, N, buf, ws, cs, t_begin,
                           convert_us);
  }

  /** @brief The handles one dense FFN is registered as: chunk c of the
   *  intermediate dimension is the expert pair (gate_up [K x 2w] with the
   *  gate columns first, down [w x N]) of the MoE layer kernel. */
  struct DenseHandles {
    std::vector<uint32_t> h_gu, h_dn;
    unsigned int w = 0; /**< columns of I per chunk */
  };

  /** @brief Columns of the intermediate dimension per chunk: the largest
   *  divisor of I that is a multiple of 32 and at most 1792 -- this
   *  model's expert width, so the MoE layer kernel lays VTCM out exactly
   *  as it does for the experts (gate_up 3.5 MiB, down 1.75). 0 if none. */
  static unsigned int denseChunkCols(unsigned int I) {
    for (unsigned int w = std::min(I, 1792u); w >= 32u; w -= 32u) {
      if (I % w == 0)
        return w;
    }
    return 0;
  }

  /** @brief Converts and registers a dense FFN's three Q4_0x4 weights as
   *  I / w expert pairs (doc 51). The gate_up chunk takes column slices of
   *  gate and up, so its column sums are the full ones; the down chunk
   *  takes a row slice, so its column sums are recomputed over those rows
   *  -- the kernel's zero-point correction is per chunk. Cached by the up
   *  weight's pointer. */
  const DenseHandles &get_or_register_dense(void *up, void *gate, void *down,
                                            remote_handle64 session, uint32_t K,
                                            uint32_t I, uint32_t N,
                                            const float *up_s = nullptr,
                                            const float *gate_s = nullptr,
                                            const float *down_s = nullptr) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = dense_cache_.find(up);
    if (it != dense_cache_.end())
      return it->second;

    const unsigned int w = denseChunkCols(I);
    if (w == 0) {
      throw std::invalid_argument(
        "gemm_q4_0_dense_ffn_fp32: intermediate size " + std::to_string(I) +
        " has no chunk width that is a multiple of 32");
    }
    uint64_t t_begin = HtpProfile::nowUs();
    std::vector<int8_t> up_rm(static_cast<size_t>(K) * I),
      gate_rm(static_cast<size_t>(K) * I), down_rm(static_cast<size_t>(I) * N);
    std::vector<float> up_s_v(I), gate_s_v(I), down_s_v(N);
    std::vector<int32_t> up_c(I), gate_c(I), down_c(N);
    // QS4CX weights (scales given) are a bit rearrangement, as the FC
    // path's; Q4_0 ones a requantization.
    if (up_s) {
      htp_qs4cx_from_packed(up, up_s, K, I, up_rm.data(), up_s_v.data(),
                            up_c.data());
      htp_qs4cx_from_packed(gate, gate_s, K, I, gate_rm.data(), gate_s_v.data(),
                            gate_c.data());
      htp_qs4cx_from_packed(down, down_s, I, N, down_rm.data(), down_s_v.data(),
                            down_c.data());
    } else {
      htp_qs4cx_from_q4_0x4(up, K, I, up_rm.data(), up_s_v.data(), up_c.data());
      htp_qs4cx_from_q4_0x4(gate, K, I, gate_rm.data(), gate_s_v.data(),
                            gate_c.data());
      htp_qs4cx_from_q4_0x4(down, I, N, down_rm.data(), down_s_v.data(),
                            down_c.data());
    }
    uint64_t convert_us = HtpProfile::nowUs() - t_begin;

    DenseHandles dh;
    dh.w = w;
    for (uint32_t c0 = 0; c0 < I; c0 += w) {
      // gate_up chunk: [K x 2w], gate columns then up columns -- the pair
      // layout hexkl_mm_u8i4_moe.c's epilogue reads (gate j with up w+j).
      std::vector<int8_t> gu(static_cast<size_t>(K) * 2 * w);
      for (uint32_t k = 0; k < K; ++k) {
        std::memcpy(gu.data() + static_cast<size_t>(k) * 2 * w,
                    gate_rm.data() + static_cast<size_t>(k) * I + c0, w);
        std::memcpy(gu.data() + static_cast<size_t>(k) * 2 * w + w,
                    up_rm.data() + static_cast<size_t>(k) * I + c0, w);
      }
      std::vector<float> gus(gate_s_v.begin() + c0, gate_s_v.begin() + c0 + w);
      gus.insert(gus.end(), up_s_v.begin() + c0, up_s_v.begin() + c0 + w);
      std::vector<int32_t> guc(gate_c.begin() + c0, gate_c.begin() + c0 + w);
      guc.insert(guc.end(), up_c.begin() + c0, up_c.begin() + c0 + w);
      if (c0 != 0)
        t_begin = HtpProfile::nowUs();
      dh.h_gu.push_back(registerRm(static_cast<char *>(gate) + c0, session,
                                   gu.data(), K, 2 * w, gus, guc, t_begin,
                                   convert_us));
      convert_us = 0;

      // down chunk: rows c0 .. c0 + w, contiguous in the row-major weight.
      const int8_t *dn = down_rm.data() + static_cast<size_t>(c0) * N;
      std::vector<int32_t> dnc(N, 0);
      for (uint32_t r = 0; r < w; ++r) {
        for (uint32_t n = 0; n < N; ++n)
          dnc[n] += dn[static_cast<size_t>(r) * N + n];
      }
      std::vector<float> dns(down_s_v);
      dh.h_dn.push_back(registerRm(static_cast<char *>(down) + c0, session, dn,
                                   w, N, dns, dnc, HtpProfile::nowUs(), 0));
    }
    return dense_cache_.emplace(up, std::move(dh)).first->second;
  }

  bool supports_gemm_q4_0_dense_ffn_fp32() const override { return true; }

  void gemm_q4_0_dense_ffn_fp32(void *up, void *gate, void *down,
                                const float *act, float *out, unsigned int M,
                                unsigned int K, unsigned int I, unsigned int N,
                                bool gelu, const float *pre_gamma,
                                const float *post_gamma, float eps,
                                const float *up_scale, const float *gate_scale,
                                const float *down_scale) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const DenseHandles &dh = get_or_register_dense(
      up, gate, down, session, K, I, N, up_scale, gate_scale, down_scale);
    // Every chunk is an "expert" that takes every row with weight 1: the
    // kernel zero-fills out and scatter-adds each chunk's down result into
    // it, which is exactly the sum over the intermediate dimension.
    const size_t chunks = dh.h_gu.size();
    std::vector<unsigned int> row_index(chunks * M), row_count(chunks, M);
    std::vector<float> row_weight(chunks * M, 1.0f);
    for (size_t c = 0; c < chunks; ++c) {
      for (unsigned int r = 0; r < M; ++r)
        row_index[c * M + r] = r;
    }
    invokeMoeLayer(session, dh.h_gu, dh.h_dn, row_index, row_count, row_weight,
                   act, out, M, K, dh.w, N, /*kind=*/1, gelu ? 1u : 0u,
                   pre_gamma, post_gamma, eps);
  }

  bool register_q4_0_dense_ffn(void *up, void *gate, void *down, unsigned int K,
                               unsigned int I, unsigned int N,
                               const float *up_scale, const float *gate_scale,
                               const float *down_scale) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_dense(up, gate, down, session, K, I, N, up_scale,
                          gate_scale, down_scale);
    return true;
  }

  /** @brief Registers a conv block's two Q4_0x4 weights (doc 51 section
   *  2). in_proj goes through get_or_register_fc, whose column slices at
   *  K = 2048 are exactly C = 2048 wide, so its three handles ARE a, b
   *  and c and the kernel's split is free; a shape where the slices do
   *  not fall on the thirds is refused rather than re-sliced.
   *  ponytail: fcSliceCols(K) == C is this model's coincidence. Another
   *  shape needs its own slicing here (three registerRm calls at c0 = 0,
   *  C, 2C), not a change to the kernel. Cached by in_proj's pointer. */
  const ConvHandles &get_or_register_conv(void *in_proj, void *out_proj,
                                          remote_handle64 session, uint32_t K,
                                          uint32_t C, uint32_t N,
                                          const float *in_scale = nullptr,
                                          const float *out_scale = nullptr,
                                          bool weights_wh = false) {
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      auto it = conv_cache_.find(in_proj);
      if (it != conv_cache_.end())
        return it->second;
    }
    const FcHandles &in =
      get_or_register_fc(in_proj, session, K, 3 * C, in_scale, weights_wh);
    const FcHandles &out =
      get_or_register_fc(out_proj, session, C, N, out_scale, weights_wh);
    if (in.handles.size() != 3 || in.cols[0] != C || in.cols[1] != C ||
        in.cols[2] != C || out.handles.size() != 1) {
      throw std::invalid_argument(
        "gemm_q4_0_conv_block_fp32: in_proj [" + std::to_string(K) + " x " +
        std::to_string(3 * C) + "] slices into " +
        std::to_string(in.handles.size()) + " handles of " +
        std::to_string(fcSliceCols(K)) + " columns, not the three of " +
        std::to_string(C) + " the conv block kernel takes");
    }
    std::lock_guard<std::mutex> lock(handle_mutex_);
    // Optional extra WH image, prepared once during model registration.
    // Keep the original image for HMX prefill and any shared FC handle.
    auto existing = conv_cache_.find(in_proj);
    if (existing != conv_cache_.end())
      return existing->second;
    ConvHandles ch;
    ch.h_in = in.handles;
    ch.h_out = out.handles[0];
    const char *prepack = std::getenv("NNTR_HTP_CONV_PREPACK");
    if (prepack && std::atoi(prepack) != 0) {
      sendMoeOptsOnce(session);
      if (applied_moe_flags_ & HTP_CONV_FLAG_M1_HVX) {
        const uint64_t start = HtpProfile::nowUs();
        const uint32_t input_tiles = htp_conv_chunk_tiles(applied_moe_flags_);
        const uint32_t output_tiles =
          htp_conv_out_chunk_tiles(applied_moe_flags_);
        const uint64_t bytes = 3u * whBytes(K, C) + whBytes(C, N);
        uint32_t chunk = 0, off = 0;
        if (bytes > UINT32_MAX || !ensureArena(session) ||
            !placeConvDecode(session, static_cast<uint32_t>(bytes), &chunk,
                             &off))
          throw std::runtime_error(
            "HTP Conv decode prepack arena allocation failed");
        // VTCM and the DMA ring belong to the same session as inference.
        std::lock_guard<std::mutex> invoke_lock(invoke_mutex_);
        uint32_t prepared = 0;
        for (uint32_t i = 0; i < 4u; ++i) {
          const uint32_t h = i == 3u ? ch.h_out : ch.h_in[i];
          const int err = nntr_hvx_weight_prepare_conv_decode(
            session, h, arena_chunks_[chunk].dsp_id, off,
            i == 3u ? output_tiles : input_tiles);
          if (err != AEE_SUCCESS) {
            for (uint32_t j = 0; j < prepared; ++j)
              nntr_hvx_weight_prepare_conv_decode(session, ch.h_in[j], 0, 0, 0);
            throw std::runtime_error("HTP Conv decode prepack failed: err=" +
                                     std::to_string(err));
          }
          ++prepared;
          off += static_cast<uint32_t>(i == 3u ? whBytes(C, N) : whBytes(K, C));
        }
        std::fprintf(
          stderr,
          "[HTP] conv decode prepack: bytes=%llu in_tiles=%u "
          "out_tiles=%u prepare_us=%llu\n",
          static_cast<unsigned long long>(bytes), input_tiles, output_tiles,
          static_cast<unsigned long long>(HtpProfile::nowUs() - start));
      }
    }
    return conv_cache_.emplace(in_proj, std::move(ch)).first->second;
  }

  bool supports_gemm_q4_0_conv_block_fp32() const override { return true; }

  void gemm_q4_0_conv_block_fp32(void *in_proj, const float *conv_w,
                                 void *out_proj, const float *act, float *out,
                                 float *state, unsigned int M, unsigned int K,
                                 unsigned int C, unsigned int N) override {
    try {
      const remote_handle64 session =
        static_cast<remote_handle64>(HtpBackend::global().handle());
      const ConvHandles &ch =
        get_or_register_conv(in_proj, out_proj, session, K, C, N);
      invokeConvBlock(session, ch, conv_w, act, out, state, M, K, C, N);
    } catch (...) {
      discard_deferred_norm();
      throw;
    }
  }

  bool register_q4_0_conv_block(void *in_proj, void *out_proj, unsigned int K,
                                unsigned int C, unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_conv(in_proj, out_proj, session, K, C, N);
    return true;
  }

  bool supports_gemm_qs4cx_conv_block_fp32(bool weights_wh) const override {
    (void)weights_wh;
    return true;
  }

  void gemm_qs4cx_conv_block_fp32(void *in_proj, const float *conv_w,
                                  void *out_proj, const float *act, float *out,
                                  float *state, unsigned int M, unsigned int K,
                                  unsigned int C, unsigned int N,
                                  const float *in_scale, const float *out_scale,
                                  bool weights_wh) override {
    try {
      const remote_handle64 session =
        static_cast<remote_handle64>(HtpBackend::global().handle());
      const ConvHandles &ch = get_or_register_conv(
        in_proj, out_proj, session, K, C, N, in_scale, out_scale, weights_wh);
      invokeConvBlock(session, ch, conv_w, act, out, state, M, K, C, N);
    } catch (...) {
      discard_deferred_norm();
      throw;
    }
  }

  bool register_qs4cx_conv_block(void *in_proj, void *out_proj, unsigned int K,
                                 unsigned int C, unsigned int N,
                                 const float *in_scale, const float *out_scale,
                                 bool weights_wh) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_conv(in_proj, out_proj, session, K, C, N, in_scale,
                         out_scale, weights_wh);
    return true;
  }

  /** Declared here rather than beside arena_chunks_ at the bottom of the
     class: registerFromArena takes an ArenaEntry by reference, and a
     parameter type has to be complete where the function is declared, unlike
     a member a function BODY refers to. Only an HTP build compiles this
     file, so a host build cannot catch it. */
  /** @brief One uncached ION buffer the DSP has mapped, filled front to
   *  back. Never reused or rewritten: a weight placed here keeps its bytes
   *  for the process lifetime, which is also why the DSP can borrow them
   *  and why no cache line in either direction can go stale. */
  struct ArenaChunk {
    std::unique_ptr<HtpRpcBuffer> buf;
    uint32_t dsp_id; /**< what nntr_hvx_arena_attach called it */
    size_t used;     /**< bump pointer */
  };

  /** @brief Where one weight sits, and its three small arrays when they
   *  are passed to the DSP by value (the resident path). An expert slot
   *  keeps them in the arena after the WH bytes instead (readWeight) and
   *  leaves these empty. */
  struct ArenaEntry {
    uint32_t chunk;
    uint32_t off;
    uint32_t K, N;
    std::vector<float> w_scale;
    std::vector<int32_t> colsum_w;
    std::vector<float> bias;
  };

  /** @brief One expert's place in the arena: gate_up at off, down at
   *  off + its 4 KB-rounded length (register_qs4cx_wh_expert_file). */
  struct ExpertSlot {
    uint32_t chunk = 0;
    uint32_t off = 0;
    /** The retired expert's handles, still registered on the DSP until the
     *  next load into this slot swaps them out; kNoHandle when none. */
    uint32_t h_gu = kNoHandle, h_dn = kNoHandle;
  };
  struct ExpertResident {
    ExpertSlot slot;
    const void *key_dn;
    uint32_t h_gu, h_dn;
  };

  /** @brief pread that finishes: 0, or the errno, or -1 at end of file.
   *  Does not throw, so a worker thread can call it. */
  static int preadAll(int fd, void *dst, size_t len, uint64_t off) {
#if defined(__linux__) || defined(__ANDROID__)
    if (weightDirectIO()) {
      static WeightDirectFiles direct_files;
      return direct_files.read(fd, dst, len, off);
    }
#endif
    uint8_t *p = static_cast<uint8_t *>(dst);
    while (len != 0) {
      const ssize_t n = ::pread(fd, p, len, static_cast<off_t>(off));
      if (n < 0) {
        if (errno == EINTR)
          continue;
        return errno;
      }
      if (n == 0)
        return -1;
      p += n;
      len -= static_cast<size_t>(n);
      off += static_cast<uint64_t>(n);
    }
    return 0;
  }

  /** @brief A short read into a weight is a wrong matmul, never a warning. */
  static void throwPread(int rc, const char *what) {
    if (rc == 0)
      return;
    if (rc < 0) {
      throw std::runtime_error(std::string("pread(") + what +
                               ") hit end of file: the model file is "
                               "shorter than its weights");
    }
    throw std::runtime_error(std::string("pread(") + what +
                             ") failed: " + std::strerror(rc));
  }

  /** @brief One expert on its way into a slot: where it comes from, where
   *  it goes, what the file said about it, and how the read went. */
  struct StagedExpert {
    ExpertFileDesc d;
    ExpertSlot slot;
    /** The slot's address, resolved when the slot was taken, so a reader
     *  thread never indexes arena_chunks_ while it might grow. */
    uint8_t *base;
    ArenaEntry gu, dn;
    int rc; /**< preadAll's result for the whole expert */
    std::shared_ptr<HostExpertBank::Lease> host;
  };

  /** @note Call with handle_mutex_ held. */
  StagedExpert stageExpert(remote_handle64 session, const ExpertFileDesc &d) {
    auto host = hostExpertLease(d);
    if (host && free_expert_slots_.empty())
      throw std::logic_error("host expert promotion requires a freed slot");
    const ExpertSlot slot = takeExpertSlot(session, d);
    return StagedExpert{d,
                        slot,
                        arena_chunks_[slot.chunk].buf->data() + slot.off,
                        {},
                        {},
                        0,
                        std::move(host)};
  }

  /** @brief A free slot, or a new one bumped into a chunk (the load, and
   *  the first prefill before the pool is full). One expert shape per
   *  process: a released slot is reused as is.
   *  @note Call with handle_mutex_ held. */
  ExpertSlot takeExpertSlot(remote_handle64 session, const ExpertFileDesc &d) {
    if (d.fd < 0) {
      throw std::runtime_error("register_qs4cx_wh_expert_file: the virtual "
                               "expert weight carries no model file fd");
    }
    if (!ensureArena(session)) {
      throw std::runtime_error(
        "QS4CX_WH weights need the DSP arena, which this device did not "
        "provide (no rpcmem_to_fd or no fastrpc_mmap).");
    }
    const uint32_t slot_bytes = expertStride(d.K, 2 * d.inter, d.weight_bits) +
                                expertStride(d.inter, d.N_out, d.weight_bits);
    if (expert_slot_bytes_ == 0) {
      expert_slot_bytes_ = slot_bytes;
      expert_slot_shape_ = {d.K, d.inter, d.N_out, d.weight_bits};
    } else if (expert_slot_bytes_ != slot_bytes) {
      throw std::runtime_error("register_qs4cx_wh_expert_file: expert shape "
                               "differs from the pool's");
    }
    expert_shapes_uniform_ &=
      expert_slot_shape_ ==
      std::array<unsigned, 4>{d.K, d.inter, d.N_out, d.weight_bits};
    ExpertSlot slot;
    // A new chunk is sized to the slots still to come when the caller said
    // how many there will be (reserve_qs4cx_wh_expert_slots), else a full
    // chunk as before. newChunk rounds up to its 64 MiB grain.
    const size_t left = expert_slots_wanted_ > expert_slots_placed_
                          ? expert_slots_wanted_ - expert_slots_placed_
                          : 0;
    const size_t want =
      left != 0 ? std::min(kArenaChunkMax, left * slot_bytes) : kArenaChunkMax;
    if (!free_expert_slots_.empty()) {
      slot = free_expert_slots_.back();
      free_expert_slots_.pop_back();
    } else if (place(session, slot_bytes, want, &slot.chunk, &slot.off)) {
      ++expert_slots_placed_;
    } else {
      throw std::runtime_error(
        "HTP arena: cannot place an expert slot (" +
        std::to_string(slot_bytes >> 20) + " MiB). " +
        (arena_fail_.empty() ? "no chunk was attempted" : arena_fail_) +
        ". mapped=" + std::to_string(arenaBytes() >> 20) + " MiB in " +
        std::to_string(arena_chunks_.size()) +
        " chunks, RSS=" + std::to_string(rssKb() >> 10) + " MB");
    }
    return slot;
  }

  /** @brief One weight's span in an expert slot: its WH bytes, then its N
   *  f32 scales and N i32 column sums (doc 52 section 10.30: the DSP takes
   *  them from here, so a swap carries offsets and nothing else), rounded
   *  to a page. +44 KB on a 5.25 MB expert. */
  static bool compactExperts() {
    static const bool enabled = [] {
      const char *v = std::getenv("NNTR_HTP_QS2_VTCM");
      return !v || std::strcmp(v, "0") != 0;
    }();
    return enabled;
  }

  static const char *preparedQs2Dir() {
    const char *dir = std::getenv("NNTR_HTP_QS2_PREPARED_DIR");
    return dir && *dir ? dir : nullptr;
  }

  static uint32_t expertStride(uint32_t K, uint32_t N, unsigned bits) {
    const bool compact = bits == 2 && compactExperts();
    if (compact && preparedQs2Dir() && N % 128u)
      throw std::invalid_argument("offline QS2: N must be divisible by 128");
    return (static_cast<uint32_t>(whBytes(K, N)) / (compact ? 2u : 1u) +
            8u * N + (compact ? 4u : 0u) + 4095u) &
           ~4095u;
  }

  /** @brief Reads both weights of @a st into its slot. No lock and no
   *  throw -- the slot is this expert's alone until registerStaged -- so a
   *  background thread can run it. @return 0, errno, or -1 at EOF. */
  int readExpert(StagedExpert &st, bool use_pool, bool demand = false) {
    struct ReleaseHost {
      StagedExpert &st;
      ~ReleaseHost() { st.host.reset(); }
    } release_host{st};
    const int rc = readExpertPart(st, false, use_pool, demand);
    return rc ? rc : readExpertPart(st, true, use_pool, demand);
  }

  /** Each half has separate metadata and a disjoint destination. No registry
   *  or arena-container access occurs on the reader threads. */
  int readExpertPart(StagedExpert &st, bool down, bool use_pool,
                     bool demand = false) {
    const ExpertFileDesc &d = st.d;
    const uint32_t offset =
      down ? expertStride(d.K, 2 * d.inter, d.weight_bits) : 0u;
    ArenaEntry &entry = down ? st.dn : st.gu;
    entry.chunk = st.slot.chunk;
    entry.off = st.slot.off + offset;
    if (st.host) {
      const auto &record = st.host->record();
      // The bank contains only initialized compact bytes. Leave page padding
      // untouched, exactly like the existing validated direct read path.
      const size_t bytes = down ? record.dn_bytes : record.gu_bytes;
      const int rc = st.host->copyPart(down, st.base + offset, bytes);
      if (rc)
        return rc;
      entry.K = down ? d.inter : d.K;
      entry.N = down ? d.N_out : 2u * d.inter;
      return 0;
    }
    return readWeight(d.fd, down ? d.off_dn : d.off_gu, down ? d.inter : d.K,
                      down ? d.N_out : 2u * d.inter, st.base + offset, entry,
                      use_pool, d.weight_bits, demand);
  }

  /** @brief Registers a read expert and files it. On any failure the slot
   *  goes back to the free list and this throws.
   *  @note Call with handle_mutex_ held. */
  void registerStaged(remote_handle64 session, StagedExpert &st) {
    const ExpertFileDesc &d = st.d;
    uint32_t h_gu = kNoHandle, h_dn = kNoHandle;
    int err = AEE_SUCCESS;
    try {
      throwPread(st.rc, "expert weight");
      // One round trip: register the pair just read and release the pair
      // retired from this slot, if any (release_qs4cx_wh_expert). On error
      // the DSP has changed nothing, so the slot goes back to the free list
      // still holding its retired pair.
      err = d.weight_bits == 2 && compactExperts()
              ? (preparedQs2Dir() ? nntr_hvx_weight_swap_u2lut_single_arena
                                  : nntr_hvx_weight_swap_u2i4_arena)(
                  session, st.slot.h_gu, st.slot.h_dn, d.K, d.inter, d.N_out,
                  arena_chunks_[st.slot.chunk].dsp_id, st.gu.off, st.dn.off,
                  &h_gu, &h_dn)
              : nntr_hvx_weight_swap_u8i4_arena(
                  session, st.slot.h_gu, st.slot.h_dn, d.K, d.inter, d.N_out,
                  arena_chunks_[st.slot.chunk].dsp_id, st.gu.off, st.dn.off,
                  st.gu.w_scale.data(), static_cast<int>(st.gu.w_scale.size()),
                  st.gu.colsum_w.data(),
                  static_cast<int>(st.gu.colsum_w.size()), st.dn.w_scale.data(),
                  static_cast<int>(st.dn.w_scale.size()), st.dn.colsum_w.data(),
                  static_cast<int>(st.dn.colsum_w.size()), &h_gu, &h_dn);
      if (err != AEE_SUCCESS) {
        uint32_t session_info[7] = {};
        nntr_hvx_session_info(session, session_info, 7);
        std::fprintf(stderr,
                     "[HTP] expert swap session hmx=%u vtcm=%u path=%u\n",
                     session_info[0], session_info[1], session_info[5]);
        char code[16];
        std::snprintf(code, sizeof(code), "0x%08x", static_cast<unsigned>(err));
        throw std::runtime_error(
          std::string("HTP expert swap failed: K=") + std::to_string(d.K) +
          " inter=" + std::to_string(d.inter) + " N=" +
          std::to_string(d.N_out) + " bits=" + std::to_string(d.weight_bits) +
          " compact=" + std::to_string(compactExperts()) + " err=" + code +
          " (a skel older than test/htp/nntr_hvx.idl answers this call with "
          "an error: rebuild and push libnntr_hvx_skel.so)");
      }
    } catch (...) {
      free_expert_slots_.push_back(st.slot);
      throw;
    }
    fileRegistered(st, h_gu, h_dn);
  }

  /** @brief Files an expert the DSP has just registered.
   *  @note Call with handle_mutex_ held. */
  void fileRegistered(StagedExpert &st, uint32_t h_gu, uint32_t h_dn) {
    const ExpertFileDesc &d = st.d;
    st.slot.h_gu = st.slot.h_dn = kNoHandle; // released by the swap
    handle_cache_[d.key_gu] = h_gu;
    handle_cache_[d.key_dn] = h_dn;
    experts_.emplace(d.key_gu, ExpertResident{st.slot, d.key_dn, h_gu, h_dn});
  }

  /**
   * @brief registerStaged for several experts in ONE round trip (doc 52
   *        section 10.23): a prefill layer's read-ahead batch, or one call's
   *        misses. The experts the DSP swapped are filed; one whose read
   *        failed, and every one from the first the DSP refused on, goes
   *        back to the free list with its retired pair, which the DSP still
   *        holds. The first failure is thrown after that.
   * @note Call with handle_mutex_ held.
   */
  void registerStagedBatch(remote_handle64 session,
                           std::vector<StagedExpert> &sts,
                           bool allow_defer = false) {
    if (expert_updates_failed_)
      throw std::runtime_error("expert update failed; restart the session");
    // Only compact, equal-shape slots with already allocated handles can be
    // published early. Their bytes are complete and exclusively owned here;
    // the next MoE dispatch commits their DSP metadata before using them.
    const bool defer =
      allow_defer && decode_expert_updates_ && expertUpdateMode() &&
      expert_shapes_uniform_ && compactExperts() && !preparedQs2Dir() &&
      !(forwardSwitch() && !graph_words_.empty()) && !sts.empty() &&
      pending_expert_updates_.size() + sts.size() <=
        HTP_DSPQ_MAX_EXPERT_UPDATES &&
      std::all_of(sts.begin(), sts.end(), [&](const StagedExpert &st) {
        const auto &d = sts.front().d;
        return st.rc == 0 && st.d.weight_bits == 2 &&
               st.slot.h_gu != kNoHandle && st.slot.h_dn != kNoHandle &&
               st.d.K == d.K && st.d.inter == d.inter &&
               st.d.N_out == d.N_out &&
               expertUpdatesMatch(d.K, d.inter, d.N_out);
      });
    if (defer) {
      pending_expert_updates_.reserve(HTP_DSPQ_MAX_EXPERT_UPDATES);
      for (auto &st : sts) {
        pending_expert_updates_.push_back(st);
        fileRegistered(st, st.slot.h_gu, st.slot.h_dn);
        ++expert_updates_queued_;
      }
      return;
    }
    if (allow_defer)
      flushExpertUpdates(session);
    std::exception_ptr first;
    std::vector<StagedExpert *> ok;
    for (StagedExpert &st : sts) {
      if (st.rc == 0) {
        ok.push_back(&st);
        continue;
      }
      free_expert_slots_.push_back(st.slot);
      if (!first) {
        try {
          throwPread(st.rc, "expert weight");
        } catch (...) {
          first = std::current_exception();
        }
      }
    }
    if (!ok.empty()) {
      const ExpertFileDesc &d0 = ok[0]->d;
      const size_t n = ok.size();
      std::vector<uint32_t> og(n), od(n), ar(n), ofg(n), ofd(n);
      std::vector<uint32_t> hg(n, kNoHandle), hd(n, kNoHandle);
      std::vector<float> gs, ds;
      std::vector<int32_t> gc, dc;
      for (size_t i = 0; i < n; ++i) {
        const StagedExpert &st = *ok[i];
        og[i] = st.slot.h_gu;
        od[i] = st.slot.h_dn;
        ar[i] = arena_chunks_[st.slot.chunk].dsp_id;
        ofg[i] = st.gu.off;
        ofd[i] = st.dn.off;
        gs.insert(gs.end(), st.gu.w_scale.begin(), st.gu.w_scale.end());
        gc.insert(gc.end(), st.gu.colsum_w.begin(), st.gu.colsum_w.end());
        ds.insert(ds.end(), st.dn.w_scale.begin(), st.dn.w_scale.end());
        dc.insert(dc.end(), st.dn.colsum_w.begin(), st.dn.colsum_w.end());
      }
      const int ni = static_cast<int>(n);
      uint32_t done = 0;
      int32_t err = AEE_SUCCESS;
      const int rc =
        d0.weight_bits == 2 && compactExperts()
          ? (preparedQs2Dir() ? nntr_hvx_weight_swap_batch_u2lut_single_arena
                              : nntr_hvx_weight_swap_batch_u2i4_arena)(
              session, d0.K, d0.inter, d0.N_out, og.data(), ni, od.data(), ni,
              ar.data(), ni, ofg.data(), ni, ofd.data(), ni, hg.data(), ni,
              hd.data(), ni, &done, &err)
          : nntr_hvx_weight_swap_batch_u8i4_arena(
              session, d0.K, d0.inter, d0.N_out, og.data(), ni, od.data(), ni,
              ar.data(), ni, ofg.data(), ni, ofd.data(), ni, gs.data(),
              static_cast<int>(gs.size()), gc.data(),
              static_cast<int>(gc.size()), ds.data(),
              static_cast<int>(ds.size()), dc.data(),
              static_cast<int>(dc.size()), hg.data(), ni, hd.data(), ni, &done,
              &err);
      if (rc != AEE_SUCCESS) { // refused whole: the DSP changed nothing
        done = 0;
        err = rc;
      }
      for (size_t i = 0; i < n; ++i) {
        if (i < done)
          fileRegistered(*ok[i], hg[i], hd[i]);
        else
          free_expert_slots_.push_back(ok[i]->slot);
      }
      if (done < n && !first) {
        char code[16];
        std::snprintf(code, sizeof(code), "0x%08x", static_cast<unsigned>(err));
        first = std::make_exception_ptr(std::runtime_error(
          std::string("nntr_hvx_weight_swap_batch_u8i4_arena failed: err=") +
          code + " at expert " + std::to_string(done) + " of " +
          std::to_string(n) +
          " (a skel older than test/htp/nntr_hvx.idl answers this call with "
          "an error: rebuild and push libnntr_hvx_skel.so)"));
      }
    }
    if (first)
      std::rethrow_exception(first);
  }

  /** 0: original, 1: merge at dispatch, 2: queue with MoE, 3: batch DMA too.
   * The optimization is opt-in; arbitrary expert shapes and the first use
   * of a physical slot continue through the synchronous registration API. */
  static unsigned expertUpdateMode() {
    static const unsigned mode = [] {
      const char *v = std::getenv("NNTR_HTP_EXPERT_UPDATE_MODE");
      return v ? static_cast<unsigned>(std::max(0, std::min(3, std::atoi(v))))
               : 0u;
    }();
    return mode;
  }

  /** Caller holds handle_mutex_. */
  bool expertUpdatesMatch(unsigned K, unsigned inter, unsigned N) const {
    return std::all_of(pending_expert_updates_.begin(),
                       pending_expert_updates_.end(), [&](const auto &st) {
                         return st.d.K == K && st.d.inter == inter &&
                                st.d.N_out == N;
                       });
  }

  /** Caller holds handle_mutex_; ordinary RPC fallback also implements the
   * merged-registration control variant. Stable slot handles do not change. */
  void flushExpertUpdates(remote_handle64 session) {
    if (pending_expert_updates_.empty())
      return;
    auto updates = std::move(pending_expert_updates_);
    pending_expert_updates_.clear();
    const auto start = HtpProfile::nowUs();
    try {
      registerStagedBatch(session, updates);
    } catch (...) {
      expert_updates_failed_ = true;
      throw;
    }
    ++expert_update_rpc_batches_;
    expert_update_rpc_us_ += HtpProfile::nowUs() - start;
  }

  std::vector<htp_dspq_expert_update>
  expertUpdatePacket(unsigned K, unsigned inter, unsigned N) {
    if (expertUpdateMode() < 2)
      return {};
    std::lock_guard<std::mutex> lock(handle_mutex_);
    if (expert_updates_failed_ || !expertUpdatesMatch(K, inter, N))
      throw std::logic_error("invalid pending expert update");
    std::vector<htp_dspq_expert_update> packet;
    packet.reserve(pending_expert_updates_.size());
    for (const auto &st : pending_expert_updates_)
      packet.push_back({st.slot.h_gu, st.slot.h_dn,
                        arena_chunks_[st.slot.chunk].dsp_id, st.gu.off,
                        st.dn.off});
    return packet;
  }

  static bool hostExpertProfile() {
    const char *v = std::getenv("NNTR_MOE_IO_PROFILE");
    return v && std::atoi(v) != 0;
  }

  std::shared_ptr<HostExpertBank> hostExpertBank() {
    std::lock_guard<std::mutex> lock(prefetch_mutex_);
    return host_expert_bank_;
  }

  std::shared_ptr<HostExpertBank::Lease>
  hostExpertLease(const ExpertFileDesc &d) {
    auto bank = hostExpertBank();
    return bank ? bank->take(d) : nullptr;
  }

  void waitHostExpertReads() {
    auto bank = hostExpertBank();
    if (!bank)
      return;
    const uint64_t begin = hostExpertProfile() ? HtpProfile::nowUs() : 0;
    bank->waitReads();
    if (hostExpertProfile())
      host_expert_wait_us_ += HtpProfile::nowUs() - begin;
  }

  void readHostExpert(const std::shared_ptr<HostExpertBank> &bank) {
    bank->readOne([&](HostExpertBank::Record &record) {
      const uint64_t begin = HtpProfile::nowUs();
      ArenaEntry gu, dn;
      const auto &d = record.d;
      int rc = readWeight(record.fd(), d.off_gu, d.K, 2u * d.inter,
                          record.payload.get(), gu, false, 2u, false);
      if (!rc)
        rc = readWeight(record.fd(), d.off_dn, d.inter, d.N_out,
                        record.payload.get() + record.gu_bytes, dn, false, 2u,
                        false);
      if (HtpProfile::global().level())
        HtpProfile::global().addPrefetchRead(HtpProfile::nowUs() - begin);
      return rc;
    });
  }

  /** @brief Starts the readers once, pinned to NNTR_MOE_PREFETCH_CPUS or
   *  else to every core but the caller's. */
  void startPrefetchReaders() {
    if (!prefetch_readers_.empty())
      return;
    const PrefetchKnobs &knobs = prefetchKnobs();
    prefetch_caller_cpus_ = cpuBit();
    std::vector<int> cpus = knobs.cpus;
    if (cpus.empty() && prefetch_caller_cpus_ != 0) {
      const long n = sysconf(_SC_NPROCESSORS_CONF);
      for (int c = 0; c < n && c < 32; ++c)
        if ((prefetch_caller_cpus_ & (1u << c)) == 0)
          cpus.push_back(c);
    }
    for (size_t t = 0; t < knobs.readers; ++t)
      prefetch_readers_.emplace_back([this, cpus] {
        pinToCpus(cpus);
        prefetchReaderLoop();
      });
  }

  void prefetchReaderLoop() {
    for (;;) {
      std::pair<PrefetchBatch *, size_t> job;
      std::shared_ptr<HostExpertBank> bank;
      {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_cv_.wait(lock, [this] {
          return prefetch_stop_ || !prefetch_jobs_.empty() ||
                 (host_expert_bank_ && host_expert_bank_->hasWork());
        });
        if (prefetch_stop_)
          return;
        if (host_expert_bank_ && host_expert_bank_->hasWork())
          bank = host_expert_bank_;
        else if (!prefetch_jobs_.empty()) {
          job = prefetch_jobs_.front();
          prefetch_jobs_.pop_front();
        } else {
          continue; // Another worker claimed the last host record.
        }
      }
      prefetch_reader_cpus_.fetch_or(cpuBit());
      if (bank) {
        readHostExpert(bank);
        continue;
      }
      const uint64_t t0 = HtpProfile::nowUs();
      StagedExpert &st = job.first->experts[job.second];
      const bool from_host = static_cast<bool>(st.host);
      static const bool decode_mmap = [] {
        const char *v = std::getenv("NNTR_MOE_DECODE_PREFETCH_MMAP");
        return !v || std::atoi(v) != 0;
      }();
      try {
        st.rc =
          readExpert(st, /*use_pool=*/false, job.first->decode && decode_mmap);
      } catch (...) {
        st.rc = EIO; // Required predictions surface errors during registration.
      }
      HtpProfile &profile = HtpProfile::global();
      if (profile.level() != 0 && !from_host)
        profile.addPrefetchRead(HtpProfile::nowUs() - t0);
      {
        std::lock_guard<std::mutex> lock(prefetch_mutex_);
        ++job.first->done;
        if (job.second < 64)
          job.first->done_mask |= uint64_t(1) << job.second;
      }
      prefetch_done_cv_.notify_all();
    }
  }

  /** @brief Reads one QS4CX_WH weight from the model file into the arena
   *  at @a arena_dst: the nibbles, then the N scales and N column sums
   *  that follow them in the file, laid out as the DSP registry expects
   *  (f32 scales, then i32 column sums -- the file keeps the sums as f32,
   *  so they pass through a small buffer). @a e gets the shape only; its
   *  arrays stay empty, which is what tells the swap call the arrays are
   *  in the arena. whBytes(K, N) is the nibble half exactly:
   *  QS4CX_Tensor::size() counts N * ceil(K / 2) and K is a multiple of
   *  32 here. @return 0, errno, or -1 at end of file. */
  int readWeight(int fd, uint64_t off, uint32_t K, uint32_t N,
                 uint8_t *arena_dst, ArenaEntry &e, bool use_pool,
                 unsigned int weight_bits = 4, bool demand = false) {
    const size_t nib = whBytes(K, N);
    if (weight_bits == 2) {
      // Keep file and shared LRU compressed by default; DSP expands in VTCM.
      // The diagnostic ARM expansion uses the same file and quantization.
      // Readers own their scratch; no uncached arena reads are needed.
      thread_local std::vector<uint8_t> packed, expanded;
      const size_t compact = nib / 2;
      if (compactExperts() && !preparedQs2Dir()) {
        static const unsigned map_mode = [] {
#if defined(__linux__) || defined(__ANDROID__)
          const char *v = std::getenv("NNTR_HTP_QS2_MMAP_READ");
          // 1: synchronous demand only; 2: also preload and prefetch.
          return v ? static_cast<unsigned>(
                       std::max(0, std::min(2, std::atoi(v))))
                   : 0u;
#else
          return 0u;
#endif
        }();
        const bool mapped =
          !weightDirectIO() && (map_mode == 2u || (map_mode == 1u && demand));
        static const bool direct = [] {
          const char *v = std::getenv("NNTR_HTP_QS2_DIRECT_READ");
          return v && std::atoi(v) != 0;
        }();
        static const bool profile = [] {
          const char *v = std::getenv("NNTR_MOE_IO_PROFILE");
          return v && std::atoi(v) != 0;
        }();
        HtpQs2ReadStats stats;
        const int rc = htpReadQs2Compact(
          [this, fd, mapped](void *dst, size_t len, uint64_t at) {
#if defined(__linux__) || defined(__ANDROID__)
            if (mapped)
              return qs2_mapped_files_.read(fd, dst, len, at);
#endif
            return preadAll(fd, dst, len, at);
          },
          off, K, N, arena_dst, direct || mapped, packed,
          profile ? &stats : nullptr);
        if (profile) {
          qs2_reads_.fetch_add(1, std::memory_order_relaxed);
          qs2_read_us_.fetch_add(stats.read_us, std::memory_order_relaxed);
          qs2_copy_us_.fetch_add(stats.copy_us, std::memory_order_relaxed);
          qs2_metadata_us_.fetch_add(stats.metadata_us,
                                     std::memory_order_relaxed);
          qs2_copied_bytes_.fetch_add(stats.copied_bytes,
                                      std::memory_order_relaxed);
        }
        if (!rc) {
          e.K = K;
          e.N = N;
        }
        return rc;
      }
      packed.resize(compact + 4 + 2 * static_cast<size_t>(N) * sizeof(float));
      if (compactExperts() && preparedQs2Dir()) {
        try {
          const std::string path =
            std::string(preparedQs2Dir()) + "/" + std::to_string(off) + "_" +
            std::to_string(K) + "_" + std::to_string(N) + ".nntri2";
          readPreparedQs2Cached(path, fd, off, K, N, packed, arena_dst);
        } catch (const std::exception &error) {
          std::fprintf(stderr, "[HTP] %s\n", error.what());
          return EINVAL;
        }
      } else {
        const int rc = preadAll(fd, packed.data(), packed.size(), off);
        if (rc != 0)
          return rc;
      }
      if (compactExperts()) {
        const int8_t *pal =
          reinterpret_cast<const int8_t *>(packed.data() + compact);
        for (unsigned i = 0; i < 4; ++i)
          if (pal[i] < -8 || pal[i] > 7)
            return EINVAL;
        if (!preparedQs2Dir())
          std::memcpy(arena_dst, packed.data(), compact);
        std::memcpy(arena_dst + compact, packed.data() + compact + 4,
                    sizeof(float) * N);
        std::vector<int32_t> sums(N);
        for (uint32_t i = 0; i < N; ++i) {
          float value;
          std::memcpy(&value,
                      packed.data() + compact + 4 + sizeof(float) * (N + i),
                      sizeof(value));
          if (!std::isfinite(value) || std::fabs(value) > 8.0 * K)
            return EINVAL;
          sums[i] = static_cast<int32_t>(value);
        }
        std::memcpy(arena_dst + compact + sizeof(float) * N, sums.data(),
                    sizeof(int32_t) * N);
        std::memcpy(arena_dst + compact + 8u * N, pal, 4);
        e.K = K;
        e.N = N;
        return 0;
      }
      expanded.resize(nib);
      if (!wh2Expand(packed.data(),
                     reinterpret_cast<const int8_t *>(packed.data() + compact),
                     K, N, expanded.data()))
        return EINVAL;
      std::memcpy(arena_dst, expanded.data(), nib);
      const uint8_t *scales = packed.data() + compact + 4;
      std::memcpy(arena_dst + nib, scales, sizeof(float) * N);
      std::vector<int32_t> sums(N);
      for (uint32_t i = 0; i < N; ++i) {
        float value;
        std::memcpy(&value, scales + sizeof(float) * (N + i), sizeof(value));
        if (!std::isfinite(value) || std::fabs(value) > 8.0 * K)
          return EINVAL;
        sums[i] = static_cast<int32_t>(value);
      }
      std::memcpy(arena_dst + nib + sizeof(float) * N, sums.data(),
                  sizeof(int32_t) * N);
      e.K = K;
      e.N = N;
      return 0;
    }
    if (weight_bits != 4)
      return EINVAL;
    // [doc 52 sections 10.7, 10.9] The nibble read is 82% of a miss and
    // capped near 4.9 GB/s by the uncached mapping whatever the thread
    // count: 8 slices bought 18%. Kept for the synchronous miss; the
    // prefetch readers are already one thread per expert. Page-sized
    // slices, so no two threads write the same page; a worker cannot throw
    // across parallel_for, so each slice keeps its result.
    std::atomic<int> first_rc{0};
    auto slice_read = [&](uint8_t *dst, size_t len, uint64_t at) {
      const int rc = preadAll(fd, dst, len, at);
      if (rc == 0)
        return;
      int expected = 0;
      first_rc.compare_exchange_strong(expected, rc);
    };
    if (use_pool) {
      auto &tm = ThreadManager::Global();
      const size_t n_slices =
        std::min<size_t>(tm.getComputeThreadCount(), kExpertReadSlicesMax);
      const size_t slice =
        (((nib + n_slices - 1) / n_slices) + 4095u) & ~size_t(4095u);
      tm.parallel_for(0, n_slices, [&](size_t i) {
        const size_t b = i * slice;
        if (b < nib)
          slice_read(arena_dst + b, std::min(slice, nib - b), off + b);
      });
    } else {
      slice_read(arena_dst, nib, off);
    }
    if (first_rc.load() != 0)
      return first_rc.load();
    std::vector<float> tail(2 * static_cast<size_t>(N));
    const int rc =
      preadAll(fd, tail.data(), tail.size() * sizeof(float), off + nib);
    if (rc != 0)
      return rc;
    std::vector<int32_t> colsum(N);
    for (uint32_t i = 0; i < N; ++i)
      colsum[i] = static_cast<int32_t>(tail[N + i]);
    // Two sequential stores into the uncached mapping, never a read back.
    std::memcpy(arena_dst + nib, tail.data(), sizeof(float) * N);
    std::memcpy(arena_dst + nib + sizeof(float) * N, colsum.data(),
                sizeof(int32_t) * N);
    e.K = K;
    e.N = N;
    return 0;
  }

  /**
   * @brief Hands the OS back the pages of a weight that is now in the arena.
   *
   * The arena copy doubles this model's 3.9 GB of expert weights, and the
   * device refuses the fourth 1 GiB ION chunk long before the last layer
   * registers (measured: 1170 of 1408 weights, doc 46 section 36.3). The ARM
   * copy is dead the moment the memcpy above returns -- the matmul reads the
   * arena -- so this drops it. MADV_DONTNEED rather than free() because every
   * weight is a slice of ONE pool allocation that nobody may free piecemeal;
   * see whSourcePageRange for why the range is rounded inward and why the
   * pointer stays a valid handle_cache_ key afterwards.
   *
   * A read of these bytes after this point returns zeros, which would be a
   * silently wrong matmul. Nothing reads them: the only other reader is the
   * layer's per-expert CPU loop, and that path throws on a QS4CX_WH weight
   * (FloatTensor::dot) rather than computing with zeros. Set
   * NNTR_HTP_KEEP_ARM_WEIGHTS=1 to keep them anyway when bisecting a wrong
   * answer -- the model then needs both copies resident.
   */
  void releaseArmSource(void *src, size_t len) {
#if defined(__linux__)
    static const bool keep =
      std::getenv("NNTR_HTP_KEEP_ARM_WEIGHTS") != nullptr;
    if (keep)
      return;
    const long ps = sysconf(_SC_PAGESIZE);
    if (ps <= 0)
      return;
    uintptr_t begin = 0;
    size_t span = 0;
    whSourcePageRange(src, len, static_cast<size_t>(ps), &begin, &span);
    if (span != 0)
      madvise(reinterpret_cast<void *>(begin), span, MADV_DONTNEED);
#else
    (void)src;
    (void)len;
#endif
  }

  /**
   * @brief Registers a weight the offline quantizer already put in WH
   *        layout, by copying it into the arena as-is.
   *
   * No conversion, no DSP bake, no cache file: the bytes on disk are the
   * bytes the matmul reads (doc 46 section 35), and the DSP borrows them out
   * of the arena rather than keeping a heap copy. This is the whole of what
   * registration costs for a QS4CX_WH model.
   *
   * @param matAdata  whBytes(K, N) of WH nibbles
   * @param matAscale N scales, with N column sums immediately after them --
   *                  QS4CX_WH_Tensor's layout, which is why no separate
   *                  pointer is passed
   */
  uint32_t get_or_register_wh(void *matAdata, const float *matAscale,
                              remote_handle64 session, uint32_t K, uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;
    // A null scale is the layer's way of saying "this is a virtual expert's
    // key, the bytes are in your slot pool" (register_qs4cx_wh_expert_file).
    // Not finding it is a bookkeeping bug -- the caller's LRU let an expert
    // reach the call without acquiring it -- and reading the key as bytes
    // would compute a wrong answer quietly.
    if (matAscale == nullptr) {
      throw std::runtime_error(
        "QS4CX_WH expert weight (" + std::to_string(K) + "x" +
        std::to_string(N) +
        ") is not resident: register_qs4cx_wh_expert_file must precede the "
        "call for a virtual expert");
    }

    const uint64_t t_begin = HtpProfile::nowUs();
    // There is no other way to register these. The DSP-heap path bakes its
    // input, which would rearrange bytes that are already arranged, so a
    // model quantized to QS4CX_WH needs the arena and saying so plainly
    // beats computing a wrong answer quietly.
    if (!ensureArena(session)) {
      throw std::runtime_error(
        "QS4CX_WH weights need the DSP arena, which this device did not "
        "provide (no rpcmem_to_fd or no fastrpc_mmap). Quantize the model as "
        "QS4CX to use the conversion path instead.");
    }

    const uint32_t wh_len = static_cast<uint32_t>(whBytes(K, N));
    uint32_t chunk = 0, off = 0;
    // want = kArenaChunkMax, not wh_len: these weights arrive one at a time
    // with no total to size a chunk from, so every chunk is made full size
    // and filled by bump before the next. 3696 MiB of weights is 15 chunks,
    // against the 3840 MiB a clean process can map (doc 46 section 41) and
    // NNTR_HVX_MAX_ARENAS slots.
    if (!place(session, wh_len, kArenaChunkMax, &chunk, &off)) {
      // Everything the next step needs, because getting it costs a device
      // round: which of the four calls refused, how much was mapped when it
      // did, and this process's RSS -- which is how a run says whether
      // releaseArmSource gave the ARM copies back. Host RAM and DSP address
      // space fail the same way here and have nothing in common as fixes.
      throw std::runtime_error(
        "HTP arena: cannot register a " + std::to_string(K) + "x" +
        std::to_string(N) + " weight (" + std::to_string(wh_len >> 20) +
        " MiB). " +
        (arena_fail_.empty() ? "no chunk was attempted" : arena_fail_) +
        ". mapped=" + std::to_string(arenaBytes() >> 20) + " MiB in " +
        std::to_string(arena_chunks_.size()) +
        " chunks, RSS=" + std::to_string(rssKb() >> 10) + " MB");
    }
    std::memcpy(arena_chunks_[chunk].buf->data() + off, matAdata, wh_len);

    ArenaEntry e;
    e.chunk = chunk;
    e.off = off;
    e.K = K;
    e.N = N;
    e.w_scale.assign(matAscale, matAscale + N);
    e.colsum_w.resize(N);
    e.bias.assign(N, 0.0f); // FC weights carry no bias tensor
    // The column sums are the N floats after the scales, written there by
    // the quantizer. They are whole numbers small enough to be exact in
    // f32 -- a sum of at most K values in [-8, 7] -- so this conversion is
    // lossless, and the registry wants int32.
    const float *colsum_f = matAscale + N;
    for (uint32_t i = 0; i < N; ++i) {
      e.colsum_w[i] = static_cast<int32_t>(colsum_f[i]);
    }

    const uint32_t handle = registerFromArena(session, e, K, N, t_begin);
    if (handle == kNoHandle) {
      throw std::runtime_error("weight_register_u8i4_arena rejected a " +
                               std::to_string(K) + "x" + std::to_string(N) +
                               " WH weight");
    }
    // Only now: e.w_scale and e.colsum_w are already copies, so the source
    // buffer -- whose scales sit just past the nibbles -- has no reader left.
    releaseArmSource(matAdata, wh_len);
    // Still a usable key after releaseArmSource: the pages went, the mapping
    // did not, so no later allocation can take this address and turn a miss
    // into a hit on somebody else's weight.
    handle_cache_.emplace(matAdata, handle);
    return handle;
  }

  uint32_t get_or_register_qs4cx(void *matAdata, const float *matAscale,
                                 remote_handle64 session, uint32_t K,
                                 uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    return get_or_register_qs4cx_unlocked(matAdata, matAscale, session, K, N);
  }

  /** @note Call with handle_mutex_ already held. */
  uint32_t get_or_register_qs4cx_unlocked(void *matAdata,
                                          const float *matAscale,
                                          remote_handle64 session, uint32_t K,
                                          uint32_t N) {
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

    const uint64_t t_begin = HtpProfile::nowUs();

    // Convert on the host, bake on the DSP, keep the baked bytes on the DSP
    // heap: the 1.89 GB ceiling Gate 0 measured, and the reason QS4CX_WH
    // exists. A QS4CX model that fits under it still runs this way; one that
    // does not has to be quantized as QS4CX_WH, which needs no conversion,
    // no bake, and no DSP heap at all (doc 46 section 35).
    HtpRpcBuffer q_w4_i8(static_cast<size_t>(K) * N);
    std::vector<float> w_scale(N);
    std::vector<int32_t> colsum_w(N);
    const uint64_t t_convert = HtpProfile::nowUs();
    htp_qs4cx_from_packed(matAdata, matAscale, K, N,
                          reinterpret_cast<int8_t *>(q_w4_i8.data()),
                          w_scale.data(), colsum_w.data());
    const uint64_t convert_us = HtpProfile::nowUs() - t_convert;

    return register_locked(matAdata, session, K, N, q_w4_i8, w_scale, colsum_w,
                           t_begin, convert_us);
  }

  /** @brief Sentinel for "the cache did not have it"; a real handle is a
   *  table slot index, and 0 is a valid one. */
  static constexpr uint32_t kNoHandle = 0xFFFFFFFFu;

  /**
   * @brief Decides, once, whether weights can be registered out of the arena.
   *
   * Nothing is mapped here: the first weight's place() makes the first chunk.
   * All this answers is whether this device gives out the two calls an arena
   * is made of -- an fd for a host buffer and a way to attach it.
   *
   * @return true if weights can be registered out of the arena. false means
   *         the ordinary DSP-heap path, which still works for QS4CX -- it
   *         just has the 1.89 GB ceiling Gate 0 measured -- and which
   *         get_or_register_wh refuses, because WH bytes have no other home.
   * @note   Call with handle_mutex_ already held.
   */
  bool ensureArena(remote_handle64 session) {
    (void)session;
    if (arena_state_ != ARENA_UNTRIED)
      return arena_state_ == ARENA_ON;

    const HtpRpcMemApi &api = HtpRpcMemApi::get();
    arena_state_ =
      (api.to_fd != nullptr && api.mmap != nullptr) ? ARENA_ON : ARENA_OFF;
    return arena_state_ == ARENA_ON;
  }

  /**
   * @brief Finds room for @a bytes in the arena, making a chunk if needed.
   *
   * @param want how much more is expected to follow, so a new chunk is
   *             sized for the run rather than for this one weight
   * @note  Call with handle_mutex_ already held.
   */
  bool place(remote_handle64 session, uint32_t bytes, size_t want,
             uint32_t *chunk, uint32_t *off) {
    return placeOn(session, CDSP_DOMAIN_ID, arena_chunks_, chunk_cap_, bytes,
                   want, chunk, off);
  }

  /** Reserve separate 64 MiB chunks for optional Conv decode images.
   * Ordinary FC registration uses any spare arena room, so sharing its bump
   * allocator would also move unrelated weights from DSP heap to the arena.
   * Mark these chunks fully used there; track Conv's own bump pointer here.
   * They still belong to arena_chunks_ for the existing teardown hook. */
  bool placeConvDecode(remote_handle64 session, uint32_t bytes, uint32_t *chunk,
                       uint32_t *off) {
    for (auto &entry : conv_decode_chunks_) {
      const size_t size = arena_chunks_[entry.first].buf->size();
      if (entry.second <= size && bytes <= size - entry.second) {
        *chunk = entry.first;
        *off = entry.second;
        entry.second += bytes;
        return true;
      }
    }
    if (!newChunkOn(session, CDSP_DOMAIN_ID, arena_chunks_, conv_decode_cap_,
                    bytes, size_t(64) << 20))
      return false;
    *chunk = static_cast<uint32_t>(arena_chunks_.size() - 1u);
    *off = 0;
    arena_chunks_.back().used = arena_chunks_.back().buf->size();
    conv_decode_chunks_.emplace_back(*chunk, bytes);
    return true;
  }

  /** @brief place() on any session's chunk list: [#132 Part B E3] S2's FC
   *  arena is the same path on S2's handle and effective domain. */
  bool placeOn(remote_handle64 session, int domain,
               std::vector<ArenaChunk> &chunks, size_t &cap, uint32_t bytes,
               size_t want, uint32_t *chunk, uint32_t *off) {
    if (placeExistingOn(chunks, bytes, chunk, off))
      return true;
    if (!newChunkOn(session, domain, chunks, cap, bytes, want))
      return false;
    chunks.back().used = bytes;
    *chunk = static_cast<uint32_t>(chunks.size() - 1);
    *off = 0;
    return true;
  }

  /** @brief place() without the new chunk: room in a mapped chunk or
   *  nothing. What the FC slices use (get_or_register_fc), so that they
   *  never map address space the remaining heap registrations need. */
  bool placeExisting(uint32_t bytes, uint32_t *chunk, uint32_t *off) {
    return placeExistingOn(arena_chunks_, bytes, chunk, off);
  }
  bool placeExistingOn(std::vector<ArenaChunk> &chunks, uint32_t bytes,
                       uint32_t *chunk, uint32_t *off) {
    for (size_t c = 0; c < chunks.size(); ++c) {
      // 4 KB rather than the 512 the DSP requires: a weight that starts on
      // a page boundary is one the DMA never splits across a page for
      // alignment reasons alone, and the waste is under a part in 400.
      const size_t at = (chunks[c].used + 4095u) & ~size_t(4095u);
      if (at + bytes <= chunks[c].buf->size()) {
        chunks[c].used = at + bytes;
        *chunk = static_cast<uint32_t>(c);
        *off = static_cast<uint32_t>(at);
        return true;
      }
    }
    return false;
  }

  /** @brief Allocates one uncached ION buffer and attaches it to the DSP.
   *  @note  Call with handle_mutex_ already held. */
  /**
   * @brief Largest arena chunk to ask the DSP for.
   *
   * 256 MiB, and the size is the whole fix for the memory wall (doc 46
   * section 41). A clean process maps 3840 MiB in 256 MiB steps; the model
   * with 1 GiB chunks stopped at 3072 and was then refused at every size
   * down to 64 MiB. A flat byte budget cannot produce both. A bump
   * allocator that aligns each mapping to its own size can: 1 GiB chunks
   * land at 1G, 2G, 3G and push the cursor to the 4 GB end of the PD's
   * address space, so the 768 MiB below the first one is never reachable
   * again. 256 MiB chunks waste none of it.
   *
   * The model needs 3696 MiB of weights. That leaves ~144 MiB of margin,
   * which place() protects by scanning every chunk for room -- a 1.75 MiB
   * down weight fits tails a 3.5 MiB gate_up cannot.
   *
   * ponytail: if a bigger model needs more, the next thing to take is the
   * DSP heap, which answered AEE_ENOMEMORY after 182 MiB with 3840 mapped
   * -- heap and mappings share the one 4 GB space, so there is no second
   * budget to find, only this one to stop wasting.
   */
  static constexpr size_t kArenaChunkMax = size_t(256) << 20;
  /** @brief This process's resident set, in KB, or 0 if it cannot be read.
   *  The one number that says whether releaseArmSource actually gave the
   *  pages back -- MADV_DONTNEED never reports failure for a range it simply
   *  did not reclaim. */
  static unsigned long long rssKb() {
#if defined(__linux__)
    std::FILE *f = std::fopen("/proc/self/statm", "r");
    if (f == nullptr)
      return 0;
    unsigned long long total = 0, resident = 0;
    const int n = std::fscanf(f, "%llu %llu", &total, &resident);
    std::fclose(f);
    if (n != 2)
      return 0;
    return resident * 4u; // statm counts pages; 4 KB on every target here
#else
    return 0;
#endif
  }

  /** @brief Total bytes the arena has mapped to the DSP so far. */
  size_t arenaBytes() const {
    size_t total = 0;
    for (const ArenaChunk &c : arena_chunks_)
      total += c.buf->size();
    return total;
  }

  /**
   * @brief Attaches one arena chunk of exactly @a size bytes.
   *
   * @return true on success. On failure arena_fail_ says which of the four
   *         calls refused and with what, because they mean four unrelated
   *         things: rpcmem_alloc is the host's ION heap, fastrpc_mmap is the
   *         DSP's address space, arena_attach is the DSP's own arena table.
   */
  bool tryChunk(remote_handle64 session, size_t size) {
    return tryChunkOn(session, CDSP_DOMAIN_ID, arena_chunks_, size);
  }
  bool tryChunkOn(remote_handle64 session, int domain,
                  std::vector<ArenaChunk> &chunks, size_t size) {
    const unsigned long long rss_before = rssKb();
    // Uncached, because the DSP maps this once and the host keeps writing
    // into it afterwards -- with an uncached CPU mapping those writes reach
    // DDR with no flush to remember. The host never reads it back, so the
    // cost is on the write side only. The probe that passed Gate 0c wrote
    // BEFORE attaching and used a cached buffer, so this ordering is the
    // one thing section 34 rests on that the probe did not show; the
    // ArenaUncachedWriteAfterMap test is what answers it.
    auto buf = std::make_unique<HtpRpcBuffer>(size, HTP_RPC_FLAGS_UNCACHED);
    // [#132 Part B E5g] the buffer of a refused mapping lives until this
    // one exists, so the retry gets another fd: E5g's retries at 128 and 64
    // MiB after a refused 256 were refused AEE_EALREADY on the freed
    // buffer's fd number, which the driver still held
    last_failed_.reset();
    if (!buf->isIon()) {
      arena_fail_ = "rpcmem_alloc(" + std::to_string(size >> 20) +
                    " MiB) failed -- the HOST ION heap is out, so the ARM "
                    "copies are what to shrink";
      return false;
    }
    const int fd = buf->fd();
    if (fd < 0) {
      arena_fail_ = "rpcmem_to_fd returned no fd for an ION buffer";
      return false;
    }

    const HtpRpcMemApi &api = HtpRpcMemApi::get();
    // FASTRPC_MAP_FD, not 0: 0 is FASTRPC_MAP_STATIC, which is the driver's
    // mapping for a buffer passed as a call argument and is not tagged with
    // the fd, so HAP_mmap_get on the DSP refuses it. Three device runs went
    // on that (doc 46 section 32.10).
    const int merr = api.mmap(domain, fd, buf->data(), 0, size, FASTRPC_MAP_FD);
    if (merr != 0) {
      // the driver may keep the fd's entry after a refused map: drop it
      if (api.munmap != nullptr)
        api.munmap(domain, fd, buf->data(), size);
      last_failed_ = std::move(buf);
      arena_fail_ =
        "fastrpc_mmap(" + std::to_string(size >> 20) +
        " MiB) failed: err=" + std::to_string(merr) +
        " -- the host buffer exists, so this is the DSP side (its address "
        "space or the driver's mapping table), not host memory";
      return false;
    }

    uint32_t dsp_id = 0;
    const int aerr =
      nntr_hvx_arena_attach(session, fd, static_cast<uint32_t>(size), &dsp_id);
    if (aerr != AEE_SUCCESS) {
      // Hex as well as decimal: AEEStdErr offsets every code by 0x80000400
      // under __hexagon__, so 0x8000040d is the one that reads as a plain
      // AEE_EBADSTATE and a decimal print hides that.
      char err[80];
      std::snprintf(err, sizeof(err), "%zu MiB) failed: err=%d (0x%08x)",
                    size >> 20, aerr, static_cast<unsigned>(aerr));
      arena_fail_ = "nntr_hvx_arena_attach(" + std::string(err) +
                    " -- mapped, but the DSP refused it; NNTR_HVX_MAX_ARENAS "
                    "or HAP_mmap_get";
      if (api.munmap != nullptr)
        api.munmap(domain, fd, buf->data(), size);
      return false;
    }
    chunks.push_back(ArenaChunk{std::move(buf), dsp_id, 0});
    if (&chunks == &arena_chunks_ && chunks.size() == 1) {
      std::shared_ptr<std::vector<ArenaChunk>> a = s1_arena_;
      HtpBackend::global().atCloseLast(
        [a, session] { s1ArenaTeardown(session, *a); });
    }
    if (HtpProfile::global().level() != 0) {
      size_t total = 0;
      for (const ArenaChunk &c : chunks)
        total += c.buf->size();
      std::printf("[HTP] arena chunk %zu: %zu MiB, dsp_id=%u, mapped total "
                  "%zu MiB, RSS %llu -> %llu MB%s\n",
                  chunks.size() - 1, size >> 20, dsp_id, total >> 20,
                  rss_before >> 10, rssKb() >> 10,
                  domain == CDSP_DOMAIN_ID ? "" : " (S2)");
    }
    return true;
  }

  /**
   * @brief [#132 Part B E5i] The S1 arena's close hook, after the queues and
   *        the token drivers (atCloseLast: after every atClose hook): the
   *        skel releases the graph, the weights and every arena
   *        (arenas_release), then each chunk is fastrpc_munmap'd, every rc
   *        checked, before the session closes -- the S1Ceiling probe's
   *        order. E5f-E5h: about one app run in 20, A or E, left the next
   *        process one 256 MiB window short when the arena was left mapped
   *        to process exit (and its buffers rpcmem_free'd, still mapped, by
   *        whichever singleton died first).
   */
  static void s1ArenaTeardown(remote_handle64 session,
                              std::vector<ArenaChunk> &chunks) {
    if (chunks.empty())
      return;
    const HtpRpcMemApi &mem = HtpRpcMemApi::get();
    uint32_t put = 0, unmapped = 0;
    size_t bytes = 0;
    const int rc = nntr_hvx_arenas_release(session, &put);
    for (auto c = chunks.rbegin(); c != chunks.rend(); ++c) {
      bytes += c->buf->size();
      const int u = mem.munmap != nullptr
                      ? mem.munmap(CDSP_DOMAIN_ID, c->buf->fd(), c->buf->data(),
                                   c->buf->size())
                      : -1;
      if (u == 0)
        ++unmapped;
      else
        std::fprintf(stderr,
                     "[HTP] arena: fastrpc_munmap(chunk dsp_id=%u, %zu MiB) "
                     "failed: 0x%x\n",
                     c->dsp_id, c->buf->size() >> 20, static_cast<unsigned>(u));
    }
    std::fprintf(stderr,
                 "[HTP] arena: chunks unmapped %u/%zu mib=%zu (release rc=0x%x "
                 "put=%u)\n",
                 unmapped, chunks.size(), bytes >> 20,
                 static_cast<unsigned>(rc), put);
    chunks.clear(); // rpcmem_free, each buffer unmapped first
  }

  /**
   * @brief Adds a chunk, halving the request until one is accepted.
   *
   * The DSP refused a fourth 1 GiB mapping after three were accepted (doc 46
   * section 39), so this ceiling is measured in whole GiB and the last GiB
   * before it is left on the floor. Halving turns a ceiling that happens to
   * fall between two chunk sizes into one the arena can fill up to, and the
   * size the run settles on is itself the measurement -- there is no other
   * way to learn where the DSP's address space actually ends.
   *
   * chunk_cap_ remembers what was refused, so a size known not to fit is not
   * asked for again on the next weight. A failure to map is not free: the
   * host buffer is allocated and released each time.
   */
  bool newChunk(remote_handle64 session, uint32_t bytes, size_t want) {
    return newChunkOn(session, CDSP_DOMAIN_ID, arena_chunks_, chunk_cap_, bytes,
                      want);
  }
  bool newChunkOn(remote_handle64 session, int domain,
                  std::vector<ArenaChunk> &chunks, size_t &cap, uint32_t bytes,
                  size_t want) {
    static constexpr size_t kGrain = size_t(64) << 20;
    /** Below this a chunk holds too few weights to be worth an arena slot,
     *  and the DSP's table is not unbounded. [#132 Part B E5g] 16 MiB (was
     *  64): the tail of the MoE arena, ~112 MiB above 14 x 256, may have
     *  to go in pieces on a boot whose last 256 MiB window is taken. */
    static constexpr size_t kFloor = size_t(16) << 20;

    size_t size = (std::max(want, size_t(bytes)) + kGrain - 1) & ~(kGrain - 1);
    size = std::min(std::max(size, kGrain), cap);
    if (size < bytes) {
      arena_fail_ = "a weight is larger than a whole chunk";
      return false; // not this model
    }

    while (true) {
      arena_fail_.clear();
      if (tryChunkOn(session, domain, chunks, size))
        return true;
      const size_t half = size / 2;
      if (half < kFloor || half < bytes)
        return false; // arena_fail_ holds the last refusal, which is the one
      if (HtpProfile::global().level() != 0) {
        std::printf("[HTP] arena: %zu MiB refused, retrying at %zu MiB (%s)\n",
                    size >> 20, half >> 20, arena_fail_.c_str());
      }
      size = half;
      cap = size;
    }
  }

  /** @brief Registers a weight whose bytes are already in the arena.
   *  @return the handle, or kNoHandle if the DSP refused it.
   *  @param  t_begin caller's start time, or 0 on the miss path -- there
   *          register_locked already counted this weight and counting it
   *          again would report two registrations for one weight.
   *  @note   Call with handle_mutex_ already held. */
  uint32_t registerFromArena(remote_handle64 session, const ArenaEntry &e,
                             uint32_t K, uint32_t N, uint64_t t_begin) {
    // The file named a shape; the caller wants one. A directory holding
    // another model's files would otherwise register the wrong weight,
    // since the path is keyed on a hash of bytes this process never read.
    if (e.K != K || e.N != N)
      return kNoHandle;

    uint32_t handle = 0;
    const uint64_t t_rpc = HtpProfile::nowUs();
    const int err = nntr_hvx_weight_register_u8i4_arena(
      session, K, N, arena_chunks_[e.chunk].dsp_id, e.off, e.w_scale.data(),
      static_cast<int>(N), e.colsum_w.data(), static_cast<int>(N),
      e.bias.data(), static_cast<int>(N), &handle);
    const uint64_t rpc_us = HtpProfile::nowUs() - t_rpc;
    if (err != AEE_SUCCESS)
      return kNoHandle;

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0 && t_begin != 0)
      profile.addRegister(HtpProfile::nowUs() - t_begin, /*convert_us=*/0,
                          rpc_us, /*ion=*/true);
    return handle;
  }

  /** @brief Register a converted weight and cache its handle.
   *  @note  Call with handle_mutex_ already held.
   *  @param t_begin   caller's start timestamp, for the profile's
   *                   registration total (unused when profiling is off)
   *  @param convert_us microseconds the caller spent in htp_qs4cx_from_* */
  uint32_t register_locked(void *key, remote_handle64 session, uint32_t K,
                           uint32_t N, HtpRpcBuffer &q_w4_i8,
                           std::vector<float> &w_scale,
                           std::vector<int32_t> &colsum_w, uint64_t t_begin,
                           uint64_t convert_us) {
    std::vector<float> bias(N, 0.0f); // FC weights carry no bias tensor

    uint32_t handle = 0;
    const uint64_t t_rpc = HtpProfile::nowUs();
    const int err = nntr_hvx_weight_register_u8i4(
      session, K, N, reinterpret_cast<int8_t *>(q_w4_i8.data()),
      static_cast<int>(q_w4_i8.size()), w_scale.data(), static_cast<int>(N),
      colsum_w.data(), static_cast<int>(N), bias.data(), static_cast<int>(N),
      &handle);
    const uint64_t rpc_us = HtpProfile::nowUs() - t_rpc;
    if (err != AEE_SUCCESS) {
      throw std::runtime_error("nntr_hvx_weight_register_u8i4 failed: err=" +
                               std::to_string(err));
    }
    HtpProfile &profile = HtpProfile::global();
    if (profile.level() != 0) {
      profile.addRegister(HtpProfile::nowUs() - t_begin, convert_us, rpc_us,
                          q_w4_i8.isIon());
    }

    // Kept resident for the process lifetime -- Stage 6 (residency, see
    // 40_moe_ffn_htp_task.md) is what has to bound this table's size and
    // add release for the full-checkpoint case; not needed yet.
    handle_cache_.emplace(key, handle);
    return handle;
  }

  std::mutex handle_mutex_;
  std::unordered_map<const void *, uint32_t> handle_cache_;
  /** FC weights by data pointer -> their handle(s); see get_or_register_fc.
   *  Values are never erased or moved, so the references it hands out stay
   *  valid (std::unordered_map keeps node addresses across rehash). */
  std::unordered_map<const void *, FcHandles> fc_cache_;
  std::unordered_map<const void *, uint64_t> prepared_fc_hashes_;
  /** Dense FFNs by their up weight's pointer; see get_or_register_dense. */
  std::unordered_map<const void *, DenseHandles> dense_cache_;
  /** Conv blocks by their in_proj's pointer; see get_or_register_conv. */
  std::unordered_map<const void *, ConvHandles> conv_cache_;

  /** S1's arena, shared with its close hook (s1ArenaTeardown): the two
   *  singletons' destruction order is not fixed. */
  std::shared_ptr<std::vector<ArenaChunk>> s1_arena_ =
    std::make_shared<std::vector<ArenaChunk>>();
  std::vector<ArenaChunk> &arena_chunks_ = *s1_arena_;
  std::vector<std::pair<uint32_t, uint32_t>> conv_decode_chunks_;
  size_t conv_decode_cap_ = size_t(64) << 20;
  /** Expert slots given back by release_qs4cx_wh_expert, all of
   *  expert_slot_bytes_; a miss takes one of these before it bumps a
   *  chunk, so after the load the chunk count never moves. */
  std::vector<ExpertSlot> free_expert_slots_;
  std::unordered_map<const void *, ExpertResident> experts_;
  std::vector<StagedExpert> pending_expert_updates_;
  bool decode_expert_updates_ = false, expert_updates_failed_ = false;
  uint64_t expert_updates_queued_ = 0, expert_updates_abandoned_ = 0;
  uint64_t expert_update_fused_batches_ = 0, expert_update_rpc_batches_ = 0;
  uint64_t expert_update_rpc_us_ = 0, expert_update_dsp_us_ = 0;
  size_t expert_slot_bytes_ = 0;
  std::array<unsigned, 4> expert_slot_shape_{};
  bool expert_shapes_uniform_ = true;
  /** reserve_qs4cx_wh_expert_slots's count, and slots placed so far. */
  size_t expert_slots_wanted_ = 0, expert_slots_placed_ = 0;
  /** Slices of one expert weight's file read: 3.5 MiB over 8 is 448 KiB a
   *  pread, past which the per-call cost stops paying for the split. */
  static constexpr size_t kExpertReadSlicesMax = 8;
  /** Experts read in the background between prefetch _begin and _end, and
   *  the threads reading them -- one expert per thread at a time. Four:
   *  the write rate into the arena stops scaling well before that (doc 52
   *  section 10.9), and the ARM side has its own work after the call. */
  std::vector<StagedExpert> prefetch_;
  /** One _begin's experts; done counts the ones read. */
  struct PrefetchBatch {
    std::vector<StagedExpert> experts;
    bool decode = false;
    size_t done = 0;
    uint64_t done_mask =
      0; /**< First 64 jobs, for decode deadline diagnostics. */
    uint64_t recycle_mask = 0; /**< Discarded jobs still holding slots. */
  };
  /** Queued batches, oldest first, and their unread experts, all under
   *  prefetch_mutex_. A batch's address is stable until _end takes it. */
  std::mutex prefetch_mutex_;
  std::condition_variable prefetch_cv_, prefetch_done_cv_;
  std::deque<std::unique_ptr<PrefetchBatch>> prefetch_batches_;
  std::deque<std::unique_ptr<PrefetchBatch>> prefetch_retired_;
  std::deque<std::pair<PrefetchBatch *, size_t>> prefetch_jobs_;
  std::mutex host_scope_mutex_;
  std::shared_ptr<HostExpertBank> host_expert_bank_; // prefetch_mutex_
  uint64_t host_expert_epoch_ = 0;
  std::atomic<uint64_t> host_expert_wait_us_{0};
  bool prefetch_stop_ = false;
  std::atomic<uint64_t> qs2_reads_{0}, qs2_read_us_{0}, qs2_copy_us_{0},
    qs2_metadata_us_{0}, qs2_copied_bytes_{0};
#if defined(__linux__) || defined(__ANDROID__)
  HtpQs2MappedFiles qs2_mapped_files_;
#endif
  uint64_t qs2_demand_batches_ = 0, qs2_parallel_batches_ = 0,
           qs2_max_misses_ = 0;
  std::vector<std::thread> prefetch_readers_;
  std::atomic<uint32_t> prefetch_reader_cpus_{0};
  uint32_t prefetch_caller_cpus_ = 0;

  /**
   * @brief [doc 52 section 10.18] Measurement switches for the prefetch
   *        readers, read once. Section 10.13 found the readers make almost
   *        no progress while the layer call runs and the call's host side
   *        grows by 7 ms; section 10.19 traced that to the readers
   *        sharing the caller's core. Not defaults.
   *  - NNTR_MOE_PREFETCH_READERS=<n>: reader threads (4)
   *  - NNTR_MOE_PREFETCH_CPUS=<a,b,..>: pin the readers to these cores
   *    (default: every core but the caller's)
   */
  struct PrefetchKnobs {
    size_t readers = 4;
    std::vector<int> cpus;
  };
  static const PrefetchKnobs &prefetchKnobs() {
    static const PrefetchKnobs k = [] {
      PrefetchKnobs r;
      if (const char *v = std::getenv("NNTR_MOE_PREFETCH_READERS"))
        r.readers = std::max<size_t>(1, std::strtoul(v, nullptr, 10));
      if (const char *v = std::getenv("NNTR_MOE_PREFETCH_CPUS")) {
        for (const char *p = v; *p != '\0';) {
          char *end = nullptr;
          const long c = std::strtol(p, &end, 10);
          if (end == p)
            break;
          if (c >= 0 && c < 32)
            r.cpus.push_back(static_cast<int>(c));
          p = (*end == ',') ? end + 1 : end;
        }
      }
      return r;
    }();
    return k;
  }
  /** @brief This thread's core as a bit, 0 where that is not known. */
  static uint32_t cpuBit() {
#if defined(__linux__)
    const int c = sched_getcpu();
    return (c >= 0 && c < 32) ? (1u << c) : 0u;
#else
    return 0u;
#endif
  }
  static void pinToCpus(const std::vector<int> &cpus) {
#if defined(__linux__)
    if (cpus.empty())
      return;
    cpu_set_t set;
    CPU_ZERO(&set);
    for (int c : cpus)
      CPU_SET(c, &set);
    sched_setaffinity(0, sizeof(set), &set);
#else
    (void)cpus;
#endif
  }
  enum ArenaState { ARENA_UNTRIED, ARENA_ON, ARENA_OFF };
  ArenaState arena_state_ = ARENA_UNTRIED;
  /** Why the last newChunk refused, in words, for the throw that follows. */
  std::string arena_fail_;
  /** Largest chunk still worth asking for; only ever shrinks. */
  size_t chunk_cap_ = kArenaChunkMax;
  /** [#132 Part B E5g] The last refused chunk's buffer, freed after the
   *  next one is allocated (tryChunkOn). */
  std::unique_ptr<HtpRpcBuffer> last_failed_;

  // [#132 Part B E5i] Unmapped by the close hook s1ArenaTeardown (it was
  // left mapped to process exit until E5h). ponytail: a process that loads
  // and unloads models still keeps every arena to its end; a reload would
  // need the same hook at model teardown.

  // ION-backed activation/output scratch, reused across calls, one buffer
  // per size class (stage) -- see invokeLayer's comment. Guarded by the
  // same mutex that serializes every call into the one HTP session.
  std::mutex invoke_mutex_;
  std::once_flag moe_opts_once_;   /**< sendMoeOptsOnce */
  uint32_t applied_moe_flags_ = 0; /**< Echoed session options. */
  // [#85] The decode op list from set_decode_graph_desc, its MoE ops in
  // list order, how many have their handles bound (first-pass call order),
  // the op each layer's first gate_up handle belongs to, and whether
  // graph_init has run on the session.
  std::mutex graph_mutex_;
  std::vector<uint32_t> graph_words_;
  std::vector<uint32_t> moe_ops_;
  size_t moe_bound_ = 0;
  std::unordered_map<uint32_t, uint32_t> moe_op_by_handle_;
  bool graph_inited_ = false;
  bool graph_short_warned_ = false;
  // [#130] The stretch tables of section 3.2: per op the maximal resident
  // run it belongs to, per kind the resident ops in list order (what the
  // hooks' counters index), per op whether its parameter is bound and,
  // for CONV1D_GATE, the position its DSP state is valid for; per
  // attention ordinal the DSP cache's length (the kernel's kv_len,
  // mirrored). pending_in_ is the row of the stretch's first op (QK_NORM,
  // ADD, or a MOE op with its routing) until the stretch's last hook runs
  // it (#132's role dispatch). Touched by the model's thread only.
  uint32_t resident_mask_ = 0;
  std::vector<uint32_t> stretch_start_, stretch_end_, attn_ordinal_;
  std::vector<uint32_t> kind_ops_[HTP_OP_KIND_N];
  uint32_t kind_next_[HTP_OP_KIND_N] = {0};
  uint32_t cur_pos_ = HTP_GRAPH_NO_OP;
  bool row_bound_ = false; /**< every MoE op bound when this row began */
  std::vector<uint8_t> param_bound_;
  std::vector<uint32_t> conv_next_pos_;
  std::vector<uint32_t> kv_len_;
  bool rope_bound_ = false;
  bool attn_registered_ = false;
  uint32_t seed_ordinal_ = HTP_GRAPH_NO_OP;
  uint32_t pending_op_ = HTP_GRAPH_NO_OP;
  std::vector<float> pending_in_;
  // [#132] the routing a MOE op that starts a stretch was handed
  std::vector<unsigned int> pending_ri_, pending_rc_;
  std::vector<float> pending_rw_;
  // [#132 Part B] the model's Q4_0 weights for the Q4M1 kinds, in list
  // order, until bindQ4m1; the handles registered for them
  struct Q4Pending {
    const void *data;
    uint32_t K, N;
    bool canonical;
  };
  std::vector<Q4Pending> q4_pending_;
  std::vector<uint32_t> q4m1_handles_;
  /** @brief The lm_head's slices by weight (lmHeadHandles); ok false: it
   *  could not be placed and stays on the CPU. */
  struct LmHeadSlices {
    bool ok = false;
    std::vector<uint32_t> h;
  };
  std::map<const void *, LmHeadSlices> lm_head_;
  uint32_t first_resident_op_ = HTP_GRAPH_NO_OP;
  uint64_t fwd_calls_ = 0;      /**< nntr_hvx_forward* calls */
  uint64_t fwd_tokens_ = 0;     /**< of them at the first resident op */
  uint64_t cpu_fc_skipped_ = 0; /**< [#132 Part B] decode_row_resident's
                                     true answers: CPU FC calls skipped */
  StagingPool act_pool_;
  std::array<uint64_t, 8> math_calls_{};
  uint64_t sigmoid_calls_ = 0;
  uint64_t fused_router_calls_ = 0, fused_qkv_calls_ = 0;
  struct MathTiming {
    uint64_t calls = 0, prepare_us = 0, rpc_us = 0, dsp_us = 0, copyout_us = 0,
             bytes = 0;
  };
  std::array<std::array<MathTiming, 8>, 2> math_timing_{};
  HtpActivationReuse moe_input_reuse_;
  uint64_t prepared_moe_cookie_ = 0, prepared_moe_lease_ = 0;
  StagingPool out_pool_;
  StagingPool
    rope_prefill_pool_[6];    /**< independent Q/K/V/phase/cache staging */
  StagingPool rope_pool_;     /**< the fused projection call's RoPE rows */
  StagingPool x2_pool_;       /**< the MoE call's post-FFN second addend */
  StagingPool attn_q_pool_;   /**< the row-blocked attention's f32 Q */
  StagingPool attn_out_pool_; /**< and its f32 output */
  /** FP16 prefill scratch, protected by invoke_mutex_ and separate from q2. */
  StagingPool attn_native_k_pool_, attn_native_v_pool_;
  StagingPool attn_q8_input_pool_, attn_q8_output_pool_;
  std::mutex attn_kind_mutex_;
  std::map<int, unsigned int> attn_cache_kinds_;
  StagingPool attn_f16_q_pool_;
  StagingPool attn_f16_out_pool_;
  /** @brief ION copies of the MoE routers' gate weights, by address. */
  std::map<const float *, std::unique_ptr<HtpRpcBuffer>> router_w_;
  std::map<const float *, uint32_t> router_u8i8_; /**< gate -> u8i8 handle */
  /** [#141] The M==1 MoE call's dspqueue; null until the first such call
      unless NNTR_HTP_DSPQ=0. */
  std::shared_ptr<DspqMoe> dspq_;
  /** [#132 Part B E3] The two-session path's state, shared with its
   *  HtpBackend close hook (which runs after this object is gone). */
  struct E2eState {
    remote_handle64 h1 = 0, h2 = 0;
    int dom2 = -1;
    std::vector<ArenaChunk> arena; /**< S2's FC arena (Q4M1, attached) */
    size_t arena_cap = kArenaChunkMax;
    size_t attach_bytes = 0;
    std::vector<uint32_t> q4m1;         /**< S2's q4m1_attach handles */
    std::unique_ptr<HtpRpcBuffer> mbox; /**< the page both sessions map */
    bool mbox1 = false, mbox2 = false, drv1 = false, drv2 = false;
    bool graph2 = false;
    std::shared_ptr<DspqMoe> q1, q2; /**< S1's (dspq_) and S2's queues */
    uint32_t tok = 0;
    uint64_t tokens = 0, hops = 0, wait1_us = 0, wait2_us = 0, pcyc1 = 0,
             pcyc2 = 0, id_checked = 0, id_mismatch = 0;
    uint64_t wall1_us = 0, wall2_us = 0, wall1_pcyc = 0, wall2_pcyc = 0,
             token_us = 0; /**< per side; token_us: the ARM's round trip */
    /** [#194 L0] the hops' own latency per side (post to wake-up), the
     *  ARM's time inside tokenForward, and between one tokenForward's end
     *  and the next one's start (the layer walk, sampler, tokenizer, print;
     *  arm_n intervals) */
    uint64_t hop1_us = 0, hop2_us = 0, fwd_us = 0, arm_us = 0, arm_n = 0,
             last_exit_us = 0;
    /** [#194 L0] wake split on the shared system counter: ARM post to the
     *  DSP thread's read (per session), S2's write to the ARM's read, S2's
     *  packet handling around its token (out - in) */
    int64_t disp1_us = 0, disp2_us = 0, ret2_us = 0, inout2_us = 0;
    uint64_t kind1[HTP_OP_KIND_N] = {0}, kind2[HTP_OP_KIND_N] = {0};
    uint32_t rounds = 0, spin_us = 0;
  };
  /** @brief The mailbox page: HEXKL_MBOX_BYTES (16 896) rounded to the
   *  #178 probe's 64 KiB. */
  static constexpr size_t kMboxBytes = size_t(64) << 10;
  bool e2e_ = false; /**< NNTR_HTP_E2E=1 and a description set */
  std::shared_ptr<E2eState> e2e_st_;
  bool q4m1_bound_ = false; /**< the Q4M1 weights registered at load (E3) */
  size_t q4m1_left_ = 0;    /**< S2 arena bytes still to place */
  bool want_logits_ = true; /**< set_decode_logits */
  std::vector<uint32_t> ban_, ban_sent_; /**< set_decode_ban, S2's LM_BAN */
  bool have_id_ = false;                 /**< take_decode_token_id */
  uint32_t pending_id_ = 0;
  /** Conv-only uncached staging; never shared with cached FC/MoE buffers. */
  StagingPool conv_act_pool_;
  StagingPool conv_out_pool_;
  /** invokeConvBlock's conv_w and bidirectional state, one small ION buffer. */
  std::unique_ptr<HtpRpcBuffer> conv_buf_;

  // invokeLayerU8In's scratch: the AH-packed activation (ION-backed, same
  // reasoning as act_pool_/out_pool_) and the small per-row scale/zp arrays
  // it produces alongside it (plain heap -- a few KB at most, not worth
  // ION's pin/map bookkeeping).
  std::unique_ptr<HtpRpcBuffer> act_ah_buf_;
  std::vector<float> act_scale_scratch_;
  std::vector<int32_t> act_zp_scratch_;

#ifndef NNTR_HTP_INPROC
  // The attention entries below have no in-process stand-in (their
  // kernels need HMX fp16 and intrinsics hvx_emu does not emulate), so
  // the host twin leaves them on the CPU defaults.
public:
  bool supports_sdpa_fp16_kvcache() const override {
    return HtpBackend::global().enabled();
  }

  // The KV cache the op below reads: in rpcmem, FastRPC hands the used
  // range to the DSP by mapping, not by copying it every call.
  void *alloc_shared(size_t bytes) override {
    return HtpBackend::global().alloc_shared(bytes);
  }
  void free_shared(void *block) override {
    HtpBackend::global().free_shared(block);
  }

  // --- quantized (int8 / int4) KV cache resident on the DSP ---

  bool supports_kv_cache_q() const override {
    return HtpBackend::global().enabled();
  }

  int kv_cache_q_register(unsigned int kind, unsigned int max_rows,
                          unsigned int n_head_kv,
                          unsigned int head_dim) override {
    HtpBackend &hb = HtpBackend::global();
    // head_dim up to 512 is the registry's limit; the column-blocked
    // kernel below checks its own 256 at call time.
    if (!hb.enabled() || (kind > 1 && kind != HTP_KV_KIND_F16) ||
        max_rows == 0 || max_rows > 0xFFFFu || n_head_kv == 0 ||
        head_dim == 0 || (head_dim % 32) != 0 || head_dim > 512) {
      return -1;
    }
    uint32_t h = 0;
    const int err =
      nntr_hvx_kv_register_q(static_cast<remote_handle64>(hb.handle()), kind,
                             max_rows, n_head_kv, head_dim, &h);
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized KV cache: register failed: 0x%x", err);
      return -1;
    }
    {
      std::lock_guard<std::mutex> lock(attn_kind_mutex_);
      attn_cache_kinds_[static_cast<int>(h)] = kind;
    }
    return static_cast<int>(h);
  }

  bool kv_cache_q_append(int handle, unsigned int row0, unsigned int n_rows,
                         unsigned int kv_stride, const uint16_t *k_rows,
                         const uint16_t *v_rows) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_rows == 0 || kv_stride == 0) {
      return false;
    }
    const int len = static_cast<int>(n_rows * kv_stride);
    const int err = nntr_hvx_kv_append_q(
      static_cast<remote_handle64>(hb.handle()), static_cast<uint32_t>(handle),
      row0, k_rows, len, v_rows, len);
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized KV cache: append failed: 0x%x", err);
      return false;
    }
    return true;
  }

  void kv_cache_q_release(int handle) override {
    HtpBackend &hb = HtpBackend::global();
    {
      std::lock_guard<std::mutex> lock(attn_kind_mutex_);
      attn_cache_kinds_.erase(handle);
    }
    if (hb.enabled() && handle >= 0) {
      nntr_hvx_kv_release_q(static_cast<remote_handle64>(hb.handle()),
                            static_cast<uint32_t>(handle));
    }
  }

  bool sdpa_q8_npu(int handle, unsigned row0, unsigned rows, const uint16_t *k,
                   const uint16_t *v, const float *q, unsigned n_q,
                   unsigned cache_from, unsigned cache_to, unsigned q_heads,
                   unsigned kv_heads, unsigned hd, unsigned window,
                   float softcap, const float *sinks, float *out) {
    if (!q || !out || n_q > 65535u || rows > 65535u || q_heads > 4096u)
      return false;
    const uint64_t q_count = uint64_t(n_q) * q_heads * hd;
    const uint64_t kv_count = uint64_t(rows) * kv_heads * hd;
    if (q_count > INT32_MAX / sizeof(float) ||
        kv_count > INT32_MAX / sizeof(uint16_t))
      return false;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    auto &backend = HtpBackend::global();
    const int64_t start = now_us();
    const size_t q_bytes = size_t(q_count) * sizeof(float);
    const size_t kv_bytes = size_t(kv_count) * sizeof(uint16_t);
    const float *rpc_q = q;
    const uint16_t *rpc_k = k, *rpc_v = v;
    if (!backend.isSharedRange(q, q_bytes)) {
      rpc_q =
        reinterpret_cast<float *>(stage(attn_f16_q_pool_, q_bytes).data());
      stagedMemcpy(const_cast<float *>(rpc_q), q, q_bytes);
    }
    if (rows && !backend.isSharedRange(k, kv_bytes)) {
      rpc_k = reinterpret_cast<uint16_t *>(
        stage(attn_native_k_pool_, kv_bytes).data());
      stagedMemcpy(const_cast<uint16_t *>(rpc_k), k, kv_bytes);
    }
    if (rows && !backend.isSharedRange(v, kv_bytes)) {
      rpc_v = reinterpret_cast<uint16_t *>(
        stage(attn_native_v_pool_, kv_bytes).data());
      stagedMemcpy(const_cast<uint16_t *>(rpc_v), v, kv_bytes);
    }
    const bool direct = backend.isSharedRange(out, q_bytes);
    float *result =
      direct
        ? out
        : reinterpret_cast<float *>(stage(attn_f16_out_pool_, q_bytes).data());
    const int64_t prepared = now_us();
    uint32_t stats[20] = {};
    const int err = nntr_hvx_attn_q8_npu_step(
      static_cast<remote_handle64>(backend.handle()), uint32_t(handle), row0,
      rpc_k, int(kv_count), rpc_v, int(kv_count), n_q, cache_from, cache_to,
      q_heads, window, softcap, rpc_q, int(q_count), sinks,
      sinks ? int(q_heads) : 0, result, int(q_count), stats, 20);
    const int64_t returned = now_us();
    if (!direct && err == AEE_SUCCESS)
      stagedMemcpy(out, result, q_bytes);
    if (attn_trace_enabled())
      ml_logi("HTP attn trace q8npu: n_q=%u cache=%u..%u err=0x%x "
              "prepare_us=%lld rpc_us=%lld wall_us=%lld dsp_us=%u "
              "quant_us=%u append_us=%u attn_us=%u direct_out=%u",
              n_q, cache_from, cache_to, err,
              static_cast<long long>(prepared - start),
              static_cast<long long>(returned - prepared),
              static_cast<long long>(now_us() - start), stats[3], stats[4],
              stats[0], stats[1], unsigned(direct));
    return err == AEE_SUCCESS;
  }

  bool sdpa_q8_transport(int handle, unsigned int row0, unsigned int n_rows,
                         const uint16_t *k, const uint16_t *v, const float *q,
                         unsigned int n_q, unsigned int cache_from,
                         unsigned int cache_to, unsigned int n_head_q,
                         unsigned int n_head_kv, unsigned int hd,
                         unsigned int window, float softcap, const float *sinks,
                         float *out) {
    if (!q || !out || n_q > 65535u || n_rows > 65535u || n_head_q > 4096u ||
        n_head_kv > n_head_q || hd > 256u)
      return false;
    std::lock_guard<std::mutex> lock(invoke_mutex_);
    const int64_t start = now_us();
    const size_t kv_heads = static_cast<size_t>(n_rows) * n_head_kv;
    const size_t q_heads = static_cast<size_t>(n_q) * n_head_q;
    const size_t kv_bytes = kv_heads * hd, q_bytes = q_heads * hd;
    const size_t metadata = (kv_heads * 2u + q_heads * 2u) * sizeof(float);
    const size_t input_bytes = metadata + 2u * kv_bytes + q_bytes;
    if (input_bytes > static_cast<size_t>(INT32_MAX) ||
        q_bytes > static_cast<size_t>(INT32_MAX) / sizeof(float))
      return false;
    uint8_t *input = reinterpret_cast<uint8_t *>(
      stage(attn_q8_input_pool_, input_bytes).data());
    float *scales = reinterpret_cast<float *>(input);
    float *qs = scales + 2u * kv_heads;
    int32_t *qz = reinterpret_cast<int32_t *>(qs + q_heads);
    uint8_t *k8 = input + metadata, *v8 = k8 + kv_bytes, *q8 = v8 + kv_bytes;
    for (size_t i = 0; i < kv_heads; ++i) {
      if (!htp_quant_kv_i8(k + i * hd, hd, k8 + i * hd, scales[2u * i]) ||
          !htp_quant_kv_i8(v + i * hd, hd, v8 + i * hd, scales[2u * i + 1u]))
        return false;
    }
    for (size_t i = 0; i < q_heads; ++i)
      if (!htp_quant_query_u8(q + i * hd, hd, q8 + i * hd, qs[i], qz[i]))
        return false;
    float *result = reinterpret_cast<float *>(
      stage(attn_q8_output_pool_, q_bytes * sizeof(float)).data());
    const int64_t prepared = now_us();
    uint32_t stats[16] = {0};
    const char *hmx_env = std::getenv("NNTR_HTP_ATTN_Q8_HMX");
    const unsigned mode =
      hmx_env ? static_cast<unsigned>(std::atoi(hmx_env)) : 0u;
    const bool hmx_decode = n_q == 1 && mode != 0u;
    if (mode > 3u)
      throw std::invalid_argument("NNTR_HTP_ATTN_Q8_HMX must be 0..3");
    const char *bc_env = std::getenv("NNTR_HTP_ATTN_Q8_BC");
    const unsigned bc = bc_env ? static_cast<unsigned>(std::atoi(bc_env)) : 0u;
    const int err =
      hmx_decode
        ? nntr_hvx_attn_q8_hmx_step(
            static_cast<remote_handle64>(HtpBackend::global().handle()),
            static_cast<uint32_t>(handle), row0, n_rows, n_q, cache_from,
            cache_to, n_head_q, window, softcap, k8, static_cast<int>(kv_bytes),
            v8, static_cast<int>(kv_bytes), scales,
            static_cast<int>(2u * kv_heads), q8, static_cast<int>(q_bytes), qs,
            static_cast<int>(q_heads), qz, static_cast<int>(q_heads), sinks,
            sinks ? static_cast<int>(n_head_q) : 0, result,
            static_cast<int>(q_bytes), stats, 16,
            mode == 1u   ? 0u
            : mode == 2u ? 1u
                         : 3u,
            bc)
        : nntr_hvx_attn_q8_step(
            static_cast<remote_handle64>(HtpBackend::global().handle()),
            static_cast<uint32_t>(handle), row0, n_rows, n_q, cache_from,
            cache_to, n_head_q, window, softcap, k8, static_cast<int>(kv_bytes),
            v8, static_cast<int>(kv_bytes), scales,
            static_cast<int>(2u * kv_heads), q8, static_cast<int>(q_bytes), qs,
            static_cast<int>(q_heads), qz, static_cast<int>(q_heads), sinks,
            sinks ? static_cast<int>(n_head_q) : 0, result,
            static_cast<int>(q_bytes), stats, attn_trace_enabled() ? 16 : 4);
    if (hmx_decode && err != AEE_SUCCESS)
      throw std::runtime_error("Explicit Q8 HMX decode failed: " +
                               std::to_string(err));
    const int64_t returned = now_us();
    if (err == AEE_SUCCESS)
      stagedMemcpy(out, result, q_bytes * sizeof(float));
    if (attn_trace_enabled())
      ml_logi(
        "HTP attn trace q8wire: n_q=%u cache=%u..%u err=0x%x "
        "prepare_us=%lld rpc_us=%lld wall_us=%lld dsp_us=%u "
        "append_us=%u attn_us=%u input_bytes=%llu output_bytes=%llu path=%s",
        n_q, cache_from, cache_to, err,
        static_cast<long long>(prepared - start),
        static_cast<long long>(returned - prepared),
        static_cast<long long>(now_us() - start), stats[3], stats[0], stats[1],
        static_cast<unsigned long long>(input_bytes),
        static_cast<unsigned long long>(q_bytes * sizeof(float)),
        hmx_decode ? "HMX" : (n_q < 5 ? "HVX" : "HMX-prefill"));
    if (attn_trace_enabled() && (n_q >= 5 || hmx_decode) && err == AEE_SUCCESS)
      ml_logi("HTP attn phases q8wire: n_q=%u qprep_us=%u dma_us=%u qk_us=%u "
              "dequant_us=%u softmax_us=%u pquant_us=%u pv_us=%u "
              "oupd_us=%u store_us=%u total_us=%u kcycles=%u blocks=%u",
              n_q, stats[4], stats[5], stats[6], stats[7], stats[8], stats[9],
              stats[10], stats[11], stats[12], stats[13], stats[14], stats[15]);
    if (err != AEE_SUCCESS)
      ml_logw("HTP producer-quantized attention failed: 0x%x", err);
    return err == AEE_SUCCESS;
  }

  bool sdpa_q_kvcache(int handle, unsigned int append_row0,
                      unsigned int append_rows, unsigned int kv_stride,
                      const uint16_t *k_rows, const uint16_t *v_rows,
                      const float *q, unsigned int q_stride, unsigned int n_q,
                      unsigned int cache_from, unsigned int cache_to,
                      unsigned int n_head_q, unsigned int n_head_kv,
                      unsigned int head_dim, unsigned int window, float softcap,
                      const float *sinks, float *out,
                      unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 256 || cache_to < cache_from + n_q || cache_to > 0xFFFFu ||
        q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        (append_rows != 0 &&
         (kv_stride != n_head_kv * head_dim || !k_rows || !v_rows))) {
      return false;
    }
    const char *q8_wire = std::getenv("NNTR_HTP_ATTN_Q8_WIRE");
    bool q8_cache = false;
    if (q8_wire && std::atoi(q8_wire) != 0) {
      std::lock_guard<std::mutex> lock(attn_kind_mutex_);
      const auto it = attn_cache_kinds_.find(handle);
      q8_cache = it != attn_cache_kinds_.end() && it->second == 0u;
    }
    const char *npu_quant = std::getenv("NNTR_HTP_ATTN_Q8_NPU");
    if (q8_cache && n_q >= 5u && head_dim <= 256u &&
        (!npu_quant || std::atoi(npu_quant) != 0))
      return sdpa_q8_npu(handle, append_row0, append_rows, k_rows, v_rows, q,
                         n_q, cache_from, cache_to, n_head_q, n_head_kv,
                         head_dim, window, softcap, sinks, out);
    if (q8_cache)
      return sdpa_q8_transport(
        handle, append_row0, append_rows, k_rows, v_rows, q, n_q, cache_from,
        cache_to, n_head_q, n_head_kv, head_dim, window, softcap, sinks, out);
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int sinks_len = sinks ? static_cast<int>(n_head_q) : 0;
    const int rows_len = static_cast<int>(append_rows * kv_stride);
    uint32_t stats[20] = {0};
    const bool native_decode =
      (static_cast<uint32_t>(handle) & HTP_KV_F16_HANDLE) && n_q == 1;
    std::unique_lock<std::mutex> lock(invoke_mutex_, std::defer_lock);
    const float *rpc_q = q;
    float *rpc_out = out;
    const uint16_t *rpc_k = k_rows, *rpc_v = v_rows;
    if (native_decode) {
      lock.lock();
      const size_t bytes = static_cast<size_t>(q_len) * sizeof(float);
      rpc_q = reinterpret_cast<float *>(stage(attn_f16_q_pool_, bytes).data());
      rpc_out =
        reinterpret_cast<float *>(stage(attn_f16_out_pool_, bytes).data());
      stagedMemcpy(const_cast<float *>(rpc_q), q, bytes);
      if (rows_len) {
        const size_t kv_bytes =
          static_cast<size_t>(rows_len) * sizeof(uint16_t);
        rpc_k = reinterpret_cast<uint16_t *>(
          stage(attn_native_k_pool_, kv_bytes).data());
        rpc_v = reinterpret_cast<uint16_t *>(
          stage(attn_native_v_pool_, kv_bytes).data());
        stagedMemcpy(const_cast<uint16_t *>(rpc_k), k_rows, kv_bytes);
        stagedMemcpy(const_cast<uint16_t *>(rpc_v), v_rows, kv_bytes);
      }
    }
    const int64_t t0 = now_us();
    const int err = nntr_hvx_attn_q_step(
      h, static_cast<uint32_t>(handle), append_row0, rpc_k, rows_len, rpc_v,
      rows_len, n_q, cache_from, cache_to, n_head_q, window, softcap, rpc_q,
      q_len, sinks, sinks_len, rpc_out, q_len, stats,
      attn_trace_enabled() ? 20 : 8);
    if (native_decode && err == AEE_SUCCESS)
      stagedMemcpy(out, rpc_out, static_cast<size_t>(q_len) * sizeof(float));
    if (attn_trace_enabled()) {
      ml_logi("HTP attn trace q: handle=%d n_q=%u cache=%u..%u append=%u "
              "err=0x%x wall_us=%lld dsp_us=%u append_us=%u attn_us=%u "
              "quant_us=%u stage_us=%u bake_us=%u",
              handle, n_q, cache_from, cache_to, append_rows, err,
              static_cast<long long>(now_us() - t0), stats[3], stats[0],
              stats[1], stats[4], stats[5], stats[6]);
    }
    if (attn_trace_enabled() && n_q >= 5 && err == AEE_SUCCESS)
      ml_logi("HTP attn phases q: n_q=%u qprep_us=%u dma_us=%u qk_us=%u "
              "dequant_us=%u softmax_us=%u pquant_us=%u pv_us=%u "
              "oupd_us=%u store_us=%u total_us=%u kcycles=%u blocks=%u",
              n_q, stats[8], stats[9], stats[10], stats[11], stats[12],
              stats[13], stats[14], stats[15], stats[16], stats[17], stats[18],
              stats[19]);
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized attention step failed: 0x%x; CPU fallback", err);
      return false;
    }
    return true;
  }

  // --- the row-blocked kernel over a fixed-scale int8 cache ---

  bool supports_kv_cache_q2() const override {
    // Opt out of A16/KV8 to compare the per-row A8/KV8 kernels. Select
    // before registering caches: the two paths use different KV scales.
    const char *wire = std::getenv("NNTR_HTP_ATTN_Q8_WIRE");
    if (wire && std::atoi(wire) != 0)
      return false;
    const char *q2 = std::getenv("NNTR_HTP_ATTN_Q2");
    return HtpBackend::global().enabled() && (!q2 || std::atoi(q2) != 0);
  }

  bool kv_cache_q_set_fixed_scales(int handle, unsigned int n_head_kv,
                                   unsigned int head_dim, const float *s_k,
                                   const float *s_v) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || !s_k || !s_v || n_head_kv == 0 ||
        head_dim == 0) {
      return false;
    }
    const int err = nntr_hvx_kv_set_fixed_scales_q(
      static_cast<remote_handle64>(hb.handle()), static_cast<uint32_t>(handle),
      s_k, static_cast<int>(n_head_kv), s_v,
      static_cast<int>(n_head_kv * head_dim));
    if (err != AEE_SUCCESS) {
      ml_logw("HTP quantized KV cache: fixed scales failed: 0x%x", err);
      return false;
    }
    return true;
  }

  bool sdpa_q2_kvcache(int handle, unsigned int append_row0,
                       unsigned int append_rows, unsigned int kv_stride,
                       const uint16_t *k_rows, const uint16_t *v_rows,
                       const uint16_t *q, const float *q_enc,
                       unsigned int q_stride, unsigned int n_q,
                       unsigned int cache_from, unsigned int cache_to,
                       unsigned int n_head_q, unsigned int n_head_kv,
                       unsigned int head_dim, unsigned int window,
                       uint16_t *out, const float *out_enc,
                       unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || handle < 0 || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 512 || cache_to < cache_from + n_q || cache_to > 0xFFFFu ||
        q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        !q || !q_enc || !out || !out_enc ||
        (append_rows != 0 &&
         (kv_stride != n_head_kv * head_dim || !k_rows || !v_rows))) {
      return false;
    }
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int rows_len = static_cast<int>(append_rows * kv_stride);
    const int enc_len = static_cast<int>(2 * n_head_q);
    uint32_t stats[12] = {0};
    const int64_t t0 = now_us();
    // Q and the output through ION, as the FC path stages its rows: from
    // plain heap FastRPC moved 2 x 16.8 MB a 1024-row layer, 7-12 ms of an
    // 18-24 ms call around 11.5 ms on the DSP (doc 57 section 9.17). A
    // memcpy each way here is ~1 ms (u16 since PR 4343's a16 path: half).
    const size_t q_bytes = static_cast<size_t>(q_len) * sizeof(uint16_t);
    uint16_t *q_ion =
      reinterpret_cast<uint16_t *>(stage(attn_q_pool_, q_bytes).data());
    uint16_t *out_ion =
      reinterpret_cast<uint16_t *>(stage(attn_out_pool_, q_bytes).data());
    stagedMemcpy(q_ion, q, q_bytes);
    const int err = nntr_hvx_attn_q2_step(
      h, static_cast<uint32_t>(handle), append_row0, k_rows, rows_len, v_rows,
      rows_len, n_q, cache_from, cache_to, n_head_q, window, q_ion, q_len,
      q_enc, enc_len, out_enc, enc_len, out_ion, q_len, stats, 12);
    if (err == AEE_SUCCESS) {
      stagedMemcpy(out, out_ion, q_bytes);
    }
    if (attn_trace_enabled()) {
      // stats (nntr_hvx_attn_q2_step): append, kernel, -, total, then the
      // append's quant/stage/bake, rows appended, the kernel's qk, softmax,
      // pv and worker wait, in microseconds. wall - total is transport.
      ml_logi("HTP attn trace q2: n_q=%u cache=%u..%u err=0x%x wall_us=%lld "
              "dsp_us: append=%u (quant=%u stage=%u bake=%u rows=%u) "
              "kernel=%u (qk=%u softmax=%u pv=%u wait=%u) total=%u",
              n_q, cache_from, cache_to, err,
              static_cast<long long>(now_us() - t0), stats[0], stats[4],
              stats[5], stats[6], stats[7], stats[1], stats[8], stats[9],
              stats[10], stats[11], stats[3]);
    }
    if (err != AEE_SUCCESS) {
      ml_logw("HTP row-blocked attention step failed: 0x%x; CPU fallback", err);
      return false;
    }
    return true;
  }

  bool sdpa_fp16_kvcache(const float *q, unsigned int q_stride,
                         const uint16_t *k_cache, const uint16_t *v_cache,
                         unsigned int kv_stride, unsigned int n_q,
                         unsigned int cache_from, unsigned int cache_to,
                         unsigned int n_head_q, unsigned int n_head_kv,
                         unsigned int head_dim, unsigned int window,
                         float softcap, const float *sinks, float *out,
                         unsigned int out_stride) override {
    HtpBackend &hb = HtpBackend::global();
    if (!hb.enabled() || n_q == 0 || n_head_kv == 0 ||
        (n_head_q % n_head_kv) != 0 || head_dim == 0 || (head_dim % 32) != 0 ||
        head_dim > 512 || cache_to < cache_from + n_q || cache_to > 0xFFFFu) {
      return false;
    }
    // The IDL takes dense [n_q][n_head_q*head_dim] and
    // [cache_to][n_head_kv*head_dim]; anything strided differently is the
    // CPU's.
    if (q_stride != n_head_q * head_dim || out_stride != n_head_q * head_dim ||
        kv_stride != n_head_kv * head_dim) {
      return false;
    }
    const remote_handle64 h = static_cast<remote_handle64>(hb.handle());
    const int q_len = static_cast<int>(n_q * n_head_q * head_dim);
    const int kv_len = static_cast<int>(cache_to * n_head_kv * head_dim);
    const int sinks_len = sinks ? static_cast<int>(n_head_q) : 0;
    uint32_t stats[8] = {0};

    int err;
    const int64_t t0 = now_us();
    const bool trace = attn_trace_enabled();
    int64_t stage_in_us = 0, rpc_us = 0, stage_out_us = 0;
    bool staged = false;
    static const bool decode_hmx_enabled = [] {
      const char *env = std::getenv("NNTR_HTP_ATTN_DECODE_HMX");
      return env && std::atoi(env) != 0;
    }();
    // Explicit HMX comparison for the measured LFM2 shape.
    const bool decode_hmx =
      decode_hmx_enabled && n_q == 1 && head_dim == 64 && n_head_q == 32 &&
      n_head_kv == 8 && cache_to >= 256 && !window && softcap == 0.0f && !sinks;
    const bool decode_hvx =
      !decode_hmx && n_q < kDecodeMaxRows && head_dim <= kDecodeMaxHeadDim;
    if (n_q == 1) {
      static std::once_flag banner;
      std::call_once(banner, [decode_hvx] {
        std::fprintf(stderr, "[HTP] attention M1: %s\n",
                     decode_hvx ? "HVX" : "HMX");
      });
    }
    {
      // Keep the cached ION buffers alive through the synchronous RPC and
      // output copy. Dedicated pools avoid aliasing q2's scratch buffers.
      std::unique_lock<std::mutex> lock(invoke_mutex_, std::defer_lock);
      const float *rpc_q = q;
      float *rpc_out = out;
      const size_t q_bytes = static_cast<size_t>(q_len) * sizeof(float);
      if (attn_f16_ion_enabled() || decode_hmx || decode_hvx) {
        lock.lock();
        HtpRpcBuffer &q_buf = stage(attn_f16_q_pool_, q_bytes);
        HtpRpcBuffer &out_buf = stage(attn_f16_out_pool_, q_bytes);
        // rpcmem may fall back to malloc. In that case the original RPC
        // path avoids adding two copies without a shared-memory benefit.
        if (q_buf.isIon() && out_buf.isIon()) {
          rpc_q = reinterpret_cast<float *>(q_buf.data());
          rpc_out = reinterpret_cast<float *>(out_buf.data());
          stagedMemcpy(q_buf.data(), q, q_bytes);
          staged = true;
        }
      }
      const int64_t rpc_start = trace ? now_us() : 0;
      if (trace)
        stage_in_us = rpc_start - t0;
      if (decode_hvx) {
        err = nntr_hvx_attn_f16_decode(
          h, n_q, cache_from, cache_to, n_head_q, n_head_kv, head_dim, window,
          softcap, rpc_q, q_len, k_cache, kv_len, v_cache, kv_len, sinks,
          sinks_len, rpc_out, q_len, stats, 8);
      } else {
        err = nntr_hvx_attn_f16_prefill(
          h, n_q, cache_from, cache_to, n_head_q, n_head_kv, head_dim, window,
          /*br=*/0, /*bc=*/0, softcap, rpc_q, q_len, k_cache, kv_len, v_cache,
          kv_len, sinks, sinks_len, rpc_out, q_len, stats, 8);
      }
      const int64_t rpc_end = trace ? now_us() : 0;
      if (staged && err == AEE_SUCCESS)
        stagedMemcpy(out, rpc_out, q_bytes);
      if (trace) {
        rpc_us = rpc_end - rpc_start;
        stage_out_us = now_us() - rpc_end;
      }
    }
    if (trace) {
      // HMX exposes per-stage times; HVX exposes only total kernel time.
      const uint32_t decode_total_us = decode_hvx ? stats[0] : 0u;
      if (decode_hvx)
        stats[0] = 0;
      ml_logi(
        "HTP attn trace f16: n_q=%u cache=%u..%u err=0x%x wall_us=%lld "
        "dsp_us: qprep=%u dma=%u tile=%u qk=%u softmax=%u pv=%u "
        "store=%u blocks=%u ion=%u stage_in_us=%lld rpc_us=%lld "
        "stage_out_us=%lld path=%s decode_total_us=%u",
        n_q, cache_from, cache_to, err, static_cast<long long>(now_us() - t0),
        stats[0], stats[1], stats[2], stats[3], stats[4], stats[5], stats[6],
        stats[7], static_cast<unsigned int>(staged),
        static_cast<long long>(stage_in_us), static_cast<long long>(rpc_us),
        static_cast<long long>(stage_out_us), decode_hvx ? "HVX" : "HMX",
        decode_total_us);
    }
    if (err != AEE_SUCCESS) {
      // AEE_EUNSUPPORTED is the skel saying this part has no fp16 HMX;
      // anything else is a transport or shape failure. Either way the
      // caller recomputes on the CPU, so this is a warning, not an error.
      ml_logw("HTP attention (%s) failed: 0x%x; CPU fallback for this call",
              n_q < kDecodeMaxRows ? "decode" : "prefill", err);
      return false;
    }
    return true;
  }
#endif /* NNTR_HTP_INPROC */
};

ComputeOps *get_htp_ops() {
  static HtpComputeOps instance;
  return &instance;
}

bool htp_rope_qkv_prefill(unsigned rows, unsigned qw, unsigned kw, unsigned hd,
                          const float *cs, float *q, const float *k,
                          const float *v, uint16_t *ko, uint16_t *vo) {
  return static_cast<HtpComputeOps *>(get_htp_ops())
    ->ropeQkvPrefill(rows, qw, kw, hd, cs, q, k, v, ko, vo);
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
