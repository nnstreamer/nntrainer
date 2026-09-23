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
#include <cpu_ops_table.h>
#include <htp_act_quant.h>
#include <htp_backend.h>
#include <htp_q4_0_convert.h>
#include <htp_rpcmem.h>
#include <htp_wh_layout.h>
#include <htp_wh_palette.h>
#include <swiglu_det.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include <remote.h>

#include <nntr_hvx.h>

namespace nntrainer {

namespace {

/**
 * @brief Slots mm_u8i4_layer_timed fills.
 *
 * Restated here because the ARM side cannot include the DSP source that
 * declares them (nntr_hvx_mm_u8i4.c). The entry point rejects a stale count
 * with AEE_EBADPARM rather than truncating, so a drift shows up as a failed
 * call, not as wrong numbers.
 */
enum {
  HTP_T_DSP_TOTAL = 0,
  HTP_T_QUANT,
  HTP_T_DEQUANT,
  HTP_T_ACC_READ,
  HTP_T_ACC_COPY,
  HTP_T_DRAIN,
  HTP_T_ACC_STRIDE,
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
                         int kind = 0) {
    std::lock_guard<std::mutex> lock(mutex_);
    Bucket &b = buckets_[std::make_tuple(K, N_out, M == 1, kind)];
    ++b.calls;
    b.rows += M;
    b.host_us += host_us;
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
    }
  }

  /** ARM-side staging: the memcpy of the activation into the rpcmem buffer
   *  and of the result back out. It sits OUTSIDE every host_us window
   *  above (those start after the copy, so that transport = host - dsp
   *  stays a FastRPC number), which meant it was invisible -- and at this
   *  model's shapes it is ~0.9 MB per expert, 64 times per layer. Counted
   *  here so "HTP host time" stops understating what the path costs. */
  void addStaging(uint64_t us, uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    staging_us_ += us;
    staging_bytes_ += bytes;
  }

  ~HtpProfile() {
    if (level_ != 0)
      dump();
  }

private:
  struct Bucket {
    uint64_t calls = 0;
    uint64_t rows = 0; /**< summed M, so prefill batching is visible */
    uint64_t host_us = 0;
    uint64_t dsp_us = 0;
    uint64_t quant_us = 0;
    uint64_t swiglu_us = 0; /**< fused calls only; 0 elsewhere */
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
  };

  HtpProfile() {
    const char *env = std::getenv("NNTR_HTP_PROFILE");
    level_ = (env != nullptr) ? std::atoi(env) : 0;
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
      static const char *const kind_name[] = {"M>1", "M>1 dense", "M>1 conv",
                                              "M>1 FC"};
      const int kind = std::get<3>(entry.first);
      const Bucket &b = entry.second;
      std::fprintf(stderr,
                   "[HTP-PROFILE]   K=%-5u N=%-5u %-9s calls=%-7llu "
                   "rows=%-8llu host=%9.1f ms (%7.1f us/call)",
                   k, n, decode ? "M==1" : kind_name[kind],
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
        /* mm is the residual, so every named stage has to be subtracted --
           scatter included, or the MoE layer call's scatter time would be
           reported as matmul. The conv block call's SWIGLU column is the
           exception: there it is WORKER time inside its background stage
           (hexkl_conv_block.c's cb_stage_probe_add), work that ran under
           the HMX rather than instead of it, so subtracting it made the
           residual negative -- 3.3 ms of hidden work against a 4.0 ms
           call. Left in the sum for every other call, where SWIGLU is
           the fused layer's own synchronous elementwise pass. */
        const double hidden_per = (kind == 2) ? swiglu_per : 0.0;
        const double mm_per =
          dsp_per -
          (quant_per + (swiglu_per - hidden_per) + dequant_per + acc_per +
           drain_per + scatter_per + stage_per + gather_per + requant_per +
           mm_meas_per + drain_dn_per + push_per + alloc_per);
        std::fprintf(stderr,
                     "  dsp=%7.1f us/call (%4.1f%%) transport=%7.1f us/call"
                     "  [quant %.1f gather %.1f requant %.1f swiglu%s %.1f "
                     "dequant %.1f acc %.1f drain %.1f+%.1f push %.1f "
                     "scatter %.1f alloc %.1f "
                     "stage %.1f mm %.1f | rest<=%.1f (%.1f%% of host) "
                     "blocks=%llu]",
                     dsp_per, host_per > 0.0 ? 100.0 * dsp_per / host_per : 0.0,
                     host_per - dsp_per, quant_per, gather_per, requant_per,
                     hidden_per != 0.0 ? "(hidden)" : "", swiglu_per,
                     dequant_per, acc_per, drain_per, drain_dn_per, push_per,
                     scatter_per, alloc_per, stage_per, mm_meas_per, mm_per,
                     host_per > 0.0 ? 100.0 * mm_per / host_per : 0.0,
                     (unsigned long long)b.blocks);
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
  }

  int level_ = 0;
  int qos_mode_ = 0;
  std::mutex mutex_;
  uint64_t reg_calls_ = 0;
  uint64_t reg_ion_calls_ = 0;
  uint64_t reg_total_us_ = 0;
  uint64_t staging_us_ = 0;
  uint64_t staging_bytes_ = 0;
  uint64_t convert_us_ = 0;
  uint64_t rpc_us_ = 0;
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

inline void stagedMemcpy(void *dst, const void *src, size_t bytes) {
  HtpProfile &p = HtpProfile::global();
  if (p.level() == 0) {
    std::memcpy(dst, src, bytes);
    return;
  }
  const uint64_t t0 = HtpProfile::nowUs();
  std::memcpy(dst, src, bytes);
  p.addStaging(HtpProfile::nowUs() - t0, bytes);
}

} // namespace

class HtpComputeOps : public CpuComputeOps {
public:
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
    const unsigned int step = fcMaxRows(K);
    for (unsigned int m0 = 0; m0 < M; m0 += step) {
      const unsigned int m = std::min(step, M - m0);
      invokeLayer(session, fh.handles.data(),
                  static_cast<int>(fh.handles.size()),
                  matBdata + static_cast<size_t>(m0) * K,
                  matCdata + static_cast<size_t>(m0) * N, m, N, K, &fh.cols);
    }
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

  // A QS4CX weight was quantized once, straight from FP32, and already
  // holds the int4 values this registry wants -- so the seam is
  // htp_qs4cx_from_packed's bit rearrangement plus a colsum, not a second
  // quantization. That is both more accurate (measured on this branch:
  // mean_abs_err 0.0333 vs 0.0451 through Q4_0, a 26% cut) and cheaper to
  // load (95.9 -> 59.9 ms for a 2048x3584 gate_up, 77.2 -> 22.2 ms for a
  // 1792x2048 down, x86 host). Prefer feeding the FFN QS4CX; the Q4_0
  // entry above stays for weights shared with the CPU path.
  bool supports_gemm_qs4cx_accel_fp32() const override { return true; }

  void gemm_qs4cx_accel_fp32(void *matAdata, float *matAscale, float *matBdata,
                             float *matCdata, unsigned int M, unsigned int N,
                             unsigned int K) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const uint32_t handle =
      get_or_register_qs4cx(matAdata, matAscale, session, K, N);

    // ponytail: see gemm_q4_0_accel_fp32's identical comment -- reverted
    // from invokeLayerU8In for the same reason.
    invokeLayer(session, &handle, 1, matBdata, matCdata, M, N, K);
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
                                 unsigned int N_out,
                                 unsigned int w_bits) override {
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
    std::vector<uint32_t> h_gu(n_experts), h_dn(n_experts);
    for (size_t e = 0; e < n_experts; ++e) {
      if (w_bits != 0u) {
        h_gu[e] = get_or_register_wh(gate_up_data[e], gate_up_scale[e], session,
                                     K, 2 * inter, w_bits);
        h_dn[e] = get_or_register_wh(down_data[e], down_scale[e], session,
                                     inter, N_out, w_bits);
      } else {
        h_gu[e] = get_or_register_qs4cx(gate_up_data[e], gate_up_scale[e],
                                        session, K, 2 * inter);
        h_dn[e] = get_or_register_qs4cx(down_data[e], down_scale[e], session,
                                        inter, N_out);
      }
    }
    invokeMoeLayer(session, h_gu, h_dn, row_index, row_count, row_weight, act,
                   out, M, K, inter, N_out);
  }

  /** Same registration the layer call above does on first use, keyed by
   *  the same data pointer, so the forward-time call is a cache hit. */
  bool register_qs4cx_weight(void *data, const float *scale, unsigned int K,
                             unsigned int N, unsigned int w_bits) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    if (w_bits != 0u) {
      get_or_register_wh(data, scale, session, K, N, w_bits);
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
  };
  static HtpRpcBuffer &stage(StagingPool &pool, size_t bytes) {
    size_t cls = size_t(64) << 10;
    while (cls < bytes)
      cls <<= 1;
    auto &slot = pool.by_class[cls];
    if (!slot)
      slot = std::make_unique<HtpRpcBuffer>(cls);
    return *slot;
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
                      const std::vector<float *> *dsts = nullptr) {
    if (dsts) {
      size_t off = 0;
      for (size_t i = 0; i < dsts->size(); ++i) {
        const size_t block = static_cast<size_t>(M) * (*blocks)[i];
        stagedMemcpy((*dsts)[i], out_cat + off, block * sizeof(float));
        off += block;
      }
      return;
    }
    if (!blocks || blocks->size() < 2) {
      stagedMemcpy(dst, out_cat, static_cast<size_t>(M) * N * sizeof(float));
      return;
    }
    size_t off = 0;
    unsigned int c0 = 0;
    for (unsigned int n : *blocks) {
      for (unsigned int r = 0; r < M; ++r) {
        stagedMemcpy(dst + static_cast<size_t>(r) * N + c0,
                     out_cat + off + static_cast<size_t>(r) * n,
                     static_cast<size_t>(n) * sizeof(float));
      }
      off += static_cast<size_t>(M) * n;
      c0 += n;
    }
  }

  /** @param blocks N of each handle in call order, when @a handles are the
   *  slices of one weight and matCdata is its whole [M x N], or separate
   *  weights whose [M x N_i] blocks go to @a dsts[i]; NULL when each
   *  handle's output stays a block of its own (or there is only one). */
  void invokeLayer(remote_handle64 session, const uint32_t *handles,
                   int num_handles, float *matBdata, float *matCdata,
                   unsigned int M, unsigned int N, unsigned int K,
                   const std::vector<unsigned int> *blocks = nullptr,
                   const std::vector<float *> *dsts = nullptr) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N);
    const auto shape = [&]() {
      return " (M=" + std::to_string(M) + " K=" + std::to_string(K) +
             " N=" + std::to_string(N) +
             " handles=" + std::to_string(num_handles) + ")";
    };

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act_f32 = reinterpret_cast<float *>(
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float)).data());
    float *out_cat = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());
    stagedMemcpy(act_f32, matBdata,
                 static_cast<size_t>(act_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    if (profile.level() == 0) {
      const int err =
        nntr_hvx_mm_u8i4_layer(session, M, K, handles, num_handles, act_f32,
                               act_len, out_cat, out_len);
      if (err != AEE_SUCCESS) {
        throw std::runtime_error("nntr_hvx_mm_u8i4_layer failed: err=" +
                                 std::to_string(err) + shape());
      }
      copyOut(matCdata, out_cat, M, N, blocks, dsts);
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
    copyOut(matCdata, out_cat, M, N, blocks, dsts);
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
                      unsigned int inter, unsigned int N_out, int kind = 0) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N_out);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act_f32 = reinterpret_cast<float *>(
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float)).data());
    float *out_f32 = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());
    stagedMemcpy(act_f32, act, static_cast<size_t>(act_len) * sizeof(float));

    // The five small sequences (handles, routing) go from the heap. Tried
    // from one rpcmem buffer (doc 51 section 2.26): transport 627 -> 605
    // us, noise. The gap to the FC calls' ~400 is the call's length, not
    // its arguments -- a call longer than the poll window (htp_backend.cpp,
    // 5 ms) ends in the interrupt-driven wait and pays its wake-up.

    HtpProfile &profile = HtpProfile::global();
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    // NNTR_HTP_PROFILE=3 runs the call several times on the same input and
    // keeps the fastest. Two runs with no functional change between them
    // differed by 4.5 ms of DSP time (doc 46 section 23.2) -- the
    // compute-bound stage identical, the DDR-touching ones all moving
    // together, so the device's memory system, not the code. Several of the
    // changes on the list are worth 2 ms, which that noise floor swallows.
    // Repeating inside one call puts the comparison at the same temperature
    // and the same contention, and the output is unchanged because the
    // input is.
    const int reps = (profile.level() >= 3) ? 5 : 1;
    uint64_t best_elapsed = UINT64_MAX;
    int err = AEE_SUCCESS;
    for (int rep = 0; rep < reps && err == AEE_SUCCESS; ++rep) {
      uint32_t rep_stage[HTP_MOE_N_STAGES] = {0};
      const uint64_t t0 = profile.level() ? HtpProfile::nowUs() : 0;
      err = timed ? nntr_hvx_mm_u8i4_moe_layer_timed(
                      session, M, K, inter, N_out, h_gu.data(),
                      static_cast<int>(h_gu.size()), h_dn.data(),
                      static_cast<int>(h_dn.size()), row_index.data(),
                      static_cast<int>(row_index.size()), row_count.data(),
                      static_cast<int>(row_count.size()), row_weight.data(),
                      static_cast<int>(row_weight.size()), act_f32, act_len,
                      out_f32, out_len, rep_stage, HTP_MOE_N_STAGES)
                  : nntr_hvx_mm_u8i4_moe_layer(
                      session, M, K, inter, N_out, h_gu.data(),
                      static_cast<int>(h_gu.size()), h_dn.data(),
                      static_cast<int>(h_dn.size()), row_index.data(),
                      static_cast<int>(row_index.size()), row_count.data(),
                      static_cast<int>(row_count.size()), row_weight.data(),
                      static_cast<int>(row_weight.size()), act_f32, act_len,
                      out_f32, out_len);
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
        " failed: err=" + std::to_string(err) + hint);
    }
    stagedMemcpy(out, out_f32, static_cast<size_t>(out_len) * sizeof(float));
    if (profile.level()) {
      profile.addInvokeMoeLayer(M, K, N_out, elapsed,
                                timed ? stage_us : nullptr, kind);
    }
  }

  /** @brief [doc 51 section 2] One call for a whole conv block.
   *
   *  The activation goes over once and the output comes back once; the
   *  four M x C intermediates the block has stay on the DSP. conv_w (24 KB)
   *  rides along by value each call rather than being registered: it is
   *  under a hundredth of the activation. The state (2 x C) comes back in
   *  its own small buffer, not a size class of out_pool_: at a short M the
   *  output would land in the same 64 KiB class and the two would alias. */
  void invokeConvBlock(remote_handle64 session, const ConvHandles &ch,
                       const float *conv_w, const float *act, float *out,
                       float *state, unsigned int M, unsigned int K,
                       unsigned int C, unsigned int N_out) {
    const int act_len = static_cast<int>(M) * static_cast<int>(K);
    const int out_len = static_cast<int>(M) * static_cast<int>(N_out);
    const int conv_len = 3 * static_cast<int>(C);
    const int state_len = 2 * static_cast<int>(C);

    std::lock_guard<std::mutex> lock(invoke_mutex_);
    float *act_f32 = reinterpret_cast<float *>(
      stage(act_pool_, static_cast<size_t>(act_len) * sizeof(float)).data());
    float *out_f32 = reinterpret_cast<float *>(
      stage(out_pool_, static_cast<size_t>(out_len) * sizeof(float)).data());
    const size_t small_bytes =
      static_cast<size_t>(conv_len + state_len) * sizeof(float);
    if (!conv_buf_ || conv_buf_->size() < small_bytes)
      conv_buf_ = std::make_unique<HtpRpcBuffer>(small_bytes);
    float *conv_f32 = reinterpret_cast<float *>(conv_buf_->data());
    float *state_f32 = conv_f32 + conv_len;
    stagedMemcpy(act_f32, act, static_cast<size_t>(act_len) * sizeof(float));
    std::memcpy(conv_f32, conv_w,
                static_cast<size_t>(conv_len) * sizeof(float));

    HtpProfile &profile = HtpProfile::global();
    uint32_t stage_us[HTP_MOE_N_STAGES] = {0};
    const bool timed = profile.level() >= 2;
    const uint64_t t0 = profile.level() ? HtpProfile::nowUs() : 0;
    const int err =
      timed ? nntr_hvx_mm_u8i4_conv_block_timed(
                session, M, K, C, N_out, ch.h_in.data(),
                static_cast<int>(ch.h_in.size()), ch.h_out, conv_f32, conv_len,
                act_f32, act_len, out_f32, out_len, state_f32, state_len,
                stage_us, HTP_MOE_N_STAGES)
            : nntr_hvx_mm_u8i4_conv_block(
                session, M, K, C, N_out, ch.h_in.data(),
                static_cast<int>(ch.h_in.size()), ch.h_out, conv_f32, conv_len,
                act_f32, act_len, out_f32, out_len, state_f32, state_len);
    const uint64_t elapsed = profile.level() ? HtpProfile::nowUs() - t0 : 0;
    if (err != AEE_SUCCESS) {
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
    stagedMemcpy(out, out_f32, static_cast<size_t>(out_len) * sizeof(float));
    std::memcpy(state, state_f32,
                static_cast<size_t>(state_len) * sizeof(float));
    if (profile.level()) {
      profile.addInvokeMoeLayer(M, K, N_out, elapsed,
                                timed ? stage_us : nullptr, /*kind=*/2);
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
    constexpr unsigned int kSliceBytes = 2u << 20;
    const unsigned int k_tiles = K / 32u;
    const unsigned int n_tiles =
      k_tiles == 0 ? 0 : (kSliceBytes / 512u) / k_tiles;
    return n_tiles < 1u ? 32u : n_tiles * 32u;
  }

  /** @brief get_or_register for an FC weight of any width: one handle when
   *  it fits fcSliceCols, else one per column slice, converted once and
   *  registered slice by slice under keys inside the weight (its data
   *  pointer plus the slice's first column -- distinct, and valid as long
   *  as the weight is). Cached by the weight pointer like every handle. */
  const FcHandles &get_or_register_fc(void *matAdata, remote_handle64 session,
                                      uint32_t K, uint32_t N) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = fc_cache_.find(matAdata);
    if (it != fc_cache_.end())
      return it->second;

    FcHandles fh;
    const unsigned int cap = fcSliceCols(K);
    if (N <= cap) {
      fh.handles.push_back(get_or_register_unlocked(matAdata, session, K, N));
      fh.cols.push_back(N);
    } else {
      std::vector<int8_t> full(static_cast<size_t>(K) * N);
      std::vector<float> w_scale(N);
      std::vector<int32_t> colsum_w(N);
      // The first slice's profile entry carries the conversion, so its
      // clock starts before it; the later slices' start with their copy.
      uint64_t t_begin = HtpProfile::nowUs();
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
                                            uint32_t I, uint32_t N) {
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
    std::vector<float> up_s(I), gate_s(I), down_s(N);
    std::vector<int32_t> up_c(I), gate_c(I), down_c(N);
    htp_qs4cx_from_q4_0x4(up, K, I, up_rm.data(), up_s.data(), up_c.data());
    htp_qs4cx_from_q4_0x4(gate, K, I, gate_rm.data(), gate_s.data(),
                          gate_c.data());
    htp_qs4cx_from_q4_0x4(down, I, N, down_rm.data(), down_s.data(),
                          down_c.data());
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
      std::vector<float> gus(gate_s.begin() + c0, gate_s.begin() + c0 + w);
      gus.insert(gus.end(), up_s.begin() + c0, up_s.begin() + c0 + w);
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
      std::vector<float> dns(down_s);
      dh.h_dn.push_back(registerRm(static_cast<char *>(down) + c0, session, dn,
                                   w, N, dns, dnc, HtpProfile::nowUs(), 0));
    }
    return dense_cache_.emplace(up, std::move(dh)).first->second;
  }

  bool supports_gemm_q4_0_dense_ffn_fp32() const override { return true; }

  void gemm_q4_0_dense_ffn_fp32(void *up, void *gate, void *down,
                                const float *act, float *out, unsigned int M,
                                unsigned int K, unsigned int I,
                                unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const DenseHandles &dh =
      get_or_register_dense(up, gate, down, session, K, I, N);
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
                   act, out, M, K, dh.w, N, /*kind=*/1);
  }

  bool register_q4_0_dense_ffn(void *up, void *gate, void *down, unsigned int K,
                               unsigned int I, unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_dense(up, gate, down, session, K, I, N);
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
                                          uint32_t C, uint32_t N) {
    {
      std::lock_guard<std::mutex> lock(handle_mutex_);
      auto it = conv_cache_.find(in_proj);
      if (it != conv_cache_.end())
        return it->second;
    }
    const FcHandles &in = get_or_register_fc(in_proj, session, K, 3 * C);
    const FcHandles &out = get_or_register_fc(out_proj, session, C, N);
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
    ConvHandles ch;
    ch.h_in = in.handles;
    ch.h_out = out.handles[0];
    return conv_cache_.emplace(in_proj, std::move(ch)).first->second;
  }

  bool supports_gemm_q4_0_conv_block_fp32() const override { return true; }

  void gemm_q4_0_conv_block_fp32(void *in_proj, const float *conv_w,
                                 void *out_proj, const float *act, float *out,
                                 float *state, unsigned int M, unsigned int K,
                                 unsigned int C, unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    const ConvHandles &ch =
      get_or_register_conv(in_proj, out_proj, session, K, C, N);
    invokeConvBlock(session, ch, conv_w, act, out, state, M, K, C, N);
  }

  bool register_q4_0_conv_block(void *in_proj, void *out_proj, unsigned int K,
                                unsigned int C, unsigned int N) override {
    const remote_handle64 session =
      static_cast<remote_handle64>(HtpBackend::global().handle());
    get_or_register_conv(in_proj, out_proj, session, K, C, N);
    return true;
  }

  /* Declared here rather than beside arena_chunks_ at the bottom of the
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

  /** @brief Where one weight sits, and the three small arrays that are not
   *  worth arena space (4 KB against 3.5 MB) and would need their own
   *  alignment rules if they were. */
  struct ArenaEntry {
    uint32_t chunk;
    uint32_t off;
    uint32_t K, N;
    std::vector<float> w_scale;
    std::vector<int32_t> colsum_w;
    std::vector<float> bias;
    /** Four int4 codes when this is a QS2CX_WH weight, empty at 4 bits. */
    std::vector<int8_t> pal;
  };

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
                              remote_handle64 session, uint32_t K, uint32_t N,
                              uint32_t w_bits = 4u) {
    std::lock_guard<std::mutex> lock(handle_mutex_);
    auto it = handle_cache_.find(matAdata);
    if (it != handle_cache_.end())
      return it->second;

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

    /* QS2CX_WH is half the bytes, with the four palette codes right after
       them -- QS2CX_WH_Tensor's layout, which is why the scale pointer the
       caller passed already skips them. */
    const uint32_t wh_len =
      static_cast<uint32_t>(w_bits == 2u ? whBytes2(K, N) : whBytes(K, N));
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

    if (w_bits == 2u) {
      const int8_t *pal = reinterpret_cast<const int8_t *>(matAdata) + wh_len;
      e.pal.assign(pal, pal + 4);
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
    if (placeExisting(bytes, chunk, off))
      return true;
    if (!newChunk(session, bytes, want))
      return false;
    arena_chunks_.back().used = bytes;
    *chunk = static_cast<uint32_t>(arena_chunks_.size() - 1);
    *off = 0;
    return true;
  }

  /** @brief place() without the new chunk: room in a mapped chunk or
   *  nothing. What the FC slices use (get_or_register_fc), so that they
   *  never map address space the remaining heap registrations need. */
  bool placeExisting(uint32_t bytes, uint32_t *chunk, uint32_t *off) {
    for (size_t c = 0; c < arena_chunks_.size(); ++c) {
      // 4 KB rather than the 512 the DSP requires: a weight that starts on
      // a page boundary is one the DMA never splits across a page for
      // alignment reasons alone, and the waste is under a part in 400.
      const size_t at = (arena_chunks_[c].used + 4095u) & ~size_t(4095u);
      if (at + bytes <= arena_chunks_[c].buf->size()) {
        arena_chunks_[c].used = at + bytes;
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
    const unsigned long long rss_before = rssKb();
    // Uncached, because the DSP maps this once and the host keeps writing
    // into it afterwards -- with an uncached CPU mapping those writes reach
    // DDR with no flush to remember. The host never reads it back, so the
    // cost is on the write side only. The probe that passed Gate 0c wrote
    // BEFORE attaching and used a cached buffer, so this ordering is the
    // one thing section 34 rests on that the probe did not show; the
    // ArenaUncachedWriteAfterMap test is what answers it.
    auto buf = std::make_unique<HtpRpcBuffer>(size, HTP_RPC_FLAGS_UNCACHED);
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
    const int merr =
      api.mmap(CDSP_DOMAIN_ID, fd, buf->data(), 0, size, FASTRPC_MAP_FD);
    if (merr != 0) {
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
        api.munmap(CDSP_DOMAIN_ID, fd, buf->data(), size);
      return false;
    }
    arena_chunks_.push_back(ArenaChunk{std::move(buf), dsp_id, 0});
    if (HtpProfile::global().level() != 0) {
      std::printf("[HTP] arena chunk %zu: %zu MiB, dsp_id=%u, mapped total "
                  "%zu MiB, RSS %llu -> %llu MB\n",
                  arena_chunks_.size() - 1, size >> 20, dsp_id,
                  arenaBytes() >> 20, rss_before >> 10, rssKb() >> 10);
    }
    return true;
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
    static constexpr size_t kGrain = size_t(64) << 20;
    /** Below this a chunk holds too few weights to be worth an arena slot,
     *  and the DSP's table is not unbounded. */
    static constexpr size_t kFloor = size_t(64) << 20;

    size_t size = (std::max(want, size_t(bytes)) + kGrain - 1) & ~(kGrain - 1);
    size = std::min(std::max(size, kGrain), chunk_cap_);
    if (size < bytes) {
      arena_fail_ = "a weight is larger than a whole chunk";
      return false; // not this model
    }

    while (true) {
      arena_fail_.clear();
      if (tryChunk(session, size))
        return true;
      const size_t half = size / 2;
      if (half < kFloor || half < bytes)
        return false; // arena_fail_ holds the last refusal, which is the one
      if (HtpProfile::global().level() != 0) {
        std::printf("[HTP] arena: %zu MiB refused, retrying at %zu MiB (%s)\n",
                    size >> 20, half >> 20, arena_fail_.c_str());
      }
      size = half;
      chunk_cap_ = size;
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
    const int err =
      e.pal.empty()
        ? nntr_hvx_weight_register_u8i4_arena(
            session, K, N, arena_chunks_[e.chunk].dsp_id, e.off,
            e.w_scale.data(), static_cast<int>(N), e.colsum_w.data(),
            static_cast<int>(N), e.bias.data(), static_cast<int>(N), &handle)
        : nntr_hvx_weight_register_u2i4_arena(
            session, K, N, arena_chunks_[e.chunk].dsp_id, e.off, e.pal.data(),
            4, e.w_scale.data(), static_cast<int>(N), e.colsum_w.data(),
            static_cast<int>(N), e.bias.data(), static_cast<int>(N), &handle);
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
  /** Dense FFNs by their up weight's pointer; see get_or_register_dense. */
  std::unordered_map<const void *, DenseHandles> dense_cache_;
  /** Conv blocks by their in_proj's pointer; see get_or_register_conv. */
  std::unordered_map<const void *, ConvHandles> conv_cache_;

  std::vector<ArenaChunk> arena_chunks_;
  enum ArenaState { ARENA_UNTRIED, ARENA_ON, ARENA_OFF };
  ArenaState arena_state_ = ARENA_UNTRIED;
  /** Why the last newChunk refused, in words, for the throw that follows. */
  std::string arena_fail_;
  /** Largest chunk still worth asking for; only ever shrinks. */
  size_t chunk_cap_ = kArenaChunkMax;

  // Deliberately never fastrpc_munmap'd: the DSP's close() puts every arena
  // back, and the kernel reclaims the ION buffers at process exit. The
  // destruction order of HtpBackend's singleton and this one is not fixed,
  // so unmapping here could touch a session that is already closed.
  // ponytail: a process that loads and unloads models would leak an arena
  // per model. The fix is an explicit shutdown hook on HtpBackend that runs
  // before it closes the session, not a destructor here.

  // ION-backed activation/output scratch, reused across calls, one buffer
  // per size class (stage) -- see invokeLayer's comment. Guarded by the
  // same mutex that serializes every call into the one HTP session.
  std::mutex invoke_mutex_;
  StagingPool act_pool_;
  StagingPool out_pool_;
  /** invokeConvBlock's conv_w in and state out, one small ION buffer. */
  std::unique_ptr<HtpRpcBuffer> conv_buf_;

  // invokeLayerU8In's scratch: the AH-packed activation (ION-backed, same
  // reasoning as act_pool_/out_pool_) and the small per-row scale/zp arrays
  // it produces alongside it (plain heap -- a few KB at most, not worth
  // ION's pin/map bookkeeping).
  std::unique_ptr<HtpRpcBuffer> act_ah_buf_;
  std::vector<float> act_scale_scratch_;
  std::vector<int32_t> act_zp_scratch_;
};

ComputeOps *get_htp_ops() {
  static HtpComputeOps instance;
  return &instance;
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
