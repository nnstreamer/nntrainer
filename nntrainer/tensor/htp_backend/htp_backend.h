// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.h
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP (Hexagon Tensor Processor) backend lifecycle.
 *
 * Process-wide singleton owning the one HexKL micro-API FastRPC session a
 * process opens (nntr_hvx_open against the skel PR #4256 device-verified --
 * test/htp/nntr_hvx.idl is the interface, and this file's meson.build
 * generates the same client stub from it that test/htp's gtests use).
 * If the skel isn't reachable (missing from ADSP_LIBRARY_PATH, no device,
 * driver error, ...) construction leaves the backend DISABLED and every
 * HtpComputeOps::supports_*() reports false, so callers transparently fall
 * back to the CPU path.
 *
 * This used to own a Qualcomm HexKL CPU Macro API (sdkl.h / libsdkl.so)
 * session instead. That session had no caller left -- docs/htp_attention/
 * 10_mha_htp_plan.md section 6 established there is no macro-API FC
 * dispatch left on this lineage to migrate, so it existed only to print a
 * version string -- and removing it was not optional: per the documented
 * macro/micro one-way door, a macro-API session opened after any HexKL
 * micro-API FastRPC session (this file's session, now) fails permanently.
 *
 * Compiled only when ENABLE_HEXKL is defined (meson: -Denable-htp=true).
 */

#ifndef __HTP_BACKEND_H__
#define __HTP_BACKEND_H__
#ifdef __cplusplus
#ifdef ENABLE_HEXKL

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <vector>

namespace nntrainer {

/**
 * @class HtpBackend
 * @brief Process-wide owner of the HTP (HexKL micro-API/FastRPC) session.
 */
class HtpBackend {
public:
  /**
   * @brief Access the process-wide singleton. The first call attempts
   *        nntr_hvx_open() exactly once (thread-safe).
   */
  static HtpBackend &global();

  /**
   * @brief Whether the HTP session is initialized and usable. When false,
   *        all HTP ops must defer to the CPU fallback.
   */
  bool enabled() const { return enabled_; }

  /**
   * @brief The FastRPC session handle every nntr_hvx_* call dispatches
   *        through (HtpComputeOps, from the first accelerated kernel
   *        onward). Only meaningful when enabled() is true.
   */
  uint64_t handle() const { return handle_; }

  /**
   * @brief Which FastRPC latency QoS mode the constructor's control call
   *        landed on: 2 = poll, 1 = PM, 0 = both rejected (interrupt-driven,
   *        the ~3.9ms-tail path -- see htp_compute_ops.cpp's profile dump,
   *        which prints this so a transport number is never read without
   *        knowing which mode produced it).
   */
  int qosMode() const { return qos_mode_; }

  /**
   * @brief The poll-QoS window the driver accepted, in us (NNTR_HTP_POLL_US,
   *        default 5000), or 0 when poll QoS was refused (qosMode() != 2):
   *        how long a FastRPC call spins before it blocks. [#141] The
   *        dspqueue MoE call spins its response wait for the same window.
   */
  uint32_t pollUs() const { return poll_us_; }

  /**
   * @brief DSP-visible memory: an rpcmem block, i.e. a dma-buf the rpcmem
   *        library registers with FastRPC as it allocates. Any range of it
   *        passed as a FastRPC buffer argument is then mapped into the DSP
   *        and cache-maintained instead of copied -- the KV cache goes
   *        here so attention never copies the used cache per call.
   * @return the block, or nullptr when the backend is disabled, @a bytes
   *         is 0 or above the rpcmem limit, or the allocation failed
   */
  void *alloc_shared(size_t bytes);

  /**
   * @brief Frees a block from alloc_shared(); nullptr is a no-op.
   */
  void free_shared(void *block);

  /** Whether the complete range belongs to a live shared allocation. */
  bool isSharedRange(const void *ptr, size_t bytes);

  /**
   * @brief [#141] Runs fn in ~HtpBackend, in registration order, before
   *        the session is closed: the shutdown hook for state that makes
   *        RPC calls on the session (the dspqueue thread). fn must own
   *        what it touches; HtpComputeOps may already be destroyed.
   */
  void atClose(std::function<void()> fn) { at_close_.push_back(std::move(fn)); }
  /**
   * @brief [#132 Part B E5i] As atClose, but after every atClose hook:
   *        for what the others' DSP threads use (S1's arena, released on
   *        the DSP and unmapped before the session closes).
   */
  void atCloseLast(std::function<void()> fn) {
    at_close_last_.push_back(std::move(fn));
  }

  /**
   * @brief [#132 Part B E3] NNTR_HTP_E2E=1: decode runs end to end on two
   *        cDSP sessions (plan docs/plans/132-part-b-two-session-e2e.md
   *        section 3.2). Read once; off by default, and off opens nothing
   *        more than before.
   */
  static bool e2eRequested();

  /**
   * @brief [#132 Part B E3] Opens the second session S2 (reserved beside
   *        the default one, lite-opened on its effective domain) once, when
   *        e2eRequested(); later calls return the first answer. Called only
   *        after S1's MoE arena is mapped: the E5 sitting (2026-09-30) saw
   *        S1 stop at 3584 MiB when S2 was open and mapped first, where the
   *        #178 probe (S1's 3840 mapped before S2 opened) did not.
   * @return enabled2()
   */
  bool openSecond();
  /** @brief S2 is open (openSecond succeeded); s2Error() says why not. */
  bool enabled2() const { return enabled2_; }
  /** @brief S2's remote_handle64; meaningful when enabled2(). */
  uint64_t handle2() const { return handle2_; }
  /** @brief S2's effective domain id: every fastrpc_mmap / dspqueue_create
   *  for S2 names it (the default session's is CDSP_DOMAIN_ID). */
  int effDomain2() const { return effdom2_; }
  /** @brief S2's VTCM in bytes after its open (0: the lite open found
   *  none, as on the S25 beside S1's M=1 feed). */
  uint32_t vtcm2Bytes() const { return vtcm2_bytes_; }
  /** @brief Why S2 did not open (empty when it did or was not asked for). */
  const std::string &s2Error() const { return s2_error_; }

  ~HtpBackend();

  HtpBackend(const HtpBackend &) = delete;
  HtpBackend &operator=(const HtpBackend &) = delete;

private:
  std::mutex shared_mutex_;
  std::map<uintptr_t, size_t> shared_blocks_;
  HtpBackend();

  void openSecondNow();

  bool enabled_ = false;
  uint64_t handle_ = 0; ///< remote_handle64 from nntr_hvx_open; opaque here
                        ///< so this header does not need <remote.h>.
  int qos_mode_ = 0;
  uint32_t poll_us_ = 0;
  std::vector<std::function<void()>> at_close_, at_close_last_;
  bool enabled2_ = false; ///< [#132 Part B E3] S2 open
  bool tried2_ = false;
  uint64_t handle2_ = 0;
  int effdom2_ = -1;
  uint32_t vtcm2_bytes_ = 0;
  std::string s2_error_;
};

} // namespace nntrainer

#endif // ENABLE_HEXKL
#endif // __cplusplus
#endif // __HTP_BACKEND_H__
