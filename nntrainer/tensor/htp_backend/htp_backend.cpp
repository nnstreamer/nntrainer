// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_backend.cpp
 * @date   18 Jun 2026
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 * @brief  HTP backend lifecycle implementation (HexKL micro-API FastRPC).
 */

#ifdef ENABLE_HEXKL

#include <htp_backend.h>

#include <nntrainer_log.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// remote_handle64, CDSP_DOMAIN_ID, remote_session_control -- Hexagon SDK,
// not the HexKL addon. nntr_hvx.h is generated from test/htp/nntr_hvx.idl by
// this directory's meson.build (the same IDL the DSP skel is built from).
#include <remote.h>

#include <nntr_hvx.h>

#include <climits>

#include <htp_rpcmem.h>

namespace nntrainer {

HtpBackend &HtpBackend::global() {
  static HtpBackend instance;
  return instance;
}

bool HtpBackend::e2eRequested() {
  static const bool on = [] {
    const char *e = std::getenv("NNTR_HTP_E2E");
    return e != nullptr && std::atoi(e) != 0;
  }();
  return on;
}

/**
 * [#132 Part B E3] S2, the #178 probe's sequence (unittest_hvx_two_sessions
 * Q1, set3: rc 0 in 22 ms): reserve a new session on "cdsp", its effective
 * domain and its URI, the unsigned-PD control on that domain, then
 * nntr_hvx_open on the URI -- which takes the lite path there, because S1
 * already holds the HMX and the VTCM. Never FASTRPC_SESSION_CLOSE: the
 * destructor closes the handle only (plan section 4 step E3).
 * Called from openSecond(), after S1's arena is in place (see the header).
 * ponytail: the open is synchronous (no watchdog thread as in the probe,
 * whose 10 s kill never fired in three sittings); a hang here hangs the
 * app at load, and the handoff's s2_open > 2 s stop rule reads open_ms.
 */
bool HtpBackend::openSecond() {
  if (!tried2_ && enabled_ && e2eRequested()) {
    tried2_ = true;
    openSecondNow();
    if (!enabled2_)
      std::fprintf(stderr, "[HTP] s2: open FAILED (%s)\n", s2_error_.c_str());
  }
  return enabled2_;
}

void HtpBackend::openSecondNow() {
  char dom[] = "cdsp", sname[] = "nntr_s2";
  char err[160];
  remote_rpc_reserve_new_session_t rs;
  std::memset(&rs, 0, sizeof(rs));
  rs.domain_name = dom;
  rs.domain_name_len = std::strlen(dom);
  rs.session_name = sname;
  rs.session_name_len = std::strlen(sname);
  int rc = remote_session_control(FASTRPC_RESERVE_NEW_SESSION, &rs, sizeof(rs));
  if (rc != AEE_SUCCESS) {
    std::snprintf(err, sizeof(err), "reserve rc=0x%x%s", (unsigned)rc,
                  rc == 0x73 ? " (no second session on this device)" : "");
    s2_error_ = err;
    return;
  }
  remote_rpc_effective_domain_id_t ed;
  std::memset(&ed, 0, sizeof(ed));
  ed.domain_name = dom;
  ed.domain_name_len = rs.domain_name_len;
  ed.session_id = rs.session_id;
  rc = remote_session_control(FASTRPC_GET_EFFECTIVE_DOMAIN_ID, &ed, sizeof(ed));
  if (rc != AEE_SUCCESS) {
    std::snprintf(err, sizeof(err), "effective domain rc=0x%x", (unsigned)rc);
    s2_error_ = err;
    return;
  }
  std::string mod(nntr_hvx_URI);
  std::vector<char> uri(mod.size() + 64, '\0');
  remote_rpc_get_uri_t gu;
  std::memset(&gu, 0, sizeof(gu));
  gu.domain_name = dom;
  gu.domain_name_len = rs.domain_name_len;
  gu.session_id = rs.session_id;
  gu.module_uri = &mod[0];
  gu.module_uri_len = mod.size();
  gu.uri = uri.data();
  gu.uri_len = uri.size();
  rc = remote_session_control(FASTRPC_GET_URI, &gu, sizeof(gu));
  if (rc != AEE_SUCCESS) {
    std::snprintf(err, sizeof(err), "get uri rc=0x%x", (unsigned)rc);
    s2_error_ = err;
    return;
  }
  remote_rpc_control_unsigned_module up = {
    static_cast<int>(ed.effective_domain_id), 1};
  rc = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE, &up, sizeof(up));
  if (rc != AEE_SUCCESS) {
    std::snprintf(err, sizeof(err), "unsigned PD rc=0x%x", (unsigned)rc);
    s2_error_ = err;
    return;
  }
  const auto t0 = std::chrono::steady_clock::now();
  remote_handle64 h2 = 0;
  rc = nntr_hvx_open(uri.data(), &h2);
  const double open_ms = std::chrono::duration<double, std::milli>(
                           std::chrono::steady_clock::now() - t0)
                           .count();
  if (rc != AEE_SUCCESS) {
    std::snprintf(err, sizeof(err), "nntr_hvx_open rc=0x%x after %.1f ms",
                  (unsigned)rc, open_ms);
    s2_error_ = err;
    return;
  }
  handle2_ = static_cast<uint64_t>(h2);
  effdom2_ = static_cast<int>(ed.effective_domain_id);
  enabled2_ = true;
  // res: [hmx_locked, vtcm_size, vtcm_avail_kib, max_page_kib, -, open_path,
  // hvx_units]; 0x8000040e here is a skel older than session_info
  uint32_t info[7] = {0, 0, 0, 0, 0, 0, 0};
  const int irc = nntr_hvx_session_info(h2, info, 7);
  vtcm2_bytes_ = irc == AEE_SUCCESS ? info[1] : 0u;
  std::fprintf(stderr,
               "[HTP] s2: open s1_effdom=%d s2_session=%u s2_effdom=%d "
               "open_ms=%.1f info_rc=0x%x open_path=%u hmx=%u vtcm_kib=%u "
               "hvx_units=%u uri=%s\n",
               static_cast<int>(CDSP_DOMAIN_ID),
               static_cast<unsigned>(rs.session_id), effdom2_, open_ms,
               static_cast<unsigned>(irc), info[5], info[0], info[1] >> 10,
               info[6], uri.data());
}

HtpBackend::HtpBackend() {
  // Enables the unsigned-PD CDSP session the dev/bring-up skel needs.
  // Not fatal if it fails -- a signed production skel does not need it,
  // and nntr_hvx_open below is the real pass/fail signal either way.
  remote_rpc_control_unsigned_module unsigned_pd = {CDSP_DOMAIN_ID, 1};
  int pd_err = remote_session_control(DSPRPC_CONTROL_UNSIGNED_MODULE,
                                      &unsigned_pd, sizeof(unsigned_pd));
  if (pd_err != AEE_SUCCESS) {
    ml_logw("remote_session_control(unsigned PD) failed (err=%d); "
            "continuing -- a signed skel does not need it.",
            pd_err);
  }

  const std::string uri = std::string(nntr_hvx_URI) + "&_dom=cdsp";
  remote_handle64 h = 0;
  int err = nntr_hvx_open(uri.c_str(), &h);
  if (err != AEE_SUCCESS) {
    // Graceful disable: leave enabled_ = false so supports_*() reports
    // false and callers fall back to CPU. Not fatal.
    ml_logw("nntr_hvx_open failed (err=%d); HTP backend disabled, falling "
            "back to CPU. Is libnntr_hvx_skel.so on ADSP_LIBRARY_PATH?",
            err);
    return;
  }

  handle_ = static_cast<uint64_t>(h);
  enabled_ = true;

  // Poll-mode QoS: the host half of the measured 90 -> 3,900 us transport
  // spread (docs/htp_attention/34_fc_measured.md section4 item F; the
  // control call itself is test/unittest/htp_rpc_bench.h's
  // htp_set_latency_qos, ported here because this is the first production
  // caller -- every 34_fc_measured.md transport number assumes this is on,
  // and until now nothing in the model path turned it on). The DSP-side
  // half (DCVS/apptype vote) already runs unconditionally in
  // nntr_hvx_open's session setup; this is the other half of the same pair.
  // Best effort: an SDK or device without the control just keeps the old
  // interrupt-driven behavior, logged so a slow transport number is
  // explainable rather than silently misread as the kernel being slow.
  // NNTR_HTP_POLL_US: how long the host polls for the DSP's reply before
  // falling back to the interrupt wait. Measured on device (doc 51
  // section 2.20): at 100 us the MoE layer call's transport read 2.19 ms
  // and decode's 158 us; at 5000, 1.13 ms and 83 us, which is prefill
  // -30 ms and decode 20.6 -> 23.7 TPS. 10000 was refused by the driver
  // (the control call fails and the session falls back to PM QoS, which
  // the profile's first line reports as qos_mode=1), so 5000 is the
  // default and the ceiling between the two is still to be found. The
  // cost is a core spinning for up to 5 ms per call while it waits.
  struct remote_rpc_control_latency lat;
  std::memset(&lat, 0, sizeof(lat));
  lat.enable = RPC_POLL_QOS;
  lat.latency = 5000;
  if (const char *poll_us = std::getenv("NNTR_HTP_POLL_US")) {
    lat.latency = static_cast<uint32_t>(std::strtoul(poll_us, nullptr, 10));
    ml_logi("HtpBackend: poll QoS latency %u us (NNTR_HTP_POLL_US)",
            static_cast<unsigned>(lat.latency));
  }
  int qos_err =
    remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
  if (qos_err == AEE_SUCCESS) {
    qos_mode_ = 2;
    poll_us_ = lat.latency;
  } else {
    std::memset(&lat, 0, sizeof(lat));
    lat.enable = RPC_PM_QOS;
    lat.latency = 100;
    qos_err =
      remote_handle64_control(h, DSPRPC_CONTROL_LATENCY, &lat, sizeof(lat));
    qos_mode_ = (qos_err == AEE_SUCCESS) ? 1 : 0;
  }
  if (qos_mode_ == 0) {
    ml_logw("HtpBackend: poll and PM latency QoS both rejected (err=%d); "
            "transport will pay the interrupt-wake tail (34_fc_measured.md "
            "section4 item F).",
            qos_err);
  }
}

void *HtpBackend::alloc_shared(size_t bytes) {
  // rpcmem_alloc takes an int, and no single KV cache slab approaches
  // 2 GiB. The default flags are the cached mapping: the CPU writes new K/V
  // rows into this memory every token, and FastRPC cleans the passed range
  // before each call. rpcmem comes through HtpRpcMemApi (dlsym) like every
  // other rpcmem user here, so libnntrainer.so gains no import for it.
  const HtpRpcMemApi &api = HtpRpcMemApi::get();
  if (!enabled_ || api.alloc == nullptr || bytes == 0 ||
      bytes > static_cast<size_t>(INT_MAX)) {
    return nullptr;
  }
  void *block = api.alloc(HTP_RPC_HEAP_ID_SYSTEM, HTP_RPC_FLAGS_DEFAULT,
                          static_cast<int>(bytes));
  if (!block) {
    ml_logw("rpcmem_alloc(%zu bytes) failed; this buffer stays on the heap "
            "and FastRPC copies it per call",
            bytes);
  }
  if (block) {
    std::lock_guard<std::mutex> lock(shared_mutex_);
    shared_blocks_.emplace(reinterpret_cast<uintptr_t>(block), bytes);
  }
  return block;
}

void HtpBackend::free_shared(void *block) {
  if (block) {
    std::lock_guard<std::mutex> lock(shared_mutex_);
    shared_blocks_.erase(reinterpret_cast<uintptr_t>(block));
    HtpRpcMemApi::get().free_(block);
  }
}

bool HtpBackend::isSharedRange(const void *ptr, size_t bytes) {
  if (!ptr || !bytes)
    return false;
  const uintptr_t address = reinterpret_cast<uintptr_t>(ptr);
  std::lock_guard<std::mutex> lock(shared_mutex_);
  auto it = shared_blocks_.upper_bound(address);
  if (it == shared_blocks_.begin())
    return false;
  --it;
  const size_t offset = address - it->first;
  return offset <= it->second && bytes <= it->second - offset;
}

HtpBackend::~HtpBackend() {
  if (enabled_) {
    for (auto &fn : at_close_) {
      fn();
    }
    for (auto &fn : at_close_last_) {
      fn();
    }
    // [#132 Part B E3] S2 first, after the hooks have unmapped every buffer
    // it had (#178's rule); the handle only, never FASTRPC_SESSION_CLOSE
    if (enabled2_) {
      nntr_hvx_close(static_cast<remote_handle64>(handle2_));
      enabled2_ = false;
    }
    nntr_hvx_close(static_cast<remote_handle64>(handle_));
    enabled_ = false;
  }
}

} // namespace nntrainer

#endif // ENABLE_HEXKL
