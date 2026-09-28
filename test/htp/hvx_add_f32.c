// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   hvx_add_f32.c
 * @date   03 Aug 2026
 * @brief  DSP-side implementation of the nntr_hvx FastRPC interface
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 */

#include <stdlib.h>
#include <string.h>

#include <AEEStdErr.h>
#include <HAP_farf.h>
#include <remote.h>

#include <hexagon_types.h>
#include <hvx_hexagon_protos.h>
#include <qurt.h>

#include "hexkl_micro.h"
#include "hvx_worker_pool.h"
#include "nntr_hvx.h"
#include "nntr_hvx_session.h"

#if __has_include(<HAP_power.h>)
#include <HAP_power.h>
#define NNTR_HVX_HAVE_HAP_POWER 1
#endif

/** @brief HVX vector width in bytes (128B mode). */
#define VLEN 128u
/** @brief f32 lanes per HVX vector. */
/** Kept as an int-casted expression (not the unsigned LANES form used
    elsewhere) because n_vec/i in this file's loops are int. */
#define LANES ((int)(VLEN / sizeof(float)))

int nntr_hvx_open(const char *uri, remote_handle64 *handle) {
  (void)uri;

  nntr_hvx_session *s = (nntr_hvx_session *)calloc(1, sizeof(nntr_hvx_session));
  if (!s) {
    return AEE_ENOMEMORY;
  }

  /** Bits 15:8 hold the number of 128-byte HVX contexts. Checked once for
     the session: every entry in this skel uses HVX and the unit count does
     not change while a session is open. */
  if (((qurt_hvx_get_units() >> 8) & 0xFF) == 0) {
    free(s);
    return AEE_EUNSUPPORTED;
  }

  // hw_init and the HMX lock happen once here, for the session's whole
  // lifetime, instead of per call (doc15 §3/§4) -- every other entry point
  // in this skel reaches vtcm_base/vtcm_size/config_off through the
  // session rather than re-acquiring either.
  // hexkl_micro_hw_init's arity is not stable across hexkl_addon drops: the
  // beta2 header ref_08's docs describe takes three args (the third reports
  // the HMX fp16 throughput rate), an older checkout this skel was first
  // built against took two. The header carries no version macro to branch
  // on, so this defaults to the three-arg form -- what hxkl-beta2 ships --
  // and leaves -DNNTR_HEXKL_HW_INIT_2ARG as the escape hatch for the older
  // one. Nothing here wants hmx_fp16_rate: this skel's HMX use is int8/int4
  // only. If a future drop changes the arity again, this is the one line to
  // touch.
#ifdef NNTR_HEXKL_HW_INIT_2ARG
  int res = hexkl_micro_hw_init(&s->vtcm_base, &s->vtcm_size);
#else
  uint32_t hmx_fp16_rate = 0; /* unused: int8/int4 HMX only */
  int res = hexkl_micro_hw_init(&s->vtcm_base, &s->vtcm_size, &hmx_fp16_rate);
  (void)hmx_fp16_rate;
#endif
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "nntr_hvx_open: hexkl_micro_hw_init failed: 0x%08x", res);
    free(s);
    return res;
  }
  // config_off depends only on vtcm_size (see hexkl_mm_u8i4_plan), so it is
  // computed once here rather than at every mm_u8i4_layer call.
  const uint32_t config_size = hexkl_micro_hmx_config_size();
  if (s->vtcm_size < config_size) {
    free(s);
    return AEE_ENOMEMORY;
  }
  s->config_off =
    (s->vtcm_size - config_size) & ~(HEXKL_HMX_CONFIG_ALIGNMENT - 1u);

  res = hexkl_micro_hmx_lock();
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "nntr_hvx_open: hexkl_micro_hmx_lock failed: 0x%08x", res);
    free(s);
    return res;
  }
  s->hmx_locked = 1;

  res = hexkl_micro_hmx_setup_acc_read_int32(s->vtcm_base, s->config_off);
  if (res != AEE_SUCCESS) {
    FARF(ERROR, "nntr_hvx_open: setup_acc_read_int32 failed: 0x%08x", res);
    hexkl_micro_hmx_unlock();
    free(s);
    return res;
  }

  // Sized from the real HVX context count (bits 15:8, same decode
  // nntr_hvx_add_f32 uses to check HVX is present at all) minus one, since
  // this session's own FastRPC thread uses an HVX context too when it runs
  // quant's own share of the work.
  const int hvx_units_raw = qurt_hvx_get_units();
  const uint32_t n_hvx = (qurt_hvx_get_units() >> 8) & 0xFFu;
  (void)hvx_units_raw; // used only by FARF below, which may compile to nothing
  FARF(HIGH,
       "nntr_hvx_open: qurt_hvx_get_units()=0x%08x n_hvx=%u pool_workers=%u",
       hvx_units_raw, (unsigned)n_hvx, (unsigned)(n_hvx > 1 ? n_hvx - 1 : 0));
  s->quant_pool = hvx_worker_pool_create(n_hvx > 1 ? n_hvx - 1 : 0);
  if (!s->quant_pool) {
    FARF(ERROR, "nntr_hvx_open: hvx_worker_pool_create failed");
    hexkl_micro_hmx_unlock();
    free(s);
    return AEE_ENOMEMORY;
  }

#ifdef NNTR_HVX_HAVE_HAP_POWER
  /** Vote the DSP's clocks up and its wake latency down, once, for the
   * session's lifetime. This is the DSP half of the measured 90 -> 3,900 us
   * transport spread: without a vote the core power-collapses between
   * FastRPC calls and every call pays the DCVS ramp back up. Best effort by
   * design -- a device that rejects the vote runs exactly as before, only
   * slower to wake, so failures are logged and ignored. The votes die with
   * this PD; close() does not need to unwind them. */
  {
    HAP_power_request_t req;
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_apptype;
    req.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
    if (HAP_power_set((void *)s, &req) != AEE_SUCCESS) {
      FARF(HIGH, "nntr_hvx_open: apptype vote rejected (continuing)");
    }
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_DCVS_v2;
    req.dcvs_v2.dcvs_enable = 1;
    req.dcvs_v2.dcvs_option = HAP_DCVS_V2_PERFORMANCE_MODE;
    req.dcvs_v2.set_latency = 1;
    req.dcvs_v2.latency = 100; /* us of wake latency we are willing to pay */
    req.dcvs_v2.set_dcvs_params = 1;
    req.dcvs_v2.dcvs_params.min_corner = HAP_DCVS_VCORNER_NOM;
    req.dcvs_v2.dcvs_params.target_corner = HAP_DCVS_VCORNER_TURBO;
    req.dcvs_v2.dcvs_params.max_corner = HAP_DCVS_VCORNER_TURBO;
    if (HAP_power_set((void *)s, &req) != AEE_SUCCESS) {
      FARF(HIGH, "nntr_hvx_open: DCVS vote rejected (continuing)");
    }
    /* The bus, separately from the core. The DCVS vote above raises the
       DSP clock, and mm at 17.5 ns a tile says it holds; it says nothing
       about DDR. The MoE layer's weight DMA measured 16-18 GB/s in situ
       (doc 46 section 50.7) against 38.8 for the same arena in isolation,
       and a decode token leaves the SoC mostly idle -- the bus is free to
       clock down between 2 ms bursts. This asks for it at 40 GB/s, roughly
       what the isolated probe reached. Same best-effort policy as above.
       ponytail: an unconditional vote for the session's lifetime, which
       is the measurement; if it is what moves DMA_FIRST, the product
       shape is a vote around the layer call, or a lower figure.
       -DNNTR_HVX_NO_BUS_VOTE (HEX_EXTRA_CFLAGS in build.sh) builds the
       skel without this vote so the DMA probe can measure its worth. */
#ifndef NNTR_HVX_NO_BUS_VOTE
    memset(&req, 0, sizeof(req));
    req.type = HAP_power_set_mips_bw;
    req.mips_bw.set_bus_bw = 1;
    req.mips_bw.bwBytePerSec = 40000000000ull;
    req.mips_bw.busbwUsagePercentage = 100;
    if (HAP_power_set((void *)s, &req) != AEE_SUCCESS) {
      FARF(HIGH, "nntr_hvx_open: bus bandwidth vote rejected (continuing)");
    }
#endif
  }
#endif

  *handle = (remote_handle64)s;
  return AEE_SUCCESS;
}

int nntr_hvx_close(remote_handle64 handle) {
  nntr_hvx_session *s = (nntr_hvx_session *)handle;
  if (!s) {
    return AEE_SUCCESS;
  }
  /* [#85] The graph names weight handles, so it goes before the tables. */
  hexkl_graph_free(s->graph);
  s->graph = NULL;
  for (uint32_t i = 0; i < HEXKL_MM_U8I4_MAX_WEIGHTS; ++i) {
    if (s->weights_u8i4.slots[i].in_use) {
      hexkl_weight_u8i4_release(&s->weights_u8i4, i);
    }
  }
  for (uint32_t i = 0; i < HEXKL_MM_U8I8_MAX_WEIGHTS; ++i) {
    if (s->weights_u8i8.slots[i].in_use) {
      hexkl_weight_u8i8_release(&s->weights_u8i8, i);
    }
  }
  nntr_hvx_arenas_put_all(s);
  hexkl_moe_scratch_free(&s->moe_scratch);
  /* [#81] The attention cache borrows the pool, so it goes first. */
  hvx_attn_m1_free(s->attn_m1);
  s->attn_m1 = NULL;
  hvx_worker_pool_destroy(s->quant_pool);
  int res = AEE_SUCCESS;
  if (s->hmx_locked) {
    res = hexkl_micro_hmx_unlock();
    if (res != AEE_SUCCESS) {
      FARF(ERROR, "nntr_hvx_close: hexkl_micro_hmx_unlock failed: 0x%08x", res);
    }
  }
  free(s);
  return res;
}

int nntr_hvx_add_f32(remote_handle64 handle, const float *a, int aLen,
                     const float *b, int bLen, float *c, int cLen) {
  (void)handle;

  if (aLen != bLen || aLen != cLen) {
    return AEE_EBADPARM;
  }

  // FastRPC buffers carry no vector alignment guarantee, so the unaligned
  // vector type is what keeps this from faulting on a misaligned input.
  const HVX_UVector *va = (const HVX_UVector *)a;
  const HVX_UVector *vb = (const HVX_UVector *)b;
  HVX_UVector *vc = (HVX_UVector *)c;

  const int n_vec = aLen / LANES;
  for (int i = 0; i < n_vec; ++i) {
    vc[i] = Q6_Vsf_vadd_VsfVsf(va[i], vb[i]);
  }

  // ponytail: scalar tail; a masked vector store would fold it in, add
  // that only if the tail shows up in a profile.
  for (int i = n_vec * LANES; i < aLen; ++i) {
    c[i] = a[i] + b[i];
  }

  return AEE_SUCCESS;
}
