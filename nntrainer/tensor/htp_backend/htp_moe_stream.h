// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_moe_stream.h
 * @date   30 Sep 2026
 * @brief  Shared ARM-DSP mailbox for a resident decode MoE call
 */

#ifndef __NNTRAINER_HTP_MOE_STREAM_H__
#define __NNTRAINER_HTP_MOE_STREAM_H__

#include <stddef.h>
#include <stdint.h>

#define NNTR_HVX_STREAM_MAGIC 0x5354524du /* "STRM" */
#define NNTR_HVX_STREAM_VERSION 5u
#define NNTR_HVX_STREAM_MAX_STAGES 64u
#define NNTR_HVX_STREAM_PAYLOAD_XOR 0x5a5a5a5au

#define NNTR_HVX_STREAM_MAX_EXPERTS 32u
#define NNTR_HVX_STREAM_MAX_ROUTES 4u
#define NNTR_HVX_STREAM_MAX_HIDDEN 2048u

/**
 * @brief Control line shared by the latency probe and the MoE stream.
 *
 * request_seq and done_seq are monotonically increasing. The producer fills
 * the payload, then publishes request_seq; the consumer fills the output,
 * then publishes done_seq. Keeping this block on its own 128-byte line avoids
 * polling invalidations touching the activation payload.
 */
typedef struct nntr_hvx_stream_control {
  uint32_t magic;
  uint32_t version;
  uint32_t stages;
  volatile uint32_t ready;
  volatile uint32_t request_seq;
  volatile uint32_t request_payload;
  volatile uint32_t reply_payload;
  volatile uint32_t done_seq;
  volatile uint32_t status;
  volatile uint32_t stop_requested;
  uint32_t profile_enabled;
  uint32_t poll_spin_count;
  uint32_t poll_backoff_us;
  uint32_t reserved[19];
} nntr_hvx_stream_control;

/** @brief DSP-side breakdown for one resident MoE stage. */
typedef struct nntr_moe_stream_stage_profile {
  uint32_t request_wait_us;
  uint32_t request_polls;
  uint32_t request_backoffs;
  uint32_t payload_invalidate_us;
  uint32_t kernel_us;
  uint32_t output_flush_us;
  uint32_t completion_flush_us;
} nntr_moe_stream_stage_profile;

/** @brief One M=1 MoE command and its output, in an uncached rpcmem arena. */
typedef struct nntr_moe_stream_mailbox {
  nntr_hvx_stream_control control;
  uint32_t K;
  uint32_t inter;
  uint32_t N_out;
  uint32_t n_experts;
  uint32_t n_routes;
  uint32_t reserved0[27];
  uint32_t h_gate_up[NNTR_HVX_STREAM_MAX_EXPERTS];
  uint32_t h_down[NNTR_HVX_STREAM_MAX_EXPERTS];
  uint32_t row_count[NNTR_HVX_STREAM_MAX_EXPERTS];
  uint32_t row_index[NNTR_HVX_STREAM_MAX_ROUTES];
  float row_weight[NNTR_HVX_STREAM_MAX_ROUTES];
  float act_f32[NNTR_HVX_STREAM_MAX_HIDDEN];
  float out_f32[NNTR_HVX_STREAM_MAX_HIDDEN];
  nntr_moe_stream_stage_profile profile[NNTR_HVX_STREAM_MAX_STAGES];
} nntr_moe_stream_mailbox;

#if defined(__cplusplus)
static_assert(sizeof(nntr_hvx_stream_control) == 128,
              "stream control ABI must stay one 128-byte line");
static_assert(offsetof(nntr_moe_stream_mailbox, h_gate_up) == 256,
              "stream payload header must stay two 128-byte lines");
#else
_Static_assert(sizeof(nntr_hvx_stream_control) == 128,
               "stream control ABI must stay one 128-byte line");
_Static_assert(offsetof(nntr_moe_stream_mailbox, h_gate_up) == 256,
               "stream payload header must stay two 128-byte lines");
#endif

#endif /* __NNTRAINER_HTP_MOE_STREAM_H__ */
