// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 dlwlzzero <dlwlzzero@gmail.com>
 *
 * @file   htp_dspq_wire.h
 * @date   28 Sep 2026
 * @brief  [#141] The dspqueue packet of one M==1 MoE layer call, shared by
 *         the ARM side (htp_compute_ops.cpp) and the DSP side
 *         (test/htp/nntr_hvx_dspq.c), as pure C99
 * @see    https://github.com/nntrainer/nntrainer
 * @author dlwlzzero <dlwlzzero@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 * Plan docs/plans/141-dspq-moe.md section 3.1. Request message: the
 * header below, then h_gate_up[n_experts], h_down[n_experts],
 * row_count[n_experts], row_index[n_rows] (u32) and row_weight[n_rows]
 * (f32), in that order; buffer reference 0 is the M x K f32 activation,
 * 1 the M x N_out f32 output. Response message: {seq, rc} and, when the
 * request's HTP_DSPQ_FLAG_TIMED is set, the HTP_DSPQ_STAGES stage slots of
 * mm_u8i4_moe_layer_timed. The DSP calls the FastRPC method's own C
 * function with these arguments, so the arithmetic is the same.
 */
#ifndef __HTP_DSPQ_WIRE_H__
#define __HTP_DSPQ_WIRE_H__

#include <stdint.h>

#define HTP_DSPQ_OP_MOE 1u
#define HTP_DSPQ_OP_QUIT 2u
/** @brief [#132 Part B E2] One decode token of the session's token driver
 *  role (nntr_hvx_token.c): htp_dspq_token_req, answered with
 *  htp_dspq_token_resp. S2's packet carries buffer 0, the embedding row
 *  (op 0's width in f32), and with HTP_DSPQ_TOKEN_LOGITS buffer 1, the
 *  logits (vocab f32); S1's carries none. */
#define HTP_DSPQ_OP_TOKEN 3u
/** @brief One decode math call; buffer 0 packs x, aux, param (f32),
 * buffer 1 holds the existing inference_math output layout. */
#define HTP_DSPQ_OP_MATH 4u
typedef struct {
  uint32_t op, seq, flags, kind, M, N, chunk;
  float eps, scale;
  uint32_t x_len, aux_len, param_len, out_bytes;
} htp_dspq_math_req;

/** @brief Router input: x, optional gamma, bias. Buffer 2 is immutable W.
 * Output: logits, selected ids (u32), routing weights. */
#define HTP_DSPQ_OP_ROUTER 6u
typedef struct {
  uint32_t op, seq, flags, K, E, top_k, n_sel, gamma_len;
  float eps;
} htp_dspq_router_req;

/** Router with an input norm. Input: x, [residual], norm gamma, [router
 * gamma], bias. Output: usual router result padded to 32 floats, sum, norm. */
#define HTP_DSPQ_OP_ROUTER_NORM 7u
typedef struct {
  htp_dspq_router_req router;
  uint32_t add;
  float norm_eps;
} htp_dspq_router_norm_req;

/** Input q/k/v/cos/sin; output FP32 q then FP16 k and v. */
#define HTP_DSPQ_OP_ROPE_QKV 8u
typedef struct {
  uint32_t op, seq, flags, q_width, kv_width, head_dim;
} htp_dspq_rope_qkv_req;

#define HTP_DSPQ_FLAG_TIMED 1u
/** @brief OP_MOE: the call's act argument (gelu_tanh epilogue) is 1. */
#define HTP_DSPQ_FLAG_GELU 2u
/** OP_MOE: append a count and compact QS2 slot updates after routing.
 * Reused handles retain their shape; the queue commits metadata before FFN. */
#define HTP_DSPQ_FLAG_EXPERT_UPDATES 4u
#define HTP_DSPQ_FLAG_EXPERT_BATCH_DMA 8u
#define HTP_DSPQ_MAX_EXPERT_UPDATES 8u
typedef struct {
  uint32_t h_gu, h_dn, arena, off_gu, off_dn;
} htp_dspq_expert_update;
/** @brief OP_TOKEN: S2 writes the logits into buffer 1 (NNTR_PPL_DECODE,
 *  the shadows); without it only the id travels. */
#define HTP_DSPQ_TOKEN_LOGITS 1u
/** @brief mm_u8i4_moe_layer_timed's slot count; both sides check theirs. */
#define HTP_DSPQ_STAGES 31u
#define HTP_DSPQ_MAX_MSG 4096u
#define HTP_DSPQ_REQ_QUEUE_BYTES (16u * 1024u)
#define HTP_DSPQ_RESP_QUEUE_BYTES (4u * 1024u)
/** @brief Each of the two queue-owned ION buffers (act, out). */
#define HTP_DSPQ_BUF_BYTES (64u * 1024u)

/** Decode Conv on the HVX path only. Input: x (or packed norm inputs),
 * conv_w[3*C], state[2*C]. Output: y[N_out], optional norm result[2*K],
 * updated state[2*C]. State travels in the flushed output reference.
 * norm_kind: 0 = none, 1 = RMSNorm, 2 = add + RMSNorm. */
#define HTP_DSPQ_OP_CONV 9u
typedef struct {
  uint32_t op, seq, flags, K, C, N_out, h_in[3], h_out, norm_kind;
  float eps;
} htp_dspq_conv_req;

static inline int htp_dspq_conv_shape_valid(const htp_dspq_conv_req *h) {
  return h->K && h->C && h->N_out && h->K <= 8192u && h->C <= 8192u &&
         h->N_out <= 8192u && !(h->K % 32u) && !(h->C % 32u) &&
         !(h->N_out % 32u) && h->norm_kind <= 2u &&
         (!h->norm_kind || h->K <= 4096u);
}

static inline uint64_t htp_dspq_conv_input_bytes(const htp_dspq_conv_req *h) {
  return 4u * ((uint64_t)(h->norm_kind + 1u) * h->K + 5ull * h->C);
}

static inline uint64_t htp_dspq_conv_output_bytes(const htp_dspq_conv_req *h) {
  return 4u *
         ((uint64_t)h->N_out + (h->norm_kind ? 2ull * h->K : 0u) + 2ull * h->C);
}

/** Conv input cache, owned by the queue. Input: x, [residual], followed
 * by only the selected updates in conv_w, gamma, state order. Output is
 * unchanged: state is always returned, preserving the public in/out API.
 * A new/rebound slot requires conv_w + state and gamma when using norm.
 * The host compares contents, not pointers, before omitting an update. */
#define HTP_DSPQ_OP_CONV_CACHED 12u
#define HTP_DSPQ_CONV_CACHE_SLOTS 64u
#define HTP_DSPQ_CONV_UPDATE_W 1u
#define HTP_DSPQ_CONV_UPDATE_GAMMA 2u
#define HTP_DSPQ_CONV_UPDATE_STATE 4u
#define HTP_DSPQ_CONV_UPDATES 7u
typedef struct {
  htp_dspq_conv_req conv;
  uint32_t slot, updates;
} htp_dspq_conv_cached_req;

static inline int htp_dspq_conv_binding_equal(const htp_dspq_conv_req *a,
                                              const htp_dspq_conv_req *b) {
  return a->K == b->K && a->C == b->C && a->N_out == b->N_out &&
         a->h_in[0] == b->h_in[0] && a->h_in[1] == b->h_in[1] &&
         a->h_in[2] == b->h_in[2] && a->h_out == b->h_out;
}

static inline uint64_t
htp_dspq_conv_cached_input_bytes(const htp_dspq_conv_cached_req *h) {
  return 4ull *
         ((uint64_t)(h->conv.norm_kind == 2u ? 2u : 1u) * h->conv.K +
          ((h->updates & HTP_DSPQ_CONV_UPDATE_W) ? 3ull * h->conv.C : 0) +
          ((h->updates & HTP_DSPQ_CONV_UPDATE_GAMMA) ? h->conv.K : 0) +
          ((h->updates & HTP_DSPQ_CONV_UPDATE_STATE) ? 2ull * h->conv.C : 0));
}

/** Decode FC, optionally with the same fused norms as the FastRPC entry.
 * Buffer 0: activation (or pending norm inputs), pre/post gamma, RoPE table.
 * Buffer 1: projection output followed by pending norm's sum and result.
 * Metadata stays in the queue packet, avoiding per-argument RPC mappings. */
#define HTP_DSPQ_OP_FC 10u
#define HTP_DSPQ_FC_HANDLES 16u
typedef struct {
  uint32_t op, seq, flags, K, N, count, fused, norm_kind;
  uint32_t pre_len, post_len, rope_hd, rope_handles;
  float eps, norm_eps;
  uint32_t handles[HTP_DSPQ_FC_HANDLES];
  uint32_t chunks[HTP_DSPQ_FC_HANDLES];
  uint32_t offsets[HTP_DSPQ_FC_HANDLES];
  uint32_t strides[HTP_DSPQ_FC_HANDLES];
} htp_dspq_fc_req;

static inline int htp_dspq_fc_shape_valid(const htp_dspq_fc_req *h) {
  return h->K && h->N && h->K <= 16384u && h->N <= 16384u && !(h->K % 32u) &&
         !(h->N % 32u) && h->count && h->count <= HTP_DSPQ_FC_HANDLES &&
         h->fused <= 1u && h->norm_kind <= 2u &&
         (!h->norm_kind || (h->fused && h->K <= 4096u)) &&
         (!h->pre_len || h->pre_len == h->K) && h->post_len <= 16384u &&
         h->rope_hd <= 512u && (!h->rope_hd || !(h->rope_hd % 64u)) &&
         h->rope_handles <= h->count &&
         (h->fused ||
          !(h->pre_len || h->post_len || h->rope_hd || h->norm_kind));
}
static inline uint64_t htp_dspq_fc_input_bytes(const htp_dspq_fc_req *h) {
  return 4u * ((uint64_t)(h->norm_kind + 1u) * h->K + h->pre_len + h->post_len +
               2ull * h->rope_hd);
}
static inline uint64_t htp_dspq_fc_output_bytes(const htp_dspq_fc_req *h) {
  return 4u * ((uint64_t)h->N + (h->norm_kind ? 2ull * h->K : 0u));
}
/** OP_MATH only: buffer 2 carries aux, allowing two 64 KiB inputs without
 * enlarging every queue allocation/cache-maintenance range. */
#define HTP_DSPQ_MATH_AUX_BUFFER 2u

/** Decode QS4CX router using registered weights and the HVX M=1 FC kernel.
 * Input: x, [residual], [norm gamma], bias. Output: ids, weights, padding
 * to 32 floats, sum, normalized input retained for the next MoE call.
 * norm_kind: 0 = input already normalized, 1 = RMSNorm, 2 = add + RMSNorm. */
#define HTP_DSPQ_OP_ROUTER_Q4 11u
typedef struct {
  uint32_t op, seq, flags, K, E, w_handle, top_k, n_sel, norm_kind;
  float eps;
} htp_dspq_router_q4_req;

static inline int
htp_dspq_router_q4_shape_valid(const htp_dspq_router_q4_req *h) {
  return h->K && h->K <= 4096u && !(h->K % 32u) && h->E && h->E <= 128u &&
         !(h->E % 32u) && h->top_k && h->top_k <= h->n_sel &&
         h->n_sel <= h->E && h->norm_kind <= 2u;
}
static inline uint32_t
htp_dspq_router_q4_norm_offset(const htp_dspq_router_q4_req *h) {
  return (h->n_sel + h->top_k + 31u) & ~31u;
}
static inline uint64_t
htp_dspq_router_q4_input_bytes(const htp_dspq_router_q4_req *h) {
  return 4ull * ((uint64_t)(h->norm_kind + 1u) * h->K + h->E);
}
static inline uint64_t
htp_dspq_router_q4_output_bytes(const htp_dspq_router_q4_req *h) {
  return 4ull * (htp_dspq_router_q4_norm_offset(h) + 2ull * h->K);
}

/** @brief The request message's header, 9 x u32. */
typedef struct {
  uint32_t op, seq, flags, M, K, inter, N_out, n_experts, n_rows;
} htp_dspq_req_hdr;

/** @brief The response message; stage_us travels only for a timed call. */
typedef struct {
  uint32_t seq;
  int32_t rc;
  uint32_t stage_us[HTP_DSPQ_STAGES];
  /** Present only for a timed OP_MOE carrying expert updates. */
  uint32_t expert_update_us;
} htp_dspq_resp;

#define HTP_DSPQ_RESP_BASE_BYTES 8u

/** @brief [#132 Part B E2] OP_TOKEN's request: seq is the token number
 *  both sessions' packets carry (the mailbox's sequence values derive
 *  from it), pos the token's position. */
typedef struct {
  uint32_t op, seq, flags, pos;
} htp_dspq_token_req;

/** @brief The graph's op kinds (htp_graph_desc.h HTP_OP_KIND_N; the ARM
 *  side asserts they agree). */
#define HTP_DSPQ_TOKEN_KINDS 11u

/** @brief OP_TOKEN's response: id (S2: LM_HEAD's argmax), the hops this
 *  side made, its wait for the other side, the pcycles of its ops; the
 *  token's wall time on this side and the pcycles over it (their ratio is
 *  the clock), and the op pcycles per kind; [#194 L0] of the wait, the
 *  time from each of the other side's posts to this side's wake-up. */
typedef struct htp_dspq_token_resp_s {
  uint32_t seq;
  int32_t rc;
  uint32_t id, hops, wait_us, pcycles;
  uint32_t wall_us, wall_pcyc;
  uint32_t kind_pcyc[HTP_DSPQ_TOKEN_KINDS];
  uint32_t hop_us;
  uint32_t t_in_us, t_out_us; /**< [#194 L0] QTimer us (low 32 bits) when
                                   the packet was read / the response is
                                   written: the ARM's dispatch and return */
} htp_dspq_token_resp;

/** @brief The request message length for n_experts experts and n_rows
 *  routed rows (u64, so a hostile count cannot wrap). */
static inline uint64_t htp_dspq_req_bytes(uint32_t n_experts, uint32_t n_rows) {
  return (uint64_t)sizeof(htp_dspq_req_hdr) + 12u * (uint64_t)n_experts +
         8u * (uint64_t)n_rows;
}

#endif /* __HTP_DSPQ_WIRE_H__ */
