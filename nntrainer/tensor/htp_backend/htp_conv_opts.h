// SPDX-License-Identifier: Apache-2.0
/**
 * @file htp_conv_opts.h
 * @brief Shared host/DSP decode convolution options in the session opts word.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HTP_CONV_OPTS_H__
#define __NNTRAINER_HTP_CONV_OPTS_H__

#include <stdint.h>
#include <stdlib.h>

/** Bits 0..19 belong to the existing MoE options. The echo validates these
 * bits too, so an old skel cannot silently benchmark HMX as HVX. */
#define HTP_CONV_FLAG_M1_HVX 0x100000u
#define HTP_CONV_CHUNK_SHIFT 21u
#define HTP_CONV_CHUNK_MASK 0x600000u
#define HTP_CONV_FLAG_ROWS4 0x800000u
#define HTP_CONV_OUT_CHUNK_SHIFT 30u
#define HTP_CONV_OUT_CHUNK_MASK 0xc0000000u
#define HTP_CONV_FLAGS_KNOWN                                                   \
  (HTP_CONV_FLAG_M1_HVX | HTP_CONV_CHUNK_MASK | HTP_CONV_FLAG_ROWS4 |          \
   HTP_CONV_OUT_CHUNK_MASK)

/** @brief Number of 32-column tiles per DMA slot: 4, 8, 16 or 32. */
static inline uint32_t htp_conv_chunk_tiles(uint32_t flags) {
  return 4u << ((flags & HTP_CONV_CHUNK_MASK) >> HTP_CONV_CHUNK_SHIFT);
}

/** @brief Output chunk: zero field inherits the input; other codes select
 * 16, 24 or 32 tiles. Bits 24..29 belong to FC and QS2 GEMV options. */
static inline uint32_t htp_conv_out_chunk_tiles(uint32_t flags) {
  const uint32_t code =
    (flags & HTP_CONV_OUT_CHUNK_MASK) >> HTP_CONV_OUT_CHUNK_SHIFT;
  return code ? 8u * (code + 1u) : htp_conv_chunk_tiles(flags);
}

/** @brief NNTR_HTP_CONV_M1_HVX (default on) and NNTR_HTP_CONV_CHUNK_TILES
 * (default 8). Round the input chunk down to a supported power of two.
 * NNTR_HTP_CONV_OUT_CHUNK_TILES selects 16, 24 or 32 output tiles; unset
 * or less than 16 inherits the input chunk. */
static inline uint32_t htp_conv_opts_flags(const char *env,
                                           const char *chunk_env,
                                           const char *out_chunk_env) {
  if (env && atoi(env) == 0)
    return 0u;
  const int tiles = chunk_env ? atoi(chunk_env) : 8;
  const uint32_t code = tiles >= 32   ? 3u
                        : tiles >= 16 ? 2u
                        : tiles >= 8  ? 1u
                                      : 0u;
  const int out_tiles = out_chunk_env ? atoi(out_chunk_env) : 0;
  const uint32_t out_code = out_tiles >= 32   ? 3u
                            : out_tiles >= 24 ? 2u
                            : out_tiles >= 16 ? 1u
                                              : 0u;
  return HTP_CONV_FLAG_M1_HVX | (code << HTP_CONV_CHUNK_SHIFT) |
         (out_code << HTP_CONV_OUT_CHUNK_SHIFT);
}

#endif /* __NNTRAINER_HTP_CONV_OPTS_H__ */
