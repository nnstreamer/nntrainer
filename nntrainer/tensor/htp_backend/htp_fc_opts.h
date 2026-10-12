// SPDX-License-Identifier: Apache-2.0
/** @file htp_fc_opts.h
 * @brief Shared host/DSP M=1 projection options in the session opts word.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */
#ifndef __NNTRAINER_HTP_FC_OPTS_H__
#define __NNTRAINER_HTP_FC_OPTS_H__

#include <stdint.h>
#include <stdlib.h>

#define HTP_FC_FLAG_M1_HVX 0x1000000u
#define HTP_FC_CHUNK_SHIFT 25u
#define HTP_FC_CHUNK_MASK 0x6000000u
#define HTP_FC_FLAGS_KNOWN (HTP_FC_FLAG_M1_HVX | HTP_FC_CHUNK_MASK)

static inline uint32_t htp_fc_chunk_tiles(uint32_t flags) {
  return 4u << ((flags & HTP_FC_CHUNK_MASK) >> HTP_FC_CHUNK_SHIFT);
}

/** NNTR_HTP_FC_M1_HVX (default on), NNTR_HTP_FC_CHUNK_TILES (default 8). */
static inline uint32_t htp_fc_opts_flags(const char *env,
                                         const char *chunk_env) {
  if (env && atoi(env) == 0)
    return 0u;
  const int n = chunk_env ? atoi(chunk_env) : 8;
  const uint32_t code = n >= 32 ? 3u : n >= 16 ? 2u : n >= 8 ? 1u : 0u;
  return HTP_FC_FLAG_M1_HVX | (code << HTP_FC_CHUNK_SHIFT);
}

#endif
