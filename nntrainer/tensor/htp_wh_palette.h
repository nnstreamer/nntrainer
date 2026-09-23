// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 SeungHui Lee <shsh1004.lee@samsung.com>
 *
 * @file   htp_wh_palette.h
 * @date   23 Sep 2026
 * @brief  Restrict int4 weight codes to four levels a group -- the 2-bit
 *         expert format's arithmetic, with no format change attached
 * @see    https://github.com/nntrainer/nntrainer
 * @author SeungHui Lee <shsh1004.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * The HMX epilogue applies one weight scale per output column over the whole
 * of K (hvx_dequant_i32.h), so a 2-bit weight cannot carry a per-group scale
 * of its own the way arXiv 2511.11248's GPTQ blocks do -- a per-group scale
 * would mean draining the accumulator per group, and acc_read is already 21%
 * of the MoE call. What fits instead is a palette: two bits index four int4
 * codes, and the expansion back to the int4 lattice in VTCM is exact, so
 * everything downstream of the matmul stays bit-identical to the 4-bit path.
 *
 * That makes the accuracy question separable from the format question, which
 * is the point of this header. Restricting the codes and writing them back as
 * ordinary QS4CX_WH produces a model that runs on today's kernel and reports
 * today's perplexity for tomorrow's weights -- the gate for the whole 2-bit
 * effort, at no kernel cost.
 *
 * Two scopes, and they are not equals:
 *
 *   per_column = false  one palette per k-group, shared by all N columns.
 *                       The per-column magnitude is already carried by
 *                       w_scale[n], so the palette only has to describe the
 *                       shape of the normalized distribution -- the same
 *                       division of labour NF4 uses. Constant across a
 *                       128-byte WH vector, which is what lets the DSP
 *                       expansion be one vlut32 per output vector.
 *   per_column = true   one palette per (column, k-group). Better fit, and
 *                       the reference for how much the shared palette gives
 *                       up -- but the expansion then needs a 128-entry table
 *                       and a code-per-byte spread, about six times the work,
 *                       which costs more DMA than 2-bit saves. A control, not
 *                       a candidate.
 */

#ifndef __NNTRAINER_HTP_WH_PALETTE_H__
#define __NNTRAINER_HTP_WH_PALETTE_H__

#include "htp_wh_layout.h"

#include <cstddef>
#include <cstdint>

namespace nntrainer {

/** @brief Levels a 2-bit code can name. */
constexpr uint32_t WH_PALETTE_LEVELS = 4u;
/** @brief Distinct int4 codes, [-8, 7]. */
constexpr uint32_t WH_PALETTE_CODES = 16u;

/** @brief k-groups a K-deep tensor splits into at this group size. */
inline uint32_t whPaletteGroups(uint32_t K, uint32_t group_k) {
  const uint32_t g = (group_k == 0u || group_k > K) ? K : group_k;
  return (K + g - 1u) / g;
}

/** @brief Palettes a call to whPaletteQuantize writes (int8 entries). */
inline size_t whPaletteEntries(uint32_t K, uint32_t N, uint32_t group_k,
                               bool per_column) {
  return (size_t)whPaletteGroups(K, group_k) * (per_column ? N : 1u) *
         WH_PALETTE_LEVELS;
}

namespace detail {

/**
 * @brief The four codes minimizing weighted squared error over a histogram
 *
 * Exact, not a k-means approximation: the values are int4 codes, so there are
 * only sixteen of them, and with the levels sorted the nearest-level
 * assignment is a partition into four contiguous runs. A DP over those runs
 * (and a brute-force representative inside each) is a few hundred operations.
 *
 * @param hist  counts indexed by code + 8
 * @param out   four codes, ascending
 */
inline void whPaletteFitHist(const uint64_t hist[WH_PALETTE_CODES],
                             int8_t out[WH_PALETTE_LEVELS]) {
  /* seg_cost[i][j]: best error representing codes i..j by one code, and the
     code that does it. */
  double seg_cost[WH_PALETTE_CODES][WH_PALETTE_CODES];
  int seg_rep[WH_PALETTE_CODES][WH_PALETTE_CODES];
  for (uint32_t i = 0; i < WH_PALETTE_CODES; ++i) {
    for (uint32_t j = i; j < WH_PALETTE_CODES; ++j) {
      double best = -1.0;
      int best_r = (int)i;
      for (uint32_t r = i; r <= j; ++r) {
        double c = 0.0;
        for (uint32_t v = i; v <= j; ++v) {
          const double d = (double)v - (double)r;
          c += (double)hist[v] * d * d;
        }
        if (best < 0.0 || c < best) {
          best = c;
          best_r = (int)r;
        }
      }
      seg_cost[i][j] = best;
      seg_rep[i][j] = best_r;
    }
  }

  /* dp[m][j]: best error covering codes 0..j with m+1 segments. */
  double dp[WH_PALETTE_LEVELS][WH_PALETTE_CODES];
  uint32_t cut[WH_PALETTE_LEVELS][WH_PALETTE_CODES];
  for (uint32_t j = 0; j < WH_PALETTE_CODES; ++j) {
    dp[0][j] = seg_cost[0][j];
    cut[0][j] = 0u;
  }
  for (uint32_t m = 1; m < WH_PALETTE_LEVELS; ++m) {
    for (uint32_t j = 0; j < WH_PALETTE_CODES; ++j) {
      double best = -1.0;
      uint32_t best_i = (j >= m) ? j : m;
      for (uint32_t i = m; i <= j; ++i) {
        const double c = dp[m - 1][i - 1] + seg_cost[i][j];
        if (best < 0.0 || c < best) {
          best = c;
          best_i = i;
        }
      }
      /* j < m: fewer codes than segments left. Reuse the m-1 answer; the
         spare levels land on duplicates, which the nearest-level apply
         below handles without caring. */
      dp[m][j] = (j >= m) ? best : dp[m - 1][j];
      cut[m][j] = (j >= m) ? best_i : j;
    }
  }

  uint32_t j = WH_PALETTE_CODES - 1u;
  for (int m = (int)WH_PALETTE_LEVELS - 1; m >= 0; --m) {
    const uint32_t i = cut[m][j];
    out[m] = (int8_t)((int)seg_rep[i][j] - 8);
    if (i == 0u) {
      /* Segments left over on an under-populated histogram: pad with the
         same code so the palette always has four entries. */
      for (int mm = m - 1; mm >= 0; --mm) {
        out[mm] = out[m];
      }
      break;
    }
    j = i - 1u;
  }
}

/** @brief Nearest palette entry, ties to the lower code -- deterministic,
 *         because the host reference and the DSP table must agree exactly. */
inline int8_t whPaletteNearest(int8_t q, const int8_t pal[WH_PALETTE_LEVELS]) {
  int8_t best = pal[0];
  int best_d = (int)q - (int)pal[0];
  best_d = best_d < 0 ? -best_d : best_d;
  for (uint32_t l = 1; l < WH_PALETTE_LEVELS; ++l) {
    int d = (int)q - (int)pal[l];
    d = d < 0 ? -d : d;
    if (d < best_d) {
      best_d = d;
      best = pal[l];
    }
  }
  return best;
}

} // namespace detail

/**
 * @brief Fit palettes and snap the codes onto them, in place
 *
 * @param rm        K*N int8 codes in [-8, 7], k-major -- whPack's input
 * @param K         input width
 * @param N         output channels
 * @param group_k   k-values a palette covers; 0 or >= K means one group
 * @param per_column one palette per (column, group) instead of per group
 * @param pal       whPaletteEntries() int8 entries, ascending within each
 *                  palette; group-major, then column when per_column
 */
inline void whPaletteQuantize(int8_t *rm, uint32_t K, uint32_t N,
                              uint32_t group_k, bool per_column, int8_t *pal) {
  const uint32_t g = (group_k == 0u || group_k > K) ? K : group_k;
  const uint32_t groups = whPaletteGroups(K, group_k);

  for (uint32_t gi = 0; gi < groups; ++gi) {
    const uint32_t k0 = gi * g;
    const uint32_t k1 = (k0 + g < K) ? (k0 + g) : K;
    const uint32_t palettes = per_column ? N : 1u;

    for (uint32_t p = 0; p < palettes; ++p) {
      uint64_t hist[WH_PALETTE_CODES] = {0};
      const uint32_t n0 = per_column ? p : 0u;
      const uint32_t n1 = per_column ? (p + 1u) : N;
      for (uint32_t k = k0; k < k1; ++k) {
        const int8_t *row = rm + (size_t)k * N;
        for (uint32_t n = n0; n < n1; ++n) {
          ++hist[(uint32_t)((int)row[n] + 8)];
        }
      }

      int8_t *entry = pal + ((size_t)gi * palettes + p) * WH_PALETTE_LEVELS;
      detail::whPaletteFitHist(hist, entry);

      for (uint32_t k = k0; k < k1; ++k) {
        int8_t *row = rm + (size_t)k * N;
        for (uint32_t n = n0; n < n1; ++n) {
          row[n] = detail::whPaletteNearest(row[n], entry);
        }
      }
    }
  }
}

/* ---------------------------------------------------------------------- *
 * The 2-bit storage format itself
 * ---------------------------------------------------------------------- */

/** @brief Bytes per tile: WH_TILE*WH_TILE codes, four per byte. */
constexpr uint32_t WH_TILE2_BYTES = WH_TILE * WH_TILE / 4u;

/** @brief Bytes a K x N weight occupies as 2-bit codes in WH order. */
inline size_t whBytes2(uint32_t K, uint32_t N) {
  return static_cast<size_t>(K / WH_TILE) * (N / WH_TILE) * WH_TILE2_BYTES;
}

/**
 * @brief The 16-entry expansion table: a PAIR of codes -> one int4 byte
 *
 * This is the whole reason the format looks the way it does. A WH byte holds
 * two nibbles, and the two codes that produce them are adjacent in the code
 * stream, so a table indexed by the pair turns expansion into one lookup per
 * output byte. On HVX that is a single vlut32 over a 16-entry table -- three
 * ops an output vector, against about seventeen for a per-column palette
 * that needs the codes spread one to a byte and a 128-entry table. The
 * cheaper shape is what fits inside the DMA it is meant to save (doc 54 3).
 *
 * @param pal four int4 codes, ascending
 * @param out 16 bytes, out[(hi << 2) | lo]
 */
inline void whPaletteTable(const int8_t pal[WH_PALETTE_LEVELS], uint8_t *out) {
  for (uint32_t hi = 0; hi < WH_PALETTE_LEVELS; ++hi) {
    for (uint32_t lo = 0; lo < WH_PALETTE_LEVELS; ++lo) {
      out[(hi << 2) | lo] =
        static_cast<uint8_t>((pal[lo] & 0x0F) | ((pal[hi] & 0x0F) << 4));
    }
  }
}

/**
 * @brief Packs palette-restricted int4 codes as 2 bits each, in WH order
 *
 * Code i of a tile is whPack's nibble slot i, four to a byte, low bits
 * first: slot s lives in byte s/4 at bit 2*(s%4). That is what makes source
 * byte j expand to output bytes 2j and 2j+1 with nothing to shuffle.
 *
 * @param rm  K rows of N int8 codes, row stride N -- ALREADY restricted by
 *            whPaletteQuantize with this palette. Anything else is snapped
 *            to the nearest entry, which silently changes the weight;
 *            whPack2/whExpand2 round-tripping against whPack is the check
 *            that catches it.
 * @param out whBytes2(K, N) bytes, zeroed by this function
 */
inline void whPack2(const int8_t *rm, uint32_t K, uint32_t N,
                    const int8_t pal[WH_PALETTE_LEVELS], uint8_t *out) {
  const uint32_t k_tiles = K / WH_TILE, n_tiles = N / WH_TILE;
  for (size_t i = 0, n = whBytes2(K, N); i < n; ++i) {
    out[i] = 0;
  }
  for (uint32_t kt = 0; kt < k_tiles; ++kt) {
    for (uint32_t nt = 0; nt < n_tiles; ++nt) {
      uint8_t *tile = out + ((size_t)kt * n_tiles + nt) * WH_TILE2_BYTES;
      for (uint32_t r = 0; r < WH_TILE; ++r) {
        const int8_t *row = rm + (size_t)(kt * WH_TILE + r) * N + nt * WH_TILE;
        for (uint32_t c = 0; c < WH_TILE; ++c) {
          const uint32_t sl = whSlot(r, c);
          const int8_t q = detail::whPaletteNearest(row[c], pal);
          uint32_t code = 0;
          for (uint32_t l = 0; l < WH_PALETTE_LEVELS; ++l) {
            if (pal[l] == q) {
              code = l;
              break;
            }
          }
          tile[sl / 4u] |= static_cast<uint8_t>(code << (2u * (sl % 4u)));
        }
      }
    }
  }
}

/**
 * @brief Expands 2-bit codes back to WH int4 bytes -- the host reference
 *
 * The DSP does this in VTCM with the table above; this is the scalar twin
 * that says what those bytes have to be. Its output must equal whPack of the
 * same restricted codes, byte for byte, or the 2-bit model and the 4-bit
 * model it was measured as are not the same model.
 *
 * @param out whBytes(K, N) bytes
 */
inline void whExpand2(const uint8_t *in, uint32_t K, uint32_t N,
                      const int8_t pal[WH_PALETTE_LEVELS], uint8_t *out) {
  uint8_t table[WH_PALETTE_LEVELS * WH_PALETTE_LEVELS];
  whPaletteTable(pal, table);
  const size_t tiles = static_cast<size_t>(K / WH_TILE) * (N / WH_TILE);
  for (size_t t = 0; t < tiles; ++t) {
    const uint8_t *src = in + t * WH_TILE2_BYTES;
    uint8_t *dst = out + t * WH_TILE_BYTES;
    for (uint32_t j = 0; j < WH_TILE2_BYTES; ++j) {
      const uint8_t s = src[j];
      dst[2u * j] = table[s & 0x0Fu];
      dst[2u * j + 1u] = table[(s >> 4) & 0x0Fu];
    }
  }
}

} // namespace nntrainer

#endif /* __NNTRAINER_HTP_WH_PALETTE_H__ */
