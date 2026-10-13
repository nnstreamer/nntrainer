// SPDX-License-Identifier: Apache-2.0
#ifndef NNTR_HTP_ROPE_PREFILL_SHAPE_H
#define NNTR_HTP_ROPE_PREFILL_SHAPE_H

#include <stddef.h>
#include <stdint.h>

/** Compare ranges without overflowing an endpoint address. */
static inline int nntr_rope_ranges_overlap(const void *a, size_t an,
                                           const void *b, size_t bn) {
  const uintptr_t aa = (uintptr_t)a, bb = (uintptr_t)b;
  return aa <= bb ? bb - aa < an : aa - bb < bn;
}

/** q is in place. K/V and the phase table are read only; all other overlapping
 * ranges are unsupported so workers can own disjoint rows without scratch.
 * This guard is shared by the host, DSP entry and host transaction checks. */
static inline int nntr_rope_prefill_valid(uint32_t rows, uint32_t qw,
                                          uint32_t kw, uint32_t hd,
                                          const float *cs, float *q,
                                          const float *k, const float *v,
                                          uint16_t *ko, uint16_t *vo) {
  const uint64_t nq = (uint64_t)rows * qw, nk = (uint64_t)rows * kw;
  const uint64_t nc = (uint64_t)rows * 2u * hd;
  if (rows < 2u || hd != 64u || !qw || !kw || qw % hd || kw % hd ||
      nq > 0x7fffffffu / 4u || nk > 0x7fffffffu / 4u || nc > 0x7fffffffu / 4u ||
      !cs || !q || !k || !v || !ko || !vo)
    return 0;
  const size_t qb = (size_t)nq * 4u, kb = (size_t)nk * 4u;
  const size_t hb = (size_t)nk * 2u, cb = (size_t)nc * 4u;
  if (nntr_rope_ranges_overlap(q, qb, k, kb) ||
      nntr_rope_ranges_overlap(q, qb, v, kb) ||
      nntr_rope_ranges_overlap(q, qb, cs, cb) ||
      nntr_rope_ranges_overlap(q, qb, ko, hb) ||
      nntr_rope_ranges_overlap(q, qb, vo, hb) ||
      nntr_rope_ranges_overlap(ko, hb, k, kb) ||
      nntr_rope_ranges_overlap(ko, hb, v, kb) ||
      nntr_rope_ranges_overlap(ko, hb, cs, cb) ||
      nntr_rope_ranges_overlap(ko, hb, vo, hb) ||
      nntr_rope_ranges_overlap(vo, hb, k, kb) ||
      nntr_rope_ranges_overlap(vo, hb, v, kb) ||
      nntr_rope_ranges_overlap(vo, hb, cs, cb))
    return 0;
  return 1;
}

#endif
