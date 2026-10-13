// SPDX-License-Identifier: Apache-2.0
/* Br256 Q8 VTCM bounds and per-row attention geometry checks.
 * No Hexagon SDK or device is required.
 */
#include "hexkl_attn_q_plan.h"

#include <assert.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

static uint64_t region_cases, row_cases, block_cases, shape_cases, reject_cases;

static uint64_t round_tile(uint64_t value) {
  return (value + 2047u) / 2048u * 2048u;
}

static void region(uint32_t start, uint64_t bytes, uint64_t *cursor,
                   uint32_t total) {
  assert(start % 2048u == 0u);
  assert(start == *cursor);
  assert(bytes % 2048u == 0u);
  *cursor += bytes;
  assert(*cursor <= total);
  ++region_cases;
}

static uint32_t check_regions(const hexkl_attn_f16_shape *s,
                              const hexkl_attn_f16_tiling *t, uint32_t chunk) {
  hexkl_attn_q_layout L, old;
  assert(hexkl_attn_q_plan_wire_br256(s, t, UINT32_MAX, 1024u, chunk, &L) ==
         HEXKL_ATTN_OK);
  assert(hexkl_attn_q_plan(s, t, UINT32_MAX, 1024u, chunk, &old) ==
         HEXKL_ATTN_OK);
  assert(old.qx_f32 == L.total && L.qx_f32 == L.total);
  assert(old.total - L.total == chunk * t->g_br * s->head_dim * 4u);
  old.total = L.total;
  assert(memcmp(&old, &L, sizeof L) == 0);

  const uint64_t rt = t->g_br / 64u, ct = t->bc / 32u, dt = s->head_dim / 32u;
  uint64_t cursor = 0;
  region(L.q_ah, chunk * rt * dt * 2048u, &cursor, L.total);
  for (unsigned b = 0; b < 2u; ++b)
    region(L.kt_wh[b], round_tile(ct * dt * 1024u), &cursor, L.total);
  for (unsigned b = 0; b < 2u; ++b)
    region(L.v_wh[b], round_tile(ct * dt * 1024u), &cursor, L.total);
  region(L.acc, 8192u, &cursor, L.total);
  for (unsigned b = 0; b < 2u; ++b)
    region(L.s_hf[b], 2u * rt * ct * 2048u, &cursor, L.total);
  region(L.p_ah, rt * ct * 2048u, &cursor, L.total);
  region(L.o_f32, round_tile((uint64_t)t->g_br * s->head_dim * 4u), &cursor,
         L.total);
  assert(cursor == L.total);
  const uint64_t bank = rt * ct * 8192u;
  const uint64_t need64 = cursor + 2u * bank;
  assert(need64 <= UINT32_MAX);
  const uint32_t need = (uint32_t)need64;
  region(L.total, bank, &cursor, need);
  region(L.total + (uint32_t)bank, bank, &cursor, need);
  assert(cursor == need);

  /* Every HMX operand/readout and vector row stays in its own region. */
  for (uint32_t qi = 0; qi < chunk; ++qi)
    for (uint32_t i = 0; i < rt; ++i)
      for (uint32_t d = 0; d < dt; ++d) {
        uint64_t off = L.q_ah + ((qi * rt + i) * dt + d) * 2048u;
        assert(off % 2048u == 0 && off + 2048u <= L.kt_wh[0]);
      }
  for (uint32_t i = 0; i < rt; ++i)
    for (uint32_t col = 0; col < ct; ++col) {
      uint64_t off = L.total + (i * ct + col) * 8192u;
      assert(off + 8192u <= L.total + bank);
      assert(off + bank + 8192u <= need);
    }
  for (uint32_t r = 0; r < t->g_br; r += 4u) {
    const uint64_t i32 = r / 32u, v = (r % 32u) / 2u;
    for (uint32_t col = 0; col < ct; ++col) {
      uint64_t soff = (i32 * ct + col) * 2048u + v * 128u;
      assert(soff + 256u <= 2u * rt * ct * 2048u);
      uint64_t poff = ((r / 64u) * ct + col) * 2048u + (r % 64u) * 32u;
      assert(poff + 128u <= rt * ct * 2048u);
    }
    assert((uint64_t)(r + 4u) * s->head_dim * 4u <= L.total - L.o_f32);
  }
  /* map_meta's only alignment padding is included by its caller. */
  const uint64_t meta_bytes =
    hexkl_attn_q_meta_bytes(t->g_br, L.n_dot_tiles, chunk) + 128u;
  for (uint32_t alignment = 0; alignment < 128u; ++alignment) {
    const uint64_t scalar_end =
      alignment + 8u * (uint64_t)chunk * t->g_br + 12u * (uint64_t)t->g_br;
    const uint64_t vectors = (scalar_end + 127u) / 128u * 128u;
    assert(vectors >= scalar_end && vectors % 128u == 0u);
    assert(vectors + 128u * t->g_br <= alignment + meta_bytes);
  }
  assert(hexkl_attn_q_plan_wire_br256(s, t, need - 1u, 1024u, chunk, &L) ==
         HEXKL_ATTN_ENOMEM);
  assert(hexkl_attn_q_plan_wire_br256(s, t, need, 1024u, chunk, &L) ==
         HEXKL_ATTN_OK);
  assert(hexkl_attn_q_plan_wire_br256(s, t, need + 1u, 1024u, chunk, &L) ==
         HEXKL_ATTN_OK);
  return need;
}

static void check_geometry(void) {
  for (uint32_t n = 256u; n <= 65535u; n += 256u) {
    hexkl_attn_f16_shape s = {n, 0, n, 32, 8, 64, 0, 0};
    hexkl_attn_f16_tiling old = {128, 256, 0, 0}, wide = {256, 256, 0, 0};
    assert(hexkl_attn_q_tiling_init(&s, &old) == HEXKL_ATTN_OK);
    assert(hexkl_attn_q_tiling_init(&s, &wide) == HEXKL_ATTN_OK);
    const uint32_t chunk = hexkl_attn_q_chunk_count(&s, &wide);
    uint32_t need = 0;
    for (uint32_t ch = 1; ch <= chunk; ++ch)
      need = check_regions(&s, &wide, ch);
    for (uint32_t qb = 0; qb < n / 128u; ++qb) {
      uint32_t lo0, hi0, lo1, hi1;
      hexkl_attn_f16_kv_block_range(&s, &old, qb, &lo0, &hi0);
      hexkl_attn_f16_kv_block_range(&s, &wide, qb / 2u, &lo1, &hi1);
      assert(lo0 == lo1 && hi0 == hi1);
      for (uint32_t kb = lo0; kb < hi0; ++kb) {
        assert(hexkl_attn_f16_kv_block_unmasked(&s, &old, qb, kb) ==
               hexkl_attn_f16_kv_block_unmasked(&s, &wide, qb / 2u, kb));
        ++block_cases;
      }
      for (uint32_t r = 0; r < old.g_br; ++r) {
        uint32_t q0, g0, q1, g1;
        assert(hexkl_attn_f16_tile_row_to_qg(&s, &old, qb, r, &q0, &g0));
        assert(hexkl_attn_f16_tile_row_to_qg(
          &s, &wide, qb / 2u, (qb % 2u) * old.g_br + r, &q1, &g1));
        assert(q0 == q1 && g0 == g1);
        ++row_cases;
      }
    }
    if (n == 768u || n == 1024u || n == 2048u || n == 65280u)
      printf("n_q=%u chunk=%u staged_bytes=%u reclaimed_bytes=%u\n", n, chunk,
             need, chunk * 1024u * 64u * 4u);
  }
}

/* Every possible public cache length either satisfies the new shape or
 * remains outside it. Prefixes, tails and windows must never qualify.
 */
static void check_scope(void) {
  const hexkl_attn_f16_tiling t = {256, 256, 4, 1024};
  for (uint32_t n = 1u; n <= 65535u; ++n) {
    hexkl_attn_f16_shape s = {n, 0, n, 32, 8, 64, 0, 0};
    assert(hexkl_attn_q_wire_br256_shape(&s, &t) == (n % 256u == 0u));
    s.cache_from = 1;
    s.cache_to = n + 1u;
    assert(!hexkl_attn_q_wire_br256_shape(&s, &t));
    s.cache_from = 0;
    s.cache_to = n;
    s.window = 255;
    assert(!hexkl_attn_q_wire_br256_shape(&s, &t));
    shape_cases += 3u;
  }
}

static void reject_shape(hexkl_attn_f16_shape s, hexkl_attn_f16_tiling t) {
  hexkl_attn_q_layout L;
  assert(!hexkl_attn_q_wire_br256_shape(&s, &t));
  assert(hexkl_attn_q_plan_wire_br256(&s, &t, UINT32_MAX, 1024, 1, &L) ==
         HEXKL_ATTN_EBADPARM);
  ++reject_cases;
}

static void check_rejections(void) {
  hexkl_attn_f16_shape s = {1024, 0, 1024, 32, 8, 64, 0, 0}, x;
  hexkl_attn_f16_tiling t = {256, 256, 4, 1024}, y;
  hexkl_attn_q_layout L;
  const uint32_t bad[] = {0, 1, 255, 257, 65535, 65536, UINT32_MAX};
  for (unsigned i = 0; i < sizeof bad / sizeof *bad; ++i) {
    x = s;
    x.n_q = x.cache_to = bad[i];
    reject_shape(x, t);
    y = t;
    y.g_br = bad[i];
    reject_shape(s, y);
  }
  x = s;
  x.cache_from = 1;
  reject_shape(x, t);
  x = s;
  x.cache_to++;
  reject_shape(x, t);
  x = s;
  x.n_head_q = 16;
  reject_shape(x, t);
  x = s;
  x.n_head_kv = 4;
  reject_shape(x, t);
  x = s;
  x.head_dim = 128;
  reject_shape(x, t);
  x = s;
  x.window = 1;
  reject_shape(x, t);
  y = t;
  y.br = 128;
  reject_shape(s, y);
  y = t;
  y.bc = 512;
  reject_shape(s, y);
  y = t;
  y.g = 3;
  reject_shape(s, y);
  assert(hexkl_attn_q_plan_wire_br256(NULL, &t, UINT32_MAX, 1024, 1, &L) ==
         HEXKL_ATTN_EBADPARM);
  assert(hexkl_attn_q_plan_wire_br256(&s, NULL, UINT32_MAX, 1024, 1, &L) ==
         HEXKL_ATTN_EBADPARM);
  assert(hexkl_attn_q_plan_wire_br256(&s, &t, UINT32_MAX, 1024, 1, NULL) ==
         HEXKL_ATTN_EBADPARM);
  assert(hexkl_attn_q_plan_wire_br256(&s, &t, UINT32_MAX, 512, 1, &L) ==
         HEXKL_ATTN_EBADPARM);
  assert(hexkl_attn_q_plan_wire_br256(&s, &t, UINT32_MAX, 1024, 0, &L) ==
         HEXKL_ATTN_EBADPARM);
  assert(hexkl_attn_q_plan_wire_br256(&s, &t, UINT32_MAX, 1024, 5, &L) ==
         HEXKL_ATTN_EBADPARM);
  reject_cases += 6;
}

int main(void) {
  check_geometry();
  check_scope();
  check_rejections();
  printf("ATTN_Q_WIRE_BR256_PASS regions=%" PRIu64 " rows=%" PRIu64
         " masks=%" PRIu64 " shapes=%" PRIu64 " rejected=%" PRIu64 "\n",
         region_cases, row_cases, block_cases, shape_cases, reject_cases);
  return 0;
}
