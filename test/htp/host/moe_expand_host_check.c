// SPDX-License-Identifier: Apache-2.0
/**
 * @file moe_expand_host_check.c
 * @brief Exhaustive palettes and guarded aligned/unaligned in-place expansion.
 * @author NNTrainer contributors
 * @bug See the repository issue tracker for known issues.
 */

#include "hvx_expand_i2i4.h"
#include "hvx_worker_pool.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
void hvx_worker_pool_run(hvx_worker_pool *p, hvx_worker_pool_func f, void *c,
                         uint32_t u) {
  (void)p;
  (void)u;
  f(1, 0, c);
}
static unsigned seed = 72385u;
static unsigned next(void) { return seed = 1664525u * seed + 1013904223u; }
static int run(const int8_t pal[4], unsigned n, unsigned off,
               unsigned in_place) {
  uint8_t *buf = aligned_alloc(128, 3 * 8192 + 512),
          *original = malloc(n ? n : 1), *expected = malloc(n ? 2 * n : 1),
          table[128];
  memset(buf, 0xA5, 3 * 8192 + 512);
  uint8_t *out = buf + 128 + off;
  uint8_t *src = in_place ? out + n : buf + 2 * 8192 + 256 + off;
  for (unsigned i = 0; i < n; ++i)
    original[i] = (i < 256) ? i : next();
  memcpy(src, original, n);
  hvx_expand_i2i4_table(pal, table);
  for (unsigned g = 0; g < n; g += 128) {
    unsigned group = n - g < 128 ? n - g : 128;
    for (unsigned j = 0; j < group; ++j) {
      unsigned code = original[g + j];
      expected[2 * g + j] =
        (pal[code & 3] & 15) | ((pal[(code >> 2) & 3] & 15) << 4);
      expected[2 * g + group + j] =
        (pal[(code >> 4) & 3] & 15) | ((pal[(code >> 6) & 3] & 15) << 4);
    }
  }
  hvx_expand_i2i4(src, n, table, out);
  int bad = memcmp(out, expected, 2 * n) != 0;
  for (unsigned j = 0; j < 128 + off; ++j)
    bad |= buf[j] != 0xA5;
  for (unsigned j = 0; j < 128; ++j)
    bad |= out[2 * n + j] != 0xA5;
  if (bad)
    printf("EXPAND_FAIL n=%u offset=%u inplace=%u pal=%d,%d,%d,%d\n", n, off,
           in_place, pal[0], pal[1], pal[2], pal[3]);
  free(buf);
  free(original);
  free(expected);
  return bad;
}
int main(void) {
  unsigned cases = 0;
  for (unsigned p = 0; p < 65536; ++p) {
    int8_t pal[4];
    for (unsigned j = 0; j < 4; ++j)
      pal[j] = ((p >> (4 * j)) & 15u) - 8;
    if (run(pal, 256, 0, 1))
      return 1;
    cases++;
  }
  unsigned ns[] = {0, 1, 127, 128, 129, 255, 256, 257, 384, 511, 1536, 8192},
           offsets[] = {0, 1, 4, 127};
  for (unsigned p = 0; p < 32; ++p) {
    int8_t pal[4];
    for (unsigned j = 0; j < 4; ++j)
      pal[j] = (next() % 16) - 8;
    for (unsigned n = 0; n < 12; ++n)
      for (unsigned o = 0; o < 4; ++o)
        for (unsigned a = 0; a < 2; ++a) {
          if (run(pal, ns[n], offsets[o], a))
            return 2;
          cases++;
        }
  }
  printf("EXPAND_SCHEDULE_HOST_PASS cases=%u palettes=65536 "
         "source_code_bytes=256 aligned_unaligned_disjoint_inplace=PASS\n",
         cases);
  return 0;
}
