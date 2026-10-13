// SPDX-License-Identifier: Apache-2.0
/** @file hvx_alignment_host_alloc.h
 * @brief Portable fixture allocation with explicit 128-byte alignment.
 */
#ifndef NNTR_HVX_ALIGNMENT_HOST_ALLOC_H
#define NNTR_HVX_ALIGNMENT_HOST_ALLOC_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Store the allocation pointer with memcpy: the +4-byte fixture offset
 * deliberately need not align that slot to the host pointer width.
 */
static void *hvx_alignment_host_alloc(size_t bytes, uint32_t offset) {
  if (offset > 127u || bytes > SIZE_MAX - 255u - sizeof(void *))
    return NULL;
  void *base = malloc(bytes + 255u + sizeof(void *));
  if (!base)
    return NULL;
  const uintptr_t aligned =
    ((uintptr_t)base + sizeof(void *) + 127u) & ~(uintptr_t)127u;
  void *result = (void *)(aligned + offset);
  memcpy((char *)result - sizeof(base), &base, sizeof(base));
  return result;
}

static void hvx_alignment_host_free(void *p) {
  if (p) {
    void *base;
    memcpy(&base, (char *)p - sizeof(base), sizeof(base));
    free(base);
  }
}

#endif
