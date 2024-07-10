// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   hgemm_util.h
 * @date   01 April 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is for util functions for half-precision GEMM
 */

#include <assert.h>
#include <stdlib.h>

/**
 * @brief aligned dynamic allocation function
 *
 * @param sz amount of data to allocate
 * @return __fp16* addr of allocated memory
 */
static inline __fp16 *alignedMalloc(int sz) {
  void *addr = 0;
  int iRet = posix_memalign(&addr, 64, sz * sizeof(__fp16));
  assert(0 == iRet);
  return (__fp16 *)addr;
}
