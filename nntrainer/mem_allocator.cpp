// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    mem_allocator.cpp
 * @date    13 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is memory allocator for memory pool
 *
 */

#include <limits>
#include <mem_allocator.h>
#include <numeric>
#include <vector>
#include <nntrainer_log.h>

namespace nntrainer {

void MemAllocator::alloc(void **ptr, size_t size, size_t alignment) {
  if (size == 0)
    ml_loge("cannot allocate size = 0");

  *ptr = calloc(size, 1);
};

void MemAllocator::free(void *ptr) { free(ptr); };
} // namespace nntrainer
