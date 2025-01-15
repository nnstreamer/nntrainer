// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    mem_allocator.h
 * @date    13 Jan 2025
 * @see     https://github.com/nnstreamer/nntrainer
 * @author  Jijoong Moon <jijoong.moon@samsung.com>
 * @bug     No known bugs except for NYI items
 * @brief   This is memory allocator for memory pool
 *
 */
#ifndef __MEM_ALLOCATOR_H__
#define __MEM_ALLOCATOR_H__

#include <cstddef>
#include <memory>

namespace nntrainer {

class MemAllocator {
public:
  MemAllocator() = default;
  virtual ~MemAllocator() = default;
  virtual void alloc(void **ptr, size_t size, size_t alignment);
  virtual void free(void *ptr);
  /* virtual std::string getName() { return "cpu"; }; */
};
} // namespace nntrainer

#endif
