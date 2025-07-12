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
#include <string>

#include "tensor/shared_memory.h"

namespace nntrainer {

/**
 * @brief MemAllocator, Memory allocator class
 */
class MemAllocator {
public:
  MemAllocator() = default;
  virtual ~MemAllocator() = default;
  virtual void alloc(void **ptr, size_t size, size_t alignment);
  virtual void free(void *ptr);
  virtual std::string getName() { return "cpu"; };
};

using HostSystemAllocator = MemAllocator;

class DeviceMemoryAllocator
{
public:
  using AccessKind = nntrainer::shared_memory::DevMemAccessKind;
  using AccessCapDesc = shared_memory::AccessCapDesc;
  using AccessCapDescription = const shared_memory::AccessCapDescription &;

protected:
  DeviceMemoryAllocator(DeviceContext &ctx,
                        shared_memory::AccessCapDesc descriptor,
                        HostSystemAllocator *system_allocator);

public:
  virtual ~DeviceMemoryAllocator() = 0;

  /*
   * @brief allocate host memory on behalf an device
   */
  virtual bool allocHost(void **, size_t size, size_t alignment, AccessKind req_access) = 0;

  /*
   * @brief allocate device memory on behalf an device
   */
  virtual bool allocDevice(void **, size_t size, size_t alignment, AccessKind req_access) = 0;

  /*
   * @brief allocate device memory on behalf an device
   */
  virtual bool allocShared(void **, size_t size, size_t alignment, AccessKind req_access) = 0;

  bool canFullfillHostAlloc(size_t size, size_t alignment, AccessKind req_access) const {
    auto *self = const_cast<DeviceMemoryAllocator*>(this);
    return self->allocHost(nullptr, size, alignment, req_access);
  }

  bool canFullfillDeviceAlloc(size_t size, size_t alignment, AccessKind req_access) {
    auto *self = const_cast<DeviceMemoryAllocator*>(this);
    return self->allocDevice(nullptr, size, alignment, req_access);
  }

  bool canFullfillSharedAlloc(size_t size, size_t alignment, AccessKind req_access) {
    auto *self = const_cast<DeviceMemoryAllocator*>(this);
    return self->allocShared(nullptr, size, alignment, req_access);
  }

  const DeviceContext& ownerContext() const;
  const DeviceInfo& ownerDevInfo() const;

  virtual auto getName() -> std::string;

  auto host_capabilities() const noexcept -> AccessCapDescription;
  auto sharing_capabilities() const noexcept -> AccessCapDescription;
  auto private_capabilities() const noexcept -> AccessKind;

private:
  AccessCapDesc _possible_mem_access;
  HostSystemAllocator *_system_allocator;
  DeviceContext *_owner_context;
};

} // namespace nntrainer

#endif
