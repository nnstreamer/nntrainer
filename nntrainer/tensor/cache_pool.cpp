// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_pool.cpp
 * @date   01 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache pool class inherited from memory pool
 *
 */

#include "memory_pool.h"
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <cache_pool.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>

namespace nntrainer {

void CacheElem::swapIn() {
  std::lock_guard<std::mutex> lock(device_mutex);
  void *buf = device->getBuffer(offset, length);
  mem_data->setAddr((float *)buf);
  mem_data->setValid(true);
  active = true;

  std::string msg("CacheElem(");
  msg += device->getDevicePath() + ") #" + std::to_string(id);
  PROFILE_MEM_ALLOC(buf, length, msg);
}

void CacheElem::swapOut() {
  std::lock_guard<std::mutex> lock(device_mutex);
  void *buf = (void *)mem_data->getAddr();
  device->putBuffer(buf);
  mem_data->setAddr(nullptr);
  mem_data->setValid(false);
  active = false;

  PROFILE_MEM_DEALLOC(buf);
}

CachePool::CachePool(const std::string &name) :
  swap_device(std::make_shared<SwapDevice>(name + std::to_string(getpid()))) {}

CachePool::CachePool(const std::string &path, const std::string &name) {
  if (path.empty())
    swap_device = std::make_shared<SwapDevice>(name + std::to_string(getpid()));
  else
    swap_device =
      std::make_shared<SwapDevice>(path, name + std::to_string(getpid()));
}

CachePool::~CachePool() {
  try {
    deallocate();
  } catch (...) {
    ml_loge("Failed deallocate");
  }
}

void CachePool::allocate() {
  NNTR_THROW_IF(swap_device->isOperating(), std::runtime_error)
    << "Cache pool is already allocated";

  size_t pool_size = size();

  NNTR_THROW_IF(pool_size == 0, std::runtime_error)
    << "Allocating memory pool with size 0";

  NNTR_THROW_IF(swap_device->isOperating(), std::runtime_error)
    << "Cache pool is already allocated";

  swap_device->start(pool_size);
}

void CachePool::deallocate() {
  if (!swap_device->isOperating())
    return;

  for (auto &[id, elem] : elems)
    invalidate(id);
  swap_device->finish();
}

void CachePool::validate(unsigned int id) {
  if (!elems[id]->isActive()) {
    elems[id]->swapIn();
    actives.push_back(elems[id]);
  }
}

void CachePool::invalidate(unsigned int id) {
  if (elems[id]->isActive())
    elems[id]->swapOut();
}

std::shared_ptr<MemoryData<float>> CachePool::getMemory(unsigned int id) {
  NNTR_THROW_IF(!swap_device->isOperating(), std::invalid_argument)
    << "Allocate memory before allocation";

  int offset = getMemoryOffset().at(id - 1);
  size_t len = getMemorySize().at(id - 1);
  auto exe_order = getMemoryExecOrder().at(id - 1);
  auto mem_data = std::make_shared<MemoryData<float>>(
    id, std::bind(&CachePool::validate, this, std::placeholders::_1),
    std::bind(&CachePool::invalidate, this, std::placeholders::_1));
  auto elem = std::make_shared<CacheElem>(swap_device, id, offset, len,
                                          mem_data, exe_order);

  elems[id] = elem;

  std::string ords;
  for (auto &o : exe_order) {
    ords.append(std::to_string(o));
    ords.append(" ");
  }
  ml_logd("[%d] exe_order(%s), offset: 0x%x, len: %zu", id, ords.c_str(),
          offset, len);

  return mem_data;
}

void CachePool::flush() {
  for (auto elem : actives)
    invalidate(elem->getId());

  actives.clear();
}

void CachePool::flushExcept(unsigned int order) {
  actives.remove_if([&, order](auto elem) -> bool {
    auto exe_order = elem->getExeOrder();
    auto found = std::find(exe_order.begin(), exe_order.end(), order);
    if (found == exe_order.end()) {
      invalidate(elem->getId());
      return true;
    }
    return false;
  });
}

void CachePool::clear() {
	flush();
	deallocate();
	MemoryPool::clear();
}

bool CachePool::isAllocated() const { return swap_device->isOperating(); }

} // namespace nntrainer
