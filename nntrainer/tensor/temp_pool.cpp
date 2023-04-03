// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   temp_pool.cpp
 * @date   03 April 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Temporal data pool class inherited from memory pool
 *
 */

#include "temp_pool.h"

#include <limits>
#include <malloc.h>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <vector>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <profiler.h>

namespace {

const unsigned int TEMPPOOL_INIT_ID = 0x10000;

} // namespace

namespace nntrainer {

TempPool::TempPool() : allocated(false), next_id(TEMPPOOL_INIT_ID) {}

TempPool::~TempPool() {
  try {
    deallocate();
  } catch (...) {
    ml_loge("Failed deallocate");
  }
}

void TempPool::allocate() { allocated = true; }

void TempPool::deallocate() {
  reclaimAll();
  allocated = false;
}

unsigned int TempPool::requestMemory(size_t bytes, unsigned int start_time,
                                     unsigned int end_time,
                                     std::vector<unsigned int> exec_order,
                                     TensorLifespan lifespan, bool is_wgrad) {
  next_id++;
  data_size[next_id.load()] = bytes;

  for (auto &order : exec_order) {
    exec_orders[order].push_back(next_id.load());
  }

  return next_id.load();
}

double TempPool::planLayout(const MemoryPlanner &planner) {

  // @todo: plan layout before allocating memory
  return 1.0;
}

static unsigned int alloced = 0;
static unsigned int max_alloced = 0;
std::shared_ptr<MemoryData> TempPool::getMemory(unsigned int id) {
  data[id] = std::make_shared<MemoryData>(
    id,
    [&](unsigned int id) {
      data[id]->setAddr((float *)(new char[data_size[id]]()));
      data[id]->setValid(true);

      alloced += data_size[id];
      max_alloced = std::max(max_alloced, alloced);
    },
    [&](unsigned int id) {
      alloced -= data_size[id];
      delete[] data[id]->getAddr();
      data[id]->setValid(false);
#ifndef __ANDROID__
      malloc_trim(0);
#endif
    });

  return data[id];
}

size_t TempPool::minMemoryRequirement() {
  size_t total = 0;

  for (auto &[id, dsize] : data_size)
    total += dsize;

  return total;
}

size_t TempPool::size() { return minMemoryRequirement(); }

void TempPool::clear() {
  next_id.store(TEMPPOOL_INIT_ID);
  data.clear();
  data_size.clear();
  exec_orders.clear();
}

void TempPool::reclaim(unsigned int exec_order) {
  for (auto &id : exec_orders[exec_order]) {
    data[id]->invalidate();
  }
}

void TempPool::reclaimAll() {
  ml_logd("temp pool: max alloc size: %d", max_alloced);
  for (auto &[id, d] : data)
    d->invalidate();
}

} // namespace nntrainer
