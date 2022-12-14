// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_elem.cpp
 * @date   28 Nov 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache elem class
 *
 */

#include "cache_elem.h"

#include <stdexcept>
#include <vector>

#include <profiler.h>

namespace nntrainer {

namespace {

std::map<CachePolicy, std::string> policyToStr = {
  {WRITE_BACK, "WRITE_BACK"},           {NO_WRITE_BACK, "NO_WRITE_BACK"},
  {READ_CONSIST, "READ_CONSIST"},       {NO_READ_CONSIST, "NO_READ_CONSIST"},
  {ALWAYS_SYNCED, "ALWAYS_SYNCED"},     {TEMPORAL, "TEMPORAL"},
  {FIRST_LAST_SKIP, "FIRST_LAST_SKIP"}, {ITERATION_CONSIST, "ITER_CONSIST"}};

inline bool checkAllocOnly(CachePolicy policy, CacheElem::Options opt) {
  return ((policy & CachePolicy::NO_READ_CONSIST) ||
          ((opt & CacheElem::Options::FIRST_ACCESS) &&
           (policy & CachePolicy::FIRST_LAST_SKIP)));
}

inline bool checkDeallocOnly(CachePolicy policy, CacheElem::Options opt) {
  return ((policy & CachePolicy::NO_READ_CONSIST) ||
          ((opt & CacheElem::Options::LAST_ACCESS) &&
           (policy & CachePolicy::FIRST_LAST_SKIP)));
}

} // namespace

void CacheElem::swapIn(Options opt) {
  std::lock_guard<std::mutex> lock(device_mutex);

  opt = static_cast<Options>(opt | initial_opt);
  bool alloc_only = checkAllocOnly(policy, opt);
  void *buf = device->getBuffer(offset, length, alloc_only);

  initial_opt = Options::NONE;
  mem_data->setAddr((float *)buf);
  mem_data->setValid(true);
  active = true;

#ifdef PROFILE
  std::string msg("CacheElem(");
  msg += device->getDevicePath() + ") #" + std::to_string(id);
  PROFILE_CACHE_ALLOC(buf, length, msg, policyToStr[policy], !alloc_only);
#endif
}

void CacheElem::swapOut(Options opt) {
  std::lock_guard<std::mutex> lock(device_mutex);
  bool dealloc_only = checkDeallocOnly(policy, opt);
  void *buf = (void *)mem_data->getAddr();
  device->putBuffer(buf, dealloc_only);
  mem_data->setAddr(nullptr);
  mem_data->setValid(false);
  active = false;

#ifdef PROFILE
  PROFILE_CACHE_DEALLOC(buf, policyToStr[policy], !dealloc_only);
#endif
}

} // namespace nntrainer
