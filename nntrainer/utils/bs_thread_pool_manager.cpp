// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   bs_threadpool_manager.cpp
 * @date   20 May 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  BS threadpool manager class source file
 */

#ifndef THREAD_POOL_MANAGER_CPP
#define THREAD_POOL_MANAGER_CPP

#include "bs_thread_pool_manager.hpp"
#include <algorithm>
#include <cmath>
namespace nntrainer {

BS::thread_pool<>& ThreadPoolManager::getInstance() {
  // Constructed on first call, reused thereafter
  static BS::thread_pool<> instance{std::thread::hardware_concurrency() / 2};
  // static BS::thread_pool<> instance{std::thread::hardware_concurrency()};
  return instance;
}

} // namespace nntrainer

#endif // THREAD_POOL_MANAGER_CPP
