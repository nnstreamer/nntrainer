// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sungsik Kong <ss.kong@samsung.com>
 *
 * @file   bs_threadpool_manager.hpp
 * @date   20 May 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  BS threadpool manager class header file
 */

#ifndef THREAD_POOL_MANAGER_HPP
#define THREAD_POOL_MANAGER_HPP

#pragma once
#include "bs_thread_pool.h"

namespace nntrainer {
class ThreadPoolManager {
public:
  static BS::thread_pool<>& getInstance();

  // Prevent copying and moving  
  ThreadPoolManager(const ThreadPoolManager&) = delete;  
  ThreadPoolManager& operator=(const ThreadPoolManager&) = delete;  
  ThreadPoolManager(ThreadPoolManager&&) = delete;  
  ThreadPoolManager& operator=(ThreadPoolManager&&) = delete;  

private:  
  ThreadPoolManager() = default; // Prevent instantiation  
};
}

#endif // THREAD_POOL_MANAGER_HPP
