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
/**
 * @brief ThreadPoolManager is a singleton class that manages a thread pool
 *
 */
class ThreadPoolManager {
protected:
  static BS::thread_pool<> pool;

public:
  // Delete copy and move constructors and assignment operators
  /**
   * @brief Construct a new Thread Pool Manager object
   *
   */
  ThreadPoolManager(const ThreadPoolManager &) = delete;

  /**
   * @brief Construct a new Thread Pool Manager object
   *
   */
  ThreadPoolManager(ThreadPoolManager &&) = delete;

  /**
   * @brief Static method to access the single instance
   *
   * @return BS::thread_pool<>&
   */
  static BS::thread_pool<> &getInstance() { return pool; }

private:
  /**
   * @brief Construct a new Thread Pool Manager object
   *
   */
  ThreadPoolManager() = default;
  /**
   * @brief Destroy the Thread Pool Manager object
   *
   */
  ~ThreadPoolManager() = default;
};
} // namespace nntrainer

#endif // THREAD_POOL_MANAGER_HPP
