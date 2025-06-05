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
   * @brief Select optimal number of thread to use in K-quantized GEMM and GEMV
   *
   * @param M M for GEMM (M != 1) or GEMV (M = 1)
   * @param N N for GEMM or GEMV
   * @param K K for GEMM or GEMV
   * @return std::size_t number of thread to use
   */
  std::size_t select_k_quant_thread_count(unsigned int M, unsigned int N,
                                          unsigned int K);

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
