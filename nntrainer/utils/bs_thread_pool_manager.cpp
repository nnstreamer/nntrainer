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
/**
 * @brief Instantiate thread pool with the number of hardware concurrency.
 *
 * @return BS::thread_pool<>
 */
BS::thread_pool<> ThreadPoolManager::pool(std::thread::hardware_concurrency());

std::size_t ThreadPoolManager::select_k_quant_thread_count(unsigned int M,
                                                           unsigned int N,
                                                           unsigned int K) {
  const std::size_t max_threads = std::thread::hardware_concurrency();

  const std::size_t work_size = static_cast<std::size_t>(M * N * K);

  //  Use log-scale thresholds to reduce threads on smaller work sizes
  if (work_size < 1536 * 1536)
    return 1;
  if (work_size < 1536 * 2048)
    return 2;
  if (work_size < 2048 * 2048)
    return 4;

  std::size_t est_threads =
    static_cast<std::size_t>(std::log2(work_size / (1536 * 1536))) + 4;
  return std::min(est_threads, max_threads);
}
} // namespace nntrainer

#endif // THREAD_POOL_MANAGER_CPP
