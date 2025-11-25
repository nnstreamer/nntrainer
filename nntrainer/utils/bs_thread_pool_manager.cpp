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

std::size_t ThreadPoolManager::select_k_quant_thread_count(unsigned int M,
                                                           unsigned int N,
                                                           unsigned int K) {
  const std::size_t hw_threads =
    std::max<std::size_t>(1U, std::thread::hardware_concurrency());
  const std::size_t work_size = static_cast<std::size_t>(M) *
                                static_cast<std::size_t>(N) *
                                static_cast<std::size_t>(K);

  std::size_t est_threads = 1;

  // Use log-scale thresholds to reduce threads on smaller work sizes
  if (work_size < 1536ULL * 1536ULL) {
    est_threads = 1;
  } else if (work_size < 1536ULL * 2048ULL) {
    est_threads = 2;
  } else if (work_size < 2048ULL * 2048ULL) {
    est_threads = 4;
  } else {
    const std::size_t normalized =
      std::max<std::size_t>(1ULL, work_size / (1536ULL * 1536ULL));
    est_threads =
      static_cast<std::size_t>(std::log2(static_cast<double>(normalized))) + 4;
  }

  return std::max<std::size_t>(1ULL, std::min(est_threads, hw_threads));
}

} // namespace nntrainer

#endif // THREAD_POOL_MANAGER_CPP
