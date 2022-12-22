// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   basic_planner.cpp
 * @date   11 August 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Naive Memory Planner
 *
 */

#include <basic_planner.h>
#include <nntrainer_error.h>

namespace nntrainer {

/**
 * @copydoc MemoryPlanner::planLayout(
 * const std::vector<size_t> &memory_size,
 * const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
 * std::vector<size_t> &memory_offset,
 * std::vector<bool> &memory_is_wgrad);
 *
 * @details The basic memory planner does not incorporate any memory sharing.
 * This planner allocates independent memory for all the required memories
 * without considering their memory validity.
 *
 */
size_t BasicPlanner::planLayout(
  const std::vector<size_t> &memory_size,
  const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
  std::vector<size_t> &memory_offset,
  std::vector<bool> &memory_is_wgrad) const {

  memory_offset.resize(memory_size.size());
  size_t csum = 0;

  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {

#ifdef DEBUG
    /** skip any memory requirement, if validity is less than 1 */
    if (memory_validity[idx].second <= memory_validity[idx].first)
      throw std::runtime_error("Memory requested for non-valid duration.");
#endif

    memory_offset[idx] = csum;
    csum += memory_size[idx];
  }

  return csum;
}

} // namespace nntrainer
