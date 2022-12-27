// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_planner.h
 * @date   10 August 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is interface for the Memory Planner
 *
 */

#ifndef __MEMORY_PLANNER_H__
#define __MEMORY_PLANNER_H__

#include <string>
#include <vector>

namespace nntrainer {

/**
 * @class   MemoryPlanner
 * @brief   Memory Planner provides the plan/strategy to allocate the memory
 */
class MemoryPlanner {
public:
  /**
   * @brief MemoryPlanner destructor
   *
   */
  virtual ~MemoryPlanner() = default;

  /**
   * @brief Plan the layout for the memory allocation
   *
   * @param[in] memory_size The size of the various memories
   * @param[in] memory_validity The validity of the various memories
   * @param[out] memory_offset The offset of each memory from the base of the
   * allocated memory
   * @param[in] memory_is_wgrad The index for identification of weight gradient
   * @return The total memory required as per this strategy
   *
   * @details The minimum offset will be 0, and the maximum offset will be less
   * than the sum of the memory_size vector.
   */
  virtual size_t planLayout(
    const std::vector<size_t> &memory_size,
    const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
    std::vector<size_t> &memory_offset, std::vector<bool> &memory_is_wgrad,
    size_t n_wgrad) const = 0;

  /**
   * @brief Get type of the planner
   *
   * @return The type of the planner
   */
  virtual const std::string &getType() const = 0;
};

} // namespace nntrainer

#endif /** __MEMORY_PLANNER_H__ */
