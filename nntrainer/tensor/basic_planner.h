// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   basic_planner.h
 * @date   11 August 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Naive Memory Planner
 *
 */

#ifndef __BASIC_PLANNER_H__
#define __BASIC_PLANNER_H__

#include <vector>

#include <memory_planner.h>

namespace nntrainer {

/**
 * @class   BasicPlanner
 * @brief   Basic Memory Planner provides the basic plan for memory layout
 * @details Basic planner performs no memory optimization and provides no memory
 * sharing
 */
class BasicPlanner : public MemoryPlanner {
public:
  /**
   * @brief BasicPlanner destructor
   *
   */
  ~BasicPlanner() = default;

  /**
   * @copydoc MemoryPlanner::planLayout(
   * const std::vector<size_t> &memory_size,
   * const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
   * std::vector<size_t> &memory_offset,
   * std::vector<bool> &memory_is_wgrad);
   *
   */
  size_t planLayout(
    const std::vector<size_t> &memory_size,
    const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
    std::vector<size_t> &memory_offset, std::vector<bool> &memory_is_wgrad,
    size_t n_wgrad = 0) const;

  /**
   * @copydoc MemoryPlanner::getType() const
   *
   */
  const std::string &getType() const { return type; }

  inline static const std::string type = "basic_planner";
};

} // namespace nntrainer

#endif /** __BASIC_PLANNER_H__ */
