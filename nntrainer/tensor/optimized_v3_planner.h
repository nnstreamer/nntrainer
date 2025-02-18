// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimzied_v3_planner.h
 * @date   2 January 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Optimized V3 Memory Planner
 *
 *
 */

#ifndef __OPTIMIZED_V3_PLANNER_H_
#define __OPTIMIZED_V3_PLANNER_H_

#include <vector>

#include <memory_planner.h>

namespace nntrainer {

/**
 * @class   OptimizedV3Planner
 * @brief   Optimized V3 Memory Planner provides the optimized plan for memory
 * layout
 * @details optimized planner performs sharing of overlapping memory sharing
 * upto certain extent
 */
class OptimizedV3Planner : public MemoryPlanner {
public:
  /**
   * @brief OptimizedV3Planner destructor
   *
   */
  OptimizedV3Planner() = default;

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

  inline static const std::string type = "optimized_v3_planner";
};

} // namespace nntrainer

#endif /** __OPTIMIZED_V3_PLANNER_H_ */
