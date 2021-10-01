// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_pool.h
 * @date   10 August 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Memory Pool Class
 *
 * @todo   Support an external allocator for different backends and alignment
 * @todo   Support releaseMemory(token) - this need not release actual memory
 * until deallocate
 * @todo   Support maximum memory size for the memory pool as an argument
 * @todo support late memory request without optimization
 */

#ifndef __MEMORY_POOL_H__
#define __MEMORY_POOL_H__

#include <vector>

#include <memory_planner.h>

namespace nntrainer {

/**
 * @class   MemoryPool
 * @brief   Memory Pool provides a common pool for all the tensor memory
 */
class MemoryPool {
public:
  /**
   * @brief MemoryPool default constructor
   *
   */
  MemoryPool() : mem_pool(nullptr), pool_size(0), min_pool_size(0) {}

  /**
   * @brief Request Memory from memory pool
   *
   * @param bytes The size of the memory requested in bytes
   * @param start_time The start of the validity interval of this memory
   * @param end_time The end of the validity interval of this memory
   *
   * @return The token to get the pointer for this memory after allocation
   * @note start_time is inclusive, but end_time is exclusive
   */
  unsigned int requestMemory(size_t bytes, unsigned int start_time,
                             unsigned int end_time);

  /**
   * @brief Planner the layout with memory planner
   *
   * @param planner The memory planner to be used for finalizing the layout
   *
   * @return The efficiency of the memory layer with the given memory planner
   *
   * @details The efficiency of the planner is calculated as the ratio of the
   * theoretical minimum memory requirement divided by the memory requirement
   * given by the memory planner.
   *
   * @details planLayout can be called multiple times as this does not perform
   * any allocation but rather just plans the layout and stores the layout.
   * Subsequent call to this function will overwrite any existing layout.
   */
  double planLayout(const MemoryPlanner &planner);

  /**
   * @brief Do the allocation of memory
   *
   */
  void allocate();

  /**
   * @brief Get the allocated memory
   *
   * @param token The token received from the requestMemory
   *
   * @return The pointer of the memory
   *
   * @details This function will throw if called before allocation.
   */
  void *getMemory(unsigned int idx);

  /**
   * @brief Free all the allocated memory
   *
   */
  void deallocate();

  /**
   * @brief Get the maximum real memory requirement
   *
   * @return The real memory requirement with this strategy in bytes
   */
  size_t size();

  /**
   * @brief Get the minimum theoretical memory requirement
   *
   * @return The theoretical memory requirement with this strategy in bytes
   */
  size_t minMemoryRequirement();

  /**
   * @brief Clear the memory pool
   *
   */
  void clear();

private:
  /**
   * @brief Validate the provided layout
   */
  bool validateLayout();

  /**
   * @brief Validate the provided layout does not overflow outside the given
   * size of the memory pool
   */
  bool validateOverflow();

  /**
   * @brief Validate the provided layout so that no two memories to be used at
   * overlap interval has overlapping memories
   */
  bool validateOverlap();

  /**
   * @brief Calculate the minimum memory requirement for the given memory
   * requests
   *
   * @return the minimum memory requirement in bytes
   *
   * @note This will be theoretical minimum memory requirement ensuring that the
   * memory usages at the same time do not overlap with their validity. This
   * does not consider about the fragmentation which comes from the actual
   * memory layout.
   */
  size_t calcMinMemoryRequirement();

  /**
   * @brief Get sorted permuation for the memory requests
   *
   * @return sorted permutation
   *
   * @details Performs sorting based on the memory overlap using memory offset
   * as the start and the memory offset + memory size as the end of the
   * interval.
   */
  std::vector<unsigned int> getSortedPermutation();

  std::vector<size_t> memory_size; /**< various sizes memory requested */
  std::vector<std::pair<unsigned int, unsigned int>>
    memory_validity; /**< validity intervals for each requested memory */
  std::vector<size_t> memory_offset; /**< offsets for the memory requested */

  void *mem_pool;   /**< memory pool allocated at once */
  size_t pool_size; /**< memory requirement for this pool */

  size_t min_pool_size; /**< minimum theoretical memory requirement */
};

} // namespace nntrainer

#endif /** __MEMORY_POOL_H__ */
