// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   temp_pool.h
 * @date   03 April 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Temporal data pool class inherited from memory pool
 *
 */

#ifndef __TEMP_POOL_H__
#define __TEMP_POOL_H__

#include <atomic>
#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <cache_elem.h>
#include <memory_pool.h>
#include <swap_device.h>

namespace nntrainer {

/**
 * @class   TempPool
 * @brief   Temporal data memory
 */
class TempPool : public MemoryPool {
public:
  /**
   * @brief TempPool default constructor
   *
   * @param name name of the cache pool
   */
  explicit TempPool();

  /**
   * @brief MemoryPool destructor
   *
   */
  virtual ~TempPool();

  /**
   * @brief Do the allocation of cache
   *
   */
  virtual void allocate();

  /**
   * @brief Free all the allocated cache
   *
   */
  virtual void deallocate();

  /**
   * @brief Request Memory from memory pool
   * @note start_time is inclusive, but end_time is exclusive
   */
  virtual unsigned int requestMemory(
    size_t bytes, unsigned int start_time, unsigned int end_time,
    std::vector<unsigned int> exec_order = std::vector<unsigned int>(),
    TensorLifespan lifespan = TensorLifespan::MAX_LIFESPAN,
    bool is_wgrad = false);

  /**
   * @brief Get the allocated cache
   *
   * @param id The token received from the requestMemory
   *
   * @return The pointer of the cache
   *
   * @details This function will throw if called before allocation.
   */
  virtual std::shared_ptr<MemoryData> getMemory(unsigned int id);

  /**
   * @brief Plan the layout with memory planner
   */
  virtual double planLayout(const MemoryPlanner &planner);

  /**
   * @brief Is the cache pool allocated
   *
   * @return true if the memory is allocated, else false
   */
  virtual bool isAllocated() const { return allocated; }

  /**
   * @brief Reclaim temp memory
   *
   * @param exec_order The execution order of the graph
   */
  virtual void reclaim(unsigned int exec_order);

  /**
   * @brief Reclaim all temp memory
   */
  virtual void reclaimAll();

  /**
   * @brief Get the maximum real memory requirement
   */
  virtual size_t size();

  /**
   * @brief Clear the memory pool
   *
   */
  virtual void clear();

  /**
   * @brief Get the minimum memory requirement
   */
  virtual size_t minMemoryRequirement();

private:
  bool allocated;           /**< is the memory allocated */
  std::atomic_uint next_id; /**< The next id to be used */
  std::unordered_map<unsigned int, std::shared_ptr<MemoryData>>
    data; /**< memoty id to the memory data */
  std::unordered_map<unsigned int, size_t>
    data_size; /**< memory id to requested memroy size */
  std::unordered_map<unsigned int, std::vector<unsigned int>>
    exec_orders; /**< The execution order to memory id */
};

} // namespace nntrainer

#endif /** __CACHE_POOL_H__ */
