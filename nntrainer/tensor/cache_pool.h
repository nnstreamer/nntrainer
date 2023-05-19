// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_pool.h
 * @date   01 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache pool class inherited from memory pool
 *
 */

#ifndef __CACHE_POOL_H__
#define __CACHE_POOL_H__

#include <list>
#include <mutex>
#include <vector>

#include <cache_elem.h>
#include <memory_pool.h>
#include <swap_device.h>

namespace nntrainer {

/**
 * @class   CachePool
 * @brief   Cache memory with fixed size utilizing swap device
 */
class CachePool : public MemoryPool {
public:
  using CacheElems =
    std::map<unsigned int,
             std::shared_ptr<CacheElem>>; /**< cache id, cache elem */
  using CacheElemsIter = CacheElems::iterator;
  using ExecIds = std::vector<unsigned int>;
  using ExecIdsIter = ExecIds::iterator;

  /**
   * @brief CachePool default constructor
   *
   * @param name name of the cache pool
   */
  explicit CachePool(const std::string &name);

  /**
   * @brief CachePool constructor with cache path
   *
   */
  explicit CachePool(const std::string &path, const std::string &name);

  /**
   * @brief MemoryPool destructor
   *
   */
  virtual ~CachePool();

  /**
   * @brief Do the allocation of cache
   *
   */
  virtual void allocate() override;

  /**
   * @brief Free all the allocated cache
   *
   */
  virtual void deallocate() override;

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
   * @brief Is the cache pool allocated
   *
   * @return true if the memory is allocated, else false
   */
  virtual bool isAllocated() const override;

  /**
   * @brief Flush cache data to device
   *
   * @note it must be called only when epoch ends.
   */
  virtual void flush();

  /**
   * @brief Flush cache data to device except given order
   *
   * @param order except execution order
   */
  virtual void flushExcept(unsigned int order);

  /**
   * @brief Flush cache data to device except given order
   *
   * @param order except execution order
   */
  virtual void flushExcept(std::vector<unsigned int> order);

  /**
   * @brief Clear the memory pool
   *
   */
  virtual void clear() override;

  /**
   * @brief Load cache data by execution order
   *
   * @param order execution order
   */
  virtual void loadExec(unsigned int order);

  /**
   * @brief Load cache data by execution order
   *
   * @param order execution order
   */
  virtual void initCacheElemIter(CacheElemsIter &iter);

  /**
   * @brief Check iterator is last element
   *
   * @param order execution order
   */
  virtual bool isLastCacheElemIter(const CacheElemsIter &iter);

  /**
   * @brief Load cache data by execution order
   *
   * @param order execution order
   */
  virtual void initExecIdsIter(unsigned int order, ExecIdsIter &iter);

  /**
   * @brief Check iterator is last element
   *
   * @param order execution order
   */
  virtual bool isLastExecIdsIter(unsigned int order, const ExecIdsIter &iter);

  /**
   * @brief Load cache data by execution order
   *
   * @param order execution order
   */
  virtual bool loadExecOnce(unsigned int order, ExecIdsIter &iter);

  /**
   * @brief Unload cache data by execution order
   *
   * @param order execution order
   */
  virtual void unloadExec(unsigned int order);

  /**
   * @brief Load active cache data
   */
  virtual void loadActives();

  /**
   * @brief Unload active cache data
   */
  virtual void unloadActives();

  /**
   * @brief Get name
   *
   * @return cache pool name
   */
  virtual std::string getName() { return name; }

protected:
  /**
   * @brief validate cache element
   *
   * @param cache element id
   */
  virtual void validate(unsigned int id);

  /**
   * @brief invalidate cache element
   *
   * @param cache element id
   */
  virtual void invalidate(unsigned int id);

  /**
   * @brief Get cache policies
   *
   * @return Cache polices
   */
  std::vector<CachePolicy> &getCachePolicy() { return policies; }

private:
  std::string name;                        /**< pool name */
  std::shared_ptr<SwapDevice> swap_device; /**< swap device */
  CacheElems elems;                        /**< cache elements */

  std::list<std::shared_ptr<CacheElem>> actives;
  std::vector<CachePolicy> policies;
  std::map<unsigned int, ExecIds> exec_ids;

  std::mutex mod_mutex;
};

} // namespace nntrainer

#endif /** __CACHE_POOL_H__ */
