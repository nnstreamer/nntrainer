// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_loader.h
 * @date   10 Nov 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache loader class
 *
 */

#ifndef __CACHE_LOADER_H__
#define __CACHE_LOADER_H__

#include <algorithm>
#include <atomic>
#include <functional>
#include <future>
#include <map>

#include <cache_pool.h>
#include <task_executor.h>

namespace nntrainer {

/**
 * @class   CacheLoader
 * @brief   Cache loader from swap device
 */
class CacheLoader {
public:
  /**
   * @brief CacheLoader default constructor
   *
   */
  explicit CacheLoader(std::shared_ptr<CachePool> cache_pool);

  /**
   * @brief CacheLoader destructor
   *
   */
  virtual ~CacheLoader();

  /**
   * @brief initialize loader
   */
  virtual void init();

  /**
   * @brief finish loader
   *
   */
  virtual void finish();

  /**
   * @brief Load cache data with execution order
   *
   * @param order execution order
   */
  virtual void load(unsigned int order);

  /**
   * @brief Load cache data asynchronously with execution order
   *
   * @param order execution order
   * @param complete complete callback
   * @return async task id
   */
  virtual int loadAsync(unsigned int order,
                        TaskExecutor::CompleteCallback callback);

  /**
   * @brief Load cache data asynchronously with execution order
   *
   * @param order execution order
   * @param complete complete callback
   * @param timeout timeout time in ms
   * @return async task id
   * @note timeout_ms does not work now.
   */
  virtual int loadAsync(unsigned int order,
                        TaskExecutor::CompleteCallback callback,
                        long timeout_ms);

  /**
   * @brief Cancel async task
   *
   * @param id task id
   * @return 0 on success, otherwise negative error
   */
  virtual int cancelAsync(int id);

private:
  std::shared_ptr<CachePool> pool; /**< cache pool */
  TaskExecutor *task_executor;     /**< task executor */
};

} // namespace nntrainer

#endif /** __CACHE_LOADER_H__ */
