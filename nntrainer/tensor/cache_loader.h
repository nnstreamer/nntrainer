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
#include <queue>
#include <task_executor.h>

namespace nntrainer {

/**
 * @enum   LoadState
 * @brief  enum to describe Load State
 */
enum class LoadState { Idle, Loading, Loaded, Unloading };

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
   * @brief Load cache data asynchronously in Tensor
   *
   * @param id Tensor id
   * @return async task id
   */
  virtual int loadTensor(unsigned int id);

  /**
   * @brief unLoad cache data asynchronously in Tensor
   *
   * @param id Tensor id
   * @return async task id
   */
  virtual int unloadTensor(unsigned int id);

  /**
   * @brief Load cache data asynchronously with execution order
   *
   * @param order execution order
   * @return true if enqueue loads successfully
   */
  virtual bool loadAllinOrder(unsigned int order);

  /**
   * @brief unLoad cache data asynchronously with execution order
   *
   * @param order execution order
   * @return true if enqueue unloads successfully
   */
  virtual bool unloadAllinOrder(unsigned int order);
  /**
   *
   * @param order execution order
   * @param complete complete callback
   * @return async task id
   */
  virtual int flushAsync(unsigned int order,
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
  virtual int flushAsync(unsigned int order,
                         TaskExecutor::CompleteCallback callback,
                         long timeout_ms);

  /**
   * @brief Cancel async task
   *
   * @param id task id
   * @return 0 on success, otherwise negative error
   */
  virtual int cancelAsync(int id);

  /**
   * @brief Get number of loaded tensors
   *
   * @return number of loaded tensors
   */
  virtual unsigned int getNumLoadedTensors();

  /**
   * @brief set Inactive elems in order
   *
   */

  unsigned int Inactive(unsigned int order);

  /**
   * @brief setup FSU for the given execution order.
   * This function will reset Actives at the given order.
   *
   */
  virtual void setupFSU();

  /**
   * @brief check data of order is loaded
   *
   * @param order execution order
   */
  virtual bool checkFsuLoadComplete(unsigned int order);

  /**
   * @brief Load cache data with execution order for FSU
   *
   * @param order execution order
   */
  virtual int loadFsuAsync(unsigned int order, unsigned int look_ahead);

  /**
   * @brief wait for the load tasks in order are complete
   * @param order execution order
   * @return true if all load tasks complete
   *
   */
  bool checkAllLoadComplete(unsigned int order);

  /**
   * @brief wait for the load task with id is complete
   * @param id tensor id
   * @return true if load tas complete
   *
   */
  bool checkLoadComplete(unsigned int id);

  /**
   * @brief wait for the unload tasks in order are complete
   * @param order execution order
   * @return true if all unload tasks complete
   *
   */
  bool checkAllUnloadComplete(unsigned int order);

  /**
   * @brief wait for the unload task with id is complete
   * @param id tensor id
   * @return true if unload tas complete
   *
   */
  bool checkUnloadComplete(unsigned int id);

  /**
   * @brief get Loading / Unloading Status of tensor with id
   * @param id tensor id
   * @return LoadState
   *
   */
  LoadState getState(int id) const;

  /**
   * @brief unload all the tensors and clear cache elem
   * @return true if unload tas complete
   *
   */
  void flush();

private:
  std::shared_ptr<CachePool> pool;    /**< cache pool */
  TaskExecutor *load_task_executor;   /**< task executor */
  TaskExecutor *unload_task_executor; /**< task executor */

  std::mutex load_lock;
  std::mutex thread_lock;
  mutable std::mutex state_mutex;
  std::unordered_map<int, LoadState> states;
};

} // namespace nntrainer

#endif /** __CACHE_LOADER_H__ */
