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
 * @class ThreadPool
 * @brief ThreadPool class to manage multiple threads
 */
class ThreadPool {
public:
  /**
   * @brief constructor of ThreadPool
   *
   * @param num_threads number of threads
   */
  ThreadPool(size_t num_threads);

  /**
   * @brief destructor of ThreadPool
   */
  ~ThreadPool();

  /**
   *
   * @tparam F Function that work
   * @tparam Args args for the function
   * @param f functions
   * @param args arguments for the function
   * @return
   */
  template <class F, class... Args>
  std::future<typename std::invoke_result<F, Args...>::type>
  EnqueueJob(F &&f, Args &&...args);

private:
  size_t num_threads_;
  std::vector<std::thread> worker_threads_;
  std::queue<std::function<void()>> jobs_;
  std::condition_variable cv_job_q_;
  std::mutex m_job_q_;

  bool stop_all;

  /**
   * Thread Worker
   */
  void WorkerThread();
};

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
   * @brief Load cache data asynchronously with execution order
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

private:
  std::shared_ptr<CachePool> pool;    /**< cache pool */
  TaskExecutor *load_task_executor;   /**< task executor */
  TaskExecutor *unload_task_executor; /**< task executor */

  ThreadPool *thread_pool_;
  std::map<int, std::future<bool>> order_to_future;

  std::mutex load_lock;
  std::mutex thread_lock;
};

} // namespace nntrainer

#endif /** __CACHE_LOADER_H__ */
