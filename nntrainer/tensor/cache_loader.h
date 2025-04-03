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

class ThreadPool {
 public:
  ThreadPool(size_t num_threads);
  ~ThreadPool();

  template <class F, class... Args>
  std::future<typename std::result_of<F(Args...)>::type> EnqueueJob(
    F&& f, Args&&... args);

 private:
  size_t num_threads_;
  std::vector<std::thread> worker_threads_;
  std::queue<std::function<void()>> jobs_;
  std::condition_variable cv_job_q_;
  std::mutex m_job_q_;

  bool stop_all;

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
  bool order_load_done(unsigned int order);
  int loadFsuAsync(unsigned int order, unsigned int look_ahead);

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
  unsigned int Inactive(unsigned int order);

private:
  std::shared_ptr<CachePool> pool;    /**< cache pool */
  TaskExecutor *load_task_executor;   /**< task executor */
  TaskExecutor *unload_task_executor; /**< task executor */
  ThreadPool *thread_pool_;
  std::map<int, std::future<bool>> order_to_future;
  std::map<int, std::promise<bool>> order_to_promise;

  std::mutex load_lock;
  std::mutex thread_lock;
};

} // namespace nntrainer

#endif /** __CACHE_LOADER_H__ */
