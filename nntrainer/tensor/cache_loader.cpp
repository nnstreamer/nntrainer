// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_loader.cpp
 * @date   10 Nov 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache loader class
 *
 */

#include "cache_loader.h"
#include "task.h"
#include "task_executor.h"

#include <cache_pool.h>
#include <climits>
#include <cstdint>
#include <exception>
#include <memory>
#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

CacheLoader::CacheLoader(std::shared_ptr<CachePool> cache_pool) :
  pool(cache_pool),
  load_task_executor(nullptr),
  unload_task_executor(nullptr),
  thread_pool_(nullptr) {}

CacheLoader::~CacheLoader() {
  if (load_task_executor)
    delete load_task_executor;
  if (unload_task_executor)
    delete unload_task_executor;
  if (thread_pool_)
    delete thread_pool_;
}

void CacheLoader::init() {

  if (load_task_executor == nullptr)
    load_task_executor = new TaskExecutor(pool->getName());
  if (unload_task_executor == nullptr)
    unload_task_executor = new TaskExecutor(pool->getName());
  if (thread_pool_ == nullptr)
    thread_pool_ = new ThreadPool(1);
}

void CacheLoader::finish() {
  delete load_task_executor;
  load_task_executor = nullptr;
  delete unload_task_executor;
  unload_task_executor = nullptr;
  delete thread_pool_;
  thread_pool_ = nullptr;
}

void CacheLoader::load(unsigned int order) { pool->loadExec(order); }

int CacheLoader::loadAsync(unsigned int order,
                           TaskExecutor::CompleteCallback complete) {
  return loadAsync(order, complete, LONG_MAX);
}

int CacheLoader::loadAsync(unsigned int order,
                           TaskExecutor::CompleteCallback complete,
                           long timeout_ms) {
  if (!load_task_executor) {
    ml_loge("init is needed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  Task::Work work = [&](std::atomic_bool &running, void *data) {
    unsigned int exe_order = (unsigned int)(std::uintptr_t)data;

    // pool->flushExcept({exe_order - 1, exe_order});
    pool->loadExec(exe_order);

    return ML_ERROR_NONE;
  };

  auto task =
    std::make_shared<TaskAsync<>>(work, (void *)(std::uintptr_t)order);
  task->setTimeout(timeout_ms);

  return load_task_executor->run(task, complete);
}

int CacheLoader::flushAsync(unsigned int order,
                            TaskExecutor::CompleteCallback complete) {
  return flushAsync(order, complete, LONG_MAX);
}

int CacheLoader::flushAsync(unsigned int order,
                            TaskExecutor::CompleteCallback complete,
                            long timeout_ms) {
  if (!unload_task_executor) {
    ml_loge("init is needed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  Task::Work work = [&](std::atomic_bool &running, void *data) {
    unsigned int exe_order = (unsigned int)(std::uintptr_t)data;

    // pool->flushExcept({exe_order - 1, exe_order});
    pool->flushExcept(exe_order);

    return ML_ERROR_NONE;
  };

  auto task =
    std::make_shared<TaskAsync<>>(work, (void *)(std::uintptr_t)order);
  task->setTimeout(timeout_ms);

  return unload_task_executor->run(task, complete);
}

int CacheLoader::cancelAsync(int id) {
  try {
    load_task_executor->cancel(id);
  } catch (const std::exception &e) {
    ml_loge("CacheLoader(%s): failed to cancel(%d): %s",
            pool->getName().c_str(), id, e.what());
    return ML_ERROR_UNKNOWN;
  }

  return ML_ERROR_NONE;
}

unsigned int CacheLoader::getNumLoadedTensors() {
  return pool->getNumLoadedTensors();
}

bool CacheLoader::checkFsuLoadComplete(unsigned int order) {
  std::lock_guard<std::mutex> lock(load_lock);
  order_to_future[order].wait();
  if (order_to_future[order].get() == true) {
    return true;
  }
  return false;
}

int CacheLoader::loadFsuAsync(unsigned int order, unsigned int look_ahead) {

  auto load_work = [&](unsigned int exe_order) {
    pool->loadExec(exe_order);
    return true;
  };

  order_to_future[order] = thread_pool_->EnqueueJob(load_work, order);

  return 0;
}

void CacheLoader::setupFSU() {
  pool->setupFSU();
  order_to_future.clear();
  order_to_future.clear();
}

ThreadPool::ThreadPool(size_t num_threads) :
  num_threads_(num_threads), stop_all(false) {
  worker_threads_.reserve(num_threads_);
  for (size_t i = 0; i < num_threads_; ++i) {
    worker_threads_.emplace_back([this]() { this->WorkerThread(); });
  }
}

void ThreadPool::WorkerThread() {
  while (true) {
    std::unique_lock<std::mutex> lock(m_job_q_);
    cv_job_q_.wait(lock, [this]() { return !this->jobs_.empty() || stop_all; });
    if (stop_all && this->jobs_.empty()) {
      return;
    }

    std::function<void()> job = std::move(jobs_.front());
    jobs_.pop();
    lock.unlock();

    job();
  }
}

ThreadPool::~ThreadPool() {
  stop_all = true;
  cv_job_q_.notify_all();

  for (auto &t : worker_threads_) {
    t.join();
  }
}

template <class F, class... Args>
std::future<typename std::invoke_result<F, Args...>::type>
ThreadPool::EnqueueJob(F &&f, Args &&...args) {
  if (stop_all) {
    throw std::runtime_error("ThreadPool Stop all");
  }

  using return_type = typename std::invoke_result<F, Args...>::type;
  auto job = std::make_shared<std::packaged_task<return_type()>>(
    std::bind(std::forward<F>(f), std::forward<Args>(args)...));
  std::future<return_type> job_result_future = job->get_future();
  {
    std::lock_guard<std::mutex> lock(m_job_q_);
    jobs_.push([job]() { (*job)(); });
  }
  cv_job_q_.notify_one();

  return job_result_future;
}

} // namespace nntrainer
