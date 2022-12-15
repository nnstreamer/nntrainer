// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   task_executor.h
 * @date   04 Nov 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Task executor class
 *
 */

#ifndef __TASK_EXECUTOR_H__
#define __TASK_EXECUTOR_H__

#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <list>
#include <memory>
#include <mutex>
#include <thread>
#include <unistd.h>

#include <task.h>

namespace nntrainer {

/**
 * @class   TaskExecutor
 * @brief   task executor class
 */
class TaskExecutor {
public:
  /**
   * @brief Complete error types
   */
  enum CompleteStatus {
    SUCCESS,
    FAIL_CANCEL,
    FAIL_TIMEOUT,
    FAIL,
  };

  /**
   * @brief Complete callback type which will be called when task is completed
   */
  using CompleteCallback =
    std::function<void(int, CompleteStatus)>; /**< (task id, status) */

  template <typename T = std::chrono::milliseconds>
  using TaskInfo =
    std::tuple<int, std::shared_ptr<TaskAsync<T>>, CompleteCallback,
               std::atomic_bool, std::promise<CompleteStatus>>;
  /**< (task id, task, complete callback, running, complete promise) */

  /**
   * @brief TaskExecutor constructor
   *
   */
  explicit TaskExecutor(const std::string &name = "");

  /**
   * @brief TaskExecutor destructor
   *
   */
  virtual ~TaskExecutor();

  /**
   * @brief Run task
   *
   * @param task task to be run
   * @return 0 on Success, else negative error
   */
  virtual int run(std::shared_ptr<Task> task);

  /**
   * @brief Run task asynchronously
   *
   * @param task async task to be run
   * @param callback complete callback
   * @return id of requested task.
   *         negative on fail
   */
  template <typename T>
  int run(std::shared_ptr<TaskAsync<T>> task, CompleteCallback callback) {
    int id = ids.load();
    {
      std::scoped_lock lock(task_mutex);
      task_queue.emplace_back(id, task, callback, true,
                              std::promise<CompleteStatus>());
    }
    task_cv.notify_one();
    ids++;

    return id;
  }

  /**
   * @brief Cancel task
   *
   * @param id task id returned from @ref TaskExecutorAsync::run
   */
  virtual void cancel(int id);

  /**
   * @brief Cancel all task
   */
  virtual void cancelAll(void);

  /**
   * @brief Clean all non-running tasks from managed list
   *
   * @note Do not use this inside of the complete or worker callback
   */
  virtual void clean(void);

protected:
  /**
   * @brief task worker for asynchronous task
   *
   * @param id task id
   * @param task asynchronous task
   */
  template <typename T> CompleteStatus worker(TaskInfo<T> &info) {
    auto &running = std::get<std::atomic_bool>(info);
    auto task = std::get<std::shared_ptr<TaskAsync<T>>>(info);
    if (task->started())
      return CompleteStatus::FAIL;

    auto work = task->getWork();
    auto data = task->getData();

    task->setState(Task::State::PROCESSING);

    return handleWork(running, work, data);
  }

  /**
   * @brief handle work for asynchronous task
   *
   * @param running running flag
   * @param work work function
   * @param data data to be passed to work function
   * @return CompleteStatus
   */
  virtual CompleteStatus handleWork(std::atomic_bool &running, Task::Work &work,
                                    void *data);

  static std::atomic_int32_t ids;
  std::string name;
  bool run_thread;
  bool wait_complete;

  std::list<TaskInfo<>> task_queue;

  std::condition_variable task_cv;
  std::condition_variable comp_cv;
  std::thread task_thread;
  std::mutex task_mutex;
};

} // namespace nntrainer

#endif /** __TASK_EXECUTOR_H__ */
