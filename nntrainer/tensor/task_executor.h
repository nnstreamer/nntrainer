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
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <unistd.h>

#include <nntrainer_error.h>
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

  /**
   * @brief TaskExecutor constructor
   *
   */
  explicit TaskExecutor() : ids(1) {}

  /**
   * @brief TaskExecutor destructor
   *
   */
  virtual ~TaskExecutor() = default;

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

    tasks[id] = std::make_tuple(
      std::async(std::launch::async, &TaskExecutor::worker<T>, this, id, task),
      std::async(std::launch::async, &TaskExecutor::completeChecker<T>, this,
                 id, task),
      callback, CompleteStatus::SUCCESS, true);

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
  template <typename T>
  void worker(int id, std::shared_ptr<TaskAsync<T>> task) {
    if (task->started())
      return;

    auto work = task->getWork();
    auto data = task->getData();

    task->setState(Task::State::PROCESSING);

    handleWork(id, work, data);
  }

  virtual void handleWork(int id, Task::Work &work, void *data);

  /**
   * @brief Check task is completed
   *
   * @param id task id
   * @param task asynchronous task
   */
  template <typename T>
  void completeChecker(int id, std::shared_ptr<TaskAsync<T>> task) {
    // wait until tasks are fully mapped (max 1 sec)
    int retry = 10;
    while (retry && (tasks.find(id) == tasks.end())) {
      usleep(100 * 1000);
      retry--;
    }

    NNTR_THROW_IF(tasks.find(id) == tasks.end(), std::runtime_error)
      << "tasks was not correctly mapped";

    auto status = std::get<0>(tasks[id]).wait_for(T(task->getTimeout()));
    handleCompleteStatus(id, status);

    task->setState(Task::State::DONE);
  }

  /**
   * @brief handles complete status
   *
   * @param id task id
   * @param status asynchronous worker's finishing status
   */
  virtual void handleCompleteStatus(int id, const std::future_status status);

  std::atomic_int32_t ids; /**< for id generation */
  std::map<int, std::tuple<std::future<void>, std::future<void>,
                           CompleteCallback, CompleteStatus, std::atomic_bool>>
    tasks; /**< (task id, (future, complete future, callback, complete status,
              running)) */
};

} // namespace nntrainer

#endif /** __TASK_EXECUTOR_H__ */
