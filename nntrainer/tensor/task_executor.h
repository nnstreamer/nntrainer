// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   task_executor.h
 * @date   04 April 2025
 * @brief  This file contains a task executor
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Task executor class
 *
 */

#ifndef __TASK_EXECUTOR_H__
#define __TASK_EXECUTOR_H__

#include <atomic>
#include <condition_variable>
#include <functional>
#include <future>
#include <iostream>
#include <map>
#include <mutex>
#include <queue>
#include <task.h>
#include <thread>
#include <unordered_set>
#include <vector>

namespace nntrainer {

/**
 * @brief This is call back for load/unload Task
 *
 */
using TaskCallback = std::function<void(void *)>;

/**
 * @class TaskExecutor Class
 * @brief This is load / unload Task Executor with thread pool
 *
 */
class TaskExecutor {
public:
  /**
   * @enum  Temperal Enum for CompeleteStatus
   *
   */
  enum CompleteStatus {
    SUCCESS,
    FAIL_CANCEL,
    FAIL_TIMEOUT,
    FAIL,
  };

  /**
   * @struct To describe Task
   *
   */
  struct TaskDesc {
    int id;
    TaskCallback callback;
    void *data;
  };

  /**
   * @enum  Temperal definition for callback
   *
   */
  using CompleteCallback =
    std::function<void(int, CompleteStatus,
                       std::future<CompleteStatus>)>; /**< (task id, status) */

  template <typename T = std::chrono::milliseconds>
  using TaskInfo =
    std::tuple<int, std::shared_ptr<TaskAsync<T>>, CompleteCallback,
               std::atomic_bool, std::promise<CompleteStatus>>;
  /**< (task id, task, complete callback, running, complete promise) */

  /**
   * @brief Constructor of TaskExecutor
   *
   */
  TaskExecutor(std::string name = "",
               size_t thread_count = std::thread::hardware_concurrency());

  /**
   * @brief Destructor of TaskExecutor
   *
   */
  ~TaskExecutor();

  /**
   * @brief submit Task
   *
   */
  int submit(TaskCallback cb, void *data = nullptr);

  /**
   * @brief Cancel Task
   *
   */
  bool cancel(int id);

  /**
   * @brief Wait to complete
   *
   */
  void wait(int id);

  /**
   * @brief Wait to complete Tasks in vectors
   *
   */
  void waitAll(const std::vector<int> &ids);

  /**
   * @brief check done of task id
   *
   */
  bool isDone(int id);

  /**
   * @brief check done all the tasks in vector
   *
   */
  bool isAllDone(const std::vector<int> &ids);

  /**
   * @brief Submit mutiple tasks
   *
   */
  void submitTasks(const std::vector<TaskDesc> &tasks);

  /**
   * @brief release Task
   *
   */
  void releaseTask(int id);

private:
  /**
   * @brief Definition of  Task
   *
   */
  struct Task {
    int id;
    std::promise<void> promise;
    TaskCallback callback;
    void *data = nullptr;
  };

  /**
   * @brief Create Worker Thread
   *
   */
  void worker_thread();

  /**
   * @brief Get Next Task Id for protect the overflow
   *
   */
  int getNextTaskId() {
    if (!reusable_ids.empty()) {
      int id = reusable_ids.front();
      reusable_ids.pop();
      return id;
    }
    return next_task_id.fetch_add(1);
  }

  std::string name;
  std::vector<std::thread> workers;
  std::queue<Task> task_queue;
  std::map<int, std::shared_ptr<std::atomic_bool>> cancel_map;
  std::map<int, std::shared_future<void>> future_map;
  std::map<int, bool> task_started;
  std::mutex queue_mutex;
  std::condition_variable cond_var;
  std::condition_variable task_started_cv;
  std::atomic<bool> stop;
  std::unordered_set<int> queued_ids;
  std::queue<int> reusable_ids;
  std::atomic<int> next_task_id{0};
};

} // namespace nntrainer

#endif /** __TASK_EXECUTOR_H__ */
