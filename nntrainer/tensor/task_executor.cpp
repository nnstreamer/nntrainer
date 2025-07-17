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

#include "task_executor.h"

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

TaskExecutor::TaskExecutor(std::string n, size_t thread_count) :
  name(n), stop(false) {
  for (size_t i = 0; i < thread_count; ++i) {
    workers.emplace_back([this] { this->worker_thread(); });
  }
}

TaskExecutor::~TaskExecutor() {
  {
    std::unique_lock<std::mutex> lock(queue_mutex);
    stop = true;
  }

  cond_var.notify_all();
  for (std::thread &t : workers) {
    if (t.joinable())
      t.join();
  }
}

void TaskExecutor::worker_thread() {

  while (true) {
    Task task;
    {
      std::unique_lock<std::mutex> lock(queue_mutex);
      cond_var.wait(lock, [this]() { return stop || !task_queue.empty(); });

      if (stop && task_queue.empty()) {
        return;
      }

      task = std::move(task_queue.front());
      task_queue.pop();
      task_started[task.id] = true;
      task_started_cv.notify_all();

      // we are not going to remove the Done Tasks.
      // we exeplicitly call release tasks. until then, we keep the results and
      // not going to submit that task again
      // queued_ids.erase(task.id);
    }

    try {
      task.callback(task.data);
      task.promise.set_value();
    } catch (...) {
      ml_loge("[%s] : [Error ] Task ID %d threw an exception\n", name.c_str(),
              task.id);
    }
  }
}

int TaskExecutor::submit(TaskCallback cb, void *data) {

  auto canceled = std::make_shared<std::atomic_bool>(false);
  auto promise = std::make_shared<std::promise<void>>();
  std::shared_future<void> fut = promise->get_future().share();
  int id = getNextTaskId();

  {
    std::lock_guard<std::mutex> lock(queue_mutex);

    if (future_map.count(id)) {
      if (!future_map[id].valid()) {
        ml_loge("[%s] : [Error] Future is not valid : Task id - %d\n",
                name.c_str(), id);
      }
      auto status = future_map[id].wait_for(std::chrono::seconds(0));
      if (status != std::future_status::ready) {
        ml_logi("[%s] : Task id - %d is still active\n", name.c_str(), id);
        return id;
      }
    }

    Task task{id, std::move(*promise), cb, data};

    future_map[id] = fut;

    task_queue.push(std::move(task));
  }
  cond_var.notify_one();
  return id;
}

void TaskExecutor::submitTasks(const std::vector<TaskDesc> &tasks) {
  for (const auto &task : tasks) {
    submit(task.callback, task.data);
  }
}

bool TaskExecutor::cancel(int id) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  auto it = cancel_map.find(id);
  if (it != cancel_map.end()) {
    *(it->second) = true;
    return true;
  }
  return false;
}

void TaskExecutor::wait(int id) {
  std::shared_future<void> fut;
  {
    std::unique_lock<std::mutex> lock(queue_mutex);

    task_started_cv.wait(
      lock, [&] { return task_started.count(id) && task_started[id]; });

    auto it = future_map.find(id);
    if (it == future_map.end() || !it->second.valid()) {
      return;
    }
    fut = it->second;
  }
  try {
    fut.wait();
  } catch (const std::future_error &e) {
    ml_loge("[%s] : exception while waiting on future : %s\n", name.c_str(),
            e.what());
  }
}

void TaskExecutor::waitAll(const std::vector<int> &ids) {
  std::vector<std::shared_future<void>> futures;
  {
    std::lock_guard<std::mutex> lock(queue_mutex);
    for (int id : ids) {
      auto it = future_map.find(id);
      if (it != future_map.end()) {
        futures.push_back(it->second);
      } else {
        ml_logw("[%s] : Task ID is not found : %d\n", name.c_str(), id);
      }
    }
  }

  for (auto &fut : futures) {
    try {
      fut.wait();
    } catch (const std::exception &e) {
      ml_loge("[%s] : exception while waiting on future : %s\n", name.c_str(),
              e.what());
    }
  }
}

bool TaskExecutor::isDone(int id) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  auto it = future_map.find(id);
  if (it == future_map.end())
    return false;
  return it->second.wait_for(std::chrono::seconds(0)) ==
         std::future_status::ready;
}

bool TaskExecutor::isAllDone(const std::vector<int> &ids) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  for (int id : ids) {
    isDone(id);
  }
  return true;
}

void TaskExecutor::releaseTask(int id) {
  std::lock_guard<std::mutex> lock(queue_mutex);
  future_map.erase(id);
  cancel_map.erase(id);
  reusable_ids.push(id);
}

} // namespace nntrainer
