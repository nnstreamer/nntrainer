// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   task_executor.cpp
 * @date   04 Nov 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Task executor class
 *
 */

#include "task_executor.h"

#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <tuple>

#include <nntrainer_error.h>
#include <nntrainer_log.h>

namespace nntrainer {

int TaskExecutor::run(std::shared_ptr<Task> task) {
  auto work = task->getWork();
  std::atomic_bool running(true);
  return work(running, task->getData());
}

void TaskExecutor::cancel(int id) {
  NNTR_THROW_IF(tasks.find(id) == tasks.end(), std::invalid_argument)
    << "there is no task id: " << id;

  std::get<std::atomic_bool>(tasks[id]).store(false);
}

void TaskExecutor::cancelAll(void) {
  for (auto &[id, task_data] : tasks) {
    std::get<std::atomic_bool>(task_data).store(false);
  }
}

void TaskExecutor::clean(void) {
  for (auto it = tasks.begin(); it != tasks.end();) {
    auto running = std::get<std::atomic_bool>(it->second).load();
    if (running == false)
      tasks.erase(it);
    it++;
  }
}

void TaskExecutor::handleWork(int id, Task::Work &work, void *data) {
  CompleteStatus status = CompleteStatus::SUCCESS;

  try {
    int ret = work(std::get<std::atomic_bool>(tasks[id]), data);
    if (ret != 0)
      status = CompleteStatus::FAIL_CANCEL;
  } catch (const std::exception &e) {
    ml_loge("AsyncTask(%d): failed while running task: %s", id, e.what());
    status = CompleteStatus::FAIL;
  }

  std::get<CompleteStatus>(tasks[id]) = status;
}

void TaskExecutor::handleCompleteStatus(int id,
                                        const std::future_status status) {
  if (status != std::future_status::ready) {
    ml_loge("Task(%d): timeout reached", id);

    cancel(id);

    /* wait for cancel is complete */
    auto status_cancel =
      std::get<0>(tasks[id]).wait_for(std::chrono::seconds(1));
    if (status_cancel != std::future_status::ready) {
      ml_loge(
        "AsyncTask(%d): fatal problem. cancel does not work. please check", id);
      throw std::runtime_error("cancel does not work");
    }

    std::get<CompleteStatus>(tasks[id]) = CompleteStatus::FAIL_TIMEOUT;
  }

  try {
    auto callback = std::get<CompleteCallback>(tasks[id]);
    auto complete_status = std::get<CompleteStatus>(tasks[id]);
    callback(id, complete_status);
  } catch (const std::exception &e) {
    ml_loge("AsyncTask(%d): failed while processing user callback: %s", id,
            e.what());
  }

  std::get<std::atomic_bool>(tasks[id]).store(false);
}

} // namespace nntrainer
