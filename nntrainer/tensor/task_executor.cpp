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

#include <algorithm>
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

std::atomic_int32_t TaskExecutor::ids(1);

TaskExecutor::TaskExecutor(const std::string &n) :
  name(n), run_thread(true), wait_complete(false) {
  task_thread = std::thread([&]() {
    ml_logd("Task Thread(%s): start thread", name.c_str());
    while (run_thread) {
      std::unique_lock lk(task_mutex);
      if (!task_cv.wait_for(lk, std::chrono::milliseconds(500),
                            [&] { return !task_queue.empty(); }))
        continue;

      auto &task_info = task_queue.front();
      lk.unlock();

      const auto &id = std::get<int>(task_info);
      const auto &callback = std::get<CompleteCallback>(task_info);

      auto status = worker(task_info);
      callback(id, status);

      lk.lock();
      task_queue.pop_front();
      lk.unlock();
    }
    ml_logd("Task Thread(%s): finish thread", name.c_str());
  });
}

TaskExecutor::~TaskExecutor() {
  run_thread = false;

  task_thread.join();
}

int TaskExecutor::run(std::shared_ptr<Task> task) {
  auto work = task->getWork();
  std::atomic_bool running(true);
  return work(running, task->getData());
}

void TaskExecutor::cancel(int id) {
  std::scoped_lock lock(task_mutex);

  auto it = std::find_if(task_queue.begin(), task_queue.end(),
                         [&](auto &info) { return std::get<int>(info) == id; });

  if (it != task_queue.end())
    std::get<std::atomic_bool>(*it).store(false);
}

void TaskExecutor::cancelAll(void) {
  for (auto &task_info : task_queue) {
    std::get<std::atomic_bool>(task_info).store(false);
  }
}

void TaskExecutor::clean(void) {
  for (auto it = task_queue.begin(); it != task_queue.end();) {
    auto running = std::get<std::atomic_bool>(*it).load();
    if (running == false)
      it = task_queue.erase(it);
    else
      it++;
  }
}

TaskExecutor::CompleteStatus TaskExecutor::handleWork(std::atomic_bool &running,
                                                      Task::Work &work,
                                                      void *data) {
  CompleteStatus status = CompleteStatus::SUCCESS;

  try {
    int ret = work(running, data);
    if (ret != 0)
      status = CompleteStatus::FAIL_CANCEL;
  } catch (const std::exception &e) {
    ml_loge("TaskExecutor(%s): failed while running task: %s", name.c_str(),
            e.what());
    status = CompleteStatus::FAIL;
  }

  return status;
}

} // namespace nntrainer
