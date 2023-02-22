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
  task_executor(nullptr) {}

CacheLoader::~CacheLoader() {
  if (task_executor)
    delete task_executor;
}

void CacheLoader::init() {
  if (task_executor)
    return;

  task_executor = new TaskExecutor(pool->getName());
}

void CacheLoader::finish() {
  delete task_executor;
  task_executor = nullptr;
}

void CacheLoader::load(unsigned int order) { pool->loadExec(order); }

int CacheLoader::loadAsync(unsigned int order,
                           TaskExecutor::CompleteCallback complete) {
  return loadAsync(order, complete, LONG_MAX);
}

int CacheLoader::loadAsync(unsigned int order,
                           TaskExecutor::CompleteCallback complete,
                           long timeout_ms) {
  if (!task_executor) {
    ml_loge("init is needed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  Task::Work work = [&](std::atomic_bool &running, void *data) {
    unsigned int exe_order = (unsigned int)(std::uintptr_t)data;

    pool->flushExcept({exe_order - 1, exe_order});
    pool->loadExec(exe_order);

    return ML_ERROR_NONE;
  };

  auto task =
    std::make_shared<TaskAsync<>>(work, (void *)(std::uintptr_t)order);
  task->setTimeout(timeout_ms);

  return task_executor->run(task, complete);
}

int CacheLoader::cancelAsync(int id) {
  try {
    task_executor->cancel(id);
  } catch (const std::exception &e) {
    ml_loge("CacheLoader(%s): failed to cancel(%d): %s",
            pool->getName().c_str(), id, e.what());
    return ML_ERROR_UNKNOWN;
  }

  return ML_ERROR_NONE;
}

} // namespace nntrainer
