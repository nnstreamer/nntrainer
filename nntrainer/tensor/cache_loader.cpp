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
  unload_task_executor(nullptr) {}

CacheLoader::~CacheLoader() {
  if (load_task_executor)
    delete load_task_executor;
  if (unload_task_executor)
    delete unload_task_executor;
}

void CacheLoader::init() {
  if (load_task_executor == nullptr)
    load_task_executor = new TaskExecutor("loadPool", 2);
  if (unload_task_executor == nullptr)
    unload_task_executor = new TaskExecutor("UnloadPool", 2);
}

void CacheLoader::finish() {
  delete load_task_executor;
  load_task_executor = nullptr;
  delete unload_task_executor;
  unload_task_executor = nullptr;
}

void CacheLoader::load(unsigned int order) { loadAllinOrder(order); }

bool CacheLoader::loadAllinOrder(unsigned int order) {
  if (!load_task_executor) {
    ml_loge("init is needed");
    return false;
  }

  std::set<unsigned int> exec_id = pool->getExecIDs(order);

  for (auto &id : exec_id) {
    loadTensor(id);
  }

  return true;
}

int CacheLoader::loadTensor(unsigned int id) {
  if (!load_task_executor) {
    ml_loge("init is needed");
    return ML_ERROR_INVALID_PARAMETER;
  }
  checkUnloadComplete(id);

  std::lock_guard<std::mutex> lock(state_mutex);

  if (states[id] == LoadState::Loading || states[id] == LoadState::Loaded)
    return -1;

  states[id] = LoadState::Loading;

  int load_task_id = load_task_executor->submit(
    [this, id](void *data) {
      pool->loadTensor(id);
      std::lock_guard<std::mutex> lock(this->state_mutex);
      this->states[id] = LoadState::Loaded;
    },
    (void *)(std::uintptr_t)id);

  pool->getCacheElem(id)->setLoadTaskID(load_task_id);

  return load_task_id;
}

bool CacheLoader::unloadAllinOrder(unsigned int order) {
  if (!load_task_executor) {
    ml_loge("init is needed");
    return false;
  }

  std::set<unsigned int> exec_id = pool->getExecIDs(order);

  for (auto &id : exec_id) {
    unloadTensor(id);
  }

  return true;
}

int CacheLoader::unloadTensor(unsigned int id) {
  if (!load_task_executor) {
    ml_loge("init is needed");
    return ML_ERROR_INVALID_PARAMETER;
  }

  checkLoadComplete(id);

  std::lock_guard<std::mutex> lock(state_mutex);

  if (states[id] != LoadState::Loaded)
    return -1;

  states[id] = LoadState::Unloading;

  int unload_task_id = load_task_executor->submit(
    [this, id](void *data) {
      pool->unloadTensor(id);
      std::lock_guard<std::mutex> lock(this->state_mutex);
      this->states[id] = LoadState::Idle;
    },
    (void *)(std::uintptr_t)id);

  pool->getCacheElem(id)->setUnloadTaskID(unload_task_id);
  return unload_task_id;
}

LoadState CacheLoader::getState(int id) const {
  std::lock_guard<std::mutex> lock(state_mutex);
  auto it = states.find(id);
  if (it == states.end())
    return LoadState::Idle;
  return it->second;
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

  std::set<unsigned int> exec_id = pool->getExecIDs(order);

  for (auto &id : exec_id) {
    unloadTensor(id);
  }

  return 0;
}
void CacheLoader::flush() {
  std::list<std::shared_ptr<CacheElem>> actives = pool->getActiveElems();

  for (auto &elem : actives) {
    auto id = elem->getId();
    unloadTensor(id);
  }

  for (auto &elem : actives) {
    auto id = elem->getId();
    checkUnloadComplete(id);
  }

  pool->flush();
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

unsigned int CacheLoader::inActive(unsigned int order) {
  std::set<unsigned int> exec_id = pool->getExecIDs(order);
  for (auto &id : exec_id) {
    std::shared_ptr<CacheElem> elem = pool->getCacheElem(id);
    std::list<std::shared_ptr<CacheElem>> actives = pool->getActiveElems();
    int load_task_id = elem->getLoadTaskID();
    if (load_task_id >= 0) {
      load_task_executor->releaseTask(load_task_id);
      elem->setLoadTaskID(-1);
      states[id] = LoadState::Unloading;
    }
    actives.remove(elem);
    elem->inActive();
  }
  return 0;
}

bool CacheLoader::checkAllLoadComplete(unsigned int order) {

  std::set<unsigned int> exec_id = pool->getExecIDs(order);

  for (auto &id : exec_id) {
    checkLoadComplete(id);
  }
  return true;
}

bool CacheLoader::checkAllUnloadComplete(unsigned int order) {

  std::set<unsigned int> exec_id = pool->getExecIDs(order);

  for (auto &id : exec_id) {
    checkUnloadComplete(id);
  }
  return true;
}

bool CacheLoader::checkLoadComplete(unsigned int id) {
  std::shared_ptr<CacheElem> elem = pool->getCacheElem(id);
  int unload_task_id = elem->getUnloadTaskID();
  int load_task_id = elem->getLoadTaskID();
  if (unload_task_id >= 0) {
    load_task_executor->releaseTask(unload_task_id);
    elem->setUnloadTaskID(-1);
  }

  if (load_task_id >= 0) {
    load_task_executor->wait(load_task_id);
  }

  return true;
}

bool CacheLoader::checkUnloadComplete(unsigned int id) {
  std::shared_ptr<CacheElem> elem = pool->getCacheElem(id);
  int unload_task_id = elem->getUnloadTaskID();
  int load_task_id = elem->getLoadTaskID();
  if (load_task_id >= 0) {
    load_task_executor->releaseTask(load_task_id);
    elem->setLoadTaskID(-1);
  }
  if (unload_task_id >= 0) {
    load_task_executor->wait(unload_task_id);
  }
  return true;
}

} // namespace nntrainer
