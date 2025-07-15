// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file   fsu_weight_pool.h
 * @date   10 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  FSU Weight Pool
 *
 */

#include <fsu_weight_pool.h>

namespace nntrainer {

FsuWeightPool::FsuWeightPool() : fd(-1), load_batch_size(1) {
  printf("FsuWeightPool created \n");
  load_task_executor = new TaskExecutor("loadPool", 8);
}

FsuWeightPool::~FsuWeightPool() {
  try {
    FsuWeightPool::deallocate();
  } catch (...) {
    ml_loge("Failed deallocate");
  }
  if (load_task_executor) {
    delete load_task_executor;
    load_task_executor = nullptr;
  }
}

void FsuWeightPool::weightFileOpen() {
  if (fd > 0)
    return;
  fd = open(weight_file_path.c_str(), O_RDWR | O_CREAT, 0666UL);
  NNTR_THROW_IF(fd < 0, std::runtime_error)
    << "[FSU_ELEM] Open file Failed : " << weight_file_path;
}

void FsuWeightPool::weightFileClose() {
  if (fd < 0) {
    return;
  }
  close(fd);
  fd = -1;
}

void FsuWeightPool::setWeightOffset(
  std::vector<std::pair<size_t, size_t>> offsets) {
  int id_idx = 1;
  for (auto element : offsets) {
    elements[id_idx].start_offset = element.first;
    elements[id_idx].weight_len = element.second;
    id_idx++;
  }
}

void FsuWeightPool::allocate() {

  size_t pool_size = size();

  NNTR_THROW_IF(pool_size == 0, std::runtime_error)
    << "Allocating memory pool with size 0";

  MemoryPool::allocateFSU();
}

void FsuWeightPool::deallocate() { MemoryPool::deallocate(); }

unsigned int FsuWeightPool::requestMemory(size_t bytes, unsigned int start_time,
                                          unsigned int end_time,
                                          std::vector<unsigned int> exec_order,
                                          TensorLifespan lifespan,
                                          bool is_wgrad) {
  auto id = MemoryPool::requestMemory(bytes, start_time, end_time, exec_order,
                                      lifespan, is_wgrad);
  return id;
}

std::shared_ptr<MemoryData> FsuWeightPool::getMemory(unsigned int id) {

  auto exe_order = getMemoryExecOrder().at(id - 1);

  void *memory_ptr = nullptr;
  memory_ptr = getMemoryPtrs().at(id - 1);

  auto mem_data = std::make_shared<MemoryData>(
    id, std::bind(&FsuWeightPool::validate, this, std::placeholders::_1),
    nullptr, memory_ptr);
  elements[id] = {id, memory_ptr, false, 0, 0, mem_data, -1, LoadState::Idle};
  auto &o = exe_order[0];
  order_to_exec_ids[o].insert(id);
  max_exec_id = std::max(max_exec_id, id);
  return mem_data;
}

void FsuWeightPool::clear() {
  deallocate();
  MemoryPool::clear();
}

void FsuWeightPool::validate(unsigned int id) {
  // printf("validate start id : %d -- element_info ", id);
  // printf("id : %d, memory_ptr : %x, active : %d, start_offset : %zu, "
         // "weight_len : %zu, load_state %d \n",
         // elements[id].id, elements[id].memory_ptr, elements[id].active,
         // elements[id].start_offset, elements[id].weight_len,
         // elements[id].load_state);
  std::vector<unsigned int> ids_to_load;
  bool should_load = false;
  {
    std::lock_guard<std::mutex> lock(id_bank_mutex);
    if (!elements[id].active) {
      id_bank.push_back(id);
    }

    if (id_bank.size() == load_batch_size || id == max_exec_id) {
      ids_to_load = id_bank;
      id_bank.clear();
      should_load = true;

    }
  }
  if (should_load) {
    loadFromFile(ids_to_load);
  }
}

void FsuWeightPool::loadFromFile(std::vector<unsigned int> ids) {
  NNTR_THROW_IF(fd <= 0, std::runtime_error)
    << "[FSU_ELEM] LoadFromFile failed : Device is not started";
  if (ids.empty()) {
    return;
  }
#if defined(_WIN32)
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  auto page_size = sysInfo.dwAllocationGranularity;
#else
  auto page_size = sysconf(_SC_PAGE_SIZE);
#endif

  auto total_len = 0;
  auto start_offset = elements[ids.front()].start_offset;

  std::vector<size_t> weight_lengths;
  weight_lengths.reserve(ids.size());

  for (auto id : ids) {
    size_t len = elements[id].weight_len;
    weight_lengths.push_back(len);
    total_len += len;
  }

  size_t off = (start_offset / page_size) * page_size;
  size_t diff = start_offset - off;
  size_t len = total_len + diff;
  // printf("mmap | off: %zu, len: %zu\n", off, len);
  char *ptr = static_cast<char *>(
    mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, off));

  std::vector<size_t> cumultative_offsets;
  cumultative_offsets.reserve(ids.size() + 1);

  size_t current_offset = 0;
  cumultative_offsets.push_back(0);
  for (size_t l : weight_lengths) {
    cumultative_offsets.push_back(current_offset);
    current_offset += l;
  }

  unsigned int num_threads = 1;//std::thread::hardware_concurrency();
  num_threads = std::min(num_threads, static_cast<unsigned int>(ids.size()));

  std::vector<std::thread> mem_cpy_threads;
  mem_cpy_threads.reserve(num_threads);

  // memcpy
  int thread_idx = 0;
  for (auto id : ids) {
    thread_idx++;
    mem_cpy_threads.emplace_back(
      [this, &id, &cumultative_offsets, ptr, diff, thread_idx]() {
        void *now_ptr =
          static_cast<void *>(ptr + diff + cumultative_offsets[thread_idx]);
        memcpy(elements[id].memory_ptr, now_ptr, elements[id].weight_len);
        elements[id].mem_data->setAddr((void *)elements[id].memory_ptr);
        elements[id].mem_data->setValid(true);
        elements[id].active = true;

      });
  }

  for (auto &t : mem_cpy_threads) {
    if (t.joinable()) {
      t.join();
    }
  }

  const auto ret = munmap(ptr, len);
}

bool FsuWeightPool::loadAllinOrder(unsigned int order) {
  std::set<unsigned int> exec_ids = order_to_exec_ids[order];
  for (auto &id : exec_ids) {
    {
      std::lock_guard<std::mutex> lock(state_mutex);
      if (elements[id].load_state == LoadState::Loading ||
          elements[id].load_state == LoadState::Loaded) {
        return -1;
      }
      elements[id].load_state = LoadState::Loading;
    }
    int load_task_id_ = load_task_executor->submit(
      [this, id](void *data) {
        this->validate(id);
        std::lock_guard<std::mutex> lock(this->state_mutex);
        this->elements[id].load_state = LoadState::Loaded;
        // printf("memcpy done for [%d] \n", id);
      },
      (void *)(std::uintptr_t)id);

    elements[id].load_task_id = load_task_id_;
  }
  return true;
}

void FsuWeightPool::inActive(unsigned int order) {
  auto exec_ids = order_to_exec_ids[order];

  for (auto &id : exec_ids) {
    // printf(">>>> inActive id : %d\n ", id);
    int load_task_id_ = elements[id].load_task_id;
    if (load_task_id_ >= 0) {
      load_task_executor->releaseTask(load_task_id_);
      elements[id].load_task_id = -1;
      elements[id].load_state = LoadState::Unloading;
      elements[id].active = false;
    }
  }
}

bool FsuWeightPool::checkAllLoadComplete(unsigned int order) {
  std::set<unsigned int> exec_id = order_to_exec_ids[order];

  for (auto &id : exec_id) {
    int load_task_id = elements[id].load_task_id;
    // printf("checkAllLoadComplete id : %d ", id);
    if (load_task_id >= 0) {
      if (elements[id].load_state != LoadState::Loaded) {
        // printf("!!!!!!!!!!!!!!!waiting ~~ id %d\n", id);
        load_task_executor->wait(load_task_id);
      }
    }
  }
  return true;
}

} // namespace nntrainer
