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

#ifndef FSU_WEIGHT_POOL_H
#define FSU_WEIGHT_POOL_H

#include <cache_loader.h>
#include <task_executor.h>
#include <memory_data.h>
#include <memory_pool.h>

#include <future>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

#if defined(_WIN32)
#include "utils/mman_windows.h"
#include <io.h>
using ssize_t = std::make_signed_t<size_t>;
#define O_SYNC 0UL
#else
#include <sys/mman.h>
#include <unistd.h>
#endif


namespace nntrainer {
/**
 * @brief FSU Element information for process load
 *
 */
struct ElementInfo {
  unsigned int id{};
  void *memory_ptr{};
  bool active{};
  size_t start_offset{};
  size_t weight_len{};
  std::shared_ptr<MemoryData> mem_data;
  int load_task_id{};
  LoadState load_state;
};

/**
 * @class   FsuWeightPool
 * @brief   Flash Storage Utilization Pool
 */
class FsuWeightPool : public MemoryPool {

public:
  using ExecIds = std::set<unsigned int>;

  explicit FsuWeightPool();

  ~FsuWeightPool() override;

  void weightFileOpen();

  void weightFileClose();

  void allocate() override;

  void deallocate() override;

  void inActive(unsigned int order);

  unsigned int requestMemory(
    size_t bytes, unsigned int start_time, unsigned int end_time,
    std::vector<unsigned int> exec_order = std::vector<unsigned int>(),
    TensorLifespan lifespan = TensorLifespan::MAX_LIFESPAN,
    bool is_wgrad = false) override;

  std::shared_ptr<MemoryData> getMemory(unsigned int id) override;

  void clear() override;

  void loadTensor(unsigned int order);

  void setFsuWeightPath(std::string path) override {
    weight_file_path = path;
    weightFileOpen();
  }

  bool checkAllLoadComplete(unsigned int order);
  void loadFromFile(std::vector<unsigned int> ids);
  bool loadAllinOrder(unsigned int order);
  void setWeightOffset(std::vector<std::pair<size_t, size_t>> offsets);
protected:
  void validate(unsigned int id);

private:
  std::string weight_file_path; /**< Weight file path */
  int fd;                       /**< File fd */
  std::unordered_map<unsigned int, ElementInfo>
    elements;                         /**< Element info dict */
  TaskExecutor *load_task_executor;   /**< task executor */
  std::unordered_map<unsigned int, ExecIds> order_to_exec_ids;
  std::vector<unsigned int> id_bank;
  unsigned int max_exec_id;
  unsigned int load_batch_size;

  std::mutex state_mutex;
  std::mutex id_bank_mutex;
};
} // namespace nntrainer
#endif // FSU_WEIGHT_POOL_H
