// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file   fsu_elem.h
 * @date   09 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache elem class V2 - for FSU(Flash Storage Utilization)
 *         For now Cache_elem V2 only for Inference Mode
 *
 */

#ifndef FSU_ELEM_H
#define FSU_ELEM_H
#include <memory_data.h>
#include <swap_device.h>

#include <memory>
#include <shared_mutex>
#include <string>
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

struct ElementInfo {
  unsigned int id;
  void *memory_ptr;
  bool active;
  size_t start_offset;
  size_t weight_len;
  std::shared_ptr<MemoryData> mem_data;
};

/**
 * @class   FsuElem
 * @brief   Fsu element
 */
class FsuElem {
public:
  explicit FsuElem() : load_task_id(-1), unload_task_id(-1) {}

   ~FsuElem() = default;

  // Cache Elem Functions
  void fsuStart();
  void fsuFinish();

  void loadFromFile(std::vector<unsigned int> ids);

  bool isActive(unsigned int id) { return elements[id].active; }

  void inActive(unsigned int id) { elements[id].active = false; }

  void setLoadTaskID(int id) { load_task_id = id; }

  int getLoadTaskID() { return load_task_id; }

  void setUnloadTaskID(int id) { unload_task_id = id; }

  int getUnloadTaskID() { return unload_task_id; }

  // swap_device functions


 void getBuffer(std::vector<unsigned int> ids);

  const std::string getDevicePath() const { return dev_path; }

  void setDevicePath(std::string file_path) { dev_path = file_path; }

  void setWeightOffset(unsigned int id,
                       const std::pair<size_t, size_t> &offset) {
    elements[id].start_offset = offset.first;
    elements[id].weight_len = offset.second;
  }

private:
  int load_task_id;
  int unload_task_id;
  std::unordered_map<unsigned int, ElementInfo> elements;

  // Swap Device's private variable
  std::string dev_path;
  int fd;
};
} // namespace nntrainer
#endif // CACHE_ELEM_V2_H
