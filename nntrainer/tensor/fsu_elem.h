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

/**
 * @brief FSU Element information for process load
 *
 */
typedef struct ElementInfo {
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
  /**
   * @brief FsuElem Constructor
   */
  explicit FsuElem() : load_task_id(-1), unload_task_id(-1) {}

  /**
   * @brief FsuElem Destructor
   */
  ~FsuElem() = default;

  /**
   * @brief fsuStart : open fd with given file_path
   */
  void fsuStart();

  /**
   * @brief fsuFinish : close fd and release memory mapped buffer
   */
  void fsuFinish();

  /**
   * @brief Load Weight from given wieght file path
   *
   * @param ids requested weight's id
   */
  void loadFromFile(std::vector<unsigned int> ids);

  /**
   * @brief get element active status
   *
   * @param id weight id
   * @return active status
   */
  bool isActive(unsigned int id) { return elements[id].active; }

  /**
   * @brief set element's active to false
   *
   * @param id weight id
   */
  void inActive(unsigned int id) { elements[id].active = false; }

  /**
   * @brief set LoadTaskID
   *
   * @param id weight id
   */
  void setLoadTaskID(int id) { load_task_id = id; }

  /**
   * @brief get LoadTaskID
   *
   * @return LoadTaskID
   */
  int getLoadTaskID() { return load_task_id; }

  /**
   * @brief set UnloadTaskID
   *
   * @param id weight id
   */
  void setUnloadTaskID(int id) { unload_task_id = id; }

  /**
   * @brief get UnLoadTaskID
   *
   * @return unload_task_id
   */
  int getUnloadTaskID() { return unload_task_id; }

  /**
   * @brief get weight_file_path
   *
   * @return weight_file_path
   */
  const std::string getWeightFilePath() const { return weight_file_path; }

  /**
   * @brief setWeightFilePath
   *
   * @param file_path file_path that contains weights to be loaded into memory
   */
  void setWeightFilePath(std::string file_path) {
    weight_file_path = file_path;
  }

  /**
   * @brief set {start_off, weight_len} to elements[id]
   *
   * @param id weight id
   * @param offset {start_off, weight_len}
   */
  void setWeightOffset(unsigned int id,
                       const std::pair<size_t, size_t> &offset) {
    elements[id].start_offset = offset.first;
    elements[id].weight_len = offset.second;
  }

private:
  int load_task_id;   /**< load task id */
  int unload_task_id; /**< unload task id */
  std::unordered_map<unsigned int, ElementInfo>
    elements;                   /**< Element info dict */
  std::string weight_file_path; /**< Weight file path */
  int fd;                       /**< File fd */
};
} // namespace nntrainer
#endif // CACHE_ELEM_V2_H
