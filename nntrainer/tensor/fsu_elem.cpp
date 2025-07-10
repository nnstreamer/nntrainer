// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file   fsu_elem.cpp
 * @date   09 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache elem class V2 - for FSU(Flash Storage Utilization)
 *         For now Cache_elem V2 only for Inference Mode
 *
 */
#include <fsu_elem.h>

namespace nntrainer {

void FsuElem::fsuStart() {
  if (fd > 0)
    return;
  fd = open(weight_file_path.c_str(), O_RDWR | O_CREAT, 0666UL);
  NNTR_THROW_IF(fd < 0, std::runtime_error)
    << "[CacheElemV2] File to open file : " << weight_file_path;
}

void FsuElem::fsuFinish() {
  if (fd < 0) {
    return;
  }
  close(fd);
  fd = -1;
}

void FsuElem::loadFromFile(std::vector<unsigned int> ids) {
  NNTR_THROW_IF(fd <= 0, std::runtime_error)
    << "SwapDevice: Device is not started";

#if defined(_WIN32)
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  auto page_size = sysInfo.dwAllocationGranularity;
#else
  auto page_size = sysconf(_SC_PAGE_SIZE);
#endif

  auto total_len = 0;
  auto start_offset = elements[ids.front()].start_offset;
  for (auto id : ids) {
    total_len += elements[id].weight_len;
  }

  size_t off = (start_offset / page_size) * page_size;
  size_t diff = start_offset - off;
  size_t len = total_len + diff;

  char *ptr = static_cast<char *>(
    mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, off));

  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  NNTR_THROW_IF(ptr == MAP_FAILED, std::runtime_error)
    << "SwapDevice: mmap: " << SAFE_STRERROR(errno, error_buf, error_buflen);

  char *real_start_ptr = static_cast<char *>(ptr + diff);

  // memcpy
  size_t offset_sum = 0;
  for (auto id : ids) {
    void *now_ptr = static_cast<void *>(real_start_ptr + offset_sum);

    memcpy(elements[id].memory_ptr, now_ptr, elements[id].weight_len);
    offset_sum += elements[id].weight_len;

    elements[id].mem_data->setAddr((void *)elements[id].memory_ptr);
    elements[id].mem_data->setValid(true);
    elements[id].active = true;
  }

  const auto ret = munmap(ptr, len);
  NNTR_THROW_IF(ret == -1, std::runtime_error)
    << "SwapDevice: munmap: " << SAFE_STRERROR(errno, error_buf, error_buflen);
}

} // namespace nntrainer
