// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   swap_device.cpp
 * @date   01 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Swap device class implementation
 *
 */

#include <cstring>
#include <fcntl.h>
#include <malloc.h>
#include <profiler.h>
#include <stdlib.h>
#include <sys/types.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <swap_device.h>

namespace nntrainer {

void SwapDevice::start(size_t size) {
  if (fd > 0)
    return;

  fd = open(dev_path.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_SYNC, 0666UL);

  NNTR_THROW_IF(fd < 0, std::runtime_error)
    << "SwapDevice: open file: " << dev_path;

  off_t off;

  /* make sparse file */
  off = lseek(fd, size - 1, SEEK_SET);
  NNTR_THROW_IF(off < 0, std::runtime_error)
    << "SwapDevice: seek file: " << dev_path;

  ssize_t len;
  len = write(fd, "", 1);
  NNTR_THROW_IF(len != 1, std::runtime_error)
    << "SwapDevice: write file: " << dev_path;

  off = lseek(fd, 0, SEEK_SET);
  NNTR_THROW_IF(off < 0, std::runtime_error)
    << "SwapDevice: seek file: " << dev_path;
}

void *SwapDevice::getBuffer(off_t offset, size_t size, void *memory_ptr,
                            bool alloc_only) {
  NNTR_THROW_IF(fd <= 0, std::runtime_error)
    << "SwapDevice: Device is not started";

#ifdef USE_MMAP
  // page aligned
  size_t off = (offset / sysconf(_SC_PAGE_SIZE)) * sysconf(_SC_PAGE_SIZE);
  size_t diff = offset - off;
  size_t len = size + diff;

  const size_t error_buflen = 100;
  char error_buf[error_buflen];

#ifdef ENABLE_QNN
  void *ptr = mmap(memory_ptr, len, PROT_READ | PROT_WRITE,
                   MAP_SHARED | MAP_FIXED, fd, off);

  NNTR_THROW_IF(ptr == (void *)-1, std::runtime_error)
    << "SwapDevice: mmap: "
    << std::string(strerror_r(errno, error_buf, error_buflen));

  mapped[memory_ptr] = std::make_tuple(memory_ptr, len, offset, (ssize_t)size);

  ++num_loaded_tensors;

  return memory_ptr;
#else
  char *ptr = static_cast<char *>(
    mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, off));

  NNTR_THROW_IF(ptr == (void *)-1, std::runtime_error)
    << "SwapDevice: mmap: "
    << std::string(strerror_r(errno, error_buf, error_buflen));

  void *buf = static_cast<void *>(ptr + diff);
  mapped[buf] = std::make_tuple(ptr, len, offset, (ssize_t)size);

  ++num_loaded_tensors;

  return buf;
#endif

#else
  off_t off;
  ssize_t len;
  void *ptr;

  ptr = calloc(1, size);
  NNTR_THROW_IF(ptr == NULL, std::runtime_error)
    << "SwapDevice: memory alloc failed";

  if (!alloc_only) {
    off = lseek(fd, offset, SEEK_SET);
    NNTR_THROW_IF(off < 0, std::runtime_error)
      << "SwapDevice: seek file: " << dev_path;

    len = read(fd, ptr, size);
    NNTR_THROW_IF(len != (size_t)size, std::runtime_error)
      << "SwapDevice: read file: " << dev_path;
  }

  allocated[ptr] = std::make_pair(offset, (ssize_t)size);

  ++num_loaded_tensors;

  return ptr;
#endif
}

void SwapDevice::putBuffer(void *ptr, bool dealloc_only) {
  NNTR_THROW_IF(fd <= 0, std::runtime_error)
    << "SwapDevice: Device is not started";
#ifdef USE_MMAP
  int ret;

  NNTR_THROW_IF(mapped.find(ptr) == mapped.end(), std::runtime_error)
    << "Couldn't find buffer";

  off_t off;
  ssize_t len;

  auto info = mapped[ptr];
  if (!dealloc_only) {
    off = lseek(fd, std::get<2>(info), SEEK_SET);
    NNTR_THROW_IF(off < 0, std::runtime_error)
      << "SwapDevice: seek file: " << dev_path;

    ssize_t size = std::get<3>(info);
    len = write(fd, ptr, size);
    NNTR_THROW_IF(len != size, std::runtime_error)
      << "SwapDevice: write file: " << len << "::" << std::to_string(size)
      << dev_path;
  }

  ret = munmap(std::get<void *>(info), std::get<size_t>(info));
  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  NNTR_THROW_IF(ret == -1, std::runtime_error)
    << "SwapDevice: munmap: "
    << std::string(strerror_r(errno, error_buf, error_buflen));

  mapped.erase(ptr);

#ifndef __ANDROID__
  madvise(std::get<void *>(info), std::get<size_t>(info), MADV_FREE);
#endif

#else
  off_t off;
  ssize_t len;

  NNTR_THROW_IF(allocated.find(ptr) == allocated.end(), std::invalid_argument)
    << "SwapDevice: Couldn't find buffer";

  auto [offset, size] = allocated[ptr];

  if (!dealloc_only) {
    off = lseek(fd, offset, SEEK_SET);
    NNTR_THROW_IF(off < 0, std::runtime_error)
      << "SwapDevice: seek file: " << dev_path;

    len = write(fd, ptr, size);
    NNTR_THROW_IF(len != size, std::runtime_error)
      << "SwapDevice: write file: " << dev_path;
  }

  free(ptr);
  allocated.erase(ptr);

#if !defined(__ANDROID__) && !defined(_WIN32)
  malloc_trim(0);
#endif

#endif
  --num_loaded_tensors;
}

unsigned int SwapDevice::getNumLoadedTensors() { return num_loaded_tensors; }

/**
 * @brief Close device
 *
 */
void SwapDevice::finish() {
  if (fd < 0)
    return;

#ifdef USE_MMAP
  for (auto &[ptr, info] : mapped)
    free(ptr);
  mapped.clear();
#else
  for (auto &alloc : allocated)
    free(alloc.first);
  allocated.clear();
#endif

  close(fd);
  fd = -1;
  int status = std::remove(dev_path.c_str());

  NNTR_THROW_IF(status, std::runtime_error)
    << "SwapDevice: Couldn't remove " << dev_path.c_str();
}

} // namespace nntrainer
