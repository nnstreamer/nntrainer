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

#include <malloc.h>
#include <profiler.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <unistd.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <swap_device.h>

namespace nntrainer {

void SwapDevice::start(size_t size) {
  int ret;

  if (fd > 0)
    return;

  fd = open(dev_path.c_str(), O_RDWR | O_CREAT | O_TRUNC, (mode_t)0666);
  NNTR_THROW_IF(fd < 0, std::runtime_error) << "open file: " << dev_path;

  /* make sparse file */
  ret = lseek(fd, size - 1, SEEK_SET);
  NNTR_THROW_IF(ret < 0, std::runtime_error) << "seek file: " << dev_path;

  ret = write(fd, "", 1);
  NNTR_THROW_IF(ret != 1, std::runtime_error) << "write file: " << dev_path;

  ret = lseek(fd, 0, SEEK_SET);
  NNTR_THROW_IF(ret < 0, std::runtime_error) << "seek file: " << dev_path;
}

void *SwapDevice::getBuffer(int offset, size_t size) {
  NNTR_THROW_IF(fd <= 0, std::runtime_error) << "SwapDevice is not started";

#ifdef USE_MMAP
  // page aligned
  off_t off = (offset / sysconf(_SC_PAGE_SIZE)) * sysconf(_SC_PAGE_SIZE);
  int diff = (off_t)offset - off;
  size_t len = size + diff;

  char *ptr = static_cast<char *>(
    mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, off));
  NNTR_THROW_IF(ptr == (void *)-1, std::runtime_error)
    << "mmap: " << std::string(strerror(errno));

  void *buf = static_cast<void *>(ptr + diff);
  mapped[buf] = std::make_pair(ptr, len);

  return buf;
#else
  int ret;
  ssize_t len;
  void *ptr;

  ptr = calloc(1, size);
  NNTR_THROW_IF(ptr == NULL, std::runtime_error) << "memory alloc failed";

  ret = lseek(fd, offset, SEEK_SET);
  NNTR_THROW_IF(ret < 0, std::runtime_error) << "seek file: " << dev_path;

  len = read(fd, ptr, size);
  NNTR_THROW_IF(len != (ssize_t)size, std::runtime_error)
    << "read file: " << dev_path;

  allocated[ptr] = std::make_pair(offset, (ssize_t)size);

  return ptr;
#endif
}

void SwapDevice::putBuffer(void *ptr) {
  int ret;

  NNTR_THROW_IF(fd <= 0, std::runtime_error) << "SwapDevice is not started";
#ifdef USE_MMAP
  NNTR_THROW_IF(mapped.find(ptr) == mapped.end(), std::runtime_error)
    << "Couldn't find buffer";

  auto info = mapped[ptr];
  ret = munmap(std::get<void *>(info), std::get<size_t>(info));
  NNTR_THROW_IF(ret == -1, std::runtime_error)
    << "munmap: " << std::string(strerror(errno));

  mapped.erase(ptr);

#ifndef __ANDROID__
  madvise(std::get<void *>(info), std::get<size_t>(info), MADV_FREE);
#endif

#else
  NNTR_THROW_IF(allocated.find(ptr) == allocated.end(), std::invalid_argument)
    << "Couldn't find buffer";

  auto [offset, size] = allocated[ptr];

  ret = lseek(fd, offset, SEEK_SET);
  NNTR_THROW_IF(ret < 0, std::runtime_error) << "seek file: " << dev_path;

  ret = write(fd, ptr, size);
  NNTR_THROW_IF(ret != size, std::runtime_error) << "write file: " << dev_path;

  free(ptr);
  allocated.erase(ptr);

#ifndef __ANDROID__
  malloc_trim(0);
#endif

#endif
}

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
    << "Couldn't remove " << dev_path.c_str();
}

} // namespace nntrainer
