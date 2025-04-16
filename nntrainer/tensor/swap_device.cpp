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

#if defined(_WIN32)
#include <Memoryapi.h>
#include <Sysinfoapi.h>
#endif

namespace nntrainer {

void SwapDevice::start(size_t size, ml::train::ExecutionMode execution_mode) {
  if (fd_ > 0)
    return;

  if (execution_mode == ml::train::ExecutionMode::TRAIN) {
    fd_ = open(dev_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC | O_SYNC, 0666UL);
  } else {
    fd_ = open(dev_path_.c_str(), O_RDWR | O_CREAT, 0666UL);
    execution_mode_ = execution_mode;
  }
  NNTR_THROW_IF(fd_ < 0, std::runtime_error)
    << "SwapDevice: open file: " << dev_path_;

  off_t off;

  /* make sparse file */
  off = lseek(fd_, size - 1, SEEK_SET);
  NNTR_THROW_IF(off < 0, std::runtime_error)
    << "SwapDevice: seek file: " << dev_path_;

  if (execution_mode == ml::train::ExecutionMode::TRAIN) {
    ssize_t len;
    len = write(fd_, "", 1);
    NNTR_THROW_IF(len != 1, std::runtime_error)
      << "SwapDevice: write file: " << dev_path_;
  }
  off = lseek(fd_, 0, SEEK_SET);
  NNTR_THROW_IF(off < 0, std::runtime_error)
    << "SwapDevice: seek file: " << dev_path_;
}

void *SwapDevice::getBuffer(off_t offset, size_t size, void *memory_ptr,
                            bool alloc_only) {
  NNTR_THROW_IF(fd_ <= 0, std::runtime_error)
    << "SwapDevice: Device is not started";

#ifdef USE_MMAP

  if (execution_mode_ == ml::train::ExecutionMode::INFERENCE) {
    // FSU Load Weights
    auto len_offset = weight_offset_.at(offset_index_);
    auto len_offset_aligned = weight_offset_aligned_.at(offset_index_);
    size_t off = len_offset_aligned.first;
    size_t len = len_offset_aligned.second;

    void *ptr =
      mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_SHARED, fd_, off);

    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    NNTR_THROW_IF(ptr == MAP_FAILED, std::runtime_error)
      << "SwapDevice: mmap: " << SAFE_STRERROR(errno, error_buf, error_buflen);

    // MADVISE can be used to improve performance.
    // madvise(ptr, len, MADV_SEQUENTIAL);

    std::memcpy(memory_ptr, ptr, len_offset.second);
    const auto unmap_ret = munmap(ptr, len);

    NNTR_THROW_IF(unmap_ret != 0, std::runtime_error)
      << "SwapDevice: unmmap: "
      << SAFE_STRERROR(errno, error_buf, error_buflen);

    ++offset_index_;
    ++num_loaded_tensors_;

    // @todo : need to check at cache_loader & check multi thread execution
    if (offset_index_ >= (int)weight_offset_.size()) {
      offset_index_ = 0;
    }

    return memory_ptr;
  } else {
    size_t off_alignment = 0;
    size_t len_alignment = 0;
    getAlignments(off_alignment, len_alignment);

    const auto off = alignTo(offset, off_alignment);
    const auto len = alignTo(size, len_alignment);

    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    char *ptr = static_cast<char *>(
      mmap(NULL, len, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd_, off));
    NNTR_THROW_IF(ptr == (void *)-1, std::runtime_error)
      << "SwapDevice: mmap: " << SAFE_STRERROR(errno, error_buf, error_buflen);
    void *buf = static_cast<void *>(ptr + size);
    mapped_[buf] = std::make_tuple(ptr, len, offset, (ssize_t)size);

    ++num_loaded_tensors_;
    return buf;
  }
#else
  off_t off;
  ssize_t len;
  void *ptr;

  ptr = calloc(1, size);
  NNTR_THROW_IF(ptr == NULL, std::runtime_error)
    << "SwapDevice: memory alloc failed";

  if (!alloc_only) {
    off = lseek(fd_, offset, SEEK_SET);
    NNTR_THROW_IF(off < 0, std::runtime_error)
      << "SwapDevice: seek file: " << dev_path_;

    len = read(fd_, ptr, size);
    NNTR_THROW_IF(len != (size_t)size, std::runtime_error)
      << "SwapDevice: read file: " << dev_path_;
  }

  allocated_[ptr] = std::make_pair(offset, (ssize_t)size);

  ++num_loaded_tensors_;

  return ptr;
#endif
}

void SwapDevice::putBuffer(void *ptr, bool dealloc_only) {
  NNTR_THROW_IF(fd_ <= 0, std::runtime_error)
    << "SwapDevice: Device is not started";
#ifdef USE_MMAP
  if (mapped_.size() == 0) {
    return;
  }
  int ret;

  NNTR_THROW_IF(mapped_.find(ptr) == mapped_.end(), std::runtime_error)
    << "Couldn't find buffer";

  off_t off;
  ssize_t len;
  auto info = mapped_[ptr];
  if (!dealloc_only) {
    off = lseek(fd_, std::get<2>(info), SEEK_SET);
    NNTR_THROW_IF(off < 0, std::runtime_error)
      << "SwapDevice: seek file: " << dev_path_;

    ssize_t size = std::get<3>(info);
    len = write(fd_, ptr, size);
    NNTR_THROW_IF(len != size, std::runtime_error)
      << "SwapDevice: write file: " << len << "::" << std::to_string(size)
      << dev_path_;
  }

  ret = munmap(std::get<void *>(info), std::get<size_t>(info));
  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  NNTR_THROW_IF(ret == -1, std::runtime_error)
    << "SwapDevice: munmap: " << SAFE_STRERROR(errno, error_buf, error_buflen);

  mapped_.erase(ptr);

#if !defined(__ANDROID__) && !defined(_WIN32)
  madvise(std::get<void *>(info), std::get<size_t>(info), MADV_FREE);
#endif

#else
  off_t off;
  ssize_t len;

  NNTR_THROW_IF(allocated_.find(ptr) == allocated_.end(), std::invalid_argument)
    << "SwapDevice: Couldn't find buffer";

  auto [offset, size] = allocated_[ptr];

  if (!dealloc_only) {
    off = lseek(fd_, offset, SEEK_SET);
    NNTR_THROW_IF(off < 0, std::runtime_error)
      << "SwapDevice: seek file: " << dev_path_;

    len = write(fd_, ptr, size);
    NNTR_THROW_IF(len != size, std::runtime_error)
      << "SwapDevice: write file: " << dev_path_;
  }

  free(ptr);
  allocated_.erase(ptr);

#if !defined(__ANDROID__) && !defined(_WIN32)
  malloc_trim(0);
#endif

#endif
  --num_loaded_tensors_;
}

/**
 * @brief Close device
 *
 */
void SwapDevice::finish() {
  if (fd_ < 0)
    return;

#ifdef USE_MMAP
  for (auto &[ptr, info] : mapped_) {
    if (ptr)
      free(ptr);
  }
  mapped_.clear();
#else
  for (auto &alloc : allocated_)
    free(alloc.first);
  allocated_.clear();
#endif

  close(fd_);
  fd_ = -1;
  if (execution_mode_ == ml::train::ExecutionMode::TRAIN) {
    int status = std::remove(dev_path_.c_str());
    NNTR_THROW_IF(status, std::runtime_error)
      << "SwapDevice: Couldn't remove " << dev_path_.c_str();
  }
}

void SwapDevice::setWeightOffset(
  const std::vector<std::pair<size_t, size_t>> &weight_offset) {
  weight_offset_ = weight_offset;
  size_t off_alignment = 0;
  size_t len_alignment = 0;
  getAlignments(off_alignment, len_alignment);

  size_t last_offset = 0;
  for (const auto weight : weight_offset_) {
    const auto off = alignTo(last_offset, off_alignment);
    const auto len = alignTo(weight.second, len_alignment);
    weight_offset_aligned_.push_back(std::make_pair(off, len));
    last_offset = off + len;
  }
}

size_t SwapDevice::alignTo(const size_t value, const size_t alignment) const {
  if (!alignment) {
    return value;
  }

  size_t aligned_value = (value / alignment) * alignment;
  aligned_value += (value % alignment) ? alignment : 0;
  return aligned_value;
}

void SwapDevice::getAlignments(size_t &off_alignment,
                               size_t &len_alignment) const {
#if defined(_WIN32)
  SYSTEM_INFO sysInfo;
  GetSystemInfo(&sysInfo);
  off_alignment = sysInfo.dwAllocationGranularity;
  len_alignment = sysInfo.dwPageSize;
#else
  off_alignment = sysconf(_SC_PAGE_SIZE);
  len_alignment = 0;
#endif
}

} // namespace nntrainer
