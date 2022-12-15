// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   swap_device.cpp
 * @date   01 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Swap device class
 *
 */

#ifndef __SWAP_DEVICE_H__
#define __SWAP_DEVICE_H__

#include <fcntl.h>
#include <map>
#include <memory>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <system_error>
#include <unistd.h>
#include <utility>

/* Uncomment this to use mmap for swap data */
//#define USE_MMAP

namespace nntrainer {

/**
 * @class   SwapDevice
 * @brief   A device used to storing data with long access time
 */
class SwapDevice {
public:
  /**
   * @brief swap device default path
   *
   */
  const std::string swap_device_default_path = ".";

  /**
   * @brief SwapDevice default constructor
   *
   */
  explicit SwapDevice(const std::string &name) :
    dev_path(swap_device_default_path + name),
    fd(-1) {}

  /**
   * @brief SwapDevice default constructor
   *
   */
  explicit SwapDevice(const std::string &path, const std::string &name) :
    dev_path(path + "/" + name),
    fd(-1) {}

  /**
   * @brief SwapDevice destructor
   *
   */
  virtual ~SwapDevice() = default;

  /**
   * @brief Start device
   *
   * @param size The size of requested swap device space
   *
   */
  void start(size_t size);

  /**
   * @brief Allocate and get data
   *
   * @param offset Requested offset of swap device file
   * @param size Requested size.
   * @param alloc_only only allocate buffer without reading data
   *
   * @return The pointer of the swap space
   *
   */
  void *getBuffer(off_t offset, size_t size, bool alloc_only = false);

  /**
   * @brief Deallocate and put data
   *
   * @param ptr The pointer obtained from getBuffer
   * @param dealloc_only only deallocate buffer without writing data
   */
  void putBuffer(void *ptr, bool dealloc_only = false);

  /**
   * @brief Close device
   *
   */
  void finish();

  /**
   * @brief Check device is operating
   *
   * @return Device operating status
   *
   */
  bool isOperating() const { return (fd >= 0); }

  /**
   * @brief Get device path
   *
   * @return Device path
   *
   */
  const std::string getDevicePath() const { return dev_path; }

private:
  const std::string dev_path; /**< device path */
  int fd;                     /**< device file description */

#ifdef USE_MMAP
  std::map<void *, std::pair<void *, size_t>>
    mapped; /**< <pointer, <orig_pointer, size>> */
#else
  std::map<void *, std::pair<off_t, ssize_t>>
    allocated; /**< <pointer, <offset, size>> */
#endif
};

} // namespace nntrainer

#endif /** __SWAP_DEVICE_H__ */
