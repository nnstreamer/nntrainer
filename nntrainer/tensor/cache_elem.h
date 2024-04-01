// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   cache_elem.h
 * @date   28 Nov 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Cache elem class
 *
 */

#ifndef __CACHE_ELEM_H__
#define __CACHE_ELEM_H__

#include <list>
#include <mutex>

#include <memory_data.h>
#include <swap_device.h>

namespace nntrainer {

enum CachePolicy {
  WRITE_BACK = 0b0001,      /**< invalidate will write to device */
  NO_WRITE_BACK = 0b0010,   /**< invalidate will not write to device */
  READ_CONSIST = 0b0100,    /**< validate will read from device */
  NO_READ_CONSIST = 0b1000, /**< validate will not read from device */
  ALWAYS_SYNCED =
    (READ_CONSIST | WRITE_BACK), /**< Always synchronized with device */
  TEMPORAL = (NO_READ_CONSIST |
              NO_WRITE_BACK), /**< Will not be synchronized with device */
  FIRST_LAST_SKIP = 0b10000,
  /**< Will skip first read and last write */
  ITERATION_CONSIST = (FIRST_LAST_SKIP | ALWAYS_SYNCED),
  /**< Will skip first read and last write. other behaviors will be same as
     ALWAYS_SYNCED */
};

/**
 * @class   CacheElem
 * @brief   Cache element containing swap address
 */
class CacheElem {
public:
  enum Options {
    NONE = 0b0000,         /**< No option */
    FIRST_ACCESS = 0x0001, /**< First Access */
    LAST_ACCESS = 0x0010,  /**< Last Access */
  };

  /**
   * @brief CacheElem default constructor
   *
   */
  explicit CacheElem(std::shared_ptr<SwapDevice> dev, unsigned int mem_id,
                     size_t off, size_t len, std::shared_ptr<MemoryData> data,
                     CachePolicy pol = CachePolicy::ALWAYS_SYNCED) :
    initial_opt(Options::FIRST_ACCESS),
    device(dev),
    active(false),
    id(mem_id),
    offset(off),
    length(len),
    policy(pol),
    mem_data(data) {}

  /**
   * @brief CacheElem destructor
   *
   */
  virtual ~CacheElem() {}

  /**
   * @brief load data from swap device
   *
   * @param alloc_only only allocate buffer without reading data
   */
  void swapIn(Options opt = Options::NONE);

  /**
   * @brief unload data to swap device
   *
   * @param dealloc_only only deallocate buffer without writing data
   */
  void swapOut(Options opt = Options::NONE);

  /**
   * @brief unload data to swap device
   *
   * @return active status
   */
  bool isActive() const {
    std::scoped_lock lg(device_mutex);
    return active;
  }

  /**
   * @brief get length of cache element
   *
   * @return length of cache element in byte
   */
  size_t getLength() const { return length; }

  /**
   * @brief get id of cache element
   *
   * @return cache element id
   */
  unsigned int getId() const { return id; }

  /**
   * @brief reset access count
   *
   */
  void reset() { initial_opt = Options::FIRST_ACCESS; }

private:
  Options initial_opt;                  /**< accessed */
  mutable std::mutex device_mutex;      /**< protect device */
  std::shared_ptr<SwapDevice> device;   /**< swap device */
  bool active;                          /**< element is loaded */
  unsigned int id;                      /**< memory id */
  size_t offset;                        /**< element offset from swap device */
  size_t length;                        /**< element size */
  CachePolicy policy;                   /**< cache policy */
  std::shared_ptr<MemoryData> mem_data; /**< allocated memory data */
};

} // namespace nntrainer

#endif /** __CACHE_ELEM_H__ */
