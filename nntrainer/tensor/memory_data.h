// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jiho Chu <jiho.chu@samsung.com>
 *
 * @file   memory_data.h
 * @date   14 Oct 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jiho Chu <jiho.chu@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  MemoryData class
 *
 */

#ifndef __MEMORY_DATA_H__
#define __MEMORY_DATA_H__

#include <functional>

namespace nntrainer {

using MemoryDataValidateCallback = std::function<void(unsigned int)>;

/**
 * @brief  MemoryData Class
 */
class MemoryData {
public:
  /**
   * @brief  Constructor of Memory Data
   * @param[in] addr Memory data
   */
  explicit MemoryData(void *addr) :
    valid(true),
    id(0),
    address(addr),
    validate_cb([](unsigned int) {}),
    invalidate_cb([](unsigned int) {}) {}

  /**
   * @brief  Constructor of Memory Data
   * @param[in] mem_id validate callback.
   * @param[in] v_cb validate callback.
   * @param[in] i_cb invalidate callback.
   */
  explicit MemoryData(unsigned int mem_id, MemoryDataValidateCallback v_cb,
                      MemoryDataValidateCallback i_cb) :
    valid(false),
    id(mem_id),
    address(nullptr),
    validate_cb(v_cb),
    invalidate_cb(i_cb) {}

  /**
   * @brief  Deleted constructor of Memory Data
   */
  explicit MemoryData() = delete;

  /**
   * @brief  Constructor of MemoryData
   */
  explicit MemoryData(MemoryDataValidateCallback v_cb,
                      MemoryDataValidateCallback i_cb) = delete;
  /**
   * @brief  Constructor of MemoryData
   */
  explicit MemoryData(void *addr, MemoryDataValidateCallback v_cb,
                      MemoryDataValidateCallback i_cb) = delete;

  /**
   * @brief  Destructor of Memory Data
   */
  virtual ~MemoryData() = default;

  /**
   * @brief  Set address
   */
  void setAddr(void *addr) { address = addr; }

  /**
   * @brief  Get address
   */
  template <typename T = float> T *getAddr() const {
    return static_cast<T *>(address);
  }

  /**
   * @brief  Validate memory data
   */
  void validate() {
    if (valid)
      return;
    validate_cb(id);
  }

  /**
   * @brief  Invalidate memory data
   */
  void invalidate() {
    if (!valid)
      return;
    invalidate_cb(id);
  }

  /**
   * @brief  Set valid
   */
  void setValid(bool v) { valid = v; }

private:
  bool valid;
  unsigned int id;
  void *address;
  MemoryDataValidateCallback validate_cb;
  MemoryDataValidateCallback invalidate_cb;
};

} // namespace nntrainer

#endif /* __MEMORY_DATA_H__ */
