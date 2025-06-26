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
#include <iostream>

namespace nntrainer {

using MemoryDataValidateCallback = std::function<void(unsigned int)>;

/**
 * @brief  MemoryData Class
 */
class MemoryData {
public:
  /**
   * @brief  Constructor of Memory Data
   * @param[in] addres Memory data address
   * @param[in] own_memory Memory ownership flag
   */
  explicit MemoryData(void *address, bool own_memory) :
    valid_(true),
    id_(0),
    address_(address),
    own_memory_(own_memory),
    validate_cb_([](unsigned int) {}),
    invalidate_cb_([](unsigned int) {}) {

    if ((address != nullptr) && !is_aligned(address, 4)) {
      std::cout << "dupa";
    }
  }

  /**
   * @brief  Constructor of Memory Data
   * @param[in] id validate callback.
   * @param[in] validate_cb validate callback.
   * @param[in] invalidate_cb invalidate callback.
   */
  explicit MemoryData(unsigned int id, MemoryDataValidateCallback validate_cb,
                      MemoryDataValidateCallback invalidate_cb,
                      void *address = nullptr) :
    valid_(false),
    id_(id),
    address_(address),
    own_memory_(false),
    validate_cb_(validate_cb),
    invalidate_cb_(invalidate_cb) {

    if ((address != nullptr) && !is_aligned(address, 4)) {
      std::cout << "dupa";
    }
  }

  /**
   * @brief  Deleted constructor of Memory Data
   */
  explicit MemoryData() = delete;

  MemoryData(const MemoryData &) = delete;

  MemoryData &operator=(const MemoryData &) = delete;

  /**
   * @brief  Destructor of Memory Data
   */
  ~MemoryData() {
    if (own_memory_ && (address_ != nullptr)) {
      std::free(address_);
    }
  };

  /**
   * @brief  Set address
   */
  void setAddr(void *address) {
    address_ = address;
    if ((address != nullptr) && !is_aligned(address, 4)) {
      std::cout << "dupa";
    }
  }

  /**
   * @brief  Get address
   */
  template <typename T = float> T *getAddr() const {
    return static_cast<T *>(address_);
  }

  /**
   * @brief  Validate memory data
   */
  void validate() {
    if (valid_) {
      return;
    }

    if (validate_cb_ != nullptr) {
      validate_cb_(id_);
    }
  }

  /**
   * @brief  Invalidate memory data
   */
  void invalidate() {
    if (!valid_) {
      return;
    }

    if (invalidate_cb_ != nullptr) {
      invalidate_cb_(id_);
    }
  }

  /**
   * @brief  Set valid
   */
  void setValid(bool valid) { valid_ = valid; }

  bool is_aligned(const void *pointer, const std::size_t alignment) {
    return ((reinterpret_cast<uintptr_t>(pointer) % alignment) == 0);
  }

private:
  bool valid_;
  unsigned int id_;
  void *address_;
  bool own_memory_;
  MemoryDataValidateCallback validate_cb_;
  MemoryDataValidateCallback invalidate_cb_;
};

} // namespace nntrainer

#endif /* __MEMORY_DATA_H__ */
