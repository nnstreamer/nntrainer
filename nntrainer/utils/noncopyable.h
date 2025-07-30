// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   noncopyable.h
 * @date   11 July 2025
 * @brief  Base class preventing copying of derived classes
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_NONCOPYABLE_H__
#define __NNTRAINER_NONCOPYABLE_H__

namespace nntrainer {

/**
 * @class   Noncopyable
 * @brief   Use it as base class to prevent copy operations
 */
class Noncopyable {
public:
  /**
   * @brief   Default constructor
   */
  Noncopyable() = default;

  /**
   * @brief Deleting copy constructor
   *
   */
  Noncopyable(const Noncopyable &) = delete;

  /**
   * @brief Deleting assignment operator
   *
   */
  Noncopyable &operator=(const Noncopyable &) = delete;

  /**
   * @brief   Default destructor
   */
  ~Noncopyable() = default;
};

} // namespace nntrainer

#endif // __NNTRAINER_NONCOPYABLE_H__
