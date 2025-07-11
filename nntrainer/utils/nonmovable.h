// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   nonmovable.h
 * @date   11 July 2025
 * @brief  Base class preventing moving of derived classes
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Grzegorz Kisala <g.kisala@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_NONMOVABLE_H__
#define __NNTRAINER_NONMOVABLE_H__

namespace nntrainer {

/**
 * @class   Nonmovable
 * @brief   Use it as base class to prevent move operations
 */
class Nonmovable {
public:
  /**
   * @brief   Default constructor
   */
  Nonmovable() = default;

  /**
   * @brief Deleting move constructor
   *
   */
  Nonmovable(Nonmovable &&) = delete;

  /**
   * @brief Deleting move assignment operator
   *
   */
  Nonmovable &operator=(Nonmovable &&) = delete;

  /**
   * @brief   Default destructor
   */
  virtual ~Nonmovable() = default;
};

} // namespace nntrainer

#endif // __NNTRAINER_NONMOVABLE_H__
