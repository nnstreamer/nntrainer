// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   custom_properties.h
 * @date   1 October 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This file contains list of custom properties widely used across
 * custom layers
 */

#ifndef __CUSTOM_PROPERTIES_H__
#define __CUSTOM_PROPERTIES_H__

#include <base_properties.h>

namespace nntrainer {

namespace props {

/**
 * @brief indicated this layer is for smart reply application
 *
 */
class SmartReply : public Property<bool> {
public:
  /**
   * @brief Construct a new SmartReply object
   *
   */
  SmartReply(bool value = false) { set(value); }
  static constexpr const char *key = "smart_reply";
  using prop_tag = bool_prop_tag;
};

} // namespace props

} // namespace nntrainer

#endif /* __CUSTOM_PROPERTIES_H__ */
