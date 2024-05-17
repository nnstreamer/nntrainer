// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimizer_common_properties.h
 * @date   17 May 2024
 * @brief  This file contains list of common properties for optimizer
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __OPTIMIZER_COMMON_PROPERTIES_H__
#define __OPTIMIZER_COMMON_PROPERTIES_H__

#include <array>
#include <fstream>
#include <string>

#include <base_properties.h>

#ifdef __cplusplus
namespace nntrainer::props {

/**
 * @brief Beta 1 props
 *
 */
class PropsB1 : public Property<double> {
public:
  static constexpr const char *key = "beta1"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};

/**
 * @brief Beta 2 props
 *
 */
class PropsB2 : public Property<double> {
public:
  static constexpr const char *key = "beta2"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};
  
/**
 * @brief Rho props
 *
 */
class Rho : public Property<double> {
public:
  static constexpr const char *key = "rho"; /**< unique key to access */
  using prop_tag = double_prop_tag;         /**< property type */
  Rho(double value = 0.9);
};

/**
 * @brief epsilon props
 *
 */
class PropsEpsilon : public Property<double> {
public:
  static constexpr const char *key = "epsilon"; /**< unique key to access */
  using prop_tag = double_prop_tag;             /**< property type */
  PropsEpsilon(double value = 1.0e-7f);
};

/**
 * @brief pytorch reference implementation
 *
 */
class TorchRef : public Property<bool> {
public:
  static constexpr const char *key = "torch_ref"; /**< unique key to access */
  using prop_tag = bool_prop_tag;                 /**< property type */

  TorchRef(bool value = false);
};

} // namespace nntrainer::props

#endif

#endif // __MODEL_COMMON_PROPERTIES_H__
