// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tensor_wrap_specs.h
 * @date   26 July 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is specs for various tensor wrappers
 *
 */

#ifndef __TENSOR_WRAP_SPECS_H__
#define __TENSOR_WRAP_SPECS_H__

#include <tuple>

#include <tensor.h>

namespace nntrainer {

/**
 * @brief     Enumeration of Weight Regularizer
 * @todo      Update to TensorRegularizer
 */
enum class WeightRegularizer {
  L2NORM, /**< L2 norm regularization */
  NONE,   /**< no regularization */
  UNKNOWN /**< Unknown */
};

/**
 * @brief     Enumeration of Weight Initialization Type
 * @todo      support intialization from file
 */
enum class TensorInitializer {
  ZEROS,          /** Zero initialization */
  ONES,           /** One initialization */
  LECUN_NORMAL,   /** LeCun normal initialization */
  LECUN_UNIFORM,  /** uniform initialization */
  XAVIER_NORMAL,  /** Xavier normal initialization */
  XAVIER_UNIFORM, /** Xavier uniform initialization */
  HE_NORMAL,      /** He normal initialization */
  HE_UNIFORM,     /** He uniform initialization */
  NONE            /** No initialization */
};

/**
 * @brief Specification of the Weight as a tensor wrapper
 *
 * @details The tuple values are dimension, initializer, regularizer,
 * regularizer_constant, need_gradient property amd name of the tensor object.
 */
typedef std::tuple<TensorDim, TensorInitializer, WeightRegularizer, float, bool,
                   const std::string>
  WeightSpec;

/**
 * @brief Specification of the Var_Grad (trainable tensor) as a tensor wrapper
 *
 * @details The tuple values are dimension, need_gradient property, and the
 * name of the tensor object.
 */
typedef std::tuple<TensorDim, bool, const std::string> VarGradSpec;

} // namespace nntrainer

#endif /** __TENSOR_WRAP_SPECS_H__ */
