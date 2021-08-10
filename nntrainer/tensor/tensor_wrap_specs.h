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
 * @brief define the lifespan of the given tensor to reduce peak memory
 *
 */
enum TensorLifespan {
  FORWARD_FUNC_LIFESPAN,  /**< tensor must not be reset before during the
                            forward function call, eg. temporary tensors
                            needed during forward operations */
  BACKWARD_FUNC_LIFESPAN, /**< tensor must not be reset before during the
                            backward function call, eg. temporary tensors
                            needed during backward operations */
  ITERATION_LIFESPAN,     /**< tensor must not be reset until the owning layer
                            finishes its execution in the current iteration,
                            eg. hidden memory/cells of RNN */
  EPOCH_LIFESPAN,         /**< tensor must not be reset before the epoch ends */
  MAX_LIFESPAN, /**< tensor must not be reset until the end of the model
                  execution, eg. layer weights */
};

/**
 * @brief Specification of the Weight as a tensor wrapper
 *
 * @details The tuple values are dimension, initializer, regularizer,
 * regularizer_constant, need_gradient property amd name of the tensor object.
 */
typedef std::tuple<TensorDim, Tensor::Initializer, WeightRegularizer, float,
                   bool, const std::string>
  WeightSpec;

/**
 * @brief Specification of the Var_Grad (trainable tensor) as a tensor wrapper
 *
 * @details The tuple values are dimension, initializer, need_gradient property,
 * the name, and lifespan of the Var_Grad object.
 */
typedef std::tuple<TensorDim, Tensor::Initializer, bool, const std::string,
                   TensorLifespan>
  VarGradSpec;

} // namespace nntrainer

#endif /** __TENSOR_WRAP_SPECS_H__ */
