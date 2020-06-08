/* SPDX-License-Identifier: Apache-2.0-only
 * Copyright (C) 2020 Jihoon Lee <jihoon.it.lee@samsung.com>
 *
 * @file	lazy_tensor.h
 * @date	05 Jun 2020
 * @brief	A lazy evaluation calculator for tensors
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jihoon Lee <jihoon.it.lee@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __LAZY_TENSOR_H__
#define __LAZY_TENSOR_H__
#ifdef __cplusplus

#include <tensor.h>
#include <vector>

namespace nntrainer {

/**
 * @class   LazyTensor a wrapper class for lazy calculation of tensor
 * @brief   calculation is delayed until Tensor LazyTensor::run() is
 *          called, can be contructed by Tensor::chain() method
 */
class LazyTensor {
public:
  /**
   * @brief Constructor of Lazy Tensor, Tensor is copied to gaurantee
   * immutability
   */
  LazyTensor(const Tensor &from) { target.copy(from); };

  /**
   * @brief Wrapper method of add_i
   * @retval LazyTensor *this
   */
  LazyTensor add_i(float const &value);

  /**
   * @brief execute the call_chain to get the tensor
   * @retval calculated tensor
   */
  Tensor &run();

private:
  /**< handle the data as a std::vector type */
  std::vector<std::function<int(Tensor &)>> call_chain;
  Tensor target;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LAZY_TENSOR_H__ */
