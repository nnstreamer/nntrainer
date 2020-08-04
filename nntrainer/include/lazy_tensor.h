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

#include <nntrainer_error.h>
#include <tensor.h>
#include <vector>

#define FWD(...) std::forward<decltype(__VA_ARGS__)>(__VA_ARGS__)

#define _LIFT(X)                                            \
  [](nntrainer::Tensor & t,                                 \
     auto &&... args) noexcept(noexcept(t.X(FWD(args)...))) \
    ->decltype(t.X(FWD(args)...)) {                         \
    return t.X(FWD(args)...);                               \
  }

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
   * @brief     Wrapper method of add_i. see tensor.h for more detail
   * @param[in] value to be added
   * @retval    LazyTensor *this
   */
  LazyTensor &add_i(float const &value);

  /**
   * @brief     Wrapper method of add_i. see tensor.h for more detail
   * @param[in] m Tensor to be added
   * @retval    LazyTensor *this
   */
  LazyTensor &add_i(Tensor const &m, float const alpha = 1);

  /**
   * @brief     Wrapper method of subtract_i. see tensor.h for more detail
   * @param[in] m Tensor to subtract
   * @retval    LazyTensor *this
   */
  LazyTensor &subtract_i(Tensor const &m);

  /**
   * @brief     Wrapper method of subtract_i. see tensor.h for more detail
   * @param[in] value value to subtract
   * @retval    LazyTensor *this
   */
  LazyTensor &subtract_i(float const &value);

  /**
   * @brief Wrapper method of multiply_i. see tensor.h for more detail
   * @param[in] value to be added
   * @retval LazyTensor *this
   */
  LazyTensor &multiply_i(float const &value);

  /**
   * @brief     Wrapper method of multiply_i. see tensor.h for more detail
   * @param[in] m Tensor to be multiplied
   * @retval    LazyTensor *this
   */
  LazyTensor &multiply_i(Tensor const &m);

  /**
   * @brief     Wrapper method of divide_i. see tensor.h for more detail
   * @param[in] value divisor
   * @retval    LazyTensor *this
   */
  LazyTensor &divide_i(float const &value);

  /**
   * @brief     Wrapper method of divide_i. see tensor.h for more detail
   * @param[in] m Tensor to for division
   * @retval    LazyTensor *this
   */
  LazyTensor &divide_i(Tensor const &m);

  /**
   * @brief     Wrapper method of dot. see tensor.h for more detail (memcopy
   * happens)
   * @param[in] m Tensor
   * @retval    LazyTensor *this
   */
  LazyTensor &dot(Tensor const &m);

  /**
   * @brief     Wrapper method of transpose. see tensor.h for more detail
   * (memcopy happens)
   * @param[in] direction to transpose ex) 0:2:1
   * @retval    LazyTensor *this
   */
  LazyTensor &transpose(std::string direction);

  /**
   * @brief     Wrapper method of sum_by_batch. see tensor.h for more detail
   * (memcopy happens)
   * @retval    LazyTensor *this
   */
  LazyTensor &sum_by_batch();

  /**
   * @brief     Wrapper method of sum. see tensor.h for more detail (memcopy
   * happens) 0 : batch direction 1 : channel direction 2 : height direction 3 :
   * width direction
   * @retval    LazyTensor *this
   */
  LazyTensor &sum(int axis);

  /**
   * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
   * happens) 0 : batch direction 1 : channel direction 2 : height direction 3 :
   * width direction
   * @retval    LazyTensor *this
   */
  LazyTensor &average(int axis);

  /**
   * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
   * happens)
   * @retval    LazyTensor *this
   */
  LazyTensor &average();

  /**
   * @brief     apply A tensor function when predicate is true
   * @param[in] bool predicate predicate to check to determine application
   * @param[in] _Callable&& fn function to be applied
   *            (Must be wrapped with _LIFT(X) macro to resolve overload set)
   * @param[in] _Args&&... args args for fn
   * @retval    LazyTensor *this
   */
  template <typename _Callable, typename... _Args>
  LazyTensor &applyIf(bool predicate, _Callable &&fn, _Args &&... args) {
    if (predicate) {
      auto f = [&](Tensor &t) mutable -> int {
        try {
          return fn(t, std::forward<_Args>(args)...);
        } catch (std::runtime_error &e) {
          return ML_ERROR_INVALID_PARAMETER;
        }
      };
      call_chain.push_back(f);
    }

    return *this;
  }

  /**
   * @brief execute the call_chain to get the tensor
   * @retval calculated tensor
   */
  Tensor run();

private:
  /**< handle the data as a std::vector type */
  std::vector<std::function<int(Tensor &)>> call_chain;
  Tensor target;
};

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __LAZY_TENSOR_H__ */
