// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   lazy_tensor.cpp
 * @date   05 Jun 2020
 * @brief  A lazy evaluation calculator for tensors
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <lazy_tensor.h>
#include <nntrainer_error.h>

namespace nntrainer {

/**
 * @brief Wrapper method of add_i (immediate version of add)
 * @retval this
 */
LazyTensor &LazyTensor::add_i(float const &value) {
  call_chain.push_back(
    [value](Tensor &t) mutable -> int { return t.add_i(value); });
  return *this;
}
/**
 * @brief     Wrapper method of add_i. see tensor.h for more detail
 * @param[in] m Tensor to be added
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::add_i(Tensor const &m, float const alpha) {
  auto f = [&m, alpha](Tensor &t) mutable -> int { return t.add_i(m, alpha); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of subtract_i. see tensor.h for more detail
 * @param[in] m Tensor to subtract
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::subtract_i(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int { return t.subtract_i(m); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of subtract_i. see tensor.h for more detail
 * @param[in] value value to subtract
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::subtract_i(float const &value) {
  auto f = [value](Tensor &t) mutable -> int { return t.subtract_i(value); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief Wrapper method of multiply_i. see tensor.h for more detail
 * @param[in] value to be added
 * @retval LazyTensor *this
 */
LazyTensor &LazyTensor::multiply_i(float const &value) {
  auto f = [value](Tensor &t) mutable -> int { return t.multiply_i(value); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of multiply_i. see tensor.h for more detail
 * @param[in] m Tensor to be multiplied
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::multiply_i(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int { return t.multiply_i(m); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of divide_i. see tensor.h for more detail
 * @param[in] value divisor
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::divide_i(float const &value) {
  auto f = [value](Tensor &t) mutable -> int { return t.divide_i(value); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of divide_i. see tensor.h for more detail
 * @param[in] m Tensor to for division
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::divide_i(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int { return t.divide_i(m); };
  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of dot. see tensor.h for more detail (memcopy
 * happens)
 * @param[in] m Tensor
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::dot(Tensor const &m) {
  auto f = [&m](Tensor &t) mutable -> int {
    try {
      t = t.dot(m);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of transpose. see tensor.h for more detail (memcopy
 * happens)
 * @param[in] direction to transpose ex) 0:2:1
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::transpose(std::string direction) {
  auto f = [direction](Tensor &t) mutable -> int {
    try {
      t = t.transpose(direction);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of sum. see tensor.h for more detail (memcopy
 * happens)
 * @param[in] direction to transpose ex) 0:2:1
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::sum_by_batch() {
  auto f = [](Tensor &t) mutable -> int {
    try {
      t = t.sum_by_batch();
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of sum. see tensor.h for more detail (memcopy
 * happens) 0 : batch direction 1 : channel direction 2 : channel direction 3 :
 * channel direction
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::sum(int axis) {
  auto f = [axis](Tensor &t) mutable -> int {
    try {
      t = t.sum(axis);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
 * happens)
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::average(int axis) {
  auto f = [axis](Tensor &t) mutable -> int {
    try {
      t = t.average(axis);
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief     Wrapper method of average. see tensor.h for more detail (memcopy
 * happens)
 * @retval    LazyTensor *this
 */
LazyTensor &LazyTensor::average() {
  auto f = [](Tensor &t) mutable -> int {
    try {
      t = t.average();
      return ML_ERROR_NONE;
    } catch (std::runtime_error &e) {
      return ML_ERROR_INVALID_PARAMETER;
    }
  };

  call_chain.push_back(f);
  return *this;
}

/**
 * @brief execute the call_chain to evaluate
 * @retval calculated tensor
 */
Tensor LazyTensor::run() {
  int status;
  for (auto item : call_chain) {
    status = item(target);
    if (status != ML_ERROR_NONE) {
      throw std::runtime_error("Error: evaluation failed");
    }
  }
  return target;
}

} /* namespace nntrainer */
