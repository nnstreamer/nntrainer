// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   acti_func.cpp
 * @date   22 March 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Activation Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iostream>
#include <vector>

#include <acti_func.h>
#include <blas_interface.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

ActiFunc::~ActiFunc() {}

void ActiFunc::run_fn(Tensor const &input, Tensor &output) {
  _act_fn(input, output);
}

Tensor &ActiFunc::run_prime_fn(Tensor &input, Tensor &output,
                               Tensor &outgoing_derivative,
                               Tensor const &incoming_derivative) {
  return _act_prime_fn(input, output, outgoing_derivative, incoming_derivative);
}

Tensor &ActiFunc::run_prime_fn(Tensor &output, Tensor &outgoing_derivative,
                               Tensor const &incoming_derivative) {
  return _act_prime_fn(Tensor(), output, outgoing_derivative,
                       incoming_derivative);
}

bool ActiFunc::supportInPlace() const { return in_place; }

void ActiFunc::executeInPlace(bool val) {
  if (val && !supportInPlace())
    throw std::runtime_error("Error setting activation layer to work in-place");

  in_place = val;
}
}; // namespace nntrainer
