/* SPDX-License-Identifier: Apache-2.0-only */
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file	tensor_dim.cpp
 * @date	22 May 2020
 * @brief	This is Tensor Dimension Class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <assert.h>
#include <cstring>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <sstream>
#include <stdio.h>
#include <tensor_dim.h>

namespace nntrainer {

int TensorDim::setTensorDim(std::string input_shape) {
  int status = ML_ERROR_NONE;
  std::regex words_regex("[^\\s.,:;!?]+");
  auto words_begin =
    std::sregex_iterator(input_shape.begin(), input_shape.end(), words_regex);
  auto words_end = std::sregex_iterator();
  int cur_dim = std::distance(words_begin, words_end);
  if (cur_dim > 4) {
    ml_loge("Tensor Dimension should be less than 4");
    return ML_ERROR_INVALID_PARAMETER;
  }
  int cn = 0;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i) {
    dim[MAXDIM - cur_dim + cn] = std::stoi((*i).str());
    if (dim[MAXDIM - cur_dim + cn] <= 0) {
      ml_loge("Tensor Dimension should be greater than 0");
      return ML_ERROR_INVALID_PARAMETER;
    }
    cn++;
  }
  return status;
}

void TensorDim::operator=(const TensorDim &from) {
  for (int i = 0; i < 4; ++i) {
    this->dim[i] = from.dim[i];
  }
};

} /* namespace nntrainer */
