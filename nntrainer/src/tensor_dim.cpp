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

void TensorDim::resetLen() {
  feature_len = dim[1] * dim[2] * dim[3];
  len = dim[0] * feature_len;
}

void TensorDim::setTensorDim(unsigned int idx, unsigned int value) {
  dim[idx] = value;
  resetLen();
}

int TensorDim::setTensorDim(std::string input_shape) {
  int status = ML_ERROR_NONE;
  std::regex words_regex("[^\\s.,:;!?]+");
  auto words_begin =
    std::sregex_iterator(input_shape.begin(), input_shape.end(), words_regex);
  auto words_end = std::sregex_iterator();
  int cur_dim = std::distance(words_begin, words_end);
  if (cur_dim > MAXDIM) {
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
  feature_len = dim[1] * dim[2] * dim[3];
  len = dim[0] * feature_len;
  return status;
}

void TensorDim::operator=(const TensorDim &from) {
  for (int i = 0; i < MAXDIM; ++i) {
    this->dim[i] = from.dim[i];
  }
  len = from.len;
  feature_len = from.feature_len;
}

bool TensorDim::operator==(const TensorDim &rhs) const {
  for (int i = 0; i < MAXDIM; ++i) {
    if (this->dim[i] != rhs.dim[i]) {
      return false;
    }
  }

  return true;
}

std::ostream &operator<<(std::ostream &out, TensorDim const &d) {
  out << "Shape : " << d.batch() << ":" << d.channel() << ":" << d.height()
      << ":" << d.width() << std::endl;
  return out;
}

} /* namespace nntrainer */
