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

TensorDim &TensorDim::operator=(const TensorDim &rhs) {
  TensorDim tmp(rhs.batch(), rhs.channel(), rhs.height(), rhs.width());
  this->swap(*this, tmp);
  return *this;
}

TensorDim &TensorDim::operator=(TensorDim &&rhs) noexcept {
  this->swap(*this, rhs);
  return *this;
}

void TensorDim::swap(TensorDim &lhs, TensorDim &rhs) noexcept {
  std::swap(lhs.dim, rhs.dim);
  std::swap(lhs.len, rhs.len);
  std::swap(lhs.feature_len, rhs.feature_len);
}

void TensorDim::resetLen() {
  feature_len = dim[1] * dim[2] * dim[3];
  len = dim[0] * feature_len;
}

void TensorDim::setTensorDim(unsigned int idx, unsigned int value) {
  if (value == 0) {
    throw std::invalid_argument(
      "[TensorDim] Trying to assign value of 0 to tensor dim");
  }

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
    if ((MAXDIM - cur_dim + cn) > 3 || (MAXDIM - cur_dim + cn) < 0) {
      ml_loge("Tensor Dimension Setting Error");
      return ML_ERROR_INVALID_PARAMETER;
    }
    setTensorDim(MAXDIM - cur_dim + cn, std::stoi((*i).str()));
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

bool TensorDim::operator==(const TensorDim &rhs) const {
  for (int i = 0; i < MAXDIM; ++i) {
    if (this->dim[i] != rhs.dim[i]) {
      return false;
    }
  }

  return true;
}

std::ostream &operator<<(std::ostream &out, TensorDim const &d) {
  out << "Shape: " << d.batch() << ":" << d.channel() << ":" << d.height()
      << ":" << d.width() << std::endl;
  return out;
}

} /* namespace nntrainer */
