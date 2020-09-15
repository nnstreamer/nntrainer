// SPDX-License-Identifier: Apache-2.0-only
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
#include <sstream>
#include <stdio.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
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
  std::swap_ranges(std::begin(lhs.dim), std::begin(lhs.dim) + MAXDIM,
                   std::begin(rhs.dim));
  std::swap(lhs.len, rhs.len);
  std::swap(lhs.feature_len, rhs.feature_len);
}

void TensorDim::resetLen() {
  feature_len = dim[1] * dim[2] * dim[3];
  len = dim[0] * feature_len;
}

const unsigned int TensorDim::getTensorDim(unsigned int idx) const {
  if (idx >= MAXDIM)
    throw std::invalid_argument(
      "[TensorDim] Tensor Dimension index should be between 0 and 4");

  return dim[idx];
}

void TensorDim::setTensorDim(unsigned int idx, unsigned int value) {
  if (idx >= MAXDIM)
    throw std::out_of_range(
      "[TensorDim] Tensor Dimension index should be between 0 and 4");

  if (value <= 0)
    throw std::invalid_argument(
      "[TensorDim] Trying to assign value <=0 to tensor dim");

  if (len == 0) {
    for (int i = 0; i < MAXDIM; ++i) {
      dim[i] = 1;
    }
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
  if (cur_dim <= 0 || cur_dim > MAXDIM) {
    ml_loge("Tensor Dimension should be between 1 and 4");
    return ML_ERROR_INVALID_PARAMETER;
  }
  int cn = 0;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i, ++cn) {
    setTensorDim(MAXDIM - cur_dim + cn, std::stoul((*i).str()));
  }

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

unsigned int TensorDim::rank() const {
  unsigned int rank = 0;
  for (unsigned int i = 0; i < MAXDIM; i++) {
    if (dim[i] > 1)
      rank += 1;
  }
  return rank;
}

std::ostream &operator<<(std::ostream &out, TensorDim const &d) {
  out << "Shape: " << d.batch() << ":" << d.channel() << ":" << d.height()
      << ":" << d.width() << std::endl;
  return out;
}

} /* namespace nntrainer */
