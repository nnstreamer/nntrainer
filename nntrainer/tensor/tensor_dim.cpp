// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file   tensor_dim.cpp
 * @date   22 May 2020
 * @brief  This is Tensor Dimension Class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cstring>
#include <regex>
#include <sstream>
#include <stdio.h>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <tensor_dim.h>

namespace nntrainer {

TensorDim::TensorDim(const std::string &shape) : TensorDim() {
  if (setTensorDim(shape) != ML_ERROR_NONE) {
    throw std::invalid_argument("[TensorDim] Setting TensorDim failed");
  }
}

TensorDim &TensorDim::operator=(const TensorDim &rhs) {
  using std::swap;

  TensorDim tmp(rhs);
  swap(*this, tmp);
  return *this;
}

TensorDim &TensorDim::operator=(TensorDim &&rhs) noexcept {
  using std::swap;

  swap(*this, rhs);
  return *this;
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
    for (size_t i = 0; i < MAXDIM; ++i) {
      dim[i] = 1;
    }
  }

  dim[idx] = value;
  resetLen();
}

int TensorDim::setTensorDim(const std::string &input_shape) {
  int status = ML_ERROR_NONE;
  static const std::regex words_regex("[^\\s.,:;!?]+");
  auto words_begin =
    std::sregex_iterator(input_shape.begin(), input_shape.end(), words_regex);
  auto words_end = std::sregex_iterator();
  int cur_dim = std::distance(words_begin, words_end);
  if (cur_dim <= 0 || (size_t)cur_dim > MAXDIM) {
    ml_loge("Tensor Dimension should be between 1 and 4");
    return ML_ERROR_INVALID_PARAMETER;
  }
  int cn = 0;
  for (std::sregex_iterator i = words_begin; i != words_end; ++i, ++cn) {
    setTensorDim(MAXDIM - cur_dim + cn, std::stoul((*i).str()));
  }

  return status;
}

TensorDim TensorDim::transpose(const std::string &direction) const {
  int dirs[MAXDIM - 1];

  getValues(3, direction, dirs);

  const std::array<unsigned int, MAXDIM> axes{{0, (unsigned int)dirs[0] + 1,
                                               (unsigned int)dirs[1] + 1,
                                               (unsigned int)dirs[2] + 1}};

  return transpose(axes);
}

TensorDim
TensorDim::transpose(const std::array<unsigned int, MAXDIM> &axes) const {
  TensorDim tmp(*this);

  for (unsigned int i = 0; i < MAXDIM; ++i) {
    tmp.setTensorDim(i, getTensorDim(axes[i]));
  }

  return tmp;
}

bool TensorDim::operator==(const TensorDim &rhs) const {
  for (size_t i = 0; i < MAXDIM; ++i) {
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

unsigned int &TensorDim::operator[](const unsigned int index) {
  if (index >= MAXDIM)
    throw std::out_of_range(
      "[TensorDim] Tensor Dimension index should be between 0 and 4");
  return dim[index];
}

const unsigned int &TensorDim::operator[](const unsigned int index) const {
  if (index >= MAXDIM)
    throw std::out_of_range(
      "[TensorDim] Tensor Dimension index should be between 0 and 4");
  return dim[index];
}

void TensorDim::reverse() { std::reverse(dim, dim + MAXDIM); }

std::vector<int> TensorDim::getEffectiveDimension(bool dynamic) const {
  std::vector<int> eff_dim;
  eff_dim.reserve(eff_dim_flag.count());

  auto get_axis = [dynamic, this](unsigned int axis) -> int {
    if (dynamic && dyn_dim_flag[MAXDIM - axis - 1]) {
      return -1;
    }

    return dim[axis];
  };

  for (unsigned int i = 0; i < MAXDIM; ++i) {
    /// flip dim_flag to effectively match with our cognition
    /// ex) 3:5:1:1 -> 3:5, we are setting eff_dim_flag to 0b1100
    if (eff_dim_flag[MAXDIM - i - 1]) {
      eff_dim.push_back(get_axis(i));
    }
  }

  return eff_dim;
}

std::ostream &operator<<(std::ostream &out, TensorDim const &d) {
  out << "Shape: " << d.batch() << ":" << d.channel() << ":" << d.height()
      << ":" << d.width() << std::endl;
  return out;
}

} /* namespace nntrainer */
