// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	tensor_v2.cpp
 * @date	16 November 2023
 * @brief	This is Tensor class in Type erasure design
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stdio.h>

#include <lazy_tensor.h>
#include <tensor_v2.h>
#include <util_func.h>

namespace nntrainer {

TensorV2::TensorV2(std::string name_, Tformat fm, Tdatatype d_type) {
  if (d_type == Tdatatype::FP32) {
    object = std::shared_ptr<TensorBase<FloatTensor>>(
      new TensorBase<FloatTensor>(FloatTensor(name_, fm)),
      std::default_delete<TensorBase<FloatTensor>>());
  } else if (d_type == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    object = std::shared_ptr<TensorBase<HalfTensor>>(
      new TensorBase<HalfTensor>(HalfTensor(name_, fm)),
      std::default_delete<TensorBase<HalfTensor>>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

TensorV2::TensorV2(const TensorDim &d, bool alloc_now, Initializer init,
                   std::string name) {
  if (d.getDataType() == Tdatatype::FP32) {
    object = std::shared_ptr<TensorBase<FloatTensor>>(
      new TensorBase<FloatTensor>(FloatTensor(d, alloc_now, init, name)),
      std::default_delete<TensorBase<FloatTensor>>());
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    object = std::shared_ptr<TensorBase<HalfTensor>>(
      new TensorBase<HalfTensor>(HalfTensor(d, alloc_now, init, name)),
      std::default_delete<TensorBase<HalfTensor>>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

TensorV2::TensorV2(const TensorDim &d, const void *buf) {
  if (d.getDataType() == Tdatatype::FP32) {
    object = std::shared_ptr<TensorBase<FloatTensor>>(
      new TensorBase<FloatTensor>(FloatTensor(d, buf)),
      std::default_delete<TensorBase<FloatTensor>>());
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    object = std::shared_ptr<TensorBase<HalfTensor>>(
      new TensorBase<HalfTensor>(HalfTensor(d, buf)),
      std::default_delete<TensorBase<HalfTensor>>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }
}

TensorV2::TensorV2(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  object = std::shared_ptr<TensorBase<FloatTensor>>(
    new TensorBase<FloatTensor>(FloatTensor(d, t_type)),
    std::default_delete<TensorBase<FloatTensor>>());
}

#ifdef ENABLE_FP16
TensorV2::TensorV2(
  std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  object = std::shared_ptr<TensorBase<HalfTensor>>(
    new TensorBase<HalfTensor>(HalfTensor(d, t_type)),
    std::default_delete<TensorBase<HalfTensor>>());
}
#endif

void TensorV2::allocate() { object->allocate(); }

void TensorV2::deallocate() { object->deallocate(); }

bool TensorV2::isAllocated() const { return object->isAllocated(); }

const void *TensorV2::getData() const { return object->getData(); }

void *TensorV2::getData(size_t idx) const { return object->getData(idx); }

void *TensorV2::getAddress(unsigned int i) { return object->getAddress(i); }

const void *TensorV2::getAddress(unsigned int i) const {
  return object->getAddress(i);
}

void *TensorV2::getAddress(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w) {
  return getAddress(getIndex(b, c, h, w));
}

const void *TensorV2::getAddress(unsigned int b, unsigned int c, unsigned int h,
                                 unsigned int w) const {
  return getAddress(getIndex(b, c, h, w));
}

void TensorV2::setValue(float value) { object->setValue(value); }

void TensorV2::setZero() { object->setZero(); }

void TensorV2::setRandNormal(float mean, float std) {
  object->setRandNormal(mean, std);
}

void TensorV2::setRandUniform(float min, float max) {
  object->setRandUniform(min, max);
}

void TensorV2::setRandBernoulli(float probability) {
  object->setRandBernoulli(probability);
}

void TensorV2::initialize() { object->initialize(); }

void TensorV2::initialize(Initializer init) { object->initialize(init); }

void TensorV2::print(std::ostream &out) const { object->print(out); }

template <typename TensorClass>
TensorClass &TensorV2::operator=(const TensorClass &rhs) {
  object = rhs;
}

template <typename TensorClass>
TensorClass &TensorV2::operator=(TensorClass &&rhs) noexcept {
  object = rhs;
}

size_t TensorV2::getIndex(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const noexcept {
  return object->getIndex(b, c, h, w);
}

Initializer TensorV2::getInitializer() const {
  return object->getInitializer();
}

TensorDim::Format TensorV2::getFormat() const { return object->getFormat(); }

Tdatatype TensorV2::getDataType() const { return object->getDataType(); }

std::ostream &operator<<(std::ostream &out, TensorV2 const &m) {
  m.print(out);
  return out;
}

} // namespace nntrainer
