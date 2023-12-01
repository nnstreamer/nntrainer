// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	tensor_v2.cpp
 * @date	16 November 2023
 * @brief	This is Tensor class in Type erasure design
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
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

bool TensorV2::operator==(const TensorV2 &rhs) const {
  if (object->getDim() != rhs.object->getDim())
    return false;

  size_t len = object->size();

  if (len != rhs.object->size())
    return false;

  if (object->getContiguous() != rhs.object->getContiguous())
    return false;

  if (object->getStrides() != rhs.object->getStrides())
    return false;

  for (size_t i = 0; i < len; ++i) {
    /** not checking sign change is intentional to avoid float calculation
     * errors around 0 */
    const float *_data = (float *)getData(i);
    const float *_rdata = (float *)rhs.getData(i);

    if ((std::isnan(*_data) && !std::isnan(*_rdata)) ||
        (!std::isnan(*_data) && std::isnan(*_rdata)) ||
        std::fabs(*_data - *_rdata) > 1e-5)
      return false;
  }

  return true;
}

bool TensorV2::operator!=(const TensorV2 &rhs) const {
  return !(this->object == rhs.object);
}

void TensorV2::allocate() { object->allocate(); }

void TensorV2::deallocate() { object->deallocate(); }

bool TensorV2::isAllocated() const { return object->isAllocated(); }

void TensorV2::setData(const std::shared_ptr<MemoryData> buf, size_t off,
                       bool init) {
  object->setData(buf, off, init);
}

const void *TensorV2::getData() const { return object->getData(); }

void *TensorV2::getData(size_t idx) const { return object->getData(idx); }

unsigned int TensorV2::sizeofData() const { return object->sizeofData(); }

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

void TensorV2::setValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value) noexcept {
  object->setValue(b, c, h, w, value);
}

void TensorV2::addValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value, float beta) noexcept {
  object->addValue(b, c, h, w, value, beta);
}

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

size_t TensorV2::size() const { return object->size(); }

bool TensorV2::empty() const { return size() == 0; }

size_t TensorV2::bytes() const { return size() * sizeofData(); }

size_t TensorV2::getIndex(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const noexcept {
  return object->getIndex(b, c, h, w);
}

bool TensorV2::checkContinuous(unsigned int n, unsigned int np1) const {
  std::vector<unsigned int> continuous_order_nhwc = {0, 3, 1, 2};
  bool continuous = false;
  if (getFormat() == Tformat::NHWC) {
    if (continuous_order_nhwc[np1] == continuous_order_nhwc[n] + 1)
      continuous = true;
  } else {
    if (n + 1 == np1)
      continuous = true;
  }
  return continuous;
}

void TensorV2::setName(const std::string &name_) { object->setName(name_); }

const std::string &TensorV2::getName() const { return object->getName(); }

Initializer TensorV2::getInitializer() const {
  return object->getInitializer();
}

TensorDim TensorV2::getDim() const { return object->getDim(); }

const std::array<size_t, TensorDim::MAXDIM> TensorV2::getStrides() const
  noexcept {
  return object->getStrides();
}

bool TensorV2::getContiguous() const { return object->getContiguous(); }

TensorDim::TensorType TensorV2::getTensorType() const {
  return object->getTensorType();
}

size_t TensorV2::batch() const { return object->batch(); }

size_t TensorV2::channel() const { return object->channel(); }

size_t TensorV2::height() const { return object->height(); }

size_t TensorV2::width() const { return object->width(); }

uint TensorV2::getDataTypeSize() const { return object->getDataTypeSize(); }

TensorDim::Format TensorV2::getFormat() const { return object->getFormat(); }

Tdatatype TensorV2::getDataType() const { return object->getDataType(); }

TensorV2 TensorV2::apply(std::function<TensorV2(TensorV2)> f) const {
  return f(*this);
}

TensorV2 &TensorV2::apply(std::function<TensorV2 &(TensorV2, TensorV2 &)> f,
                          TensorV2 &output) const {
  return f(*this, output);
}

std::ostream &operator<<(std::ostream &out, TensorV2 const &m) {
  m.print(out);
  return out;
}

} // namespace nntrainer
