// SPDX-License-Identifier: Apache-2.0
/**
 * @file	half_tensor.cpp
 * @date	01 December 2023
 * @brief	This is a HalfTensor class for 16-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <half_tensor.h>

namespace nntrainer {

HalfTensor::HalfTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::FP16) {}

void HalfTensor::allocate() {}

void HalfTensor::deallocate() {}

void *HalfTensor::getData() const { return (void *)0; }

void *HalfTensor::getData(size_t idx) const { return (void *)0; }

void *HalfTensor::getAddress(unsigned int i) { return (void *)0; }

const void *HalfTensor::getAddress(unsigned int i) const {
  return (const void *)0;
}

void HalfTensor::setValue(float value) {}

void HalfTensor::setValue(unsigned int batch, unsigned int c, unsigned int h,
                          unsigned int w, float value) {}

void HalfTensor::setZero() {}

void HalfTensor::initialize() {}

void HalfTensor::initialize(Initializer init) {}

void HalfTensor::print(std::ostream &out) const {}

} // namespace nntrainer
