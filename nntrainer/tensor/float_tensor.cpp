// SPDX-License-Identifier: Apache-2.0
/**
 * @file	float_tensor.cpp
 * @date	01 December 2023
 * @brief	This is FloatTensor class for 32-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <float_tensor.h>

namespace nntrainer {

FloatTensor::FloatTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::FP32) {}

void FloatTensor::allocate() {}

void FloatTensor::deallocate() {}

void *FloatTensor::getData() const { return (void *)0; }

void *FloatTensor::getData(size_t idx) const { return (void *)0; }

void *FloatTensor::getAddress(unsigned int i) { return (void *)0; }

const void *FloatTensor::getAddress(unsigned int i) const {
  return (const void *)0;
}

void FloatTensor::setValue(float value) {}

void FloatTensor::setValue(unsigned int batch, unsigned int c, unsigned int h,
                           unsigned int w, float value) {}

void FloatTensor::setZero() {}

void FloatTensor::initialize() {}

void FloatTensor::initialize(Initializer init) {}

void FloatTensor::print(std::ostream &out) const {}

} // namespace nntrainer
