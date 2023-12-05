// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_base.cpp
 * @date	04 December 2023
 * @brief	This is Tensor base class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <tensor_base.h>

namespace nntrainer {

void TensorBase::putData() const {
  if (!data)
    return;

  data->invalidate();
}

size_t TensorBase::getIndex(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w) const noexcept {
  if (getFormat() == Tformat::NCHW) {
    return (b * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]);
  } else {
    return (b * strides[0] + h * strides[1] + w * strides[2] + c * strides[3]);
  }
}

} // namespace nntrainer
