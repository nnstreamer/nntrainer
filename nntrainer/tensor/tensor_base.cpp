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

TensorBase::TensorBase(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name_) :
  TensorBase(name_, d.getFormat()) {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();
    initializer = init;
  }
}

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

void TensorBase::allocateSrcTensor() {
  if (src_tensor) {
    data = src_tensor->tensor()->data;
    offset = src_tensor->tensor()->offset + src_tensor->offset();
  }
}

void TensorBase::createSharedDataTensor(const TensorBase *src, TensorBase *dest,
                                        size_t offset) const {
  /**
   * - If src already has data allocated, then directly make dest tensor based
   * on the src tensor.
   * - If src->data does not exist (meaning tensor does not memory allocated),
   * and src->src_tensor does not exist (meaning the src tensor does not depened
   * on another tensor), then create a SrcSharedTensor around the src.
   * - If src->src_tensor exists, then use the src->src_tensor to create the
   *  required SrcSharedTensor to avoid recursive dependency.
   *
   * @note src->data and src->src_tensor CAN co-exist. src->src_tensor is stored
   * if the batch size of src is updated and needs reallocation.
   */
  dest->data = nullptr;
  if (src->data) {
    dest->src_tensor = std::make_shared<SrcSharedTensorBase>(src, offset);
    dest->allocate();
  } else if (!src->src_tensor)
    dest->src_tensor = std::make_shared<SrcSharedTensorBase>(src, offset);
  else
    dest->src_tensor = std::make_shared<SrcSharedTensorBase>(
      src->src_tensor->tensor(), offset + src->src_tensor->offset());
}

TensorBase *TensorBase::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                            bool reset_stride,
                                            const std::string &name_) {
  TensorBase *ret = this;
  if (dim_.getFormat() != ret->dim.getFormat())
    throw std::invalid_argument("Tensor format does not match");

  ret->dim = dim_;
  if (!name_.empty())
    ret->name = name_;

  if (dim_.getDataLen() + offset > dim.getDataLen())
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");

  if (reset_stride)
    ret->strides = ret->dim.computeStrides();

  TensorDim new_match_dim = dim_;
  new_match_dim.batch(dim.batch());
  if (new_match_dim != dim && !reset_stride)
    ret->contiguous = false;

  /**
   * In this case, its the caller's responsibility to ensure that allocate() is
   * called for the output tensor before operating on the output tensor.
   */
  createSharedDataTensor(this, ret, offset);

  return ret;
}

} // namespace nntrainer
