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
#include <tensor_v2.h>

namespace nntrainer {

/**
 * @struct External Loop Info for broadcasted info
 * @brief External Loop Info for broadcasted iteration. Please refer to
 * DISABLED_private_external_loop_n in unittest_nntrainer_tensor.
 * @note This should better be implemented in iterator fashion before used
 * extensively.
 */
struct TensorBase::BroadcastInfoV2 {

  /**
   * @brief Construct a new External Loop Info object
   *
   */
  BroadcastInfoV2() :
    buffer_size(0),
    buffer_axis(-1),
    strides{0, 0, 0, 0},
    tensor_type(nntrainer::TensorDim::TensorType()) {}

  unsigned int buffer_size; /**< virtual size of the buffer */
  int buffer_axis;          /**< the smallest axis that should be looped.
                                 -1 means no loop needed*/
  std::array<unsigned int, TensorDim::MAXDIM>
    strides; /**< modified strides for the loop */
  nntrainer::TensorDim::TensorType tensor_type;
};

TensorBase::TensorBase(const TensorDim &d, bool alloc_now, Initializer init,
                       std::string name_) :
  TensorBase(name_, d.getFormat()) {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();
    initializer = init;
  }
}

bool TensorBase::operator==(const TensorBase &rhs) const {
  if (this->dim != rhs.dim)
    return false;

  if (size() != rhs.size())
    return false;

  if (contiguous != rhs.contiguous)
    return false;

  if (strides != rhs.strides)
    return false;

  return true;
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

TensorBase::BroadcastInfoV2
TensorBase::computeBroadcastInfo(const TensorV2 &m) const {
  if (m.size() > this->size())
    throw exception::not_supported("broadcasting *this is not supported");

  const TensorDim m_dim = m.getDim();

  BroadcastInfoV2 e;
  e.tensor_type = getTensorType();

  uint continuity[4] = {0, 1, 2, 3};
  if (getFormat() == Tformat::NHWC) {
    continuity[1] = 2;
    continuity[2] = 3;
    continuity[3] = 1;
  }

  /// checking if given Tensor's can be broadcasted
  for (unsigned int i = 0; i < TensorDim::MAXDIM; ++i) {
    if (dim.getTensorDim(continuity[i]) == m_dim.getTensorDim(continuity[i])) {
      e.strides[i] = m.getStrides()[i];
      continue;
    }

    /// If given dimension is 1, it could be reused, the stride remaining 0
    /// Need to check if dim[i] == 1 && m_dim[i] == 1 first though
    /// If so, strides should not change
    if (m_dim.getTensorDim(continuity[i]) == 1) {
      continue;
    }

    std::stringstream ss;
    ss << "[computeBroadcastInfo] broadcasting only allowed for "
          "dimension value of 1 \n"
       << "this: " << dim << "target: " << m_dim;
    throw std::invalid_argument(ss.str().c_str());
  }

  /// calculate inner loop size
  e.buffer_size = 1;
  e.buffer_axis = -1;
  e.strides[3] = m.getStrides()[3];

  /// initiate buffer info with matching dimension strategy
  for (int axis = 3; axis >= 0; --axis) {
    if (dim.getTensorDim(continuity[axis]) !=
        m_dim.getTensorDim(continuity[axis])) {
      e.buffer_axis = axis;
      break;
    }

    e.buffer_size *= dim.getTensorDim(continuity[axis]);
  }

  /// check strategy that uses consecutive ones
  if (m_dim.getTensorDim(continuity[3]) == 1) {
    unsigned int inner_loop_size = 1;
    int axis;
    for (axis = 3; axis >= 0; --axis) {
      if (m_dim.getTensorDim(continuity[axis]) != 1) {
        break;
      }

      inner_loop_size *= dim.getTensorDim(continuity[axis]);
    }

    /// if consecutive-one strategy has bigger chunk size, replace the
    /// information
    if (inner_loop_size > e.buffer_size) {
      e.buffer_axis = axis;
      e.buffer_size = inner_loop_size;
      e.strides[3] = 0;
    }
  }

  return e;
}

} // namespace nntrainer
