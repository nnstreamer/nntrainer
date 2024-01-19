// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_v2.cpp
 * @date	01 December 2023
 * @brief	This is a TensorV2 class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <float_tensor.h>
#include <tensor_v2.h>

#ifdef ENABLE_FP16
#include <half_tensor.h>
#endif

namespace nntrainer {

TensorV2::TensorV2(std::string name_, Tformat fm, Tdatatype d_type) {
  itensor = nullptr;

  if (d_type == Tdatatype::FP32) {
    itensor = new FloatTensor(name_, fm);
  } else if (d_type == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = new HalfTensor(name_, fm);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument(
      "Error: TensorV2 cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

TensorV2::TensorV2(const TensorDim &d, bool alloc_now, Initializer init,
                   std::string name) {
  itensor = nullptr;

  if (d.getDataType() == Tdatatype::FP32) {
    itensor = new FloatTensor(d, alloc_now, init, name);
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = new HalfTensor(d, alloc_now, init, name);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument(
      "Error: TensorV2 cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

TensorV2::TensorV2(const TensorDim &d, const void *buf) {
  itensor = nullptr;

  if (d.getDataType() == Tdatatype::FP32) {
    itensor = new FloatTensor(d, buf);
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = new HalfTensor(d, buf);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else {
    throw std::invalid_argument(
      "Error: TensorV2 cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

TensorV2::TensorV2(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  itensor = new FloatTensor(d, t_type.format);
}

#ifdef ENABLE_FP16
TensorV2::TensorV2(
  std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  itensor = new HalfTensor(d, t_type.format);
}
#endif

bool TensorV2::operator==(const TensorV2 &rhs) const {
  /// compares tensor information
  if (*itensor == *rhs.itensor) {
    /// compares tensor data
    if (getDataType() == Tdatatype::FP32) {
      return *dynamic_cast<FloatTensor *>(itensor) ==
             *dynamic_cast<FloatTensor *>(rhs.itensor);
    } else if (getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
      return *dynamic_cast<HalfTensor *>(itensor) ==
             *dynamic_cast<HalfTensor *>(rhs.itensor);
#else
      throw std::invalid_argument(
        "Error: HalfTensor cannot be created or used when FP16 is not enabled. "
        "Please check if the tensor data type is set properly.");
#endif
    }
  }
  return false;
}

void TensorV2::allocate() { itensor->allocate(); }

void TensorV2::deallocate() { itensor->deallocate(); }

bool TensorV2::isAllocated() { return itensor->isAllocated(); }

void TensorV2::setValue(float value) { itensor->setValue(value); }

void TensorV2::setValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value) {
  itensor->setValue(b, c, h, w, value);
}

void TensorV2::setZero() { itensor->setZero(); }

void TensorV2::setRandNormal(float mean, float stddev) {
  itensor->setRandNormal(mean, stddev);
}

void TensorV2::setRandUniform(float min, float max) {
  itensor->setRandUniform(min, max);
}

void TensorV2::setRandBernoulli(float probability) {
  itensor->setRandBernoulli(probability);
}

void TensorV2::initialize() { itensor->initialize(); }

void TensorV2::initialize(Initializer init) { itensor->initialize(init); }

TensorV2 TensorV2::apply(std::function<TensorV2(TensorV2)> f) const {
  return f(*this);
}

TensorV2 &TensorV2::apply(std::function<TensorV2 &(TensorV2, TensorV2 &)> f,
                          TensorV2 &output) const {
  return f(*this, output);
}

int TensorV2::multiply_i(float const &value) { return ML_ERROR_NONE; }

TensorV2 TensorV2::multiply(float const &value) const {
  TensorV2 t;
  return multiply(value, t);
}

TensorV2 &TensorV2::multiply(float const &value, TensorV2 &out) const {
  return out;
}

int TensorV2::multiply_i(TensorV2 const &m, const float beta) {
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::multiply(TensorV2 const &m, const float beta) const {
  TensorV2 t("", this->getFormat());
  return multiply(m, t, beta);
}

TensorV2 &TensorV2::multiply(TensorV2 const &m, TensorV2 &output,
                             const float beta) const {
  return output;
}

void TensorV2::print(std::ostream &out) const { itensor->print(out); }

void TensorV2::putData() const { itensor->putData(); }

void TensorV2::copy(const TensorV2 &from) {
  /// @todo enable copy to non-contiguous tensor
  if (!itensor->getContiguous()) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (from.size() != 0 && size() == from.size() &&
      getDataType() == from.getDataType()) {
    // if tensor size and data type match, copy data
    itensor->copy(from);
  } else {
    // replace with a new tensor that are the same with the given tensor
    if (from.getDataType() == ml::train::TensorDim::DataType::FP32) {
      TensorV2 t = TensorV2(from.getDim(), from.getData<float>());
      swap(t, *this);
    } else if (from.getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      TensorV2 t = TensorV2(from.getDim(), from.getData<_FP16>());
      swap(t, *this);
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }
  }
}

void TensorV2::copyData(const TensorV2 &from) { itensor->copyData(from); }

void TensorV2::copy_with_stride(const TensorV2 &from) {
  if (itensor->getDim() == from.getDim()) {
    // if the tensor dim matches, copy the data
    copy(from);
  } else {
    // replace with a new tensor that has the same data as the given tensor
    TensorV2 t = TensorV2(from.getDim(), true);
    for (unsigned int b = 0; b < t.batch(); ++b) {
      for (unsigned int c = 0; c < t.channel(); ++c) {
        for (unsigned int h = 0; h < t.height(); ++h) {
          for (unsigned int w = 0; w < t.width(); ++w) {
            if (getDataType() == ml::train::TensorDim::DataType::FP32) {
              t.setValue(b, c, h, w, from.getValue<float>(b, c, h, w));
            } else if (getDataType() == ml::train::TensorDim::DataType::FP16) {
              /// @todo remove #ifdef ENABLE_FP16
#ifdef ENABLE_FP16
              t.setValue(b, c, h, w, from.getValue<_FP16>(b, c, h, w));
#else
              throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
            }
          }
        }
      }
    }
    swap(t, *this);
  }
}

void TensorV2::reshape(const TensorDim &d) { itensor->reshape(d); }

TensorDim TensorV2::getDim() const { return itensor->getDim(); }

TensorDim::TensorType TensorV2::getTensorType() const {
  return itensor->getTensorType();
};

Initializer TensorV2::getInitializer() const {
  return itensor->getInitializer();
}

TensorDim::Format TensorV2::getFormat() const { return itensor->getFormat(); }

Tdatatype TensorV2::getDataType() const { return itensor->getDataType(); }

const bool TensorV2::getContiguous() const noexcept {
  return itensor->getContiguous();
}

const std::array<size_t, TensorDim::MAXDIM> TensorV2::getStrides() const
  noexcept {
  return itensor->getStrides();
}

bool TensorV2::checkContinuous(unsigned int np1, unsigned int np2) const {
  if (np1 < 0 || np1 > 3 || np2 < 0 || np2 > 3) {
    throw std::invalid_argument(
      "Error: Input value must be within the range of 0 to 3.");
  }

  if (getFormat() == Tformat::NCHW) {
    if (np1 + 1 == np2)
      return true;
  } else {
    std::vector<unsigned int> continuous_order_nhwc = {0, 3, 1, 2};
    if (continuous_order_nhwc[np2] == continuous_order_nhwc[np1] + 1)
      return true;
  }

  return false;
}

void TensorV2::setName(const std::string &name_) { itensor->setName(name_); }

const std::string &TensorV2::getName() const { return itensor->getName(); }

size_t TensorV2::getIndex(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const noexcept {
  return itensor->getIndex(b, c, h, w);
}

size_t TensorV2::size() const { return itensor->size(); }

bool TensorV2::empty() const { return itensor->empty(); }

size_t TensorV2::bytes() const { return itensor->bytes(); }

size_t TensorV2::batch() const { return itensor->batch(); }

size_t TensorV2::channel() const { return itensor->channel(); }

size_t TensorV2::height() const { return itensor->height(); }

size_t TensorV2::width() const { return itensor->width(); }

void TensorV2::createSharedDataTensor(const TensorV2 &src, TensorV2 &dest,
                                      size_t offset) const {
  itensor->createSharedDataTensor(src.itensor, dest.itensor, offset);
}

TensorV2 TensorV2::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                       bool reset_stride,
                                       const std::string &name_) const {
  TensorV2 ret = *this;
  ret.itensor = itensor->getSharedDataTensor(dim_, offset, reset_stride, name_);
  return ret;
}

} // namespace nntrainer
