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
    itensor = std::shared_ptr<FloatTensor>(new FloatTensor(name_, fm),
                                           std::default_delete<FloatTensor>());
  } else if (d_type == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = std::shared_ptr<HalfTensor>(new HalfTensor(name_, fm),
                                          std::default_delete<HalfTensor>());
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
    itensor =
      std::shared_ptr<FloatTensor>(new FloatTensor(d, alloc_now, init, name),
                                   std::default_delete<FloatTensor>());
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor =
      std::shared_ptr<HalfTensor>(new HalfTensor(d, alloc_now, init, name),
                                  std::default_delete<HalfTensor>());
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
    itensor = std::shared_ptr<FloatTensor>(new FloatTensor(d, buf),
                                           std::default_delete<FloatTensor>());
  } else if (d.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = std::shared_ptr<HalfTensor>(new HalfTensor(d, buf),
                                          std::default_delete<HalfTensor>());
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
  itensor = std::shared_ptr<FloatTensor>(new FloatTensor(d, t_type.format),
                                         std::default_delete<FloatTensor>());
}

#ifdef ENABLE_FP16
TensorV2::TensorV2(
  std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
  ml::train::TensorDim::TensorType t_type) {
  itensor = std::shared_ptr<HalfTensor>(new HalfTensor(d, t_type.format),
                                        std::default_delete<HalfTensor>());
}
#endif

bool TensorV2::operator==(const TensorV2 &rhs) const {
  /// compares tensor information
  if (*itensor == *rhs.itensor) {
    /// compares tensor data
    if (getDataType() == Tdatatype::FP32) {
      return *std::dynamic_pointer_cast<FloatTensor>(itensor) ==
             *std::dynamic_pointer_cast<FloatTensor>(rhs.itensor);
    } else if (getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
      return *std::dynamic_pointer_cast<HalfTensor>(itensor) ==
             *std::dynamic_pointer_cast<HalfTensor>(rhs.itensor);
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

void TensorV2::addValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, float value, float beta) noexcept {
  itensor->addValue(b, c, h, w, value, beta);
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

int TensorV2::multiply_i_strided(TensorV2 const &m, const float beta) {
  try {
    this->multiply_strided(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::multiply_strided(TensorV2 const &m, const float beta) const {
  TensorV2 t;
  return this->multiply_strided(m, t, beta);
}

TensorV2 &TensorV2::multiply_strided(TensorV2 const &m, TensorV2 &output,
                                     const float beta) const {
  itensor->multiply_strided(m, output, beta);
  return output;
}

int TensorV2::multiply_i(float const &value) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  return itensor->multiply_i(value);
}

TensorV2 TensorV2::multiply(float const &value) const {
  TensorV2 t;
  return multiply(value, t);
}

TensorV2 &TensorV2::multiply(float const &value, TensorV2 &out) const {
  itensor->multiply(value, out);
  return out;
}

int TensorV2::multiply_i(TensorV2 const &m, const float beta) {
  try {
    this->multiply(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::multiply(TensorV2 const &m, const float beta) const {
  TensorV2 t("", this->getFormat());
  return multiply(m, t, beta);
}

TensorV2 &TensorV2::multiply(TensorV2 const &m, TensorV2 &output,
                             const float beta) const {
  itensor->multiply(m, output, beta);
  return output;
}

int TensorV2::divide_i(float const &value) {
  if (value == 0.0f) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  this->divide(value, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::divide(float const &value) const {
  TensorV2 output("", getFormat(), getDataType());
  return divide(value, output);
}

TensorV2 &TensorV2::divide(float const &value, TensorV2 &output) const {
  /// @todo add unittest, ZeroDivisionError
  if (value == 0.0f) {
    std::stringstream ss;
    ss << "[Tensor] divide by value failed, value: " << value;
    throw std::invalid_argument(ss.str().c_str());
  }
  itensor->divide(value, output);
  return output;
}

int TensorV2::divide_i(TensorV2 const &m) {
  try {
    this->divide(m, *this);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::divide(TensorV2 const &m) const {
  TensorV2 output("", getFormat(), getDataType());
  return this->divide(m, output);
}

TensorV2 &TensorV2::divide(TensorV2 const &m, TensorV2 &output) const {
  NNTR_THROW_IF(!getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot divide";
  itensor->divide(m, output);
  return output;
}

int TensorV2::add_i_strided(TensorV2 const &input, const float beta) {
  try {
    this->add_strided(input, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

TensorV2 TensorV2::add_strided(TensorV2 const &input, const float beta) const {
  TensorV2 output("", getFormat(), getDataType());
  return this->add_strided(input, output, beta);
}

TensorV2 &TensorV2::add_strided(TensorV2 const &input, TensorV2 &output,
                                const float beta) const {
  CREATE_V2_IF_EMPTY_DIMS(output, getDim(), nullptr);

  if (size() != input.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided addition does not support broadcasting");

  itensor->add_strided(input, output, beta);

  return output;
}

int TensorV2::add_i(float const &value) {
  this->add(value, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::add(float const &value) const {
  TensorV2 t("", getFormat(), getDataType());
  return add(value, t);
}

TensorV2 &TensorV2::add(float const &value, TensorV2 &output) const {
  itensor->add(value, output);
  return output;
}

int TensorV2::add_i(TensorV2 const &m, float const alpha) {
  try {
    this->add(m, *this, alpha);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::add(TensorV2 const &m, float const alpha) const {
  TensorV2 t("", getFormat(), getDataType());
  return this->add(m, t, alpha);
}

TensorV2 &TensorV2::add(TensorV2 const &m, TensorV2 &output,
                        float const alpha) const {
  NNTR_THROW_IF(!itensor->getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot add";
  itensor->add(m, output, alpha);
  return output;
}

int TensorV2::subtract_i(float const &value) {
  this->subtract(value, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::subtract(float const &value) const {
  TensorV2 output("", getFormat(), getDataType());
  return subtract(value, output);
}

TensorV2 &TensorV2::subtract(float const &value, TensorV2 &output) const {
  itensor->subtract(value, output);
  return output;
}

int TensorV2::subtract_i(TensorV2 const &m) { return add_i(m, -1); }

TensorV2 TensorV2::subtract(TensorV2 const &m) const { return add(m, -1); }

TensorV2 &TensorV2::subtract(TensorV2 const &m, TensorV2 &output) const {
  return add(m, output, -1);
}

int TensorV2::pow_i(float exponent) {
  pow(exponent, *this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::pow(float exponent) const {
  TensorV2 output("", getFormat(), getDataType());
  return pow(exponent, output);
}

TensorV2 &TensorV2::pow(float exponent, TensorV2 &output) const {
  itensor->pow(exponent, output);
  return output;
}

int TensorV2::erf_i() {
  erf(*this);
  return ML_ERROR_NONE;
}

TensorV2 TensorV2::erf() const {
  TensorV2 output("", getFormat(), getDataType());
  return erf(output);
}

TensorV2 &TensorV2::erf(TensorV2 &output) const {
  itensor->erf(output);
  return output;
}

TensorV2 TensorV2::dot(TensorV2 const &input, bool trans, bool trans_in) const {
  TensorV2 output("", this->getFormat(), this->getDataType());
  dot(input, output, trans, trans_in);

  return output;
}

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor
 * in case of trans is false.
 */
TensorV2 &TensorV2::dot(TensorV2 const &input, TensorV2 &output, bool trans,
                        bool trans_in, float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot dot product.";

  itensor->dot(input, output, trans, trans_in, beta);
  return output;
}

TensorV2 &TensorV2::dot_deriv_wrt_1(TensorV2 const &m,
                                    TensorV2 const &output_deriv, bool trans,
                                    bool trans_m, float beta) {
  bool deriv_trans_m = true;
  bool deriv_trans = false;
  /** @todo handle all cases of trans and trans_m */
  if (!trans && trans_m) {
    deriv_trans_m = false;
  }

  return output_deriv.dot(m, *this, deriv_trans, deriv_trans_m, beta);
}

/**
 * @brief compute the derivative wrt m in the m tensor
 * @note The caller tensor must be the same tensor as the one which called the
 * dot() product.
 */
TensorV2 &TensorV2::dot_deriv_wrt_2(TensorV2 &m_deriv,
                                    TensorV2 const &output_deriv, bool trans,
                                    bool trans_m, float beta) const {
  bool deriv_trans_m = false;
  bool deriv_trans = true;
  /** @todo handle all cases of trans and trans_m */

  if (!trans && trans_m) {
    output_deriv.dot(*this, m_deriv, deriv_trans, deriv_trans_m, beta);
    return m_deriv;
  } else {
    return dot(output_deriv, m_deriv, deriv_trans, deriv_trans_m, beta);
  }
}

TensorV2 &TensorV2::dotBatched(TensorV2 const &m, TensorV2 &result, bool trans,
                               bool trans_m, float beta) const {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const TensorV2 this_b = this->getBatchSlice(b, 1);
    TensorV2 m_b = m.getBatchSlice(b, 1);
    TensorV2 result_b = result.getBatchSlice(b, 1);

    this_b.dot(m_b, result_b, trans, trans_m, beta);
  }

  return result;
}

TensorV2 &TensorV2::dot_batched_deriv_wrt_1(TensorV2 const &m,
                                            TensorV2 const &output_deriv,
                                            bool trans, bool trans_m,
                                            float beta) {
  bool deriv_trans_m = true;
  bool deriv_trans = false;
  /** @todo handle all cases of trans and trans_m */
  if (!trans && trans_m) {
    deriv_trans_m = false;
  }

  return output_deriv.dotBatched(m, *this, deriv_trans, deriv_trans_m, beta);
}

TensorV2 &TensorV2::dot_batched_deriv_wrt_2(TensorV2 &m_deriv,
                                            TensorV2 const &output_deriv,
                                            bool trans, bool trans_m,
                                            float beta) const {
  bool deriv_trans_m = false;
  bool deriv_trans = true;
  /** @todo handle all cases of trans and trans_m */

  if (!trans && trans_m) {
    output_deriv.dotBatched(*this, m_deriv, deriv_trans, deriv_trans_m, beta);
    return m_deriv;
  } else {
    return dotBatched(output_deriv, m_deriv, deriv_trans, deriv_trans_m, beta);
  }
}

TensorV2 TensorV2::dropout_mask(float dropout) const {
  TensorV2 output(getDim());
  output.dropout_mask(dropout);
  return output;
}

void TensorV2::dropout_mask(float dropout) {
  /// @todo add unittest
  NNTR_THROW_IF(dropout < 0 || dropout > 1, std::invalid_argument)
    << "[Tensor::dropout_mask] Dropout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(dropout) == FP_ZERO)
    return;

  setRandUniform(0.0, 1.0);
  itensor->dropout_mask(dropout);
}

void TensorV2::filter_mask(const TensorV2 &mask_len, bool reverse) {
  /// @todo add unittest
  itensor->filter_mask(mask_len, reverse);
}

TensorV2 TensorV2::zoneout_mask(float zoneout) {
  TensorV2 output(getDim());
  zoneout_mask(output, zoneout);
  return output;
}

void TensorV2::zoneout_mask(TensorV2 &opposite, float zoneout) {
  NNTR_THROW_IF(getDim() != opposite.getDim(), std::invalid_argument)
    << "[Tensor::zoneout_mask] opposite dimension does not match";

  NNTR_THROW_IF(zoneout < 0 || zoneout > 1, std::invalid_argument)
    << "[Tensor::zoneout_mask] Zoneout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(zoneout) == FP_ZERO)
    return;

  itensor->zoneout_mask(opposite, zoneout);
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

TensorV2 TensorV2::getBatchSlice(size_t offset, unsigned int size) const {
  TensorDim dim_ = getDim();
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->getDim().getFeatureLen(),
                             true, "");
}

TensorV2 TensorV2::clone() const {
  TensorV2 output(getName(), getFormat(), getDataType());
  output.copy(*this);
  return output;
}

TensorV2 TensorV2::transpose(const std::string &direction) const {
  TensorV2 output(getDim());
  transpose(direction, output);
  return output;
}

TensorV2 &TensorV2::transpose(const std::string &direction,
                              TensorV2 &output) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot transpose.";

  if (output.getData<char>() == getData<char>()) {
    TensorV2 result = clone();
    return result.transpose(direction, output);
  }

  itensor->transpose(direction, output);

  return output;
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

const std::array<size_t, TensorDim::MAXDIM>
TensorV2::getStrides() const noexcept {
  return itensor->getStrides();
}

bool TensorV2::checkContinuous(unsigned int np1, unsigned int np2) const {
  if (np1 > 3 || np2 > 3) {
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
  itensor->createSharedDataTensor(src.itensor.get(), dest.itensor.get(),
                                  offset);
}

TensorV2 TensorV2::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                       bool reset_stride,
                                       const std::string &name_) const {
  TensorV2 ret = *this;
  itensor->getSharedDataTensor(dim_, offset, reset_stride, name_,
                               ret.itensor.get());
  return ret;
}

} // namespace nntrainer
