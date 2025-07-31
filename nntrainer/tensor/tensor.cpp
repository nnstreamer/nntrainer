// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor.cpp
 * @date	01 December 2023
 * @brief	This is a Tensor class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <char_tensor.h>
#include <float_tensor.h>
#include <lazy_tensor.h>
#include <short_tensor.h>
#include <tensor.h>

#ifdef ENABLE_FP16
#include <half_tensor.h>
#endif
namespace nntrainer {

Tensor::Tensor(std::string name_, Tformat fm, Tdatatype d_type) {
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
  } else if (d_type == Tdatatype::UINT16) {
    itensor = std::shared_ptr<ShortTensor>(new ShortTensor(name_, fm),
                                           std::default_delete<ShortTensor>());
  } else if (d_type == Tdatatype::QINT8) {
    itensor = std::shared_ptr<CharTensor>(new CharTensor(name_, fm),
                                          std::default_delete<CharTensor>());
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

Tensor::Tensor(const TensorDim &d, bool alloc_now, Initializer init,
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
  } else if (d.getDataType() == Tdatatype::UINT16) {
    itensor =
      std::shared_ptr<ShortTensor>(new ShortTensor(d, alloc_now, init, name),
                                   std::default_delete<ShortTensor>());
  } else if (d.getDataType() == Tdatatype::QINT8) {
    itensor =
      std::shared_ptr<CharTensor>(new CharTensor(d, alloc_now, init, name),
                                  std::default_delete<CharTensor>());
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

Tensor::Tensor(const TensorDim &d, const void *buf) {
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
  } else if (d.getDataType() == Tdatatype::UINT16) {
    itensor = std::shared_ptr<ShortTensor>(new ShortTensor(d, buf),
                                           std::default_delete<ShortTensor>());
  } else if (d.getDataType() == Tdatatype::QINT8) {
    itensor = std::shared_ptr<CharTensor>(new CharTensor(d, buf),
                                          std::default_delete<CharTensor>());
  } else {
    throw std::invalid_argument(
      "Error: Tensor cannot be constructed because the given d_type is not "
      "compatible with itensor. The supported d_types are: FP32, FP16 "
      "(if built with ENABLE_FP16).");
  }
}

Tensor::Tensor(const Tensor &rhs) {
  if (rhs.getDataType() == Tdatatype::FP32) {
    itensor = std::shared_ptr<FloatTensor>(new FloatTensor(*rhs.itensor),
                                           std::default_delete<FloatTensor>());
  } else if (rhs.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = std::shared_ptr<HalfTensor>(new HalfTensor(*rhs.itensor),
                                          std::default_delete<HalfTensor>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (rhs.getDataType() == Tdatatype::UINT16) {
    itensor = std::shared_ptr<ShortTensor>(new ShortTensor(*rhs.itensor),
                                           std::default_delete<ShortTensor>());
  } else if (rhs.getDataType() == Tdatatype::QINT8) {
    itensor = std::shared_ptr<CharTensor>(new CharTensor(*rhs.itensor),
                                          std::default_delete<CharTensor>());
  }
}

Tensor::Tensor(std::shared_ptr<TensorBase> rhs) {
  NNTR_THROW_IF(rhs == nullptr, std::invalid_argument)
    << "Error: received a nullptr. Tensor cannot be constructed";

  itensor = rhs;
}

Tensor &Tensor::operator=(const Tensor &rhs) {
  if (rhs.getDataType() == Tdatatype::FP32) {
    itensor = std::shared_ptr<FloatTensor>(new FloatTensor(*rhs.itensor),
                                           std::default_delete<FloatTensor>());
  } else if (rhs.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
    itensor = std::shared_ptr<HalfTensor>(new HalfTensor(*rhs.itensor),
                                          std::default_delete<HalfTensor>());
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  } else if (rhs.getDataType() == Tdatatype::UINT16) {
    itensor = std::shared_ptr<ShortTensor>(new ShortTensor(*rhs.itensor),
                                           std::default_delete<ShortTensor>());
  } else if (rhs.getDataType() == Tdatatype::QINT8) {
    itensor = std::shared_ptr<CharTensor>(new CharTensor(*rhs.itensor),
                                          std::default_delete<CharTensor>());
  }
  return *this;
}

bool Tensor::operator==(const Tensor &rhs) const {
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
    } else if (getDataType() == Tdatatype::UINT16) {
      return *std::dynamic_pointer_cast<ShortTensor>(itensor) ==
             *std::dynamic_pointer_cast<ShortTensor>(rhs.itensor);
    } else if (getDataType() == Tdatatype::QINT8) {
      return *std::dynamic_pointer_cast<CharTensor>(itensor) ==
             *std::dynamic_pointer_cast<CharTensor>(rhs.itensor);
    }
  }
  return false;
}

void Tensor::allocate() { itensor->allocate(); }

void Tensor::deallocate() { itensor->deallocate(); }

bool Tensor::isAllocated() { return itensor->isAllocated(); }

void Tensor::setValue(float value) { itensor->setValue(value); }

void Tensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  itensor->setValue(b, c, h, w, value);
}

void Tensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w, float value, float beta) noexcept {
  itensor->addValue(b, c, h, w, value, beta);
}

void Tensor::setZero() { itensor->setZero(); }

void Tensor::setRandNormal(float mean, float stddev) {
  itensor->setRandNormal(mean, stddev);
}

void Tensor::setRandUniform(float min, float max) {
  itensor->setRandUniform(min, max);
}

void Tensor::setRandBernoulli(float probability) {
  itensor->setRandBernoulli(probability);
}

void Tensor::initialize() { itensor->initialize(); }

void Tensor::initialize(Initializer init) { itensor->initialize(init); }

Tensor Tensor::apply(std::function<Tensor(Tensor)> f) const { return f(*this); }

Tensor &Tensor::apply(std::function<Tensor &(Tensor, Tensor &)> f,
                      Tensor &output) const {
  return f(*this, output);
}

int Tensor::multiply_i_strided(Tensor const &m, const float beta) {
  try {
    this->multiply_strided(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::multiply_strided(Tensor const &m, const float beta) const {
  Tensor t("", getFormat(), getDataType());
  return this->multiply_strided(m, t, beta);
}

Tensor &Tensor::multiply_strided(Tensor const &m, Tensor &output,
                                 const float beta) const {
  itensor->multiply_strided(m, output, beta);
  return output;
}

int Tensor::multiply_i(float const &value) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  return itensor->multiply_i(value);
}

Tensor Tensor::multiply(float const &value) const {
  Tensor t("", getFormat(), getDataType());
  return multiply(value, t);
}

Tensor &Tensor::multiply(float const &value, Tensor &out) const {
  itensor->multiply(value, out);
  return out;
}

int Tensor::multiply_i(Tensor const &m, const float beta) {
  try {
    this->multiply(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::multiply(Tensor const &m, const float beta) const {
  Tensor t("", this->getFormat());
  return multiply(m, t, beta);
}

Tensor &Tensor::multiply(Tensor const &m, Tensor &output,
                         const float beta) const {
  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "Tensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  NNTR_THROW_IF(!getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";
  itensor->multiply(m, output, beta);
  return output;
}

int Tensor::divide_i(float const &value) {
  if (value == 0.0f) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  this->divide(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::divide(float const &value) const {
  Tensor output("", getFormat(), getDataType());
  return divide(value, output);
}

Tensor &Tensor::divide(float const &value, Tensor &output) const {
  /// @todo add unittest, ZeroDivisionError
  if (value == 0.0f) {
    std::stringstream ss;
    ss << "[Tensor] divide by value failed, value: " << value;
    throw std::invalid_argument(ss.str().c_str());
  }
  itensor->divide(value, output);
  return output;
}

int Tensor::divide_i(Tensor const &m) {
  try {
    this->divide(m, *this);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::divide(Tensor const &m) const {
  Tensor output("", getFormat(), getDataType());
  return this->divide(m, output);
}

Tensor &Tensor::divide(Tensor const &m, Tensor &output) const {
  NNTR_THROW_IF(!getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot divide";
  itensor->divide(m, output);
  return output;
}

int Tensor::add_i_strided(Tensor const &input, const float beta) {
  try {
    this->add_strided(input, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::add_strided(Tensor const &input, const float beta) const {
  Tensor output("", getFormat(), getDataType());
  return this->add_strided(input, output, beta);
}

Tensor &Tensor::add_strided(Tensor const &input, Tensor &output,
                            const float beta) const {
  CREATE_IF_EMPTY_DIMS(output, getDim(), nullptr);

  if (size() != input.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided addition does not support broadcasting");

  itensor->add_strided(input, output, beta);

  return output;
}

int Tensor::add_i(float const &value) {
  this->add(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::add(float const &value) const {
  Tensor t("", getFormat(), getDataType());
  return add(value, t);
}

Tensor &Tensor::add(float const &value, Tensor &output) const {
  itensor->add(value, output);
  return output;
}

int Tensor::add_i(Tensor const &m, float const alpha) {
  return itensor->add_i(m, *this, alpha);
}

int Tensor::add_i_partial(unsigned int len, unsigned int addr_idx, Tensor &m,
                          unsigned int incX, unsigned int incY,
                          const Tensor alphas, unsigned int alpha_idx) {
  return itensor->add_i_partial(len, addr_idx, m, incX, incY, alphas,
                                alpha_idx);
}

Tensor Tensor::add(Tensor const &m, float const alpha) const {
  Tensor t("", getFormat(), getDataType());
  return this->add(m, t, alpha);
}

Tensor &Tensor::add(Tensor const &m, Tensor &output, float const alpha) const {
  NNTR_THROW_IF(!itensor->getContiguous() || !m.getContiguous() ||
                  !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot add";
  itensor->add(m, output, alpha);
  return output;
}

int Tensor::subtract_i(float const &value) {
  this->subtract(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::subtract(float const &value) const {
  Tensor output("", getFormat(), getDataType());
  return subtract(value, output);
}

Tensor &Tensor::subtract(float const &value, Tensor &output) const {
  itensor->subtract(value, output);
  return output;
}

int Tensor::subtract_i(Tensor const &m) { return add_i(m, -1); }

Tensor Tensor::subtract(Tensor const &m) const {
  Tensor t;
  return this->subtract(m, t);
}

Tensor &Tensor::subtract(Tensor const &m, Tensor &output) const {
  return add(m, output, -1);
}

/**
 * This is to sum the Tensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1, 1) dimension.
 */
Tensor Tensor::sum_by_batch() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  Tensor output(batch(), 1, 1, 1, this->getFormat(), getDataType());
  itensor->sum_by_batch(output);
  return output;
}

Tensor Tensor::sum(unsigned int axis, float alpha) const {
  Tensor output("", this->getFormat(), this->getDataType());
  return sum(axis, output, alpha, 0);
}

Tensor &Tensor::sum(unsigned int axis, Tensor &output, float alpha,
                    float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  itensor->sum(axis, output, alpha, beta);
  return output;
}

Tensor Tensor::sum(const std::vector<unsigned int> &axes, float alpha) const {
  Tensor output("", this->getFormat());
  return sum(axes, output, alpha);
}

Tensor &Tensor::sum(const std::vector<unsigned int> &axes, Tensor &output,
                    float alpha) const {
  if (axes.empty())
    throw std::invalid_argument("empty axes given");

  if (axes.size() == 1) {
    this->sum(axes[0], output, alpha);
  } else {

    /** club axes together */
    Tensor new_reshaped = Tensor(getDim());
    new_reshaped.copy(*this);
    std::vector<unsigned int> continuous_order = {0, 3, 1, 2};
    std::vector<unsigned int> new_axes = {axes[0]};

    for (unsigned int i = 1; i < axes.size(); ++i) {
      if (checkContinuous(axes[i - 1], axes[i])) {
        new_reshaped.mergeAxis(axes[i - 1], axes[i]);
        new_axes.back() = axes[i];
      } else {
        new_axes.push_back(axes[i]);
      }
    }

    Tensor ret = new_reshaped.sum(new_axes[0]);
    for (unsigned int i = 1; i < new_axes.size() - 1; ++i)
      ret = ret.sum(axes[i]);
    ret.sum(new_axes.back(), output, alpha);
  }
  return output;
}

Tensor Tensor::average(unsigned int axis) const {
  Tensor output("", this->getFormat(), this->getDataType());
  return average(axis, output);
}

Tensor &Tensor::average(unsigned int axis, Tensor &output) const {
  if (axis >= TensorDim::MAXDIM)
    throw std::out_of_range(
      "negative axis or axis more then MAXDIM is invalid");

  unsigned int axis_size = getDim()[axis];
  if (axis_size == 1)
    output.copy(*this);
  else
    this->sum(axis, output, 1.0 / ((float)axis_size));

  return output;
}

Tensor Tensor::average(const std::vector<unsigned int> &axes) const {
  Tensor output("", this->getFormat(), this->getDataType());
  return average(axes, output);
}

Tensor &Tensor::average(const std::vector<unsigned int> &axes,
                        Tensor &output) const {
  if (axes.empty())
    return this->average(output);

  TensorDim ret_shape(getTensorType());

  for (const auto &idx : axes) {
    if (idx >= TensorDim::MAXDIM) {
      throw std::out_of_range("axis more then MAXDIM is invalid");
    }
    ret_shape.setTensorDim(idx, getDim().getTensorDim(idx));
  }

  return this->sum(axes, output, 1.0 / (float)ret_shape.getDataLen());
}

Tensor Tensor::average() const {
  Tensor output = *this;
  unsigned int axis = 0;
  if (this->getFormat() == Tformat::NHWC) {
    output.reshape({1, getDim().getDataLen(), 1, 1, this->getTensorType()});
    axis = 1;
  } else {
    output.reshape({1, 1, 1, getDim().getDataLen(), this->getTensorType()});
    axis = 3;
  }
  return output.average(axis);
}

Tensor &Tensor::average(Tensor &output) const {
  Tensor result = *this;
  result.reshape({1, 1, 1, getDim().getDataLen()});
  return result.average(3, output);
}

int Tensor::pow_i(float exponent) {
  pow(exponent, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::pow(float exponent) const {
  Tensor output("", getFormat(), getDataType());
  return pow(exponent, output);
}

Tensor &Tensor::pow(float exponent, Tensor &output) const {
  itensor->pow(exponent, output);
  return output;
}

int Tensor::erf_i() {
  erf(*this);
  return ML_ERROR_NONE;
}

Tensor Tensor::erf() const {
  Tensor output("", getFormat(), getDataType());
  return erf(output);
}

Tensor &Tensor::erf(Tensor &output) const {
  itensor->erf(output);
  return output;
}

void Tensor::sin(Tensor &out, float alpha) {
  if (size() != out.size())
    throw std::invalid_argument("Error: Size of out of Tensor::sin must match");

  itensor->sin(out, alpha);
}

void Tensor::cos(Tensor &out, float alpha) {
  if (size() != out.size())
    throw std::invalid_argument("Error: Size of out of Tensor::cos must match");

  itensor->cos(out, alpha);
}

void Tensor::inv_sqrt_i() { itensor->inv_sqrt(*this); }

LazyTensor Tensor::chain() const { return LazyTensor(*this); }

float Tensor::l2norm() const { return itensor->l2norm(); }

void Tensor::normalization_i() {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot do normalization.";

  const float min = minValue();
  const float max = maxValue();

  if (max == min) {
    Tensor tmp = *this;
    this->subtract_i(tmp);
  } else {
    this->subtract_i(min);
    this->divide_i(max - min);
  }
}

void Tensor::standardization_i() {
  Tensor mean_by_batch = this->sum_by_batch();
  mean_by_batch.divide_i(getDim().getFeatureLen());

  this->subtract_i(mean_by_batch);
  Tensor std_dev_by_batch(batch(), 1, 1, 1, getFormat(), getDataType());
  std_dev_by_batch.setZero();

  /// @todo remove conditional statement
  if (getDataType() == ml::train::TensorDim::DataType::FP32) {
    float *std_dev = std_dev_by_batch.getData<float>();

    for (unsigned int k = 0; k < batch(); ++k) {
      Tensor sub_this = this->getBatchSlice(k, 1);
      std_dev[k] = sub_this.l2norm();
    }
  } else if (getDataType() == ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
    _FP16 *std_dev = std_dev_by_batch.getData<_FP16>();

    for (unsigned int k = 0; k < batch(); ++k) {
      Tensor sub_this = this->getBatchSlice(k, 1);
      std_dev[k] = static_cast<_FP16>(sub_this.l2norm());
    }
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
  }

  std_dev_by_batch.divide_i(getDim().getFeatureLen());
  this->divide_i(std_dev_by_batch);
}

Tensor Tensor::dot(Tensor const &input, bool trans, bool trans_in) const {
  Tensor output("", this->getFormat(), this->getDataType());
  dot(input, output, trans, trans_in);

  return output;
}

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor
 * in case of trans is false.
 */
Tensor &Tensor::dot(Tensor const &input, Tensor &output, bool trans,
                    bool trans_in, float beta) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot dot product.";

  itensor->dot(input, output, trans, trans_in, beta);
  return output;
}

Tensor &Tensor::dot_deriv_wrt_1(Tensor const &m, Tensor const &output_deriv,
                                bool trans, bool trans_m, float beta) {
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
Tensor &Tensor::dot_deriv_wrt_2(Tensor &m_deriv, Tensor const &output_deriv,
                                bool trans, bool trans_m, float beta) const {
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

Tensor &Tensor::dotBatched(Tensor const &m, Tensor &result, bool trans,
                           bool trans_m, float beta) const {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const Tensor this_b = this->getBatchSlice(b, 1);
    Tensor m_b = m.getBatchSlice(b, 1);
    Tensor result_b = result.getBatchSlice(b, 1);

    this_b.dot(m_b, result_b, trans, trans_m, beta);
  }

  return result;
}

Tensor &Tensor::dot_batched_deriv_wrt_1(Tensor const &m,
                                        Tensor const &output_deriv, bool trans,
                                        bool trans_m, float beta) {
  bool deriv_trans_m = true;
  bool deriv_trans = false;
  /** @todo handle all cases of trans and trans_m */
  if (!trans && trans_m) {
    deriv_trans_m = false;
  }

  return output_deriv.dotBatched(m, *this, deriv_trans, deriv_trans_m, beta);
}

Tensor &Tensor::dot_batched_deriv_wrt_2(Tensor &m_deriv,
                                        Tensor const &output_deriv, bool trans,
                                        bool trans_m, float beta) const {
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

Tensor Tensor::dropout_mask(float dropout) const {
  Tensor output(getDim());
  output.dropout_mask(dropout);
  return output;
}

void Tensor::dropout_mask(float dropout) {
  /// @todo add unittest
  NNTR_THROW_IF(dropout < 0 || dropout > 1, std::invalid_argument)
    << "[Tensor::dropout_mask] Dropout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(dropout) == FP_ZERO)
    return;

  setRandUniform(0.0, 1.0);
  itensor->dropout_mask(dropout);
}

void Tensor::filter_mask(const Tensor &mask_len, bool reverse) {
  /// @todo add unittest
  itensor->filter_mask(mask_len, reverse);
}

Tensor Tensor::zoneout_mask(float zoneout) {
  Tensor output(getDim());
  zoneout_mask(output, zoneout);
  return output;
}

void Tensor::zoneout_mask(Tensor &opposite, float zoneout) {
  NNTR_THROW_IF(getDim() != opposite.getDim(), std::invalid_argument)
    << "[Tensor::zoneout_mask] opposite dimension does not match";

  NNTR_THROW_IF(zoneout < 0 || zoneout > 1, std::invalid_argument)
    << "[Tensor::zoneout_mask] Zoneout rate should be between 0 and 1";

  // if the rate is zero, no change is needed
  if (std::fpclassify(zoneout) == FP_ZERO)
    return;

  itensor->zoneout_mask(opposite, zoneout);
}

std::vector<Tensor> Tensor::split(unsigned num_size, int axis) {
  NNTR_THROW_IF(num_size == 0, std::invalid_argument)
    << "num size cannot be zero";

  if (axis == -1) {
    axis = 3;
  }

  NNTR_THROW_IF(!(0 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(getDim().getTensorDim(axis) % num_size != 0,
                std::invalid_argument)
    << "axis is not divisible by num_size, axis: " << axis
    << " num size: " << num_size;

  std::vector<size_t> sizes;
  sizes.resize(num_size);

  unsigned int sz = getDim().getTensorDim(axis) / num_size;
  std::fill(sizes.begin(), sizes.end(), sz);

  return split(sizes, axis);
}

std::vector<Tensor> Tensor::split(std::vector<size_t> sizes, int axis) {
  NNTR_THROW_IF(sizes.size() == 0, std::invalid_argument)
    << "num size cannot be zero";

  NNTR_THROW_IF(!(-1 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(
    std::any_of(sizes.begin(), sizes.end(), [](size_t sz) { return !sz; }),
    std::invalid_argument)
    << "among given sizes at least one of size is 0";

  return itensor->split(sizes, axis);
}

Tensor Tensor::concat(const std::vector<Tensor> &tensors, int axis,
                      Tensor &output) {
  return itensor->concat(tensors, axis, output);
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis) {
  if (axis == -1) {
    axis = 3;
  }

  // Create an output tensor to store the concatenation result
  TensorDim out_dim = Tensor::calculateConcatOutputDim(tensors, axis);
  Tensor output = Tensor(out_dim);

  return output.concat(tensors, axis, output);
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis,
                   Tensor &output) {
  if (axis == -1) {
    axis = 3;
  }

  // Check if the given output tensor dimension is valid
  TensorDim out_dim = Tensor::calculateConcatOutputDim(tensors, axis);

  NNTR_THROW_IF(out_dim != output.getDim(), std::invalid_argument)
    << "invalid output dim for concatenation " << output.getDim()
    << "expected output dim " << out_dim;

  return output.concat(tensors, axis, output);
}

void Tensor::print(std::ostream &out) const {
  printInstance(out, this);
  itensor->print(out);
}

void Tensor::putData() const { itensor->putData(); }

void Tensor::setData(const std::shared_ptr<MemoryData> buf, size_t off,
                     bool init) {
  itensor->setMemoryData(buf, off);

  if (buf && init) {
    initialize();
  }
}

const std::shared_ptr<MemoryData> Tensor::getMemoryData() const {
  return itensor->getMemoryData();
}

size_t Tensor::getOffset() const { return itensor->getOffset(); }

void Tensor::copy(const Tensor &from) {
  /// @todo enable copy to non-contiguous tensor
  if (!itensor->getContiguous() || !from.getContiguous()) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (from.size() != 0 && size() == from.size() &&
      getDataType() == from.getDataType()) {
    // if tensor size and data type match, copy data
    itensor->copy(from);
  } else {
    Tensor t = Tensor(from.getDim(), from.getData<char>());
    swap(t, *this);
  }
}

void Tensor::copyData(const Tensor &from) { itensor->copyData(from); }

void Tensor::copy_with_stride(const Tensor &from) {
  if (itensor->getDim() == from.getDim()) {
    // If the tensor dim matches, copy the data. This also applies to
    // uncontigous tensor.
    itensor->copy_with_stride(from, *this);
  } else {
    // replace with a new tensor that has the same data as the given tensor
    Tensor t = Tensor(from.getDim(), true);
    itensor->copy_with_stride(from, t);
    swap(t, *this);
  }
}

Tensor Tensor::getBatchSlice(size_t offset, unsigned int size) const {
  TensorDim dim_ = getDim();
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->getDim().getFeatureLen(),
                             true, "");
}

Tensor Tensor::clone() const {
  Tensor output(getName(), getFormat(), getDataType());
  output.copy(*this);
  return output;
}

void Tensor::save(std::ostream &file) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot save.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, getData<char>(), sz, "[Tensor::save] operation failed");
  putData();
}

void Tensor::read(std::ifstream &file) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot read.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(file, getData<char>(), sz, "[Tensor::read] operation failed");
  putData();
}

std::vector<unsigned int> Tensor::argmax() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot get argmax.";
  return itensor->argmax();
}

float Tensor::max_abs() const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot get max_abs.";
  return itensor->max_abs();
}

float Tensor::maxValue() const { return itensor->maxValue(); }

float Tensor::minValue() const { return itensor->minValue(); }

Tensor Tensor::transpose(const std::string &direction) const {
  Tensor output(getDim());
  transpose(direction, output);
  return output;
}

Tensor &Tensor::transpose(const std::string &direction, Tensor &output) const {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous. Cannot transpose.";

  if (output.getData<char>() == getData<char>()) {
    Tensor result = clone();
    return result.transpose(direction, output);
  }

  itensor->transpose(direction, output);

  return output;
}

void Tensor::reshape(const TensorDim &d) { itensor->reshape(d); }

void Tensor::fill(const Tensor &from, bool allocate) {
  if (allocate && this->empty()) {
    this->copy(from);
    return;
  }

  if (!from.getContiguous() || !getContiguous()) {
    /// @todo enable this if needed
    throw nntrainer::exception::not_supported(
      "[Tensor::fill] non-contiguous tensors are not supported");
  }

  if (getDim() != from.getDim()) {
    throw std::invalid_argument("[Tensor::fill] dimension must be the same");
  }

  if (getStrides() != from.getStrides()) {
    /// @todo length does not represent buffer size, there should be way to
    /// get the buffer size
    throw std::invalid_argument("[Tensor::fill] buffer size must be the same");
  }

  copyData(from);
}

TensorDim Tensor::getDim() const { return itensor->getDim(); }

TensorDim::TensorType Tensor::getTensorType() const {
  return itensor->getTensorType();
};

Initializer Tensor::getInitializer() const { return itensor->getInitializer(); }

TensorDim::Format Tensor::getFormat() const { return itensor->getFormat(); }

Tdatatype Tensor::getDataType() const { return itensor->getDataType(); }

void Tensor::updateBatch(unsigned int batch) { itensor->updateBatch(batch); }

const bool Tensor::getContiguous() const noexcept {
  return itensor->getContiguous();
}

const std::array<size_t, TensorDim::MAXDIM>
Tensor::getStrides() const noexcept {
  return itensor->getStrides();
}

bool Tensor::checkContinuous(unsigned int np1, unsigned int np2) const {
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

void Tensor::setName(const std::string &name_) { itensor->setName(name_); }

const std::string &Tensor::getName() const { return itensor->getName(); }

size_t Tensor::getIndex(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w) const noexcept {
  return itensor->getIndex(b, c, h, w);
}

size_t Tensor::size() const { return itensor->size(); }

bool Tensor::empty() const { return itensor->empty(); }

size_t Tensor::bytes() const { return itensor->bytes(); }

size_t Tensor::batch() const { return itensor->batch(); }

size_t Tensor::channel() const { return itensor->channel(); }

size_t Tensor::height() const { return itensor->height(); }

size_t Tensor::width() const { return itensor->width(); }

void Tensor::mergeAxis(unsigned int axis1, unsigned int axis2) {
  NNTR_THROW_IF(!getContiguous(), std::invalid_argument)
    << getName() << " is not contiguous, cannot merge axis";

  if (axis2 != axis1 + 1)
    if (!checkContinuous(axis1, axis2))
      throw std::invalid_argument("axis2 must be axis1 + 1 for merging.");

  itensor->mergeAxis(axis1, axis2);
}

void Tensor::createSharedDataTensor(const Tensor &src, Tensor &dest,
                                    size_t offset) const {
  itensor->createSharedDataTensor(src.itensor.get(), dest.itensor.get(),
                                  offset);
}

Tensor Tensor::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                   bool reset_stride,
                                   const std::string &name_) const {
  Tensor ret = *this;
  itensor->getSharedDataTensor(dim_, offset, reset_stride, name_,
                               ret.itensor.get());
  return ret;
}

void Tensor::setTensorVar(TensorDim d, void *buf, size_t offset) {
  itensor->setTensorVar(d, buf, offset);
}

TensorDim Tensor::calculateConcatOutputDim(const std::vector<Tensor> &tensors,
                                           int axis) {
  // Check axis, in which the tensors are concatenated, is valid.
  NNTR_THROW_IF(!(-1 <= axis && axis < 4), std::invalid_argument)
    << "cannot concatenate tensors along an axis: " << axis;

  // Check if the number of input tensors is valid.
  NNTR_THROW_IF(tensors.size() <= 1, std::invalid_argument)
    << "received an invalid tensor vector. size must be greater than 1.";

  auto out_dim = tensors.front().getDim();

  // Check if all tensor data types are the same.
  for (auto &t : tensors) {
    NNTR_THROW_IF(t.getDataType() != out_dim.getDataType(),
                  std::invalid_argument)
      << "cannot concatenate tensors with different data types.";
  }

  // Compute the dimensions of an output tensor.
  out_dim.setTensorDim(axis, 1);
  NNTR_THROW_IF(!std::all_of(tensors.begin(), tensors.end(),
                             [&out_dim, axis](const Tensor &t) {
                               auto cur_dim = t.getDim();
                               cur_dim.setTensorDim(axis, 1);
                               return out_dim == cur_dim;
                             }),
                std::invalid_argument)
    << " all tensor must have the same dimension except for the axis, out_dim: "
    << out_dim << " axis : " << axis;

  auto axis_dim = std::accumulate(tensors.begin(), tensors.end(), 0u,
                                  [axis](unsigned cur, const Tensor &t) {
                                    return cur += t.getDim().getTensorDim(axis);
                                  });

  out_dim.setTensorDim(axis, axis_dim);
  return out_dim;
}

std::ostream &operator<<(std::ostream &out, Tensor const &input) {
  input.print(out);
  return out;
}

} // namespace nntrainer
