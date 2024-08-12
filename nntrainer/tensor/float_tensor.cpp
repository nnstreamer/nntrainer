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

#include <iomanip>
#include <iostream>

#include <blas_interface.h>
#include <float_tensor.h>
#include <tensor.h>
#include <util_func.h>

namespace nntrainer {

FloatTensor::FloatTensor(std::string name_, Tformat fm) :
  TensorBase(name_, fm, Tdatatype::FP32) {}

FloatTensor::FloatTensor(const TensorDim &d, bool alloc_now, Initializer init,
                         std::string name) :
  TensorBase(d, alloc_now, init, name) {
  if (alloc_now)
    allocate();
}

FloatTensor::FloatTensor(const TensorDim &d, const void *buf) :
  FloatTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

bool FloatTensor::operator==(const FloatTensor &rhs) const {
  const float *_data = (float *)getData();
  const float *_rdata = (float *)rhs.getData();
  for (size_t i = 0; i < size(); ++i) {
    /** not checking sign change is intentional to avoid float calculation
     * errors around 0 */
    if (std::isnan(_data[i]) || std::isnan(_rdata[i]) ||
        std::fabs(_data[i] - _rdata[i]) > epsilon)
      return false;
  }

  return true;
}

/// @todo support allocation by src_tensor
void FloatTensor::allocate() {
  if (empty() || data)
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    allocateSrcTensor();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data
    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new float[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<float>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

void FloatTensor::deallocate() {
  data = nullptr;
  offset = 0;
}

void *FloatTensor::getData() const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<float>() + offset;
}

void *FloatTensor::getData(size_t idx) const {
  if (!data)
    return nullptr;

  data->validate();
  return data->getAddr<float>() + offset + idx;
}

void *FloatTensor::getAddress(unsigned int i) {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((float *)getData())[i];
}

const void *FloatTensor::getAddress(unsigned int i) const {
  size_t index = getIndex(batch(), channel(), height(), width());
  if (i > index) {
    return nullptr;
  }
  return &((float *)getData())[i];
}

const float &FloatTensor::getValue(unsigned int i) const {
  return ((float *)getData())[i];
}

float &FloatTensor::getValue(unsigned int i) { return ((float *)getData())[i]; }

const float &FloatTensor::getValue(unsigned int b, unsigned int c,
                                   unsigned int h, unsigned int w) const {
  return getValue(getIndex(b, c, h, w));
}

float &FloatTensor::getValue(unsigned int b, unsigned int c, unsigned int h,
                             unsigned int w) {
  return getValue(getIndex(b, c, h, w));
}

void FloatTensor::setValue(float value) {
  float *data = (float *)getData();
  std::fill(data, data + size(), value);
}

void FloatTensor::setValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w, float value) {
  ((float *)getData())[getIndex(b, c, h, w)] = value;
}

void FloatTensor::addValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w, float value, float beta) {
  auto const &idx = getIndex(b, c, h, w);
  ((float *)getData())[idx] *= beta;
  ((float *)getData())[idx] += value;
}

void FloatTensor::setZero() {
  if (contiguous) {
    sscal(size(), 0, (float *)getData(), 1);
  } else {
    /// @todo implement apply_i
    // apply_i<float>([](float val) -> float { return 0; });
    setValue(0);
  }
}

void FloatTensor::setRandNormal(float mean, float stddev) {
  setDist<std::normal_distribution<float>>(
    std::normal_distribution<float>(mean, stddev));
}

void FloatTensor::setRandUniform(float min, float max) {
  setDist<std::uniform_real_distribution<float>>(
    std::uniform_real_distribution<float>(min, max));
}

void FloatTensor::setRandBernoulli(float probability) {
  setDist<std::bernoulli_distribution>(
    std::bernoulli_distribution(probability));
}

void FloatTensor::initialize() {
  if (empty() || !isAllocated())
    return;

  unsigned int fan_in, fan_out;

  /// @fixme: when unit is equal to one, this does not work, we need to rely on
  /// effective dimension then actual numbers here. For now, some heuristics
  /// added to infer what would be fan_in/fan_out
  if (dim.batch() * dim.channel() * dim.height() == 1) {
    fan_out = fan_in = dim.width();
  } else if (dim.batch() * dim.channel() == 1) { /// fc layer - 2-D tensor
    fan_in = dim.height();
    fan_out = dim.width();
  } else { /// conv2d filters - 4d tensor, @todo extend this to > 4
    auto field_size = dim.height() * dim.width();

    // this also handles below cases.
    // 1. fan_in = fan_out = 1 as well.
    // 2. batch == 1, channel == 1 and height == 1, theoretical rank of 1
    fan_in = dim.channel() * field_size;
    fan_out = dim.batch() * field_size;
  }

  switch (initializer) {
  case Initializer::ZEROS:
    setZero();
    break;
  case Initializer::ONES:
    setValue(1.0f);
    break;
  case Initializer::LECUN_NORMAL:
    setRandNormal(0.0f, sqrtFloat(1.0f / fan_in));
    break;
  case Initializer::XAVIER_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in + fan_out)));
    break;
  case Initializer::HE_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in)));
    break;
  case Initializer::LECUN_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(1.0f / fan_in), sqrtFloat(1.0f / fan_in));
    break;
  case Initializer::XAVIER_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in + fan_out)),
                   sqrtFloat(6.0 / (fan_in + fan_out)));
    break;
  case Initializer::HE_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in)),
                   sqrtFloat(6.0 / (fan_in)));
  default:
    break;
  }

  putData();
}

void FloatTensor::initialize(Initializer init) {
  initializer = init;
  initialize();
}

Tensor &FloatTensor::apply(std::function<float(float)> f,
                           Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  if (contiguous && output.getContiguous()) {
    const float *data = (float *)getData();
    float *rdata = output.getData<float>();

    std::transform(data, data + size(), rdata, f);
  } else if (strides[3] == 1 && output.getStrides()[3] == 1) {
    /** @todo optimize this with combining these loops where stride is 1 */
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          float *out_data = output.getAddress<float>(b, c, h, 0);
          const float *in_data = (float *)getAddress(getIndex(b, c, h, 0));
          std::transform(in_data, in_data + width(), out_data, f);
        }
      }
    }
  } else {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.setValue(b, c, h, w, f(getValue(b, c, h, w)));
          }
        }
      }
    }
  }

  return output;
}

Tensor FloatTensor::multiply_strided(Tensor const &m, Tensor &output,
                                     const float beta) const {
  CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  if (size() != m.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided multiplication does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData<float>() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<float>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  if (strides[3] != 1 || m.getStrides()[3] != 1 ||
      output.getStrides()[3] != 1 || std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.addValue(
              b, c, h, w, getValue(b, c, h, w) * m.getValue<float>(b, c, h, w),
              beta);
          }
        }
      }
    }
  } else {
    /** @todo optimize by combining these loops where stride is 1 */
    if (getFormat() == Tformat::NCHW) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            float *out_data = output.getAddress<float>(b, c, h, 0);
            const float *m_data = m.getAddress<float>(b, c, h, 0);
            const float *in_data = (float *)getAddress(getIndex(b, c, h, 0));
            std::transform(in_data, in_data + width(), m_data, out_data,
                           std::multiplies<float>());
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            float *out_data = output.getAddress<float>(b, 0, h, w);
            const float *m_data = m.getAddress<float>(b, 0, h, w);
            const float *in_data = (float *)getAddress(getIndex(b, 0, h, w));
            std::transform(in_data, in_data + channel(), m_data, out_data,
                           std::multiplies<float>());
          }
        }
      }
    }
  }

  return output;
}

int FloatTensor::multiply_i(float const &value) {
  float *data = (float *)getData();
  unsigned int len = size();

  sscal(len, value, data, 1);

  return ML_ERROR_NONE;
}

Tensor &FloatTensor::multiply(float const &value, Tensor &out) const {
  auto f = std::bind(std::multiplies<float>(), std::placeholders::_1, value);
  apply(f, out);
  return out;
}

Tensor &FloatTensor::multiply(Tensor const &m, Tensor &output,
                              const float beta) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    ele_mul(e.buffer_size, buf, m_buf, out_buf, 1, beta, e.strides[3],
            strides[3]);
  };

  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "Tensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!contiguous || !m.getContiguous() || !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  NNTR_THROW_IF(!contiguous || !m.getContiguous() || !output.getContiguous(),
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  apply_broadcast(m, f, output);
  return output;
}

Tensor &FloatTensor::divide(float const &value, Tensor &output) const {
  auto f = std::bind(std::divides<float>(), std::placeholders::_1, value);
  apply(f, output);
  return output;
}

Tensor &FloatTensor::divide(Tensor const &m, Tensor &output) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    ele_div(e.buffer_size, buf, m_buf, out_buf, 1, 0, e.strides[3], strides[3]);
  };

  apply_broadcast(m, f, output);
  return output;
}

Tensor &FloatTensor::add_strided(Tensor const &input, Tensor &output,
                                 const float beta) const {
  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(input.getData<float>() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<float>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  if (strides[3] != 1 || input.getStrides()[3] != 1 ||
      output.getStrides()[3] != 1 || std::fpclassify(beta) != FP_ZERO) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            output.setValue(b, c, h, w,
                            getValue(b, c, h, w) +
                              input.getValue<float>(b, c, h, w) * beta);
          }
        }
      }
    }
  } else {
    /** @todo optimize this with combining these loops where stride is 1 */
    if (this->getFormat() == Tformat::NCHW) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            float *out_data = output.getAddress<float>(b, c, h, 0);
            const float *in_data = input.getAddress<float>(b, c, h, 0);
            const float *_data = (float *)getAddress(getIndex(b, c, h, 0));
            std::transform(_data, _data + width(), in_data, out_data,
                           std::plus<float>());
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            float *out_data = output.getAddress<float>(b, 0, h, w);
            const float *in_data = input.getAddress<float>(b, 0, h, w);
            const float *_data = (float *)getAddress(getIndex(b, 0, h, w));
            std::transform(_data, _data + channel(), in_data, out_data,
                           std::plus<float>());
          }
        }
      }
    }
  }

  return output;
}

int FloatTensor::add_i(Tensor const &m, Tensor &output, float const alpha) {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    saxpy(e.buffer_size, alpha, m_buf, e.strides[3], out_buf, strides[3]);
  };

  try {
    apply_broadcast(m, f, output);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }
  return ML_ERROR_NONE;
}

int FloatTensor::add_i_partial(unsigned int len, unsigned int addr_idx,
                               Tensor &m, unsigned int incX, unsigned int incY,
                               const Tensor alphas, unsigned int alpha_idx) {
  saxpy(len, alphas.getValue<float>(alpha_idx), m.getData<float>(), incX,
        (float *)getAddress(addr_idx), incY);

  return ML_ERROR_NONE;
}

Tensor &FloatTensor::add(float const &value, Tensor &output) const {
  auto f = std::bind(std::plus<float>(), std::placeholders::_1, value);
  apply(f, output);
  return output;
}

Tensor &FloatTensor::add(Tensor const &m, Tensor &output,
                         float const alpha) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    ele_add(e.buffer_size, buf, m_buf, out_buf, alpha, 0, e.strides[3],
            strides[3]);
  };
  apply_broadcast(m, f, output);
  return output;
}

Tensor &FloatTensor::subtract(float const &value, Tensor &output) const {
  auto f = std::bind(std::minus<float>(), std::placeholders::_1, value);
  apply(f, output);
  return output;
}

void FloatTensor::sum_by_batch(Tensor &output) const {
  size_t feat_len = dim.getFeatureLen();
  size_t batch = dim.batch();

  const float *data = (float *)getData();
  float *out_data = output.getData<float>();

  Tensor ones(1, 1, 1, feat_len, this->getFormat());
  ones.setValue(1.0);
  sgemv(CblasRowMajor, CblasNoTrans, batch, feat_len, 1, data, feat_len,
        ones.getData<float>(), 1, 0.0, out_data, 1);
}

Tensor &FloatTensor::sum(unsigned int axis, Tensor &output, float alpha,
                         float beta) const {
  const float *data = (float *)getData();

  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  if (axis >= 4)
    throw std::out_of_range("Error: axis is invalid");

  if (dim.getDim()[axis] == 1 and alpha == 1.0 and !beta) {
    CREATE_IF_EMPTY_DIMS(output, dim);
    scopy(size(), (float *)getData(), 1, output.getData<float>(), 1);
    return output;
  }

  switch (axis) {
  case 0: {
    CREATE_IF_EMPTY_DIMS(output, 1, dim.channel(), dim.height(), dim.width(),
                         getTensorType());
    size_t feat_len = dim.getFeatureLen();
    size_t batch = dim.batch();
    Tensor ones(1, 1, 1, batch, getTensorType());
    ones.setValue(alpha);
    sgemv(CblasRowMajor, CblasTrans, batch, feat_len, 1, data, feat_len,
          ones.getData<float>(), 1, beta, output.getData<float>(), 1);
  } break;
  case 1: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], 1, dim[2], dim[3], getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = output.getDim().getDataLen();
      unsigned int t_axis = dim[1];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      sgemv(CblasRowMajor, CblasNoTrans, feat_len, t_axis, 1, data, t_axis,
            ones.getData<float>(), 1, beta, output.getData<float>(), 1);
    } else {
      unsigned int feat_len = dim[2] * dim[3];
      unsigned int t_axis = dim[1];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      float *rdata = output.getData<float>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv(CblasRowMajor, CblasTrans, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData<float>(),
              1, beta, &rdata[k * feat_len], 1);
      }
    }
  } break;
  case 2: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], dim[1], 1, dim[3], getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = dim[1] * dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      float *rdata = output.getData<float>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv(CblasRowMajor, CblasTrans, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData<float>(),
              1, beta, &rdata[k * feat_len], 1);
      }
    } else {
      unsigned int t_3 = dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);

      if (dim.getStorageOrder() == TStorageOrder::ROW_MAJOR) {
        float *rdata = output.getData<float>();
        for (unsigned int k = 0; k < dim[0]; ++k) {
          for (unsigned int c = 0; c < dim[1]; ++c) {
            unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[2];
            unsigned int ridx =
              k * output.getDim().getFeatureLen() + c * dim[3];

            sgemv(CblasRowMajor, CblasTrans, t_axis, t_3, 1, &data[idx], t_3,
                  ones.getData<float>(), 1, beta, &rdata[ridx], 1);
          }
        }
      } else {
        sgemv(CblasColMajor, CblasTrans, t_axis, output.getDim().getDataLen(),
              1, data, t_axis, ones.getData<float>(), 1, beta,
              output.getData<float>(), 1);
      }
    }
  } break;
  case 3: {
    CREATE_IF_EMPTY_DIMS(output, dim[0], dim[1], dim[2], 1,
                         this->getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int t_3 = dim[1];
      unsigned int t_axis = dim[3];
      Tensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      float *rdata = output.getData<float>();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        for (unsigned int c = 0; c < dim[2]; ++c) {
          unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[1];
          unsigned int ridx = k * output.getDim().getFeatureLen() + c * dim[1];
          sgemv(CblasRowMajor, CblasTrans, t_axis, t_3, 1, &data[idx], t_3,
                ones.getData<float>(), 1, beta, &rdata[ridx], 1);
        }
      }
    } else {
      unsigned int m = output.getDim().getDataLen();
      unsigned int n = dim[3];
      Tensor ones(1, 1, 1, n, getTensorType());
      ones.setValue(alpha);

      if (dim.getStorageOrder() == TStorageOrder::ROW_MAJOR) {
        sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, data, n,
              ones.getData<float>(), 1, beta, output.getData<float>(), 1);
      } else {
        float *rdata = output.getData<float>();

        for (unsigned int k = 0; k < dim[0]; ++k) {
          for (unsigned int c = 0; c < dim[1]; ++c) {
            unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[2];
            unsigned int ridx = k * dim[1] * dim[2] + c * dim[2];

            sgemv(CblasColMajor, CblasNoTrans, dim[2], n, 1, &data[idx], dim[2],
                  ones.getData<float>(), 1, beta, &rdata[ridx], 1);
          }
        }
      }
    }
  } break;
  default:
    throw std::out_of_range("Error: Dimension cannot exceed 3");
  }

  return output;
}

float FloatTensor::l2norm() const {
  return snrm2(size(), (float *)getData(), 1);
}

Tensor &FloatTensor::pow(float exponent, Tensor &output) const {
  auto f = [exponent](float in) { return powf(in, exponent); };
  apply(f, output);
  return output;
}

Tensor &FloatTensor::erf(Tensor &output) const {
  auto f = [](float in) { return std::erf(in); };
  apply(f, output);
  return output;
}

void FloatTensor::sin(Tensor &out, float alpha) {
  if (!contiguous) {
    auto f = [alpha](float val) -> float { return std::sin(alpha * val); };
    apply(f, out);
  } else {
    sine(size(), (float *)getData(), out.getData<float>(), alpha);
  }
}

void FloatTensor::cos(Tensor &out, float alpha) {
  if (!contiguous) {
    auto f = [alpha](float val) -> float { return std::cos(alpha * val); };
    apply(f, out);
  } else {
    cosine(size(), (float *)getData(), out.getData<float>(), alpha);
  }
}

void FloatTensor::inv_sqrt(Tensor &out) {
  if (!contiguous) {
    apply([](float val) -> float { return 1 / std::sqrt(val); }, out);
  } else {
    inv_sqrt_inplace(out.size(), out.getData<float>());
  }
}

Tensor &FloatTensor::dot(Tensor const &input, Tensor &output, bool trans,
                         bool trans_in, float beta) const {
  // Comment out with intension to support the calculation wrt. batch and height
  // direction. It supposes to have this->dim as [ BxCxH,W ] and input.dim is
  // [BxCxH,W] as well if (input.dim.rank() > 2) {
  //   throw exception::not_supported("Error: support only for rank of dot "
  //                                  "matrix <= 2");
  // }

  // Comment out with intension to support the calculation wrt. batch and height
  // direction of this tensor. It is OK as long as input is 2D
  if (trans && dim.rank() > 2) {
    ml_logw("Warning: support only for rank of dot matrix <= 2 with trans");
  }
  unsigned int first_three_flat, last_axis, input_first_three_flat,
    input_last_axis, M, N, K, lda, ldb, ldc;

  calculateFlattenDot(input, output, trans, trans_in, first_three_flat,
                      last_axis, input_first_three_flat, input_last_axis, M, N,
                      K, lda, ldb, ldc);

  const float *data = (float *)getData();
  const float *mdata = input.getData<float>();
  float *rdata = output.getData<float>();
  const float alpha = 1.0f;
  enum CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
  enum CBLAS_TRANSPOSE transB = trans_in ? CblasTrans : CblasNoTrans;

  /// shortcut handling in case of vector
  /// for vector, (1 * K) == (K * 1) in current memory layout...
  /// and plaese note that N, K, M is a fixed place holder after considering
  /// transpose.
  /// For example, there is no case like (1 * K) X (1 * K) while
  /// (1 * K) X (1 * M) can be a case
  /// case1: (1 * K) X (K * 1)
  if (M == 1 && N == 1) {
    *rdata = sdot(K, data, 1, mdata, 1) + beta * (*rdata);
  }
  /// case2: (M * K) X (K * 1)
  else if (N == 1) {
    sgemv(CblasRowMajor, transA, first_three_flat, last_axis, alpha, data, lda,
          mdata, 1, beta, rdata, 1);
  }
  /// case3: (1 * K) X (K * N) = 1 * N = R
  /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
  /// Effectively a translation of sgemv
  else if (M == 1) {
    transB = transB == CblasTrans ? CblasNoTrans : CblasTrans;
    sgemv(CblasRowMajor, transB, input_first_three_flat, input_last_axis, alpha,
          mdata, ldb, data, 1, beta, rdata, 1);
  }
  /// case others: use gemm
  else {
    sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, data, lda, mdata, ldb,
          beta, rdata, ldc);
  }

  return output;
}

void FloatTensor::copy(const Tensor &from) {
  reshape(from.getDim());
  copy(from.getData<float>());
}

void FloatTensor::copyData(const Tensor &from) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  NNTR_THROW_IF(size() != from.size(), std::invalid_argument)
    << "Size of tensor to copy must match";

  switch (from.getDataType()) {
  case ml::train::TensorDim::DataType::FP32:
    copy(from.getData<float>());
    break;
  case ml::train::TensorDim::DataType::FP16:
/// @todo remove #ifdef ENABLE_FP16
#ifdef ENABLE_FP16
    scopy(size(), from.getData<_FP16>(), 1, (float *)getData(), 1);
#else
    throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    break;
  case ml::train::TensorDim::DataType::QINT8:
    scopy_int8_to_float32(from.size(), from.getData<uint8_t>(), 1,
                          (float *)getData(), 1);
    break;
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

void FloatTensor::copy_with_stride(const Tensor &input, Tensor &output) {
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w, input.getValue<float>(b, c, h, w));
        }
      }
    }
  }
}

std::vector<unsigned int> FloatTensor::argmax() const {
  std::vector<unsigned int> result;
  const float *data = (float *)getData();
  size_t batch_size = batch();
  size_t feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto max_iter =
      std::max_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, max_iter) - (b * feature_len);
  }
  return result;
}

float FloatTensor::max_abs() const {
  const float *data = (float *)getData();
  unsigned int idx = isamax(size(), data, 1);
  return *(data + idx);
}

float FloatTensor::maxValue() const {
  const float *data = (float *)getData();
  return *std::max_element(data, data + size());
}

float FloatTensor::minValue() const {
  const float *data = (float *)getData();
  return *std::min_element(data, data + size());
}

Tensor &FloatTensor::transpose(const std::string &direction,
                               Tensor &output) const {
  unsigned int SL, SI, SJ, SK;

  output.reshape(dim.transpose(direction));

  int indexI = direction[0] - '0';
  int indexJ = direction[2] - '0';

  SL = dim.batch(), SI = dim.channel(), SJ = dim.height(), SK = dim.width();

  bool is_format_nchw = (getFormat() == Tformat::NCHW);

  const float *inptr = (float *)getData();
  float *outptr = output.getData<float>();
  switch (indexI) {
  case 0:
    if (indexJ == 1) {
      if (is_format_nchw) {
        transposeloop(l, i, j, k, SL, SI, SJ, SK);
      } else {
        transposeloop_nhwc(l, j, k, i, SL, SJ, SK, SI);
      }
    } else {
      if (is_format_nchw) {
        transposeloop(l, i, k, j, SL, SI, SK, SJ);
      } else {
        transposeloop_nhwc(l, k, j, i, SL, SK, SJ, SI);
      }
    }
    break;
  case 1:
    if (indexJ == 0) {
      if (is_format_nchw) {
        transposeloop(l, j, i, k, SL, SJ, SI, SK);
      } else {
        transposeloop_nhwc(l, i, k, j, SL, SI, SK, SJ);
      }
    } else {
      if (is_format_nchw) {
        transposeloop(l, j, k, i, SL, SJ, SK, SI);
      } else {
        transposeloop_nhwc(l, k, i, j, SL, SK, SI, SJ);
      }
    }
    break;
  case 2:
    if (indexJ == 0) {
      if (is_format_nchw) {
        transposeloop(l, k, i, j, SL, SK, SI, SJ);
      } else {
        transposeloop_nhwc(l, i, j, k, SL, SI, SJ, SK);
      }
    } else {
      if (is_format_nchw) {
        transposeloop(l, k, j, i, SL, SK, SJ, SI);
      } else {
        transposeloop_nhwc(l, j, i, k, SL, SJ, SI, SK);
      }
    }
    break;
  }

  return output;
}

void FloatTensor::dropout_mask(float dropout) {
  float scale = 1.0 / (1 - dropout);
  float *data_ = (float *)getData();
  for (unsigned int i = 0; i < size(); ++i) {
    if (data_[i] >= dropout)
      data_[i] = scale;
    else
      data_[i] = 0.0;
  }
}

void FloatTensor::filter_mask(const Tensor &mask_len, bool reverse) {
  float fill_mask_val = 0.0;
  float en_mask_val = 1.0 - fill_mask_val;

  if (reverse) {
    fill_mask_val = 1.0;
    en_mask_val = 1.0 - fill_mask_val;
  }

  setValue(fill_mask_val);

  NNTR_THROW_IF(mask_len.batch() != batch(), std::invalid_argument)
    << "Number of filter masks mismatched";

  for (unsigned int b = 0; b < batch(); b++) {
    float *addr = (float *)getAddress(getIndex(b, 0, 0, 0));
    const uint *mask_len_val = mask_len.getAddress<uint>(b, 0, 0, 0);
    std::fill(addr, addr + (*mask_len_val), en_mask_val);
  }
}

void FloatTensor::zoneout_mask(Tensor &opposite, float zoneout) {
  opposite.setRandBernoulli(zoneout);

  float *data = (float *)getData();
  float *opposite_data = opposite.getData<float>();

  for (unsigned int i = 0; i < size(); ++i) {
    if (opposite_data[i] > epsilon) {
      data[i] = 0.0f;
    } else {
      data[i] = 1.0f;
    }
  }
}

std::vector<Tensor> FloatTensor::split(std::vector<size_t> sizes, int axis) {
  size_t num_size = sizes.size();

  if (axis == -1) {
    axis = 3;
  }

  size_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0);
  NNTR_THROW_IF(dim.getTensorDim(axis) != total_size, std::invalid_argument)
    << "given sum of sizes did not match with origin tensor dim, tensor dim: "
    << dim.getTensorDim(axis) << " total size: " << total_size;

  std::vector<TensorDim> ret_dims;
  ret_dims.reserve(num_size);
  for (unsigned int i = 0; i < num_size; ++i) {
    ret_dims[i] = dim;
    ret_dims[i].setTensorDim(axis, sizes[i]);
  }

  bool is_format_nchw = (dim.getFormat() == Tformat::NCHW) ? true : false;
  std::vector<Tensor> ret;

  auto iter_value = [this, is_format_nchw](
                      std::array<size_t, 4> &loc,
                      const std::array<size_t, 4> &end_loc,
                      const std::array<size_t, 4> &reset_dim_arr) -> float & {
    auto &value = (is_format_nchw) ? getValue(loc[0], loc[1], loc[2], loc[3])
                                   : getValue(loc[0], loc[3], loc[1], loc[2]);
    for (int i = 3; i >= 0; --i) {
      loc[i]++;
      if (loc[i] == end_loc[i]) {
        loc[i] -= reset_dim_arr[i];
        continue;
      }
      break;
    }
    return value;
  };

  ret.reserve(num_size);

  unsigned int accumulated_size = 0;
  for (unsigned int i = 0; i < num_size; ++i) {
    std::array<size_t, 4> loc = {0, 0, 0, 0};

    if (is_format_nchw) {
      loc[axis] += accumulated_size;
    } else {
      if (axis == 0) {
        loc[0] += accumulated_size;
      } else if (axis == 1) {
        loc[3] += accumulated_size;
      } else if (axis == 2 || axis == 3) {
        loc[axis - 1] += accumulated_size;
      }
    }

    ret.emplace_back(ret_dims[i]);
    auto &ret_t = ret.back();

    std::array<size_t, 4> end_loc;

    if (is_format_nchw) {
      end_loc = {ret_dims[i].batch(), ret_dims[i].channel(),
                 ret_dims[i].height(), ret_dims[i].width()};
    } else {
      end_loc = {ret_dims[i].batch(), ret_dims[i].height(), ret_dims[i].width(),
                 ret_dims[i].channel()};
    }

    accumulated_size += sizes[i];

    if (is_format_nchw) {
      end_loc[axis] = accumulated_size;
    } else {
      if (axis == 0) {
        end_loc[0] = accumulated_size;
      } else if (axis == 1) {
        end_loc[3] = accumulated_size;
      } else if (axis == 2 || axis == 3) {
        end_loc[axis - 1] = accumulated_size;
      }
    }

    std::array<size_t, 4> reset_dim_arr;
    if (is_format_nchw) {
      reset_dim_arr = {ret_dims[i].batch(), ret_dims[i].channel(),
                       ret_dims[i].height(), ret_dims[i].width()};
    } else {
      reset_dim_arr = {ret_dims[i].batch(), ret_dims[i].height(),
                       ret_dims[i].width(), ret_dims[i].channel()};
    }

    ret_t.apply_i<float>(
      [&iter_value, &loc, &end_loc, &reset_dim_arr](float _) {
        return iter_value(loc, end_loc, reset_dim_arr);
      });
  }

  return ret;
}

Tensor FloatTensor::concat(const std::vector<Tensor> &tensors, int axis,
                           Tensor &output) {
  bool is_format_nchw = (tensors.front().getDim().getFormat() == Tformat::NCHW);

  auto iter_value =
    [is_format_nchw](std::array<unsigned, 4> &loc,
                     const std::array<unsigned, 4> &start_loc, Tensor &t,
                     const std::array<unsigned, 4> &ref_dim_arr) -> float & {
    auto &value = is_format_nchw
                    ? t.getValue<float>(loc[0], loc[1], loc[2], loc[3])
                    : t.getValue<float>(loc[0], loc[3], loc[1], loc[2]);

    for (int i = 3; i >= 0; --i) {
      loc[i]++;
      if (loc[i] - start_loc[i] == ref_dim_arr[i]) {
        loc[i] = start_loc[i];
        continue;
      }
      break;
    }
    return value;
  };

  std::array<unsigned, 4> loc = {0, 0, 0, 0};
  for (auto &t : tensors) {
    std::array<unsigned, 4> start_loc = loc;
    std::array<unsigned, 4> tensor_dim_arr;
    TensorDim curr_dim = t.getDim();

    tensor_dim_arr[0] = curr_dim.getTensorDim(0);
    tensor_dim_arr[1] =
      is_format_nchw ? curr_dim.getTensorDim(1) : curr_dim.getTensorDim(2);
    tensor_dim_arr[2] =
      is_format_nchw ? curr_dim.getTensorDim(2) : curr_dim.getTensorDim(3);
    tensor_dim_arr[3] =
      is_format_nchw ? curr_dim.getTensorDim(3) : curr_dim.getTensorDim(1);

    for (size_t i = 0u, sz = t.size(); i < sz; ++i) {
      iter_value(loc, start_loc, output, tensor_dim_arr) = t.getValue<float>(i);
    }

    if (is_format_nchw) {
      loc[axis] += curr_dim.getTensorDim(axis);
    } else {
      if (axis == 0) {
        loc[0] += curr_dim.getTensorDim(axis);
      } else if (axis == 1) {
        loc[3] += curr_dim.getTensorDim(axis);
      } else if (axis == 2 || axis == 3) {
        loc[axis - 1] += curr_dim.getTensorDim(axis);
      }
    }
  }

  return output;
}

void FloatTensor::print(std::ostream &out) const {
  const float *data = (float *)getData();
  unsigned int len = size();
  out << "data addr: " << data << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << data[0] << ' ' << data[1] << ' ' << data[2] << " ... "
        << data[len - 3] << ' ' << data[len - 2] << ' ' << data[len - 1] << ']'
        << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);

  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << std::setprecision(10)
                << data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  } else {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int i = 0; i < height(); i++) {
        for (unsigned int j = 0; j < width(); j++) {
          for (unsigned int l = 0; l < channel(); l++) {
            out << std::setw(10) << std::setprecision(10)
                << data[getIndex(k, l, i, j)] << " ";
          }
          out << std::endl;
        }
        out << std::endl;
      }
      out << "-------" << std::endl;
    }
  }
  out.copyfmt(init);
}

void FloatTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  scopy(size(), (float *)buf, 1, (float *)getData(), 1);
}

void FloatTensor::apply_broadcast_util(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const float *, const float *,
                     float *)>
    v_func,
  Tensor &output, const BroadcastInfo &e, int cur_axis, size_t offset,
  size_t m_offset) const {

  const float *buf = (float *)this->getData();
  const float *m_buf = m.getData<float>();
  float *out_buf = output.getData<float>();

  if (e.buffer_axis == cur_axis) {
    v_func(e, buf + offset, m_buf + m_offset, out_buf + offset);
    return;
  }

  cur_axis++;
  unsigned int continuity[4] = {0, 1, 2, 3};
  if (getFormat() == Tformat::NHWC) {
    continuity[1] = 2;
    continuity[2] = 3;
    continuity[3] = 1;
  }
  for (unsigned int i = 0; i < dim.getTensorDim(continuity[cur_axis]); ++i) {
    size_t next_offset = offset + i * strides[cur_axis];
    size_t next_m_offset = m_offset + i * e.strides[cur_axis];
    apply_broadcast_util(m, v_func, output, e, cur_axis, next_offset,
                         next_m_offset);
  }
}

void FloatTensor::apply_broadcast(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const float *, const float *,
                     float *)>
    v_func,
  Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim);

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData<float>() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<float>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  /// shortcut to cover when dimension matches
  /// note that buffer_size, the last stride is only used in v_func but it
  /// might be changed
  if (dim == m.getDim()) {
    BroadcastInfo e;
    e.buffer_size = size();
    e.strides[3] = 1;
    e.tensor_type = getTensorType();
    v_func(e, (float *)getData(), m.getData<float>(), output.getData<float>());
    return;
  }

  return apply_broadcast_util(m, v_func, output, this->computeBroadcastInfo(m));
}

} // namespace nntrainer
