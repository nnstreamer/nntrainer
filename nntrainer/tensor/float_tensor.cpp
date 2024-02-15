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

FloatTensor::FloatTensor(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
  Tformat fm) {
  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize FloatTensor from empty vector");
  }

  dim.setTensorDim(0, d.size());
  if (fm == Tformat::NCHW) {
    dim.setTensorDim(1, d[0].size());
    dim.setTensorDim(2, d[0][0].size());
    dim.setTensorDim(3, d[0][0][0].size());
  } else {
    dim.setTensorDim(2, d[0].size());
    dim.setTensorDim(3, d[0][0].size());
    dim.setTensorDim(1, d[0][0][0].size());
  }

  dim.setTensorType({fm, Tdatatype::FP32});

  strides = dim.computeStrides();
  contiguous = true;
  initializer = Initializer::NONE;

  MemoryData *mem_data =
    new MemoryData((void *)(new float[dim.getDataLen()]()));
  data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
    delete[] mem_data->getAddr<float>();
  });

  offset = 0;

  // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
  // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
  // dim[1] == height, dim[2] == width, dim[3] == channel
  if (fm == Tformat::NCHW) {
    for (unsigned int i = 0; i < batch(); ++i)
      for (unsigned int j = 0; j < channel(); ++j)
        for (unsigned int k = 0; k < height(); ++k)
          for (unsigned int l = 0; l < width(); ++l)
            this->setValue(i, j, k, l, d[i][j][k][l]);
  } else {
    for (unsigned int i = 0; i < batch(); ++i)
      for (unsigned int j = 0; j < height(); ++j)
        for (unsigned int k = 0; k < width(); ++k)
          for (unsigned int l = 0; l < channel(); ++l)
            this->setValue(i, l, j, k, d[i][j][k][l]);
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

const float FloatTensor::getValue(unsigned int i) const {
  return ((float *)getData())[i];
}

const float FloatTensor::getValue(unsigned int b, unsigned int c,
                                  unsigned int h, unsigned int w) const {
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

TensorV2 &FloatTensor::apply(std::function<float(float)> f,
                             TensorV2 &output) const {
  CREATE_V2_IF_EMPTY_DIMS(output, dim, nullptr);

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

TensorV2 FloatTensor::multiply_strided(TensorV2 const &m, TensorV2 &output,
                                       const float beta) const {
  CREATE_V2_IF_EMPTY_DIMS(output, dim, nullptr);

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
      output.getStrides()[3] != 1 || beta != 0.0) {
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

void FloatTensor::copy(const TensorV2 &from) {
  reshape(from.getDim());
  copy(from.getData<float>());
}

void FloatTensor::copyData(const TensorV2 &from) {
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
  default:
    throw std::invalid_argument("Error: Unsupported data type");
    break;
  }
}

int FloatTensor::multiply_i(float const &value) {
  float *data = (float *)getData();
  unsigned int len = size();

  sscal(len, value, data, 1);

  return ML_ERROR_NONE;
}

TensorV2 &FloatTensor::multiply(float const &value, TensorV2 &out) const {
  auto f = std::bind(std::multiplies<float>(), std::placeholders::_1, value);
  apply(f, out);
  return out;
}

TensorV2 &FloatTensor::multiply(TensorV2 const &m, TensorV2 &output,
                                const float beta) const {
  auto f = [&](const BroadcastInfoV2 &e, const float *buf, const float *m_buf,
               float *out_buf) {
    if (e.strides[3] == 1 && output.getStrides()[3] == 1 && strides[3] == 1 &&
        beta == 0.0) {
      std::transform(buf, buf + e.buffer_size, m_buf, out_buf,
                     std::multiplies<float>());
    } else {
      for (unsigned int i = 0; i < e.buffer_size; ++i) {
        *out_buf = *buf * *m_buf + beta * *out_buf;
        buf += strides[3];
        m_buf += e.strides[3];
        out_buf += output.getStrides()[3];
      }
    }
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

TensorV2 &FloatTensor::divide(float const &value, TensorV2 &output) const {
  auto f = std::bind(std::divides<float>(), std::placeholders::_1, value);
  apply(f, output);
  return output;
}

TensorV2 &FloatTensor::divide(TensorV2 const &m, TensorV2 &output) const {
  auto f = [&](const BroadcastInfoV2 &e, const float *buf, const float *m_buf,
               float *out_buf) {
    if (e.strides[3] == 1 && output.getStrides()[3] == 1 && strides[3] == 1) {
      std::transform(buf, buf + e.buffer_size, m_buf, out_buf,
                     std::divides<float>());
    } else {
      for (unsigned int i = 0; i < e.buffer_size; ++i) {
        *out_buf = *buf / *m_buf;
        buf += strides[3];
        m_buf += e.strides[3];
        out_buf += output.getStrides()[3];
      }
    }
  };

  apply_broadcast(m, f, output);
  return output;
}

TensorV2 &FloatTensor::add_strided(TensorV2 const &input, TensorV2 &output,
                                   const float beta) const {
  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(input.getData<float>() == nullptr, std::invalid_argument)
    << input.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData<float>() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  if (strides[3] != 1 || input.getStrides()[3] != 1 ||
      output.getStrides()[3] != 1 || beta != 0.0) {
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

TensorV2 &FloatTensor::add(float const &value, TensorV2 &output) const {
  auto f = std::bind(std::plus<float>(), std::placeholders::_1, value);
  apply(f, output);
  return output;
}

TensorV2 &FloatTensor::add(TensorV2 const &m, TensorV2 &output,
                           float const alpha) const {
  auto f = [&](const BroadcastInfoV2 &e, const float *buf, const float *m_buf,
               float *out_buf) {
    if (e.strides[3] == 1 && strides[3] == 1 && strides[3] == 1 && alpha == 0) {
      std::transform(buf, buf + e.buffer_size, m_buf, out_buf,
                     std::plus<float>());
    } else {
      for (unsigned int i = 0; i < e.buffer_size; ++i) {
        *out_buf = *buf + *m_buf * alpha;
        buf += strides[3];
        m_buf += e.strides[3];
        out_buf += strides[3];
      }
    }
  };
  apply_broadcast(m, f, output);
  return output;
}

TensorV2 &FloatTensor::subtract(float const &value, TensorV2 &output) const {
  throw std::logic_error("FloatTensor::subtract is not implemented yet");
  return output;
}

TensorV2 &FloatTensor::pow(float exponent, TensorV2 &output) const {
  auto f = [exponent](float in) { return powf(in, exponent); };
  apply(f, output);
  return output;
}

TensorV2 &FloatTensor::erf(TensorV2 &output) const {
  auto f = [](float in) { return std::erf(in); };
  apply(f, output);
  return output;
}

void FloatTensor::print(std::ostream &out) const {
  printInstance(out, this);
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
  TensorV2 const &m,
  std::function<void(const BroadcastInfoV2 &e, const float *, const float *,
                     float *)>
    v_func,
  TensorV2 &output, const BroadcastInfoV2 &e, int cur_axis, size_t offset,
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
  TensorV2 const &m,
  std::function<void(const BroadcastInfoV2 &e, const float *, const float *,
                     float *)>
    v_func,
  TensorV2 &output) const {
  CREATE_V2_IF_EMPTY_DIMS(output, dim);

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
    BroadcastInfoV2 e;
    e.buffer_size = size();
    e.strides[3] = 1;
    e.tensor_type = getTensorType();
    v_func(e, (float *)getData(), m.getData<float>(), output.getData<float>());
    return;
  }

  return apply_broadcast_util(m, v_func, output, this->computeBroadcastInfo(m));
}

} // namespace nntrainer
