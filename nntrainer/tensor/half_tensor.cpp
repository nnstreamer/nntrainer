// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	 half_tensor.cpp
 * @date	 09 November 2023
 * @brief	 This is HalfTensor class for calculation
 * @see		 https://github.com/nnstreamer/nntrainer
 * @author	 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		 No known bugs except for NYI items
 */

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <numeric>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stdio.h>

#include <half_tensor.h>
#include <util_func.h>

#define transposeloop(cl, ci, cj, ck, sl, si, sj, sk)                 \
  do {                                                                \
    unsigned int i, j, k, l;                                          \
    int inidx = 0, outidx = 0;                                        \
    for (cl = 0; cl < sl; cl++)                                       \
      for (ci = 0; ci < si; ci++)                                     \
        for (cj = 0; cj < sj; cj++)                                   \
          for (ck = 0; ck < sk; ck++) {                               \
            outidx = si * sj * sk * cl + sj * sk * ci + sk * cj + ck; \
            inidx = l * SI * SJ * SK + i * SJ * SK + j * SK + k;      \
            outptr[outidx] = inptr[inidx];                            \
          }                                                           \
  } while (0);

#define transposeloop_nhwc(cl, ci, cj, ck, sl, si, sj, sk)            \
  do {                                                                \
    unsigned int i, j, k, l;                                          \
    int inidx = 0, outidx = 0;                                        \
    for (cl = 0; cl < sl; cl++)                                       \
      for (ci = 0; ci < si; ci++)                                     \
        for (cj = 0; cj < sj; cj++)                                   \
          for (ck = 0; ck < sk; ck++) {                               \
            outidx = si * sj * sk * cl + sj * sk * ci + sk * cj + ck; \
            inidx = l * SJ * SK * SI + j * SK * SI + k * SI + i;      \
            outptr[outidx] = inptr[inidx];                            \
          }                                                           \
  } while (0);

namespace nntrainer {

/**
 * @struct External Loop Info for broadcasted info
 * @brief External Loop Info for broadcasted iteration. Please refer to
 * DISABLED_private_external_loop_n in unittest_nntrainer_tensor.
 * @note This should better be implemented in iterator fashion before used
 * extensively.
 */
struct HalfTensor::BroadcastInfo {

  /**
   * @brief Construct a new External Loop Info object
   *
   */
  BroadcastInfo() :
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

HalfTensor::HalfTensor(const TensorDim &d, bool alloc_now,
                       HalfTensor::Initializer init, std::string name_) :
  HalfTensor(name_, d.getFormat()) {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();
    initializer = init;
    if (alloc_now)
      allocate();
  }
}

HalfTensor::HalfTensor(const TensorDim &d, const void *buf) :
  HalfTensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

/**
 * @class SrcSharedHalfTensor
 * @brief Source of the shared tensor
 * @note Remove this class
 */
class SrcSharedHalfTensor {
public:
  /**
   * @brief   Constructor for the class
   */
  SrcSharedHalfTensor() : src(nullptr), off(0) {}

  SrcSharedHalfTensor(const HalfTensor *tensor, size_t offset) :
    src(tensor), off(offset) {}

  /**
   * @brief   Get the allocated src tensor
   */
  const HalfTensor *tensor() const {
    if (!src)
      throw std::runtime_error("Accessing empty src tensor");

    return src;
  }

  /**
   * @brief   Get the offset from the source tensor
   */
  size_t offset() const { return off; }

private:
  const HalfTensor *src; /**< HalfTensor of the source */
  size_t off;            /**< offset from the source data ptr */
};

void HalfTensor::allocate() {
  if (empty() || data)
    /// already allocated
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    data = src_tensor->tensor()->data;
    offset = src_tensor->tensor()->offset + src_tensor->offset();
    /** as this memory is shared, do NOT initialize */
  } else {
    /// allocate new memory for the tensor data

    MemoryData *mem_data;

    mem_data = new MemoryData((void *)(new _FP16[dim.getDataLen()]{}));
    data = std::shared_ptr<MemoryData>(mem_data, [](auto *mem_data) {
      delete[] mem_data->template getAddr<_FP16>();
      delete mem_data;
    });

    offset = 0;
    initialize();
  }
}

bool HalfTensor::operator==(const HalfTensor &rhs) const {
  if (this->dim != rhs.dim)
    return false;

  size_t len = size();

  if (len != rhs.size())
    return false;

  if (contiguous != rhs.contiguous)
    return false;

  if (strides != rhs.strides)
    return false;

  const _FP16 *_data = getData();
  const _FP16 *_rdata = rhs.getData();
  for (size_t i = 0; i < len; ++i) {
    if ((std::isnan((float)_data[i]) && !std::isnan((float)_rdata[i])) ||
        (!std::isnan((float)_data[i]) && std::isnan((float)_rdata[i])) ||
        std::fabs((float)(_data[i] - _rdata[i])) > epsilon)
      return false;
  }

  return true;
}

void HalfTensor::setRandNormal(float mean, float std) {
  setDist<std::normal_distribution<float>>(
    std::normal_distribution<float>(mean, std));
}

void HalfTensor::setRandUniform(float min, float max) {
  setDist<std::uniform_real_distribution<float>>(
    std::uniform_real_distribution<float>(min, max));
}

void HalfTensor::setRandBernoulli(float probability) {
  setDist<std::bernoulli_distribution>(
    std::bernoulli_distribution(probability));
}

void HalfTensor::initialize() {
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
  case HalfTensor::Initializer::ZEROS:
    setZero();
    break;
  case HalfTensor::Initializer::ONES:
    setValue(1.0f);
    break;
  case HalfTensor::Initializer::LECUN_NORMAL:
    setRandNormal(0.0f, sqrtFloat(1.0f / fan_in));
    break;
  case HalfTensor::Initializer::XAVIER_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in + fan_out)));
    break;
  case HalfTensor::Initializer::HE_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in)));
    break;
  case HalfTensor::Initializer::LECUN_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(1.0f / fan_in), sqrtFloat(1.0f / fan_in));
    break;
  case HalfTensor::Initializer::XAVIER_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in + fan_out)),
                   sqrtFloat(6.0 / (fan_in + fan_out)));
    break;
  case HalfTensor::Initializer::HE_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in)),
                   sqrtFloat(6.0 / (fan_in)));
    break;
  default:
    break;
  }

  putData();
}

int HalfTensor::multiply_i_strided(HalfTensor const &m, const float beta) {
  try {
    this->multiply_strided(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::multiply_strided(HalfTensor const &m,
                                        const float beta) const {
  HalfTensor t;
  return this->multiply_strided(m, t, beta);
}

HalfTensor &HalfTensor::multiply_strided(HalfTensor const &m,
                                         HalfTensor &output,
                                         const float beta) const {
  /** TODO: throw than create new dimenions */
  CREATE_IF_EMPTY_DIMS_HALF(output, dim, nullptr);

  if (size() != m.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided multiplication does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  // Format NCHW Case
  if (this->getFormat() == Tformat::NCHW) {
    if (strides[3] != 1 || m.strides[3] != 1 || output.strides[3] != 1 ||
        beta != 0.0) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            for (unsigned int w = 0; w < width(); ++w) {
              output.addValue(b, c, h, w,
                              getValue(b, c, h, w) * m.getValue(b, c, h, w),
                              beta);
            }
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            _FP16 *out_data = output.getAddress(b, c, h, 0);
            const _FP16 *m_data = m.getAddress(b, c, h, 0);
            const _FP16 *in_data = getAddress(b, c, h, 0);
            std::transform(in_data, in_data + width(), m_data, out_data,
                           std::multiplies<_FP16>());
          }
        }
      }
    }
  } else { // Format NHWC Case
    if (strides[3] != 1 || m.strides[3] != 1 || output.strides[3] != 1 ||
        beta != 0.0) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            for (unsigned int c = 0; c < channel(); ++c) {
              output.addValue(b, c, h, w,
                              getValue(b, c, h, w) * m.getValue(b, c, h, w),
                              beta);
            }
          }
        }
      }
    } else {
      /** @todo optimize this with combining these loops where
       * stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            _FP16 *out_data = output.getAddress(b, 0, h, w);
            const _FP16 *m_data = m.getAddress(b, 0, h, w);
            const _FP16 *in_data = getAddress(b, 0, h, w);
            std::transform(in_data, in_data + channel(), m_data, out_data,
                           std::multiplies<_FP16>());
          }
        }
      }
    }
  }

  return output;
}

int HalfTensor::add_i_strided(HalfTensor const &m, const float beta) {
  try {
    this->add_strided(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::add_strided(HalfTensor const &m,
                                   const float beta) const {
  HalfTensor t;
  return this->add_strided(m, t, beta);
}

HalfTensor &HalfTensor::add_strided(HalfTensor const &m, HalfTensor &output,
                                    const float beta) const {
  /** TODO: throw than create new dimenions */
  CREATE_IF_EMPTY_DIMS_HALF(output, dim, nullptr);

  if (size() != m.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided addition does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  // Format NCHW Case
  if (this->getFormat() == Tformat::NCHW) {
    if (strides[3] != 1 || m.strides[3] != 1 || output.strides[3] != 1 ||
        beta != 0.0) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            for (unsigned int w = 0; w < width(); ++w) {
              output.setValue(b, c, h, w,
                              getValue(b, c, h, w) +
                                m.getValue(b, c, h, w) * beta);
            }
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            _FP16 *out_data = output.getAddress(b, c, h, 0);
            const _FP16 *m_data = m.getAddress(b, c, h, 0);
            const _FP16 *in_data = getAddress(b, c, h, 0);
            std::transform(in_data, in_data + width(), m_data, out_data,
                           std::plus<_FP16>());
          }
        }
      }
    }
  } else { // Format NHWC Case
    if (strides[3] != 1 || m.strides[3] != 1 || output.strides[3] != 1 ||
        beta != 0.0) {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            for (unsigned int c = 0; c < channel(); ++c) {
              output.setValue(b, c, h, w,
                              getValue(b, c, h, w) +
                                m.getValue(b, c, h, w) * beta);
            }
          }
        }
      }
    } else {
      /** @todo optimize this with combining these loops where
       * stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            _FP16 *out_data = output.getAddress(b, 0, h, w);
            const _FP16 *m_data = m.getAddress(b, 0, h, w);
            const _FP16 *in_data = getAddress(b, 0, h, w);
            std::transform(in_data, in_data + channel(), m_data, out_data,
                           std::plus<_FP16>());
          }
        }
      }
    }
  }
  return output;
}

int HalfTensor::multiply_i(float const &value) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  /// @note this is not depending on multiply_i as there is an optimized
  /// version for multiply_i

  _FP16 *data = getData();
  unsigned int len = size();
  sscal(len, value, data, 1);
  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::multiply(float const &value) const {
  HalfTensor t;
  return multiply(value, t);
}

HalfTensor &HalfTensor::multiply(float const &value, HalfTensor &out) const {
  /// @todo add unittest
  auto f = std::bind(std::multiplies<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, out);
  return out;

  return out;
}

int HalfTensor::multiply_i(HalfTensor const &m, const float beta) {
  try {
    this->multiply(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::multiply(HalfTensor const &m, const float beta) const {
  HalfTensor t("", this->getFormat());
  return this->multiply(m, t, beta);
}

HalfTensor &HalfTensor::multiply(HalfTensor const &m, HalfTensor &output,
                                 const float beta) const {
  /**
   * @note this does not work correctly with differently strided inputs.
   * Use multiply_strided alternatively
   */
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    if (e.strides[3] == 1 && output.strides[3] == 1 && strides[3] == 1 &&
        beta == 0.0) {
      ewvm(e.buffer_size, buf, m_buf, out_buf);
    } else {
      for (unsigned int i = 0; i < e.buffer_size; ++i) {
        *out_buf = *buf * *m_buf + static_cast<_FP16>(beta) * *out_buf;
        buf += strides[3];
        m_buf += e.strides[3];
        out_buf += output.strides[3];
      }
    }
  };

  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "HalfTensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!contiguous || !m.contiguous || !output.contiguous,
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  apply_broadcast(m, f, output);
  return output;

  return output;
}

int HalfTensor::divide_i(float const &value) {
  if (value == 0.0f) {
    return ML_ERROR_INVALID_PARAMETER;
  }
  this->divide(value, *this);
  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::divide(float const &value) const {
  HalfTensor t;
  return divide(value, t);
}

HalfTensor &HalfTensor::divide(float const &value, HalfTensor &out) const {
  /// @todo add unittest, _FP16 ZeroDivisionError
  if (value == 0.0f) {
    std::stringstream ss;
    ss << "[HalfTensor] divide by value failed, value: " << value;
    throw std::invalid_argument(ss.str().c_str());
  }

  auto f = std::bind(std::divides<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, out);
  return out;

  return out;
}

int HalfTensor::divide_i(HalfTensor const &m) {
  try {
    this->divide(m, *this);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::divide(HalfTensor const &m) const {
  HalfTensor t;
  return this->divide(m, t);
}

HalfTensor &HalfTensor::divide(HalfTensor const &m, HalfTensor &output) const {
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    if (e.strides[3] == 1 && output.strides[3] == 1 && strides[3] == 1) {
      std::transform(buf, buf + e.buffer_size, m_buf, out_buf,
                     std::divides<_FP16>());
    } else {
      for (unsigned int i = 0; i < e.buffer_size; ++i) {
        *out_buf = *buf / *m_buf;
        buf += strides[3];
        m_buf += e.strides[3];
        out_buf += output.strides[3];
      }
    }
  };

  NNTR_THROW_IF(!contiguous || !m.contiguous || !output.contiguous,
                std::invalid_argument)
    << getName() << " is not contiguous, cannot divide";

  apply_broadcast(m, f, output);

  return output;
}

int HalfTensor::add_i(float const &value) {
  this->add(value, *this);
  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::add(float const &value) const {
  HalfTensor t;
  return add(value, t);
}

HalfTensor &HalfTensor::add(float const &value, HalfTensor &out) const {
  /// @todo add unittest
  auto f = std::bind(std::plus<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, out);
  return out;
}

int HalfTensor::add_i(HalfTensor const &m, float const alpha) {
  /// @todo: add axis rather doing add over the last two dimensions always
  /// operator i has optimized version
  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    saxpy(e.buffer_size, alpha, m_buf, e.strides[3], out_buf, strides[3]);
    /// @todo: saxpy is not valid for _FP16
  };

  /// @todo: enable this after add_strided supports broadcast
  // NNTR_THROW_IF(!contiguous || !m.contiguous, std::invalid_argument)
  //   << getName() << " is not contiguous, cannot add";

  try {
    apply_broadcast(m, f, *this);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::add(HalfTensor const &m, float const alpha) const {
  HalfTensor t;
  return this->add(m, t, alpha);
}

HalfTensor &HalfTensor::add(HalfTensor const &m, HalfTensor &output,
                            float const alpha) const {
  NNTR_THROW_IF(!contiguous || !m.contiguous || !output.contiguous,
                std::invalid_argument)
    << getName() << " is not contiguous, cannot add";

  auto f = [&](const BroadcastInfo &e, const _FP16 *buf, const _FP16 *m_buf,
               _FP16 *out_buf) {
    if (e.strides[3] == 1 && strides[3] == 1 && strides[3] == 1 && alpha == 0) {
      ewva(e.buffer_size, buf, m_buf, out_buf);
    } else {
      for (unsigned int i = 0; i < e.buffer_size; ++i) {
        *out_buf = *buf + *m_buf * static_cast<_FP16>(alpha);
        buf += strides[3];
        m_buf += e.strides[3];
        out_buf += strides[3];
      }
    }
  };
  apply_broadcast(m, f, output);

  return output;
}

int HalfTensor::subtract_i(float const &value) {
  this->subtract(value, *this);
  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::subtract(float const &value) const {
  HalfTensor t;
  return subtract(value, t);
}

HalfTensor &HalfTensor::subtract(float const &value, HalfTensor &out) const {
  /// @todo add unittest
  auto f = std::bind(std::minus<_FP16>(), std::placeholders::_1,
                     static_cast<_FP16>(value));
  apply(f, out);

  return out; // shouldn't reach
}

int HalfTensor::subtract_i(HalfTensor const &m) { return add_i(m, -1); }

HalfTensor HalfTensor::subtract(HalfTensor const &m) const {
  return add(m, -1);
}

HalfTensor &HalfTensor::subtract(HalfTensor const &m, HalfTensor &out) const {
  return add(m, out, -1);
}

int HalfTensor::pow_i(float exponent) {
  pow(exponent, *this);
  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::pow(float exponent) const {
  HalfTensor t;
  return pow(exponent, t);
}

HalfTensor &HalfTensor::pow(float exponent, HalfTensor &out) const {
  auto f = [exponent](_FP16 in) {
    return static_cast<_FP16>(powf(in, exponent));
  };
  apply(f, out);

  return out;
}

HalfTensor HalfTensor::getBatchSlice(size_t offset, unsigned int size) const {
  TensorDim dim_ = dim;
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->dim.getFeatureLen());
}

void HalfTensor::createSharedDataTensor(const HalfTensor &src, HalfTensor &dest,
                                        size_t offset) {
  /**
   * - If src already has data allocaed, then directly make dest tensor based on
   * the src tensor.
   * - If src.data does not exist (meaning tensor does not memory allocated),
   * and src.src_tensor does not exist (meaning the src tensor does not depened
   * on another tensor), then create a SrcSharedHalfTensor around the src.
   * - If src.src_tensor exists, then use the src.src_tensor to create the
   *  required SrcSharedHalfTensor to avoid recursive dependency.
   *
   * @note src.data and src.src_tensor CAN co-exist. src.src_tensor is stored
   * if the batch size of src is updated and needs reallocation.
   */
  dest.data = nullptr;
  if (src.data) {
    dest.src_tensor = std::make_shared<SrcSharedHalfTensor>(&src, offset);
    dest.allocate();
  } else if (!src.src_tensor)
    dest.src_tensor = std::make_shared<SrcSharedHalfTensor>(&src, offset);
  else
    dest.src_tensor = std::make_shared<SrcSharedHalfTensor>(
      src.src_tensor->tensor(), offset + src.src_tensor->offset());
}

HalfTensor HalfTensor::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                           bool reset_stride,
                                           const std::string &name_) const {
  HalfTensor ret = *this;
  if (dim_.getFormat() != ret.dim.getFormat())
    throw std::invalid_argument("HalfTensor format does not match");

  ret.dim = dim_;
  if (!name_.empty())
    ret.name = name_;

  if (dim_.getDataLen() + offset > dim.getDataLen())
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");

  if (reset_stride)
    ret.strides = ret.dim.computeStrides();

  TensorDim new_match_dim = dim_;
  new_match_dim.batch(dim.batch());
  if (new_match_dim != dim && !reset_stride)
    ret.contiguous = false;

  /**
   * In this case, its the caller's responsibility to ensure that allocate() is
   * called for the output tensor before operating on the output tensor.
   */
  createSharedDataTensor(*this, ret, offset);

  return ret;
}

std::vector<HalfTensor> HalfTensor::split(unsigned num_size, int axis) {
  NNTR_THROW_IF(num_size == 0, std::invalid_argument)
    << "num size cannot be zero";

  if (axis == -1) {
    axis = 3;
  }

  NNTR_THROW_IF(!(0 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(dim.getTensorDim(axis) % num_size != 0, std::invalid_argument)
    << "axis is not divisible by num_size, axis: " << axis
    << " num size: " << num_size;

  std::vector<size_t> sizes;
  sizes.resize(num_size);

  unsigned int sz = dim.getTensorDim(axis) / num_size;
  std::fill(sizes.begin(), sizes.end(), sz);

  return split(sizes, axis);
}

std::vector<HalfTensor> HalfTensor::split(std::vector<size_t> sizes, int axis) {
  size_t num_size = sizes.size();

  NNTR_THROW_IF(num_size == 0, std::invalid_argument)
    << "num size cannot be zero";

  if (axis == -1) {
    axis = 3;
  }

  NNTR_THROW_IF(!(0 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(
    std::any_of(sizes.begin(), sizes.end(), [](size_t sz) { return !sz; }),
    std::invalid_argument)
    << "among given sizes at least one of size is 0";

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
  std::vector<HalfTensor> ret;

  auto iter_value = [this, is_format_nchw](
                      std::array<size_t, 4> &loc,
                      const std::array<size_t, 4> &end_loc,
                      const std::array<size_t, 4> &reset_dim_arr) -> _FP16 & {
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

    ret_t.apply_i([&iter_value, &loc, &end_loc, &reset_dim_arr](_FP16 _) {
      return iter_value(loc, end_loc, reset_dim_arr);
    });
  }

  return ret;
}

HalfTensor HalfTensor::cat(const std::vector<HalfTensor> &tensors, int axis) {

  if (axis == -1) {
    axis = 3;
  }

  NNTR_THROW_IF(!(0 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(tensors.empty(), std::invalid_argument)
    << "given tensor vector is empty";

  HalfTensor ret;
  auto ref_dim = tensors.front().getDim();
  bool is_format_nchw = (ref_dim.getFormat() == Tformat::NCHW);
  ref_dim.setTensorDim(axis, 1);
  NNTR_THROW_IF(!std::all_of(tensors.begin(), tensors.end(),
                             [&ref_dim, axis](const HalfTensor &t) {
                               auto cur_dim = t.getDim();
                               cur_dim.setTensorDim(axis, 1);
                               return ref_dim == cur_dim;
                             }),
                std::invalid_argument)
    << " all tensor must have the same dimension except for the axis, ref_dim: "
    << ref_dim << " axis : " << axis;

  auto axis_dim = std::accumulate(tensors.begin(), tensors.end(), 0u,
                                  [axis](unsigned cur, const HalfTensor &t) {
                                    return cur += t.getDim().getTensorDim(axis);
                                  });

  auto iter_value =
    [is_format_nchw](std::array<unsigned, 4> &loc,
                     const std::array<unsigned, 4> &start_loc, HalfTensor &t,
                     const std::array<unsigned, 4> &ref_dim_arr) -> _FP16 & {
    auto &value = is_format_nchw ? t.getValue(loc[0], loc[1], loc[2], loc[3])
                                 : t.getValue(loc[0], loc[3], loc[1], loc[2]);

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

  auto ret_dim = ref_dim;
  ret_dim.setTensorDim(axis, axis_dim);

  ret = HalfTensor(ret_dim);

  std::array<unsigned, 4> loc = {0, 0, 0, 0};
  for (auto &t : tensors) {
    std::array<unsigned, 4> start_loc = loc;
    std::array<unsigned, 4> tensor_dim_arr;
    if (is_format_nchw) {
      tensor_dim_arr[0] = t.getDim().getTensorDim(0);
      tensor_dim_arr[1] = t.getDim().getTensorDim(1);
      tensor_dim_arr[2] = t.getDim().getTensorDim(2);
      tensor_dim_arr[3] = t.getDim().getTensorDim(3);
    } else {
      tensor_dim_arr[0] = t.getDim().getTensorDim(0);
      tensor_dim_arr[1] = t.getDim().getTensorDim(2);
      tensor_dim_arr[2] = t.getDim().getTensorDim(3);
      tensor_dim_arr[3] = t.getDim().getTensorDim(1);
    }

    for (size_t i = 0u, sz = t.size(); i < sz; ++i) {
      iter_value(loc, start_loc, ret, tensor_dim_arr) = t.getValue(i);
    }

    if (is_format_nchw) {
      loc[axis] += t.getDim().getTensorDim(axis);
    } else {
      if (axis == 0) {
        loc[0] += t.getDim().getTensorDim(axis);
      } else if (axis == 1) {
        loc[3] += t.getDim().getTensorDim(axis);
      } else if (axis == 2 || axis == 3) {
        loc[axis - 1] += t.getDim().getTensorDim(axis);
      }
    }
  }

  return ret;
}

void HalfTensor::makeSharedDataTensor(const HalfTensor &src, size_t offset) {
  if (strides != src.strides)
    throw std::invalid_argument(
      "Creating shared tensor of different stride than source tensor.");

  if (getDim().getDataLen() + offset > src.getDim().getDataLen())
    throw std::invalid_argument(
      "Creating shared tensor of different size or stride than source tensor.");

  /**
   * In this case, its the caller's responsibility to ensure that allocate() is
   * called for the output tensor before operating on the output tensor.
   */
  createSharedDataTensor(src, *this, offset);
}

void HalfTensor::apply_broadcast(
  HalfTensor const &m,
  std::function<void(const BroadcastInfo &e, const _FP16 *, const _FP16 *,
                     _FP16 *)>
    v_func,
  HalfTensor &output) const {
  CREATE_IF_EMPTY_DIMS_HALF(output, dim, nullptr);

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

  /// shortcut to cover when dimension matches
  /// note that buffer_size, the last stride is only used in v_func but it
  /// might be changed
  if (dim == m.dim) {
    BroadcastInfo e;
    e.buffer_size = size();
    e.strides[3] = 1;
    v_func(e, getData(), m.getData(), output.getData());
    return;
  }

  return apply_broadcast_util(m, v_func, output, this->computeBroadcastInfo(m));
}

void HalfTensor::apply_broadcast_util(
  HalfTensor const &m,
  std::function<void(const BroadcastInfo &e, const _FP16 *, const _FP16 *,
                     _FP16 *)>
    v_func,
  HalfTensor &output, const BroadcastInfo &e, int cur_axis, size_t offset,
  size_t m_offset) const {

  const _FP16 *buf = this->getData();
  const _FP16 *m_buf = m.getData();
  _FP16 *out_buf = output.getData();

  if (e.buffer_axis == cur_axis) {
    v_func(e, buf + offset, m_buf + m_offset, out_buf + offset);
    return;
  }

  cur_axis++;
  for (unsigned int i = 0; i < dim.getTensorDim(cur_axis); ++i) {
    size_t next_offset = offset + i * strides[cur_axis];
    size_t next_m_offset = m_offset + i * e.strides[cur_axis];
    apply_broadcast_util(m, v_func, output, e, cur_axis, next_offset,
                         next_m_offset);
  }
}

/**
 * This is to sum the HalfTensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1, 1) dimension.
 */
HalfTensor HalfTensor::sum_by_batch() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  HalfTensor ret(dim.batch(), 1, 1, 1, this->getFormat(), getDataType());
  size_t feat_len = dim.getFeatureLen();
  size_t batch = dim.batch();

  const _FP16 *data = getData();
  _FP16 *rdata = ret.getData();

  HalfTensor ones(1, 1, 1, feat_len, this->getTensorType());
  ones.setValue((_FP16)1.0);
  sgemv(CblasRowMajor, CblasNoTrans, batch, feat_len, 1, data, feat_len,
        ones.getData(), 1, 0.0, rdata, 1);

  return ret;
}

/**
 * @brief Calculate sum according to the axis.
 */
HalfTensor HalfTensor::sum(unsigned int axis, float alpha) const {
  HalfTensor ret("", this->getFormat(), this->getDataType());
  return sum(axis, ret, alpha, 0);
}

HalfTensor &HalfTensor::sum(unsigned int axis, HalfTensor &ret, float alpha,
                            float beta) const {
  const _FP16 *data = getData();

  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  if (axis >= 4)
    throw std::out_of_range("Error: axis is invalid");

  if (dim.getDim()[axis] == 1 and alpha == 1.0 and !beta) {
    CREATE_IF_EMPTY_DIMS_HALF(ret, dim);
    ret.copy(this->getData());
    return ret;
  }

  switch (axis) {
  case 0: {
    CREATE_IF_EMPTY_DIMS_HALF(ret, 1, dim.channel(), dim.height(), dim.width(),
                              this->getTensorType());
    size_t feat_len = dim.getFeatureLen();
    size_t batch = dim.batch();
    HalfTensor ones(1, 1, 1, batch, this->getTensorType());
    ones.setValue(alpha);
    sgemv(CblasRowMajor, CblasTrans, batch, feat_len, 1, data, feat_len,
          ones.getData(), 1, beta, ret.getData(), 1);
  } break;
  case 1: {
    CREATE_IF_EMPTY_DIMS_HALF(ret, dim[0], 1, dim[2], dim[3], getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int m = ret.dim.getDataLen();
      unsigned int n = dim[1];
      HalfTensor ones(1, 1, 1, n, this->getTensorType());
      ones.setValue(alpha);
      sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, data, n, ones.getData(), 1,
            beta, ret.getData(), 1);
    } else {
      unsigned int feat_len = dim[2] * dim[3];
      unsigned int t_axis = dim[1];
      HalfTensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = ret.getData();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv(CblasRowMajor, CblasTrans, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData(), 1, beta,
              &rdata[k * feat_len], 1);
      }
    }
  } break;
  case 2: {
    CREATE_IF_EMPTY_DIMS_HALF(ret, dim[0], dim[1], 1, dim[3], getTensorType());

    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = dim[1] * dim[3];
      unsigned int t_axis = dim[2];
      HalfTensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = ret.getData();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv(CblasRowMajor, CblasTrans, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData(), 1, beta,
              &rdata[k * feat_len], 1);
      }
    } else {
      unsigned int t_3 = dim[3];
      unsigned int t_axis = dim[2];
      HalfTensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = ret.getData();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        for (unsigned int c = 0; c < dim[1]; ++c) {
          unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[2];
          unsigned int ridx = k * ret.dim.getFeatureLen() + c * dim[3];
          sgemv(CblasRowMajor, CblasTrans, t_axis, t_3, 1, &data[idx], t_3,
                ones.getData(), 1, beta, &rdata[ridx], 1);
        }
      }
    }
  } break;
  case 3: {
    CREATE_IF_EMPTY_DIMS_HALF(ret, dim[0], dim[1], dim[2], 1, getTensorType());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int t_3 = dim[1];
      unsigned int t_axis = dim[3];
      HalfTensor ones(1, 1, 1, t_axis, getTensorType());
      ones.setValue(alpha);
      _FP16 *rdata = ret.getData();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        for (unsigned int c = 0; c < dim[2]; ++c) {
          unsigned int idx = k * dim.getFeatureLen() + c * dim[3] * dim[1];
          unsigned int ridx = k * ret.dim.getFeatureLen() + c * dim[1];
          sgemv(CblasRowMajor, CblasTrans, t_axis, t_3, 1, &data[idx], t_3,
                ones.getData(), 1, beta, &rdata[ridx], 1);
        }
      }
    } else {
      unsigned int m = ret.dim.getDataLen();
      unsigned int n = dim[3];
      HalfTensor ones(1, 1, 1, n, getTensorType());
      ones.setValue(alpha);
      sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, data, n, ones.getData(), 1,
            beta, ret.getData(), 1);
    }
  } break;
  default:
    throw std::out_of_range("Error: Dimension cannot exceed 3");
  }

  return ret;
}

HalfTensor HalfTensor::sum(const std::vector<unsigned int> &axes,
                           float alpha) const {
  HalfTensor ret("", this->getFormat());
  return sum(axes, ret, alpha);
}

void HalfTensor::mergeAxis(unsigned int axis1, unsigned int axis2) {
  std::vector<unsigned int> continuous_order = {0, 3, 1, 2};
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot merge axis";

  if (axis2 != axis1 + 1)
    if (!checkContinuous(axis1, axis2))
      throw std::invalid_argument("axis2 must be axis1 + 1 for merging.");

  dim.setTensorDim(axis2, dim.getTensorDim(axis1) * dim.getTensorDim(axis2));
  dim.setTensorDim(axis1, 1);
}

HalfTensor &HalfTensor::sum(const std::vector<unsigned int> &axes,
                            HalfTensor &output, float alpha) const {
  if (axes.empty())
    throw std::invalid_argument("empty axes given");

  if (axes.size() == 1) {
    this->sum(axes[0], output, alpha);
  } else {
    /** club axes together */
    HalfTensor new_reshaped = *this;
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

    HalfTensor ret = new_reshaped.sum(new_axes[0]);
    for (unsigned int i = 1; i < new_axes.size() - 1; ++i)
      ret = ret.sum(axes[i]);
    ret.sum(new_axes.back(), output, alpha);
  }

  return output;
}

HalfTensor &HalfTensor::dotBatched(HalfTensor const &m, HalfTensor &result,
                                   bool trans, bool trans_m, float beta) const {
  if (!result.isAllocated())
    throw std::invalid_argument(
      "Output tensor must be preallocated for dotBatched operation");
  for (unsigned int b = 0; b < batch(); b++) {
    /** @todo try using transpose to speedup the operation */
    const HalfTensor this_b = this->getBatchSlice(b, 1);
    HalfTensor m_b = m.getBatchSlice(b, 1);
    HalfTensor result_b = result.getBatchSlice(b, 1);

    this_b.dot(m_b, result_b, trans, trans_m, beta);
  }

  return result;
}

HalfTensor HalfTensor::dot(HalfTensor const &m, bool trans,
                           bool trans_m) const {
  HalfTensor output("", this->getFormat(), this->getDataType());
  dot(m, output, trans, trans_m);

  return output;
}
/**
 * @brief compute the derivative of this in the current tensor
 * @todo will have to see if beta effects this computation
 */
HalfTensor &HalfTensor::dot_deriv_wrt_1(HalfTensor const &m,
                                        HalfTensor const &output_deriv,
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
HalfTensor &HalfTensor::dot_deriv_wrt_2(HalfTensor &m_deriv,
                                        HalfTensor const &output_deriv,
                                        bool trans, bool trans_m,
                                        float beta) const {
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

HalfTensor &HalfTensor::dot_batched_deriv_wrt_1(HalfTensor const &m,
                                                HalfTensor const &output_deriv,
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

HalfTensor &HalfTensor::dot_batched_deriv_wrt_2(HalfTensor &m_deriv,
                                                HalfTensor const &output_deriv,
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

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor
 * in case of trans is false.
 */
HalfTensor &HalfTensor::dot(HalfTensor const &m, HalfTensor &result, bool trans,
                            bool trans_m, float beta) const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous. Cannot dot product.";

  // Comment out with intension to support the calculation wrt. batch and height
  // direction. It supposes to have this->dim as [ BxCxH,W ] and m.dim is
  // [BxCxH,W] as well if (m.dim.rank() > 2) {
  //   throw exception::not_supported("Error: support only for rank of dot "
  //                                  "matrix <= 2");
  // }

  // Comment out with intension to support the calculation wrt. batch and height
  // direction of this tensor. It is OK as long as m is 2D
  //
  if (trans && dim.rank() > 2) {
    ml_logw("Warning: support only for rank of dot matrix <= 2 with trans");
  }
  unsigned int dim1, dim2, mdim1, mdim2;
  if (getFormat() == Tformat::NHWC) {
    dim1 = batch() * height() * width();
    dim2 = channel();
    mdim1 = m.batch() * m.height() * m.width();
    mdim2 = m.channel();
  } else {
    dim1 = batch() * channel() * height();
    dim2 = width();
    mdim1 = m.batch() * m.channel() * m.height();
    mdim2 = m.width();
  }

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS_HALF(result, batch(), N, height(), width(),
                                getTensorType()); //  NHWC Result HalfTensor
    } else {
      CREATE_IF_EMPTY_DIMS_HALF(result, batch(), channel(), height(), N,
                                getTensorType());
    }

    // We are not set zero the result because of performance reason.
    // However, result is not initialized properly. There might include
    // garbage like nan. When we have to use this value as in C = alpha*A*B +
    // beta*C, then have to check garbage data of C is not effect or not.

  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS_HALF(result, batch(), N, height(), width(),
                                getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS_HALF(result, batch(), channel(), height(), N,
                                getTensorType());
    }
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS_HALF(result, 1, N, M, 1, getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS_HALF(result, 1, 1, M, N, getTensorType());
    }
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS_HALF(result, 1, N, M, 1, getTensorType());
    } else {
      CREATE_IF_EMPTY_DIMS_HALF(result, 1, 1, M, N, getTensorType());
    }
  }
  lda = dim2;
  ldb = mdim2;
  ldc = (getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  const _FP16 *data = getData();
  const _FP16 *mdata = m.getData();
  _FP16 *rdata = result.getData();
  const float alpha = 1.0f;
  enum CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
  enum CBLAS_TRANSPOSE transB = trans_m ? CblasTrans : CblasNoTrans;

  /// shortcut handling in case of vector
  /// for vector, (1 * K) == (K * 1) in current memory layout...
  /// and plaese note that N, K, M is a fixed place holder after considering
  /// transpose.
  /// For example, there is no case like (1 * K) X (1 * K) while
  /// (1 * K) X (1 * M) can be a case
  /// case1: (1 * K) X (K * 1)
  if (M == 1 && N == 1) {
    *rdata = sdot(K, data, 1, mdata, 1) + static_cast<_FP16>(beta) * (*rdata);
  }
  /// case2: (M * K) X (K * 1)
  else if (N == 1) {
    sgemv(CblasRowMajor, transA, dim1, dim2, alpha, data, lda, mdata, 1, beta,
          rdata, 1);
  }
  /// case3: (1 * K) X (K * N) = 1 * N = R
  /// = R^T = (K * N) ^T * (1 * K) ^T = (N * K) * (K * 1) = (N * K) * (1 * K)
  /// Effectively a translation of sgemv
  else if (M == 1) {
    transB = transB == CblasTrans ? CblasNoTrans : CblasTrans;
    sgemv(CblasRowMajor, transB, mdim1, mdim2, alpha, mdata, ldb, data, 1, beta,
          rdata, 1);
  }
  /// case others: use sgemm
  else {
    sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, data, lda, mdata, ldb,
          beta, rdata, ldc);
  }

  return result;
}

HalfTensor &HalfTensor::transpose(const std::string &direction,
                                  HalfTensor &out) const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous. Cannot transpose.";

  if (out.getData() == getData()) {
    HalfTensor tmp = clone();
    return tmp.transpose(direction, out);
  }

  unsigned int SL, SI, SJ, SK;

  out.reshape(dim.transpose(direction));

  int indexI = direction[0] - '0';
  int indexJ = direction[2] - '0';

  SL = dim.batch(), SI = dim.channel(), SJ = dim.height(), SK = dim.width();

  bool is_format_nchw = (getFormat() == Tformat::NCHW);

  const _FP16 *inptr = getData();
  _FP16 *outptr = out.getData();
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

  return out;
}

HalfTensor HalfTensor::transpose(const std::string &direction) const {
  HalfTensor result(dim);
  transpose(direction, result);
  return result;
}

HalfTensor HalfTensor::dropout_mask(float dropout) const {
  HalfTensor result(dim);
  result.dropout_mask(dropout);
  return result;
}

void HalfTensor::dropout_mask(float dropout) {
  setRandUniform(0.0, 1.0);

  _FP16 scale = static_cast<_FP16>(1.0 / (1 - dropout));
  _FP16 *data_ = getData();
  for (unsigned int i = 0; i < size(); ++i) {
    if (data_[i] >= dropout)
      data_[i] = scale;
    else
      data_[i] = 0;
  }
}

void HalfTensor::filter_mask(const HalfTensor &mask_len, bool reverse) {
  float fill_mask_val = 0.0;
  float en_mask_val = 1.0 - fill_mask_val;

  if (reverse) {
    fill_mask_val = 1.0;
    en_mask_val = 1.0 - fill_mask_val;
  }

  setValue(fill_mask_val);
  if (mask_len.batch() != batch())
    throw std::invalid_argument("Number of filter masks mismatched");

  for (unsigned int b = 0; b < batch(); b++) {
    _FP16 *addr = getAddress(b, 0, 0, 0);
    const uint *mask_len_val = (uint *)mask_len.getAddress(b, 0, 0, 0);
    std::fill(addr, addr + (*mask_len_val), (_FP16)en_mask_val);
  }
}

HalfTensor HalfTensor::zoneout_mask(float zoneout) {
  HalfTensor ret(getDim());
  zoneout_mask(ret, zoneout);
  return ret;
}

void HalfTensor::zoneout_mask(HalfTensor &opposite, float zoneout) {
  if (dim != opposite.dim) {
    throw std::invalid_argument(
      "[HalfTensor::zoneout_mask] opposite dimension does not match");
  }

  _FP16 zoneout_fp16 = static_cast<_FP16>(zoneout);
  opposite.setRandBernoulli(zoneout_fp16);

  _FP16 *data = getData();
  _FP16 *opposite_data = opposite.getData();

  for (unsigned int i = 0; i < size(); ++i) {
    if (opposite_data[i] > epsilon) {
      data[i] = static_cast<_FP16>(0.0);
    } else {
      data[i] = static_cast<_FP16>(1.0);
    }
  }
}

HalfTensor HalfTensor::apply(std::function<HalfTensor(HalfTensor)> f) const {
  return f(*this);
}

HalfTensor &
HalfTensor::apply(std::function<HalfTensor &(HalfTensor, HalfTensor &)> f,
                  HalfTensor &output) const {
  return f(*this, output);
}

void HalfTensor::print(std::ostream &out) const {
  printInstance(out, this);

  const _FP16 *data = getData();
  unsigned int len = size();
  out << "data addr: " << data << '\n';
  out << dim;

  if (len > 100) {
    out << '[' << (float)data[0] << ' ' << (float)data[1] << ' '
        << (float)data[2] << " ... " << (float)data[len - 3] << ' '
        << (float)data[len - 2] << ' ' << (float)data[len - 1] << ']'
        << std::endl;
    return;
  }

  std::ios init(NULL);
  init.copyfmt(out);
  float max_ = 0.0;
  float min_ = 10000000;
  if (getFormat() == Tformat::NCHW) {
    for (unsigned int k = 0; k < batch(); k++) {
      for (unsigned int l = 0; l < channel(); l++) {
        for (unsigned int i = 0; i < height(); i++) {
          for (unsigned int j = 0; j < width(); j++) {
            out << std::setw(10) << std::setprecision(10)
                << (float)this->getValue(k, l, i, j) << " ";
            if (std::isinf((float)this->getValue(k, l, i, j)))
              out << "INF or NAN " << k << ":" << l << ":" << i << ":" << j
                  << std::endl;
            if ((float)this->getValue(k, l, i, j) < min_)
              min_ = (float)this->getValue(k, l, i, j);
            if ((float)this->getValue(k, l, i, j) > max_)
              max_ = (float)this->getValue(k, l, i, j);
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
                << (float)this->getValue(k, l, i, j) << " ";
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

void HalfTensor::print_(std::ostream &out, uint opt) const {
  printInstance(out, this);

  unsigned int len = size();

  std::ios init(NULL);
  init.copyfmt(out);
  if (opt == 0) {
    if (getFormat() == Tformat::NCHW) {
      out << "{";
      for (unsigned int k = 0; k < batch(); k++) {
        out << "{";
        for (unsigned int i = 0; i < channel(); i++) {
          out << "{";
          for (unsigned int j = 0; j < height(); j++) {
            out << "{";
            for (unsigned int l = 0; l < width(); l++) {
              if (l < width() - 1)
                out << std::setw(10) << std::setprecision(10)
                    << (float)this->getValue(k, l, i, j) << ", ";
              else
                out << std::setw(10) << std::setprecision(10)
                    << (float)this->getValue(k, l, i, j);
            }
            if (j < height() - 1)
              out << "},";
            else
              out << "}";
            out << std::endl;
          }
          if (i < channel() - 1)
            out << "},";
          else
            out << "}";
          out << std::endl;
        }
        if (k < batch() - 1)
          out << "},";
        else
          out << "}";
        out << std::endl;
      }
      out << "}";
    } else {
      out << "{";
      for (unsigned int k = 0; k < batch(); k++) {
        out << "{";
        for (unsigned int i = 0; i < height(); i++) {
          out << "{";
          for (unsigned int j = 0; j < width(); j++) {
            out << "{";
            for (unsigned int l = 0; l < channel(); l++) {
              if (l < channel() - 1)
                out << std::setw(10) << std::setprecision(10)
                    << (float)this->getValue(k, l, i, j) << ", ";
              else
                out << std::setw(10) << std::setprecision(10)
                    << (float)this->getValue(k, l, i, j);
            }
            if (j < width() - 1)
              out << "},";
            else
              out << "}";
            out << std::endl;
          }
          if (i < height() - 1)
            out << "},";
          else
            out << "}";
          out << std::endl;
        }
        if (k < batch() - 1)
          out << "},";
        else
          out << "}";
        out << std::endl;
      }
      out << "}";
    }
  } else {
    for (uint i = 0; i < len; ++i) {
      out << (float)getData()[i] << ", ";
    }
  }
  out.copyfmt(init);
}

std::ostream &operator<<(std::ostream &out, HalfTensor const &m) {
  m.print(out);
  return out;
}

void HalfTensor::copy(const void *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << "HalfTensor is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  scopy(size(), (_FP16 *)buf, 1, getData(), 1);
}

void HalfTensor::copy_with_stride(const HalfTensor &from) {

  if (dim == from.getDim()) {
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            setValue(b, c, h, w, from.getValue(b, c, h, w));
          }
        }
      }
    }

  } else {
    HalfTensor t = HalfTensor(from.getDim(), true);

    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            setValue(b, c, h, w, from.getValue(b, c, h, w));
          }
        }
      }
    }

    swap(t, *this);
  }
}

void HalfTensor::copy(const HalfTensor &from) {
  // todo: enable copy to non-contiguous tensor
  if (!contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (from.size() != 0 && size() == from.size() &&
      getDataType() == from.getDataType()) {
    reshape(from.getDim());

    copy(from.getData());
  } else {
    HalfTensor t = HalfTensor(from.getDim(), from.getData());
    swap(t, *this);
  }
}

void HalfTensor::copyData(const HalfTensor &from) {
  // todo: enable copy to non-contiguous tensor
  if (!contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (size() != from.size())
    throw std::invalid_argument("Size of tensor to copy must match");

  copy(from.getData());
}

HalfTensor HalfTensor::clone() const {
  HalfTensor t;
  t.copy(*this);
  t.name = name;
  return t;
}

void HalfTensor::reshape(const TensorDim &d) {

  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot reshape.";

  NNTR_THROW_IF(d.getDataLen() != dim.getDataLen(), std::invalid_argument)
    << "[HalfTensor]: reshape cannot change the buffer size, trying reshaping "
       "\nfrom "
    << getDim() << " to " << d;

  //   dim = d;
  dim.batch(d.batch());
  dim.channel(d.channel());
  dim.height(d.height());
  dim.width(d.width());

  strides = d.computeStrides();
}

void HalfTensor::fill(const HalfTensor &from, bool alloc) {
  if (alloc && this->empty()) {
    this->copy(from);
    return;
  }

  if (!from.contiguous || !contiguous) {
    /// @todo enable this if needed
    throw nntrainer::exception::not_supported(
      "[HalfTensor::fill] non-contiguous tensors are not supported");
  }

  if (dim != from.getDim()) {
    throw std::invalid_argument(
      "[HalfTensor::fill] dimension must be the same");
  }

  if (strides != from.getStrides()) {
    /// @todo length does not represent buffer size, there should be way to
    /// get the buffer size
    throw std::invalid_argument(
      "[HalfTensor::fill] buffer size must be the same");
  }

  this->copy(from.getData());
}

void HalfTensor::save(std::ostream &file) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot save.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  //   std::vector<_FP16> temp(size());
  //   for (unsigned int i = 0; i < size(); ++i) {
  //     temp[i] = static_cast<_FP16>(getData()[i]);
  //   }

  //   checkedWrite(file, (char *)temp.data(),
  //                static_cast<std::streamsize>(size() * sizeof(_FP16)),
  //                "[HalfTensor::save] operation failed");
  checkedWrite(file, (char *)getData(), sz,
               "[FloatTensor::save] operation failed");

  putData();
}

void HalfTensor::read(std::ifstream &file, Tdatatype s_type) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot read.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(file, (char *)getData(), sz,
              "[HalfTensor::read] operation failed");
  putData();
}

/**
 * @brief Calculate average value according to the axis.
 */
HalfTensor HalfTensor::average(unsigned int axis) const {
  HalfTensor t("", this->getFormat(), this->getDataType());
  return average(axis, t);
}

/**
 * @brief Calculate average value according to the axis.
 */
HalfTensor &HalfTensor::average(unsigned int axis, HalfTensor &output) const {
  if (axis >= TensorDim::MAXDIM)
    throw std::out_of_range(
      "negative axis or axis more then MAXDIM is invalid");

  unsigned int axis_size = dim.getDim()[axis];
  if (axis_size == 1)
    output.copy(*this);
  else
    this->sum(axis, output, 1.0 / ((float)axis_size));

  return output;
}

HalfTensor HalfTensor::average(const std::vector<unsigned int> &axes) const {
  HalfTensor t("", this->getFormat(), this->getDataType());
  return average(axes, t);
}

HalfTensor &HalfTensor::average(const std::vector<unsigned int> &axes,
                                HalfTensor &output) const {
  if (axes.empty())
    return this->average(output);

  TensorDim ret_shape(getTensorType());

  for (const auto &idx : axes) {
    if (idx >= TensorDim::MAXDIM) {
      throw std::out_of_range("axis more then MAXDIM is invalid");
    }
    ret_shape.setTensorDim(idx, dim.getTensorDim(idx));
  }

  return this->sum(axes, output, 1.0 / (float)ret_shape.getDataLen());
}

/**
 * @brief Calculate average value according to the axis.
 */
HalfTensor HalfTensor::average() const {
  HalfTensor result = *this;
  unsigned int axis = 0;
  if (this->getFormat() == Tformat::NHWC) {
    result.reshape({1, dim.getDataLen(), 1, 1, this->getTensorType()});
    axis = 1;
  } else {
    result.reshape({1, 1, 1, dim.getDataLen(), this->getTensorType()});
    axis = 3;
  }
  return result.average(axis);
}

/**
 * @brief Calculate average value according to the axis.
 */
HalfTensor &HalfTensor::average(HalfTensor &output) const {
  HalfTensor result = *this;
  result.reshape({1, 1, 1, dim.getDataLen()});
  return result.average(3, output);
}

void HalfTensor::setValue(float val) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot set value.";

  _FP16 *data = getData();
  std::fill(data, data + size(), static_cast<_FP16>(val));
}

void HalfTensor::setZero() {
  if (contiguous)
    sscal(size(), 0, getData(), 1);
  else
    apply_i([](_FP16 val) -> _FP16 { return 0; });
}

std::vector<unsigned int> HalfTensor::argmax() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot get argmax.";
  std::vector<unsigned int> result;

  const _FP16 *data = getData();
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

int HalfTensor::erf_i() {
  erf(*this);
  return ML_ERROR_NONE;
}

HalfTensor HalfTensor::erf() const {
  HalfTensor t;
  return erf(t);
}

HalfTensor &HalfTensor::erf(HalfTensor &out) const {
  auto f = [](_FP16 in) {
    return static_cast<_FP16>(std::erf(static_cast<float>(in)));
  };
  apply(f, out);

  return out;
}

float HalfTensor::l2norm() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot get l2norm.";
  float ret = 0;
  unsigned int len = size();

  const _FP16 *data = getData();
  ret = snrm2(len, data, 1);

  return ret;
}

float HalfTensor::max_abs() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot get max_abs.";

  unsigned int len = size();
  float ret = 0;

  const _FP16 *data = getData();

  unsigned int idx = isamax(len, data, 1);
  ret = *(data + idx);

  return ret;
}

HalfTensor &HalfTensor::normalization(HalfTensor &output) const {
  if (output.empty())
    output = HalfTensor(dim);

  output.copy(*this);
  output.normalization_i();

  return output;
}

void HalfTensor::normalization_i() {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot do normalization.";

  const _FP16 *data = getData();

  auto bounds = std::minmax_element(data, data + size());
  const _FP16 min = *bounds.first;
  const _FP16 max = *bounds.second;

  if (max == min) {
    HalfTensor tmp = *this;
    this->subtract_i(tmp);
  } else {
    this->subtract_i(min);
    this->divide_i(max - min);
  }
}

// LazyTensor HalfTensor::chain() const { return LazyTensor(*this); }

HalfTensor &HalfTensor::standardization(HalfTensor &output) const {
  if (output.empty())
    output = HalfTensor(dim);

  output.copy(*this);
  output.standardization_i();

  return output;
}

void HalfTensor::standardization_i() {
  HalfTensor mean_by_batch = this->sum_by_batch();
  mean_by_batch.divide_i(dim.getFeatureLen());

  this->subtract_i(mean_by_batch);

  HalfTensor std_dev_by_batch(dim.batch(), 1, 1, 1, dim.getFormat(),
                              dim.getDataType());
  std_dev_by_batch.setZero();
  _FP16 *std_dev = std_dev_by_batch.getData();

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    HalfTensor sub_this = this->getBatchSlice(k, 1);
    std_dev[k] = static_cast<_FP16>(sub_this.l2norm());
  }

  std_dev_by_batch.divide_i(dim.getFeatureLen());
  this->divide_i(std_dev_by_batch);
}

HalfTensor::BroadcastInfo
HalfTensor::computeBroadcastInfo(const HalfTensor &m) const {
  if (m.size() > this->size())
    throw exception::not_supported("broadcasting *this is not supported");

  const TensorDim m_dim = m.getDim();

  BroadcastInfo e;
  e.tensor_type = getTensorType();

  uint continuity[4] = {0, 1, 2, 3};
  if (getFormat() == Tformat::NHWC) {
    continuity[1] = 2;
    continuity[2] = 3;
    continuity[3] = 1;
  }

  /// checking if given HalfTensor's can be broadcasted
  for (unsigned int i = 0; i < TensorDim::MAXDIM; ++i) {
    if (dim.getTensorDim(continuity[i]) == m_dim.getTensorDim(continuity[i])) {
      e.strides[i] = m.strides[i];
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
  e.strides[3] = m.strides[3];

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

HalfTensor HalfTensor::rotate_180(HalfTensor in) {
  HalfTensor output(in.getDim());

  output.setZero();
  for (unsigned int i = 0; i < in.batch(); ++i) {
    for (unsigned int j = 0; j < in.channel(); ++j) {
      for (unsigned int k = 0; k < in.height(); ++k) {
        for (unsigned int l = 0; l < in.width(); ++l) {
          output.setValue(
            i, j, k, l,
            in.getValue(i, j, (in.height() - k - 1), (in.width() - l - 1)));
        }
      }
    }
  }

  return output;
}

} /* namespace nntrainer */
