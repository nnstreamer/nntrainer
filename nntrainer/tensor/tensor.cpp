/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	tensor.cpp
 * @date	04 December 2019
 * @brief	This is Tensor class for calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
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
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <stdio.h>

#include <blas_interface.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tensor.h>
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

#define CREATE_IF_EMPTY_DIMS(tensor, ...) \
  do {                                    \
    if (tensor.empty()) {                 \
      tensor = Tensor(__VA_ARGS__);       \
    }                                     \
  } while (0);
namespace nntrainer {

/**
 * @struct External Loop Info for broadcasted info
 * @brief External Loop Info for broadcasted iteration. Please refer to
 * DISABLED_private_external_loop_n in unittest_nntrainer_tensor.
 * @note This should better be implemented in iterator fashion before used
 * extensively.
 */
struct Tensor::BroadcastInfo {

  /**
   * @brief Construct a new External Loop Info object
   *
   */
  BroadcastInfo() :
    buffer_size(0),
    buffer_axis(-1),
    strides{0, 0, 0, 0},
    fm(Tformat::NCHW) {}

  unsigned int buffer_size; /**< virtual size of the buffer */
  int buffer_axis;          /**< the smallest axis that should be looped.
                                 -1 means no loop needed*/
  std::array<unsigned int, TensorDim::MAXDIM>
    strides; /**< modified strides for the loop */
  Tformat fm;
};

static auto rng = [] {
  std::mt19937 rng;
  rng.seed(getSeed());
  return rng;
}();

Tensor::Tensor(const TensorDim &d, bool alloc_now, Tensor::Initializer init,
               std::string name_) :
  Tensor(name_, d.getFormat()) {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();
    initializer = init;

    if (alloc_now)
      allocate();
  }
}

Tensor::Tensor(const TensorDim &d, const float *buf) : Tensor(d, true) {
  if (d.getDataLen() != 0) {
    if (buf != nullptr)
      copy(buf);
  }
}

/**
 * @class SrcSharedTensor
 * @brief Source of the shared tensor
 */
class SrcSharedTensor {
public:
  /**
   * @brief   Constructor for the class
   */
  SrcSharedTensor() : src(nullptr), off(0) {}

  SrcSharedTensor(const Tensor *tensor, size_t offset) :
    src(tensor),
    off(offset) {}

  /**
   * @brief   Get the allocated src tensor
   */
  const Tensor *tensor() const {
    if (!src)
      throw std::runtime_error("Accessing empty src tensor");

    return src;
  }

  /**
   * @brief   Get the offset from the source tensor
   */
  size_t offset() const { return off; }

private:
  const Tensor *src; /**< Tensor of the source */
  size_t off;        /**< offset from the source data ptr */
};

void Tensor::allocate() {
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
    auto mem_data = new MemoryData<float>(new float[dim.getDataLen()]());
    data = std::shared_ptr<MemoryData<float>>(mem_data, [](auto *mem_data) {
      delete[] mem_data->getAddr();
      delete mem_data;
    });
    offset = 0;
    initialize();
  }
}

Tensor Tensor::Map(float *buf, unsigned int bytes, const TensorDim &d,
                   size_t offset) {
  if (d.getDataLen() == 0 || buf == nullptr) {
    throw std::invalid_argument(
      "[Tensor::Map] empty tensor dim is not allowed");
  }

  if (d.getDataLen() * sizeof(float) + offset > bytes) {
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");
  }

  Tensor tmp;
  tmp.dim = d;
  tmp.strides = d.computeStrides();
  /// Tensor does not own the memory
  tmp.data = std::shared_ptr<MemoryData<float>>(
    new MemoryData<float>(buf), std::default_delete<MemoryData<float>>());
  tmp.offset = offset;

  return tmp;
}

bool Tensor::operator==(const Tensor &rhs) const {
  if (this->dim != rhs.dim)
    return false;

  size_t len = size();

  if (len != rhs.size())
    return false;

  const float *data = getData();
  const float *rdata = rhs.getData();

  if (contiguous != rhs.contiguous)
    return false;

  if (strides != rhs.strides)
    return false;

  for (size_t i = 0; i < len; ++i) {
    /** not checking sign change is intentional to avoid float calculation
     * errors around 0 */
    if ((std::isnan(data[i]) && !std::isnan(rdata[i])) ||
        (!std::isnan(data[i]) && std::isnan(rdata[i])) ||
        std::fabs(data[i] - rdata[i]) > epsilon)
      return false;
  }

  return true;
}

template <typename T> void Tensor::setDist(T dist) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " Tensor is not contiguous, cannot set distribution";

  float *data = getData();
  unsigned int len = size();
  for (unsigned int i = 0; i < len; ++i) {
    data[i] = dist(rng);
  }
}

void Tensor::setRandNormal(float mean, float std) {
  setDist<std::normal_distribution<float>>(
    std::normal_distribution<float>(mean, std));
}

void Tensor::setRandUniform(float min, float max) {
  setDist<std::uniform_real_distribution<float>>(
    std::uniform_real_distribution<float>(min, max));
}

void Tensor::setRandBernoulli(float probability) {
  setDist<std::bernoulli_distribution>(
    std::bernoulli_distribution(probability));
}

void Tensor::initialize() {
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
  case Tensor::Initializer::ZEROS:
    setZero();
    break;
  case Tensor::Initializer::ONES:
    setValue(1.0f);
    break;
  case Tensor::Initializer::LECUN_NORMAL:
    setRandNormal(0.0f, sqrtFloat(1.0f / fan_in));
    break;
  case Tensor::Initializer::XAVIER_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in + fan_out)));
    break;
  case Tensor::Initializer::HE_NORMAL:
    setRandNormal(0.0f, sqrtFloat(2.0f / (fan_in)));
    break;
  case Tensor::Initializer::LECUN_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(1.0f / fan_in), sqrtFloat(1.0f / fan_in));
    break;
  case Tensor::Initializer::XAVIER_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in + fan_out)),
                   sqrtFloat(6.0 / (fan_in + fan_out)));
    break;
  case Tensor::Initializer::HE_UNIFORM:
    setRandUniform(-1.0f * sqrtFloat(6.0f / (fan_in)),
                   sqrtFloat(6.0 / (fan_in)));
    break;
  default:
    break;
  }

  putData();
}

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
  Tformat fm) {

  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize Tensor from empty vector");
  }

  // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2] ==
  // height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
  // dim[1] == height, dim[2] == width, dim[3] == channel
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

  dim.setFormat(fm);

  strides = dim.computeStrides();
  auto mem_data = new MemoryData<float>(new float[dim.getDataLen()]);
  data = std::shared_ptr<MemoryData<float>>(
    mem_data, [](auto *mem_data) { delete[] mem_data->getAddr(); });
  offset = 0;
  contiguous = true;
  initializer = Initializer::NONE;

  // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2] ==
  // height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
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
  Tensor t;
  return this->multiply_strided(m, t, beta);
}

Tensor &Tensor::multiply_strided(Tensor const &m, Tensor &output,
                                 const float beta) const {
  /** TODO: throw than create new dimenions */
  CREATE_IF_EMPTY_DIMS(output, dim);

  if (size() != m.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided multiplication does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

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
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            float *out_data = output.getAddress(b, c, h, 0);
            const float *m_data = m.getAddress(b, c, h, 0);
            const float *in_data = getAddress(b, c, h, 0);
            std::transform(in_data, in_data + width(), m_data, out_data,
                           std::multiplies<float>());
          }
        }
      }
    }
  } else {
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
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            float *out_data = output.getAddress(b, 0, h, w);
            const float *m_data = m.getAddress(b, 0, h, w);
            const float *in_data = getAddress(b, 0, h, w);
            std::transform(in_data, in_data + channel(), m_data, out_data,
                           std::multiplies<float>());
          }
        }
      }
    }
  }

  return output;
}

int Tensor::add_i_strided(Tensor const &m, const float beta) {
  try {
    this->add_strided(m, *this, beta);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::add_strided(Tensor const &m, const float beta) const {
  Tensor t;
  return this->add_strided(m, t, beta);
}

Tensor &Tensor::add_strided(Tensor const &m, Tensor &output,
                            const float beta) const {
  /** TODO: throw than create new dimenions */
  CREATE_IF_EMPTY_DIMS(output, dim);

  if (size() != m.size() || size() != output.size())
    throw std::invalid_argument(
      "Strided addition does not support broadcasting");

  NNTR_THROW_IF(getData() == nullptr, std::invalid_argument)
    << getName() << " is not allocated";
  NNTR_THROW_IF(m.getData() == nullptr, std::invalid_argument)
    << m.getName() << " is not allocated";
  NNTR_THROW_IF(output.getData() == nullptr, std::invalid_argument)
    << output.getName() << " is not allocated";

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
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            float *out_data = output.getAddress(b, c, h, 0);
            const float *m_data = m.getAddress(b, c, h, 0);
            const float *in_data = getAddress(b, c, h, 0);
            std::transform(in_data, in_data + width(), m_data, out_data,
                           std::plus<float>());
          }
        }
      }
    }
  } else {
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
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int h = 0; h < height(); ++h) {
          for (unsigned int w = 0; w < width(); ++w) {
            float *out_data = output.getAddress(b, 0, h, w);
            const float *m_data = m.getAddress(b, 0, h, w);
            const float *in_data = getAddress(b, 0, h, w);
            std::transform(in_data, in_data + channel(), m_data, out_data,
                           std::plus<float>());
          }
        }
      }
    }
  }

  return output;
}

int Tensor::multiply_i(float const &value) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  /// @note this is not depending on multiply_i as there is an optimized
  /// version for multiply_i
  float *data = getData();
  unsigned int len = size();

  sscal(len, value, data, 1);
  return ML_ERROR_NONE;
}

Tensor Tensor::multiply(float const &value) const {
  Tensor t;
  return multiply(value, t);
}

Tensor &Tensor::multiply(float const &value, Tensor &out) const {
  /// @todo add unittest
  auto f = std::bind(std::multiplies<float>(), std::placeholders::_1, value);
  return apply(f, out);
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
  return this->multiply(m, t, beta);
}

Tensor &Tensor::multiply(Tensor const &m, Tensor &output,
                         const float beta) const {
  /**
   * @note this does not work correctly with differently strided inputs.
   * Use multiply_strided alternatively
   */
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    if (e.strides[3] == 1 && output.strides[3] == 1 && strides[3] == 1 &&
        beta == 0.0) {
      std::transform(buf, buf + e.buffer_size, m_buf, out_buf,
                     std::multiplies<float>());
    } else {
      for (unsigned int i = 0; i < e.buffer_size; ++i) {
        *out_buf = *buf * *m_buf + beta * *out_buf;
        buf += strides[3];
        m_buf += e.strides[3];
        out_buf += output.strides[3];
      }
    }
  };

  NNTR_THROW_IF(m.getFormat() != this->getFormat(), std::invalid_argument)
    << "Tensor Format of " << getName() << ":"
    << ((bool)(this->getFormat()) ? "NHWC" : "NCHW") << " is not match. ("
    << ((bool)(m.getFormat()) ? "NHWC" : "NCHW") << ")";

  NNTR_THROW_IF(!contiguous || !m.contiguous || !output.contiguous,
                std::invalid_argument)
    << getName() << " is not contiguous, cannot multiply";

  apply_broadcast(m, f, output);
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
  Tensor t;
  return divide(value, t);
}

Tensor &Tensor::divide(float const &value, Tensor &out) const {
  auto f = std::bind(std::divides<float>(), std::placeholders::_1, value);
  /// @todo add unittest
  if (value == 0.0f) {
    std::stringstream ss;
    ss << "[Tensor] divide by value failed, value: " << value;
    throw std::invalid_argument(ss.str().c_str());
  }
  return apply(f, out);
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
  Tensor t;
  return this->divide(m, t);
}

Tensor &Tensor::divide(Tensor const &m, Tensor &output) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    if (e.strides[3] == 1 && output.strides[3] == 1 && strides[3] == 1) {
      std::transform(buf, buf + e.buffer_size, m_buf, out_buf,
                     std::divides<float>());
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

int Tensor::add_i(float const &value) {
  this->add(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::add(float const &value) const {
  Tensor t;
  return add(value, t);
}

Tensor &Tensor::add(float const &value, Tensor &out) const {
  /// @todo add unittest
  auto f = std::bind(std::plus<float>(), std::placeholders::_1, value);
  return apply(f, out);
}

int Tensor::add_i(Tensor const &m, float const alpha) {
  /// @todo: add axis rather doing add over the last two dimensions always
  /// operator i has optimized version
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    saxpy(e.buffer_size, alpha, m_buf, e.strides[3], out_buf, strides[3]);
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

Tensor Tensor::add(Tensor const &m, float const alpha) const {
  Tensor t;
  return this->add(m, t, alpha);
}

Tensor &Tensor::add(Tensor const &m, Tensor &output, float const alpha) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
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

  NNTR_THROW_IF(!contiguous || !m.contiguous || !output.contiguous,
                std::invalid_argument)
    << getName() << " is not contiguous, cannot add";

  apply_broadcast(m, f, output);

  return output;
}

int Tensor::subtract_i(float const &value) {
  this->subtract(value, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::subtract(float const &value) const {
  Tensor t;
  return subtract(value, t);
}

Tensor &Tensor::subtract(float const &value, Tensor &out) const {
  /// @todo add unittest
  auto f = std::bind(std::minus<float>(), std::placeholders::_1, value);
  return apply(f, out);
}

int Tensor::subtract_i(Tensor const &m) { return add_i(m, -1); }

Tensor Tensor::subtract(Tensor const &m) const { return add(m, -1); }

Tensor &Tensor::subtract(Tensor const &m, Tensor &out) const {
  return add(m, out, -1);
}

int Tensor::pow_i(float exponent) {
  pow(exponent, *this);
  return ML_ERROR_NONE;
}

Tensor Tensor::pow(float exponent) const {
  Tensor t;
  return pow(exponent, t);
}

Tensor &Tensor::pow(float exponent, Tensor &out) const {
  auto f = [exponent](float in) { return powf(in, exponent); };
  return apply(f, out);
}

Tensor Tensor::getBatchSlice(size_t offset, unsigned int size) const {
  TensorDim dim_ = dim;
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->dim.getFeatureLen());
}

void Tensor::createSharedDataTensor(const Tensor &src, Tensor &dest,
                                    size_t offset) {
  /**
   * - If src already has data allocaed, then directly make dest tensor based on
   * the src tensor.
   * - If src.data does not exist (meaning tensor does not memory allocated),
   * and src.src_tensor does not exist (meaning the src tensor does not depened
   * on another tensor), then create a SrcSharedTensor around the src.
   * - If src.src_tensor exists, then use the src.src_tensor to create the
   *  required SrcSharedTensor to avoid recursive dependency.
   *
   * @note src.data and src.src_tensor CAN co-exist. src.src_tensor is stored
   * if the batch size of src is updated and needs reallocation.
   */
  dest.data = nullptr;
  if (src.data) {
    dest.src_tensor = std::make_shared<SrcSharedTensor>(&src, offset);
    dest.allocate();
  } else if (!src.src_tensor)
    dest.src_tensor = std::make_shared<SrcSharedTensor>(&src, offset);
  else
    dest.src_tensor = std::make_shared<SrcSharedTensor>(
      src.src_tensor->tensor(), offset + src.src_tensor->offset());
}

Tensor Tensor::getSharedDataTensor(const TensorDim dim_, size_t offset,
                                   bool reset_stride,
                                   const std::string &name_) const {
  Tensor ret = *this;
  if (dim_.getFormat() != ret.dim.getFormat())
    throw std::invalid_argument("Tensor format does not match");

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

std::vector<Tensor> Tensor::split(unsigned num_size, int axis) {
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

std::vector<Tensor> Tensor::split(std::vector<size_t> sizes, int axis) {
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

  std::vector<Tensor> ret;
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

    ret_t.apply_i([&iter_value, &loc, &end_loc, &reset_dim_arr](float _) {
      return iter_value(loc, end_loc, reset_dim_arr);
    });
  }

  return ret;
}

Tensor Tensor::cat(const std::vector<Tensor> &tensors, int axis) {

  if (axis == -1) {
    axis = 3;
  }

  NNTR_THROW_IF(!(0 <= axis && axis < 4), std::invalid_argument)
    << "cannot split axis of axis: " << axis;

  NNTR_THROW_IF(tensors.empty(), std::invalid_argument)
    << "given tensor vector is empty";

  auto ref_dim = tensors.front().getDim();
  bool is_format_nchw = (ref_dim.getFormat() == Tformat::NCHW) ? true : false;
  ref_dim.setTensorDim(axis, 1);
  NNTR_THROW_IF(!std::all_of(tensors.begin(), tensors.end(),
                             [&ref_dim, axis](const Tensor &t) {
                               auto cur_dim = t.getDim();
                               cur_dim.setTensorDim(axis, 1);
                               return ref_dim == cur_dim;
                             }),
                std::invalid_argument)
    << " all tensor must have the same dimension except for the axis, ref_dim: "
    << ref_dim << " axis : " << axis;

  auto axis_dim = std::accumulate(tensors.begin(), tensors.end(), 0u,
                                  [axis](unsigned cur, const Tensor &t) {
                                    return cur += t.getDim().getTensorDim(axis);
                                  });
  auto iter_value =
    [is_format_nchw](std::array<unsigned, 4> &loc,
                     const std::array<unsigned, 4> &start_loc, Tensor &t,
                     const std::array<unsigned, 4> &ref_dim_arr) -> float & {
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

  auto ret = Tensor(ret_dim);

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

void Tensor::makeSharedDataTensor(const Tensor &src, size_t offset) {
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

void Tensor::apply_broadcast(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const float *, const float *,
                     float *)>
    v_func,
  Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim);

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
    e.fm = getFormat();
    v_func(e, getData(), m.getData(), output.getData());
    return;
  }

  return apply_broadcast_util(m, v_func, output, this->computeBroadcastInfo(m));
}

void Tensor::apply_broadcast_util(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, const float *, const float *,
                     float *)>
    v_func,
  Tensor &output, const BroadcastInfo &e, int cur_axis, size_t offset,
  size_t m_offset) const {

  const float *buf = this->getData();
  const float *m_buf = m.getData();
  float *out_buf = output.getData();

  if (e.buffer_axis == cur_axis) {
    v_func(e, buf + offset, m_buf + m_offset, out_buf + offset);
    return;
  }

  cur_axis++;
  uint continuity[4] = {0, 1, 2, 3};
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

/**
 * This is to sum the Tensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1, 1) dimension.
 */
Tensor Tensor::sum_by_batch() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  Tensor ret(dim.batch(), 1, 1, 1, this->getFormat());
  size_t feat_len = dim.getFeatureLen();
  size_t batch = dim.batch();

  const float *data = getData();
  float *rdata = ret.getData();

  Tensor ones(1, 1, 1, feat_len, this->getFormat());
  ones.setValue(1.0);
  sgemv(CblasRowMajor, CblasNoTrans, batch, feat_len, 1, data, feat_len,
        ones.getData(), 1, 0.0, rdata, 1);

  return ret;
}

/**
 * @brief Calculate sum according to the axis.
 */
Tensor Tensor::sum(unsigned int axis, float alpha) const {
  Tensor ret("", this->getFormat());
  return sum(axis, ret, alpha, 0);
}
Tensor &Tensor::sum(unsigned int axis, Tensor &ret, float alpha,
                    float beta) const {
  const float *data = getData();

  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot sum";

  if (axis >= 4)
    throw std::out_of_range("Error: axis is invalid");

  if (dim.getDim()[axis] == 1 and alpha == 1.0 and !beta) {
    CREATE_IF_EMPTY_DIMS(ret, dim);
    ret.copy(this->getData());
    return ret;
  }

  switch (axis) {
  case 0: {
    CREATE_IF_EMPTY_DIMS(ret, 1, dim[1], dim[2], dim[3], this->getFormat());

    size_t feat_len = dim.getFeatureLen();
    size_t batch = dim.batch();
    Tensor ones(1, 1, 1, batch, this->getFormat());
    ones.setValue(alpha);
    sgemv(CblasRowMajor, CblasTrans, batch, feat_len, 1, data, feat_len,
          ones.getData(), 1, beta, ret.getData(), 1);
  } break;
  case 1: {
    CREATE_IF_EMPTY_DIMS(ret, dim[0], 1, dim[2], dim[3], this->getFormat());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int m = ret.dim.getDataLen();
      unsigned int n = dim[1];
      Tensor ones(1, 1, 1, n, this->getFormat());
      ones.setValue(alpha);
      sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, data, n, ones.getData(), 1,
            beta, ret.getData(), 1);
    } else {
      unsigned int feat_len = dim[2] * dim[3];
      unsigned int t_axis = dim[1];
      Tensor ones(1, 1, 1, t_axis);
      ones.setValue(alpha);
      float *rdata = ret.getData();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv(CblasRowMajor, CblasTrans, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData(), 1, beta,
              &rdata[k * feat_len], 1);
      }
    }
  } break;
  case 2: {
    CREATE_IF_EMPTY_DIMS(ret, dim[0], dim[1], 1, dim[3], this->getFormat());

    if (this->getFormat() == Tformat::NHWC) {
      unsigned int feat_len = dim[1] * dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, this->getFormat());
      ones.setValue(alpha);
      float *rdata = ret.getData();
      for (unsigned int k = 0; k < dim[0]; ++k) {
        sgemv(CblasRowMajor, CblasTrans, t_axis, feat_len, 1,
              &data[k * dim.getFeatureLen()], feat_len, ones.getData(), 1, beta,
              &rdata[k * feat_len], 1);
      }
    } else {
      unsigned int t_3 = dim[3];
      unsigned int t_axis = dim[2];
      Tensor ones(1, 1, 1, t_axis, this->getFormat());
      ones.setValue(alpha);
      float *rdata = ret.getData();
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
    CREATE_IF_EMPTY_DIMS(ret, dim[0], dim[1], dim[2], 1, this->getFormat());
    if (this->getFormat() == Tformat::NHWC) {
      unsigned int t_3 = dim[1];
      unsigned int t_axis = dim[3];
      Tensor ones(1, 1, 1, t_axis, this->getFormat());
      ones.setValue(alpha);
      float *rdata = ret.getData();
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
      Tensor ones(1, 1, 1, n);
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

Tensor Tensor::sum(const std::vector<unsigned int> &axes, float alpha) const {
  Tensor ret("", this->getFormat());
  return sum(axes, ret, alpha);
}

void Tensor::mergeAxis(unsigned int axis1, unsigned int axis2) {
  std::vector<unsigned int> continuous_order = {0, 3, 1, 2};
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot merge axis";

  if (axis2 != axis1 + 1)
    if (!checkContinuous(axis1, axis2))
      throw std::invalid_argument("axis2 must be axis1 + 1 for merging.");

  dim.setTensorDim(axis2, dim.getTensorDim(axis1) * dim.getTensorDim(axis2));
  dim.setTensorDim(axis1, 1);
}

Tensor &Tensor::sum(const std::vector<unsigned int> &axes, Tensor &output,
                    float alpha) const {
  if (axes.empty())
    throw std::invalid_argument("empty axes given");

  if (axes.size() == 1) {
    this->sum(axes[0], output, alpha);
  } else {
    /** club axes together */
    Tensor new_reshaped = *this;
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

Tensor Tensor::dot(Tensor const &m, bool trans, bool trans_m) const {
  Tensor output("", this->getFormat());
  dot(m, output, trans, trans_m);

  return output;
}
/**
 * @brief compute the derivative of this in the current tensor
 * @todo will have to see if beta effects this computation
 */
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

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor
 * in case of trans is false.
 */
Tensor &Tensor::dot(Tensor const &m, Tensor &result, bool trans, bool trans_m,
                    float beta) const {
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
      CREATE_IF_EMPTY_DIMS(result, batch(), N, height(), width(),
                           Tformat::NHWC); //  NHWC Result Tensor
    } else {
      CREATE_IF_EMPTY_DIMS(result, batch(), channel(), height(), N);
    }

    // We are not set zero the result because of performnace reason.
    // However, result is not initialized properly. There might include
    // garbage like nan. When we have to use this value as in C = alpha*A*B +
    // beta*C, then have to check gargabe data of C is not effect or not.

  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, batch(), N, height(), width(),
                           Tformat::NHWC);
    } else {
      CREATE_IF_EMPTY_DIMS(result, batch(), channel(), height(), N);
      CREATE_IF_EMPTY_DIMS(result, batch(), channel(), height(), N);
      CREATE_IF_EMPTY_DIMS(result, batch(), channel(), height(), N);
    }
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;

    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, Tformat::NHWC);
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N);
    }
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    if (getFormat() == Tformat::NHWC) {
      CREATE_IF_EMPTY_DIMS(result, 1, N, M, 1, Tformat::NHWC);
    } else {
      CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N);
    }
  }
  lda = dim2;
  ldb = mdim2;
  ldc = (getFormat() == Tformat::NHWC) ? result.channel() : result.width();

  const float *data = getData();
  const float *mdata = m.getData();
  float *rdata = result.getData();
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
    *rdata = sdot(K, data, 1, mdata, 1) + beta * (*rdata);
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
  /// case others: use gemm
  else {
    sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, data, lda, mdata, ldb,
          beta, rdata, ldc);
  }

  return result;
}

Tensor &Tensor::transpose(const std::string &direction, Tensor &out) const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous. Cannot transpose.";

  if (out.getData() == getData()) {
    Tensor tmp = clone();
    return tmp.transpose(direction, out);
  }

  unsigned int SL, SI, SJ, SK;
  const float *inptr;
  float *outptr;

  out.reshape(dim.transpose(direction));

  int indexI = direction[0] - '0';
  int indexJ = direction[2] - '0';

  SL = dim.batch(), SI = dim.channel(), SJ = dim.height(), SK = dim.width();

  bool is_format_nchw = (getFormat() == Tformat::NCHW) ? true : false;

  inptr = getData();
  outptr = out.getData();

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

Tensor Tensor::transpose(const std::string &direction) const {
  Tensor result(dim);
  transpose(direction, result);
  return result;
}

Tensor Tensor::dropout_mask(float dropout) const {
  Tensor result(dim);
  result.dropout_mask(dropout);
  return result;
}

void Tensor::dropout_mask(float dropout) {
  setRandUniform(0.0, 1.0);
  float scale = 1.0 / (1 - dropout);
  float *data_ = getData();

  for (unsigned int i = 0; i < size(); ++i) {
    if (data_[i] >= dropout)
      data_[i] = scale;
    else
      data_[i] = 0.0;
  }
}

void Tensor::filter_mask(const Tensor &mask_len, bool reverse) {
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
    float *addr = getAddress(b, 0, 0, 0);
    const uint *mask_len_val = mask_len.getAddress<uint>(b, 0, 0, 0);
    std::fill(addr, addr + (*mask_len_val), en_mask_val);
  }
}

Tensor Tensor::zoneout_mask(float zoneout) {
  Tensor ret(getDim());
  zoneout_mask(ret, zoneout);
  return ret;
}

void Tensor::zoneout_mask(Tensor &opposite, float zoneout) {
  if (dim != opposite.dim) {
    throw std::invalid_argument(
      "[Tensor::zoneout_mask] opposite dimension does not match");
  }

  opposite.setRandBernoulli(zoneout);
  float *data = getData();
  float *opposite_data = opposite.getData();

  for (unsigned int i = 0; i < size(); ++i) {
    if (opposite_data[i] > epsilon) {
      data[i] = 0.0f;
    } else {
      data[i] = 1.0f;
    }
  }
}

int Tensor::apply_i(std::function<float(float)> f) {
  Tensor result = *this;
  apply(f, result);

  return ML_ERROR_NONE;
}

Tensor Tensor::apply(std::function<float(float)> f) const {
  Tensor result;
  return apply(f, result);
}

Tensor &Tensor::apply(std::function<float(float)> f, Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim);

  if (dim != output.dim) {
    /// @todo add unittest
    throw std::invalid_argument(
      "[Tensor::apply] output dimension does not match");
  }

  if (contiguous && output.contiguous) {
    const float *data = getData();
    float *rdata = output.getData();
    std::transform(data, data + size(), rdata, f);
  } else if (strides[3] == 1 && output.strides[3] == 1) {
    /** @todo optimize this with combining these loops where stride is 1 */
    for (unsigned int b = 0; b < batch(); ++b) {
      for (unsigned int c = 0; c < channel(); ++c) {
        for (unsigned int h = 0; h < height(); ++h) {
          float *out_data = output.getAddress(b, c, h, 0);
          const float *in_data = getAddress(b, c, h, 0);
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

Tensor Tensor::apply(std::function<Tensor(Tensor)> f) const { return f(*this); }

Tensor &Tensor::apply(std::function<Tensor &(Tensor, Tensor &)> f,
                      Tensor &output) const {
  return f(*this, output);
}

void Tensor::print(std::ostream &out) const {
  printInstance(out, this);
  const float *data = getData();

  unsigned int len = size();
  out << "data addr: " << data << '\n';
  out << dim;

  if (len > 100000) {
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
                << this->getValue(k, l, i, j) << " ";
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
                << this->getValue(k, l, i, j) << " ";
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

void Tensor::print_(std::ostream &out, uint opt) const {
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
                    << this->getValue(k, l, i, j) << ", ";
              else
                out << std::setw(10) << std::setprecision(10)
                    << this->getValue(k, l, i, j);
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
                    << this->getValue(k, l, i, j) << ", ";
              else
                out << std::setw(10) << std::setprecision(10)
                    << this->getValue(k, l, i, j);
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
      out << getData()[i] << ", ";
    }
  }
  out.copyfmt(init);
}

std::ostream &operator<<(std::ostream &out, Tensor const &m) {
  m.print(out);
  return out;
}

void Tensor::copy(const float *buf) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << "Tensor is not contiguous, cannot copy.";

  if (buf == getData()) {
    return;
  }

  scopy(size(), buf, 1, getData(), 1);
}

void Tensor::copy_with_stride(const Tensor &from) {

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
    Tensor t = Tensor(from.getDim(), true);
    for (unsigned int b = 0; b < t.batch(); ++b) {
      for (unsigned int c = 0; c < t.channel(); ++c) {
        for (unsigned int h = 0; h < t.height(); ++h) {
          for (unsigned int w = 0; w < t.width(); ++w) {
            t.setValue(b, c, h, w, from.getValue(b, c, h, w));
          }
        }
      }
    }
    swap(t, *this);
  }
}

void Tensor::copy(const Tensor &from) {
  // todo: enable copy to non-contiguous tensor
  if (!contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (from.size() != 0 && size() == from.size()) {
    reshape(from.getDim());
    copy(from.getData());
  } else {
    Tensor t = Tensor(from.getDim(), from.getData());
    swap(t, *this);
  }
}

void Tensor::copyData(const Tensor &from) {
  // todo: enable copy to non-contiguous tensor
  if (!contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (size() != from.size())
    throw std::invalid_argument("Size of tensor to copy must match");
  copy(from.getData());
}

Tensor Tensor::clone() const {
  Tensor t;
  t.copy(*this);
  t.name = name;
  return t;
}

void Tensor::reshape(const TensorDim &d) {

  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot reshape.";

  NNTR_THROW_IF(d.getDataLen() != dim.getDataLen(), std::invalid_argument)
    << "[Tensor]: reshape cannot change the buffer size, trying reshaping "
       "\nfrom "
    << getDim() << " to " << d;

  dim = d;
  strides = d.computeStrides();
}

void Tensor::fill(const Tensor &from, bool alloc) {
  if (alloc && this->empty()) {
    this->copy(from);
    return;
  }

  if (!from.contiguous || !contiguous) {
    /// @todo enable this if needed
    throw nntrainer::exception::not_supported(
      "[Tensor::fill] non-contiguous tensors are not supported");
  }

  if (dim != from.getDim()) {
    throw std::invalid_argument("[Tensor::fill] dimension must be the same");
  }

  if (strides != from.getStrides()) {
    /// @todo length does not represent buffer size, there should be way to get
    /// the buffer size
    throw std::invalid_argument("[Tensor::fill] buffer size must be the same");
  }

  this->copy(from.getData());
}

void Tensor::save(std::ostream &file) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot save.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "save size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedWrite(file, (char *)getData(), sz, "[Tensor::save] operation failed");
  putData();
}

void Tensor::read(std::ifstream &file) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot read.";

  std::streamsize sz = static_cast<std::streamsize>(bytes());

  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << "read size: " << bytes()
    << " is too big. It cannot be represented by std::streamsize";

  checkedRead(file, (char *)getData(), sz, "[Tensor::read] operation failed");
  putData();
}

/**
 * @brief Calculate average value according to the axis.
 */
Tensor Tensor::average(unsigned int axis) const {
  Tensor t("", this->getFormat());
  return average(axis, t);
}

/**
 * @brief Calculate average value according to the axis.
 */
Tensor &Tensor::average(unsigned int axis, Tensor &output) const {
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

Tensor Tensor::average(const std::vector<unsigned int> &axes) const {
  Tensor t("", this->getFormat());
  return average(axes, t);
}

Tensor &Tensor::average(const std::vector<unsigned int> &axes,
                        Tensor &output) const {
  if (axes.empty())
    return this->average(output);

  TensorDim ret_shape(getFormat());
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
Tensor Tensor::average() const {
  Tensor result = *this;
  unsigned int axis = 0;
  if (this->getFormat() == Tformat::NHWC) {
    result.reshape({1, dim.getDataLen(), 1, 1, this->getFormat()});
    axis = 1;
  } else {
    result.reshape({1, 1, 1, dim.getDataLen(), this->getFormat()});
    axis = 3;
  }
  return result.average(axis);
}

/**
 * @brief Calculate average value according to the axis.
 */
Tensor &Tensor::average(Tensor &output) const {
  Tensor result = *this;
  result.reshape({1, 1, 1, dim.getDataLen()});
  return result.average(3, output);
}

void Tensor::setValue(float val) {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot set value.";

  float *data = getData();
  std::fill(data, data + size(), val);
}

void Tensor::setZero() {
  if (contiguous)
    sscal(size(), 0, getData(), 1);
  else
    apply_i([](float val) -> float { return 0; });
}

std::vector<unsigned int> Tensor::argmax() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot get argmax.";

  const float *data = getData();
  std::vector<unsigned int> result;
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

float Tensor::l2norm() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot get l2norm.";

  unsigned int len = size();
  const float *data = getData();

  return snrm2(len, data, 1);
}

float Tensor::max_abs() const {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot get max_abs.";

  unsigned int len = size();
  const float *data = getData();

  unsigned int idx = isamax(len, data, 1);
  return *(data + idx);
}

Tensor &Tensor::normalization(Tensor &output) const {
  if (output.empty())
    output = Tensor(dim);

  output.copy(*this);
  output.normalization_i();

  return output;
}

void Tensor::normalization_i() {
  NNTR_THROW_IF(!contiguous, std::invalid_argument)
    << getName() << " is not contiguous, cannot do normalization.";

  const float *data = getData();

  auto bounds = std::minmax_element(data, data + size());
  const float min = *bounds.first;
  const float max = *bounds.second;

  if (max == min) {
    Tensor tmp = *this;
    this->subtract_i(tmp);
  } else {
    this->subtract_i(min);
    this->divide_i(max - min);
  }
}

LazyTensor Tensor::chain() const { return LazyTensor(*this); }

Tensor &Tensor::standardization(Tensor &output) const {
  if (output.empty())
    output = Tensor(dim);

  output.copy(*this);
  output.standardization_i();

  return output;
}

void Tensor::standardization_i() {
  Tensor mean_by_batch = this->sum_by_batch();
  mean_by_batch.divide_i(dim.getFeatureLen());

  this->subtract_i(mean_by_batch);

  Tensor std_dev_by_batch(dim.batch(), 1, 1, 1);
  std_dev_by_batch.setZero();
  float *std_dev = std_dev_by_batch.getData();

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    Tensor sub_this = this->getBatchSlice(k, 1);
    std_dev[k] = sub_this.l2norm();
  }

  std_dev_by_batch.divide_i(dim.getFeatureLen());
  this->divide_i(std_dev_by_batch);
}

Tensor::BroadcastInfo Tensor::computeBroadcastInfo(const Tensor &m) const {
  if (m.size() > this->size())
    throw exception::not_supported("broadcasting *this is not supported");

  const TensorDim m_dim = m.getDim();

  BroadcastInfo e;
  e.fm = getFormat();

  uint continuity[4] = {0, 1, 2, 3};
  if (getFormat() == Tformat::NHWC) {
    continuity[1] = 2;
    continuity[2] = 3;
    continuity[3] = 1;
  }

  /// checking if given Tensor's can be broadcasted
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

} /* namespace nntrainer */
