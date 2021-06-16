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

#include <assert.h>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <random>
#include <regex>
#include <sstream>
#include <stdio.h>

#include <blas_interface.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
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

#define CREATE_IF_EMPTY_DIMS(tensor, ...) \
  do {                                    \
    if (tensor.uninitialized())           \
      tensor = Tensor(__VA_ARGS__);       \
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
  BroadcastInfo() : buffer_size(0), buffer_axis(-1), strides{0, 0, 0, 0} {}

  unsigned int buffer_size; /**< virtual size of the buffer */
  int buffer_axis;          /**< the smallest axis that should be looped.
                                 -1 means no loop needed*/
  std::array<unsigned int, MAXDIM>
    strides; /**< modified strides for the loop */
};

static auto rng = [] {
  std::mt19937 rng;
  rng.seed(getSeed());
  return rng;
}();

Tensor::Tensor(const TensorDim &d, const float *buf) : Tensor() {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();

    allocate();

    if (buf != nullptr)
      copy(buf);
  }
}

Tensor::Tensor(const TensorDim &d, bool alloc_now) : Tensor() {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();

    if (alloc_now)
      allocate();
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

  SrcSharedTensor(const Tensor *tensor, unsigned int offset) :
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
  unsigned int offset() const { return off; }

private:
  const Tensor *src; /**< Tensor of the source */
  unsigned int off;  /**< offset from the source data ptr */
};

void Tensor::allocate() {
  if (data)
    /// already allocated
    return;

  if (src_tensor) {
    /// allocate data based on the source tensor
    data = std::shared_ptr<float>(src_tensor->tensor()->data,
                                  src_tensor->tensor()->data.get() +
                                    src_tensor->offset());
  } else {
    /// allocate new memory for the tensor data
    data = std::shared_ptr<float>(new float[dim.getDataLen()],
                                  std::default_delete<float[]>());
  }
}

Tensor Tensor::Map(float *buf, unsigned int size, const TensorDim &d,
                   int offset) {
  if (d.getDataLen() == 0 || buf == nullptr) {
    throw std::invalid_argument(
      "[Tensor::Map] empty tensor dim is not allowed");
  }

  if (d.getDataLen() + offset > size) {
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");
  }

  Tensor tmp;
  tmp.dim = d;
  tmp.strides = d.computeStrides();
  /// Tensor does not own the memory
  tmp.data = std::shared_ptr<float>(buf + offset, [](void *) {});

  return tmp;
}

Tensor Tensor::Map(std::shared_ptr<float> buf, unsigned int size,
                   const TensorDim &d, int offset) {
  if (d.getDataLen() == 0 || buf == nullptr) {
    throw std::invalid_argument(
      "[Tensor::Map] empty tensor dim is not allowed");
  }

  if (d.getDataLen() + offset > size) {
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");
  }

  Tensor tmp;
  tmp.dim = d;
  tmp.data = std::shared_ptr<float>(buf, buf.get() + offset);

  return tmp;
}

bool Tensor::operator==(const Tensor &rhs) const {
  if (this->dim != rhs.dim)
    return false;

  size_t len = length();

  if (len != rhs.length())
    return false;

  const float *data = getData();
  const float *rdata = rhs.getData();

  for (size_t i = 0; i < len; ++i) {
    if (std::isnan(data[i]) || std::isnan(rdata[i]) ||
        std::fabs(data[i] - rdata[i]) > epsilon)
      return false;
  }

  return true;
}

template <typename T> void Tensor::setDist(T dist) {
  float *data = getData();
  unsigned int len = length();
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

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d) {

  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize Tensor from empty vector");
  }

  dim.batch(d.size());
  dim.channel(d[0].size());
  dim.height(d[0][0].size());
  dim.width(d[0][0][0].size());
  strides = dim.computeStrides();
  data = std::shared_ptr<float>(new float[dim.getDataLen()],
                                std::default_delete<float[]>());
  is_contiguous = true;

  for (unsigned int i = 0; i < dim.batch(); ++i)
    for (unsigned int j = 0; j < dim.channel(); ++j)
      for (unsigned int k = 0; k < dim.height(); ++k)
        for (unsigned int l = 0; l < dim.width(); ++l)
          this->setValue(i, j, k, l, d[i][j][k][l]);
}

int Tensor::multiply_i(float const &value) {
  /// @note this is not depending on multiply_i as there is an optimized
  /// version for multiply_i
  float *data = getData();
  unsigned int len = length();

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

int Tensor::multiply_i(Tensor const &m) {
  try {
    this->multiply(m, *this);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::multiply(Tensor const &m) const {
  Tensor t;
  return this->multiply(m, t);
}

Tensor &Tensor::multiply(Tensor const &m, Tensor &output) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    for (unsigned int i = 0; i < e.buffer_size; ++i) {
      *out_buf = *buf * *m_buf;
      buf += strides[3];
      m_buf += e.strides[3];
      out_buf += strides[3];
    }
  };

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
    for (unsigned int i = 0; i < e.buffer_size; ++i) {
      *out_buf = *buf / *m_buf;
      buf += strides[3];
      m_buf += e.strides[3];
      out_buf += strides[3];
    }
  };

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

Tensor &Tensor::add(Tensor const &m, Tensor &out, float const alpha) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *out_buf) {
    for (unsigned int i = 0; i < e.buffer_size; ++i) {
      *out_buf = *buf + *m_buf * alpha;
      buf += strides[3];
      m_buf += e.strides[3];
      out_buf += strides[3];
    }
  };

  apply_broadcast(m, f, out);

  return out;
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

Tensor Tensor::getBatchSlice(unsigned int offset, unsigned int size) const {
  TensorDim dim_ = dim;
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->dim.getFeatureLen());
}

void Tensor::createSharedDataTensor(const Tensor &src, Tensor &dest,
                                    unsigned int offset) {
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
  if (src.data)
    dest.data = std::shared_ptr<float>(src.data, src.data.get() + offset);
  else if (!src.src_tensor)
    dest.src_tensor = std::make_shared<SrcSharedTensor>(&src, offset);
  else
    dest.src_tensor = std::make_shared<SrcSharedTensor>(
      src.src_tensor->tensor(), offset + src.src_tensor->offset());
}

Tensor Tensor::getSharedDataTensor(const TensorDim dim_, unsigned int offset,
                                   bool reset_stride) const {
  Tensor ret = *this;

  if (dim_.getDataLen() + offset > dim.getDataLen())
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");

  ret.dim = dim_;
  if (reset_stride)
    ret.strides = ret.dim.computeStrides();

  /**
   * In this case, its the caller's responsibility to ensure that allocate() is
   * called for the output tensor before operating on the output tensor.
   */
  createSharedDataTensor(*this, ret, offset);

  return ret;
}

void Tensor::makeSharedDataTensor(const Tensor &src, unsigned int offset) {
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

  /// shortcut to cover when dimension matches
  /// note that buffer_size, the last stride is only used in v_func but it
  /// might be changed
  if (dim == m.dim) {
    BroadcastInfo e;
    e.buffer_size = length();
    e.strides[3] = 1;
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
  Tensor &output, const BroadcastInfo &e, int cur_axis, unsigned int offset,
  unsigned int m_offset) const {

  const float *buf = this->getData();
  const float *m_buf = m.getData();
  float *out_buf = output.getData();

  if (e.buffer_axis == cur_axis) {
    v_func(e, buf + offset, m_buf + m_offset, out_buf + offset);
    return;
  }

  cur_axis++;
  for (unsigned int i = 0; i < dim.getTensorDim(cur_axis); ++i) {
    unsigned int next_offset = offset + i * strides[cur_axis];
    unsigned int next_m_offset = m_offset + i * e.strides[cur_axis];
    apply_broadcast_util(m, v_func, output, e, cur_axis, next_offset,
                         next_m_offset);
  }
}

/**
 * This is to sum the Tensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1, 1) dimension.
 */
Tensor Tensor::sum_by_batch() {
  Tensor ret(dim.batch(), 1, 1, 1);
  unsigned int feat_len = dim.getFeatureLen();
  unsigned int batch = dim.batch();

  const float *data = getData();
  float *rdata = ret.getData();

  Tensor ones(1, 1, 1, feat_len);
  ones.setValue(1.0);
  sgemv(CblasRowMajor, CblasNoTrans, batch, feat_len, 1, data, feat_len,
        ones.getData(), 1, 0.0, rdata, 1);

  return ret;
}

/**
 * @brief Calculate sum according to the axis.
 */
Tensor Tensor::sum(unsigned int axis, float alpha) const {
  Tensor ret;
  return sum(axis, ret, alpha);
}
Tensor &Tensor::sum(unsigned int axis, Tensor &ret, float alpha) const {
  const float *data = getData();

  if (axis >= 4)
    throw std::out_of_range("Error: axis is invalid");

  if (dim.getDim()[axis] == 1 and alpha == 1.0) {
    CREATE_IF_EMPTY_DIMS(ret, dim);
    ret.copy(*this);
    return ret;
  }

  switch (axis) {
  case 0: {
    CREATE_IF_EMPTY_DIMS(ret, 1, dim.channel(), dim.height(), dim.width());
    unsigned int feat_len = dim.getFeatureLen();
    unsigned int batch = dim.batch();
    Tensor ones(1, 1, 1, batch);
    ones.setValue(alpha);
    sgemv(CblasRowMajor, CblasTrans, batch, feat_len, 1, data, feat_len,
          ones.getData(), 1, 0.0, ret.getData(), 1);
  } break;
  case 1: {
    CREATE_IF_EMPTY_DIMS(ret, dim.batch(), 1, dim.height(), dim.width());
    unsigned int feat_len = dim.height() * dim.width();
    unsigned int channel = dim.channel();
    Tensor ones(1, 1, 1, channel);
    ones.setValue(alpha);
    float *rdata = ret.getData();
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      sgemv(CblasRowMajor, CblasTrans, channel, feat_len, 1,
            &data[k * dim.getFeatureLen()], feat_len, ones.getData(), 1, 0.0,
            &rdata[k * feat_len], 1);
    }
  } break;
  case 2: {
    CREATE_IF_EMPTY_DIMS(ret, dim.batch(), dim.channel(), 1, dim.width());
    unsigned int width = dim.width();
    unsigned int height = dim.height();
    Tensor ones(1, 1, 1, height);
    ones.setValue(alpha);
    float *rdata = ret.getData();
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      for (unsigned int c = 0; c < dim.channel(); ++c) {
        unsigned int idx =
          k * dim.getFeatureLen() + c * dim.width() * dim.height();
        unsigned int ridx = k * ret.dim.getFeatureLen() + c * dim.width();
        sgemv(CblasRowMajor, CblasTrans, height, width, 1, &data[idx], width,
              ones.getData(), 1, 0.0, &rdata[ridx], 1);
      }
    }
  } break;
  case 3: {
    CREATE_IF_EMPTY_DIMS(ret, dim.batch(), dim.channel(), dim.height(), 1);
    unsigned int m = ret.dim.getDataLen();
    unsigned int n = dim.width();
    Tensor ones(1, 1, 1, n);
    ones.setValue(alpha);
    sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, data, n, ones.getData(), 1, 0.0,
          ret.getData(), 1);
  } break;
  default:
    throw std::out_of_range("Error: Dimension cannot exceed 3");
  }
  return ret;
}

Tensor Tensor::sum(const std::vector<unsigned int> &axes, float alpha) const {
  Tensor ret;
  return sum(axes, ret, alpha);
}

Tensor &Tensor::sum(const std::vector<unsigned int> &axes, Tensor &output,
                    float alpha) const {
  if (axes.empty())
    throw std::invalid_argument("empty axes given");

  if (axes.size() == 1) {
    this->sum(axes[0], output, alpha);
  } else {
    Tensor ret = this->sum(axes[0], alpha);

    for (unsigned int i = 1; i < axes.size() - 1; ++i)
      ret = ret.sum(axes[i]);

    ret.sum(axes.back(), output);
  }

  return output;
}

Tensor Tensor::dot(Tensor const &m, bool trans, bool trans_m) const {
  Tensor output;
  dot(m, output, trans, trans_m);

  return output;
}

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor
 * in case of trans is false.
 */
Tensor &Tensor::dot(Tensor const &m, Tensor &result, bool trans, bool trans_m,
                    float beta) const {
  if (m.dim.rank() > 2) {

    throw exception::not_supported("Error: support only for rank of dot "
                                   "matrix <= 2");
  }

  // Comment out with intension to support the calculation wrt. batch and height
  // direction of this tensor. It is OK as long as m is 2D
  //
  if (trans && dim.rank() > 2) {
    ml_logw("Warning: support only for rank of dot matrix <= 2 with trans");
  }

  unsigned int dim1 = batch() * channel() * height();
  unsigned int dim2 = width();
  unsigned int mdim1 = m.batch() * m.channel() * m.height();
  unsigned int mdim2 = m.width();

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    CREATE_IF_EMPTY_DIMS(result, batch(), channel(), height(), N);

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
    CREATE_IF_EMPTY_DIMS(result, batch(), channel(), height(), N);
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N);
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    CREATE_IF_EMPTY_DIMS(result, 1, 1, M, N);
  }
  lda = dim2;
  ldb = mdim2;
  ldc = result.width();

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

  inptr = getData();
  outptr = out.getData();

  switch (indexI) {
  case 0:
    if (indexJ == 1) {
      transposeloop(l, i, j, k, SL, SI, SJ, SK);
    } else {
      transposeloop(l, i, k, j, SL, SI, SK, SJ);
    }
    break;
  case 1:
    if (indexJ == 0) {
      transposeloop(l, j, i, k, SL, SJ, SI, SK);
    } else {
      transposeloop(l, j, k, i, SL, SJ, SK, SI);
    }
    break;
  case 2:
    if (indexJ == 0) {
      transposeloop(l, k, i, j, SL, SK, SI, SJ);
    } else {
      transposeloop(l, k, j, i, SL, SK, SJ, SI);
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

int Tensor::apply_i(std::function<float(float)> f) {
  float *data = getData();

  std::transform(data, data + length(), data, f);

  return ML_ERROR_NONE;
}

Tensor Tensor::apply(std::function<float(float)> f) const {
  Tensor result;
  return apply(f, result);
}

Tensor &Tensor::apply(std::function<float(float)> f, Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim);

  const float *data = getData();
  float *rdata = output.getData();

  if (dim != output.dim) {
    /// @todo add unittest
    throw std::invalid_argument(
      "[Tensor::apply] output dimension does not match");
  }

  std::transform(data, data + length(), rdata, f);

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

  unsigned int len = length();
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
  for (unsigned int k = 0; k < dim.batch(); k++) {
    for (unsigned int l = 0; l < dim.channel(); l++) {
      for (unsigned int i = 0; i < dim.height(); i++) {
        for (unsigned int j = 0; j < dim.width(); j++) {
          out << std::setw(10) << std::setprecision(10)
              << this->getValue(k, l, i, j) << " ";
        }
        out << std::endl;
      }
      out << std::endl;
    }
    out << "-------" << std::endl;
  }
  out.copyfmt(init);
}

std::ostream &operator<<(std::ostream &out, Tensor const &m) {
  m.print(out);
  return out;
}

float *Tensor::getAddress(unsigned int i) {
  if (i > this->dim.getDataLen()) {
    ml_loge("Error: Index out of bounds");
    return nullptr;
  }

  return &getData()[i];
}

const float *Tensor::getAddress(unsigned int i) const {
  if (i > this->dim.getDataLen()) {
    ml_loge("Error: Index out of bounds");
    return nullptr;
  }

  return &getData()[i];
}

void Tensor::copy(const float *buf) noexcept {
  if (buf == getData()) {
    return;
  }

  scopy(length(), buf, 1, getData(), 1);
}

void Tensor::copy_with_stride(const Tensor &from) {
  if (from.length() != 0 && length() == from.length()) {
    reshape(from.getDim());
    for (unsigned int b = 0; b < from.batch(); ++b) {
      unsigned int from_b = b * from.strides[0];
      unsigned int t_b = b * from.channel() * from.height() * from.width();
      for (unsigned int c = 0; c < from.channel(); ++c) {
        unsigned int from_c = c * from.strides[1];
        unsigned int t_c = c * from.height() * from.width();
        for (unsigned int h = 0; h < from.height(); ++h) {
          unsigned int from_h = h * from.strides[2];
          unsigned int t_h = h * from.width();
          for (unsigned int w = 0; w < from.width(); ++w) {
            unsigned int from_w = w * from.strides[3];
            getData()[t_b + t_c + t_h + w] =
              from.getData()[from_b + from_c + from_h + from_w];
          }
        }
      }
    }
  } else {
    Tensor t = Tensor(from.getDim(), true);
    for (unsigned int b = 0; b < from.batch(); ++b) {
      unsigned int from_b = b * from.strides[0];
      for (unsigned int c = 0; c < from.channel(); ++c) {
        unsigned int from_c = c * from.strides[1];
        for (unsigned int h = 0; h < from.height(); ++h) {
          unsigned int from_h = h * from.strides[2];
          for (unsigned int w = 0; w < from.width(); ++w) {
            unsigned int from_w = w * from.strides[3];
            t.setValue(b, c, h, w,
                       from.getData()[from_b + from_c + from_h + from_w]);
          }
        }
      }
    }
    swap(t, *this);
  }
}

void Tensor::copy(const Tensor &from) {
  // todo: enable copy to non-contiguous tensor
  if (!is_contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (from.length() != 0 && length() == from.length()) {
    reshape(from.getDim());
    copy(from.getData());
  } else {
    Tensor t = Tensor(from.getDim(), from.getData());
    swap(t, *this);
  }
}

Tensor Tensor::clone() const {
  Tensor t;
  t.copy(*this);
  return t;
}

void Tensor::reshape(const TensorDim &d) {
  NNTR_THROW_IF(d.getDataLen() != dim.getDataLen(), std::invalid_argument)
    << "[Tensor]: reshape cannot change the buffer size, trying reshaping "
       "\nfrom "
    << getDim() << " to " << d;

  dim = d;
  strides = d.computeStrides();
}

void Tensor::fill(const Tensor &from, bool initialize) {
  if (initialize && this->uninitialized()) {
    this->copy(from);
    return;
  }

  if (!from.is_contiguous || !is_contiguous) {
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

void Tensor::save(std::ofstream &file) {
  checkedWrite(file, (char *)getData(), getSize(),
               "[Tensor::save] operation failed");
}

void Tensor::read(std::ifstream &file) {
  checkedRead(file, (char *)getData(), getSize(),
              "[Tensor::read] operation failed");
}

/**
 * @brief Calculate average value according to the axis.
 */
Tensor Tensor::average(unsigned int axis) const {
  if (axis >= MAXDIM)
    throw std::out_of_range(
      "negative axis or axis more then MAXDIM is invalid");

  unsigned int axis_size = dim.getDim()[axis];
  if (axis_size == 1)
    return this->clone();

  return this->sum(axis, 1.0 / ((float)axis_size));
}

Tensor Tensor::average(const std::vector<unsigned int> &axes) const {
  if (axes.empty())
    return this->average();

  TensorDim ret_shape;
  for (const auto &idx : axes) {
    if (idx >= MAXDIM) {
      throw std::out_of_range("axis more then MAXDIM is invalid");
    }
    ret_shape.setTensorDim(idx, dim.getTensorDim(idx));
  }

  return this->sum(axes, 1.0 / (float)ret_shape.getDataLen());
}

/**
 * @brief Calculate average value according to the axis.
 */
Tensor Tensor::average() const {
  Tensor result = *this;
  result.reshape({1, 1, 1, dim.getDataLen()});
  return result.average(3);
}

void Tensor::setValue(float val) {
  float *data = getData();
  std::fill(data, data + length(), val);
}

void Tensor::setZero() { sscal(length(), 0, getData(), 1); }

std::vector<unsigned int> Tensor::argmax() const {
  const float *data = getData();
  std::vector<unsigned int> result;
  unsigned int batch_size = batch();
  unsigned int feature_len = dim.getFeatureLen();

  result.resize(batch_size);

  for (unsigned int b = 0; b < batch_size; b++) {
    auto max_iter =
      std::max_element(data + b * feature_len, data + (b + 1) * feature_len);
    result[b] = std::distance(data, max_iter) - (b * feature_len);
  }

  return result;
}

float Tensor::l2norm() const {
  unsigned int len = length();
  const float *data = getData();

  return snrm2(len, data, 1);
}

float Tensor::max_abs() const {
  unsigned int len = length();
  const float *data = getData();

  unsigned int idx = isamax(len, data, 1);
  return *(data + idx);
}

Tensor &Tensor::normalization(Tensor &output) const {
  if (output.uninitialized())
    output = Tensor(dim);

  output.copy(*this);
  output.normalization_i();

  return output;
}

void Tensor::normalization_i() {
  const float *data = getData();

  auto bounds = std::minmax_element(data, data + length());
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
  if (output.uninitialized())
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
  if (m.length() > this->length())
    throw exception::not_supported("broadcasting *this is not supported");

  const TensorDim m_dim = m.getDim();

  BroadcastInfo e;

  /// checking if given Tensor's can be broadcasted
  for (unsigned int i = 0; i < MAXDIM; ++i) {
    if (dim.getTensorDim(i) == m_dim.getTensorDim(i)) {
      e.strides[i] = m.strides[i];
      continue;
    }

    /// If given dimension is 1, it could be reused, the stride remaining 0
    /// Need to check if dim[i] == 1 && m_dim[i] == 1 first though
    /// If so, strides should not change
    if (m_dim.getTensorDim(i) == 1) {
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
    if (dim.getTensorDim(axis) != m_dim.getTensorDim(axis)) {
      e.buffer_axis = axis;
      break;
    }

    e.buffer_size *= dim.getTensorDim(axis);
  }

  /// check strategy that uses consecutive ones
  if (m_dim.getTensorDim(3) == 1) {
    unsigned int inner_loop_size = 1;
    int axis;
    for (axis = 3; axis >= 0; --axis) {
      if (m_dim.getTensorDim(axis) != 1) {
        break;
      }

      inner_loop_size *= dim.getTensorDim(axis);
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
