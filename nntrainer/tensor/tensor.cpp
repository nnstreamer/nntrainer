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

/** do clone of this, perform the operation and return the output */
#define CLONE_OP_I(op, ...)                        \
  do {                                             \
    Tensor clone = this->clone();                  \
    if (clone.op(__VA_ARGS__) != ML_ERROR_NONE) {  \
      std::stringstream ss;                        \
      ss << "Error: op " << __func__ << " failed"; \
      throw std::runtime_error(ss.str());          \
    }                                              \
    return clone;                                  \
  } while (0);

namespace nntrainer {

static auto rng = [] {
  std::mt19937 rng;
  rng.seed(getSeed());
  return rng;
}();

Tensor::Tensor(const TensorDim &d, const float *buf) : Tensor() {
  if (d.getDataLen() != 0) {
    dim = d;
    strides = d.computeStrides();
    data = std::shared_ptr<float>(new float[d.getDataLen()],
                                  std::default_delete<float[]>());

    if (buf != nullptr)
      copy(buf);
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

Tensor Tensor::multiply(float const &value, Tensor &out) const {
  /// @note this is not depending on multiply_i as there is an optimized version
  /// of multiply_i
  CREATE_IF_EMPTY_DIMS(out, getDim());
  const float *data = getData();

  std::transform(data, data + length(), out.getData(),
                 std::bind2nd(std::multiplies<float>(), value));

  return out;
}

Tensor Tensor::multiply(float const &value) { CLONE_OP_I(multiply_i, value); }

int Tensor::multiply_i(float const &value) {
  float *data = getData();
  unsigned int len = length();

  sscal(len, value, data, 1);
  return ML_ERROR_NONE;
}

int Tensor::divide_i(float const &value) {
  if (value == 0.0f) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  return this->multiply_i(1.0f / value);
}

Tensor Tensor::divide(float const &value) {
  if (value == 0.0f) {
    throw std::runtime_error("Error: Divide by zero");
  }

  CLONE_OP_I(divide_i, value);
}

int Tensor::add_i(float const &value) {
  float *data = getData();

  std::transform(data, data + length(), data,
                 std::bind2nd(std::plus<float>(), value));

  return ML_ERROR_NONE;
}

Tensor Tensor::add(Tensor const &m, Tensor &out, float const alpha) const {
  CREATE_IF_EMPTY_DIMS(out, getDim());
  scopy(length(), getData(), 1, out.getData(), 1);
  out.add_i(m, alpha);

  return out;
}

Tensor Tensor::add(float const &value) { CLONE_OP_I(add_i, value); }

Tensor Tensor::add(float const &value, Tensor &out) const {
  const float *data = getData();

  CREATE_IF_EMPTY_DIMS(out, getDim());

  std::transform(data, data + length(), out.getData(),
                 std::bind2nd(std::plus<float>(), value));

  return out;
}

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
  BroadcastInfo() : strides{0, 0, 0, 0} {}

  unsigned int buffer_size; /**< virtual size of the buffer */
  int buffer_axis;          /**< the smallest axis that should be looped.
                                 -1 means no loop needed*/
  std::array<unsigned int, MAXDIM>
    strides; /**< modified strides for the loop */
};

/**
 * @brief Add Tensor Element by Element without mem copy
 * @param[in] m Tensor to be added
 * #retval #ML_ERROR_NONE  Successful
 * #retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
 * TODO: add axis rather doing add over the last two dimensions always
 */
int Tensor::add_i(Tensor const &m, float const alpha) {
  auto f = [&](const BroadcastInfo &e, float *buf, const float *m_buf) {
    saxpy(e.buffer_size, alpha, m_buf, e.strides[3], buf, strides[3]);
  };

  return operator_i(m, f);
}

Tensor Tensor::add(Tensor const &m, float const alpha) const {
  CLONE_OP_I(add_i, m, alpha);
}

int Tensor::subtract_i(Tensor const &m) { return add_i(m, -1); }

Tensor Tensor::subtract(Tensor const &m) const { return add(m, -1); }

int Tensor::subtract_i(float const &value) { return this->add_i(-value); }

Tensor Tensor::subtract(float const &value) { return this->add(-value); }

Tensor Tensor::pow(float exponent) const {
  return apply([=](float in) { return powf(in, exponent); });
}

int Tensor::pow_i(float exponent) {
  return apply_i([=](float in) { return powf(in, exponent); });
}

Tensor Tensor::getBatchSlice(unsigned int offset, unsigned int size) const {
  TensorDim dim_ = dim;
  dim_.batch(size);

  return getSharedDataTensor(dim_, offset * this->dim.getFeatureLen());
}

Tensor Tensor::getSharedDataTensor(const TensorDim dim_,
                                   unsigned int offset) const {
  Tensor ret = *this;

  if (dim_.getDataLen() + offset > dim.getDataLen())
    throw std::invalid_argument(
      "Creating shared tensor of size bigger than tensor memory.");

  ret.dim = dim_;
  ret.data = std::shared_ptr<float>(this->data, this->data.get() + offset);
  ret.strides = ret.dim.computeStrides();

  return ret;
}

int Tensor::operator_i(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, float *, const float *)> v_func) {

  BroadcastInfo e;

  /// shortcut to cover when dimension matches
  /// note that buffer_size, the last stride is only used in v_func but it might
  /// be changed
  if (dim == m.dim) {
    e.buffer_size = length();
    e.strides[3] = 1;
    v_func(e, getData(), m.getData());
    return ML_ERROR_NONE;
  }

  try {
    e = this->computeBroadcastInfo(m);
  } catch (std::exception &err) {
    ml_loge("%s %s", typeid(err).name(), err.what());
    return ML_ERROR_INVALID_PARAMETER;
  }

  return operator_i_util(m, v_func, e);
}

void Tensor::operator_(Tensor const &m,
                       std::function<void(const BroadcastInfo &e, const float *,
                                          const float *, float *)>
                         v_func,
                       Tensor &output) const {

  /// shortcut to cover when dimension matches
  /// note that buffer_size, the last stride is only used in v_func but it might
  /// be changed
  if (dim == m.dim) {
    BroadcastInfo e;
    e.buffer_size = length();
    e.strides[3] = 1;
    v_func(e, getData(), m.getData(), output.getData());
    return;
  }

  return operator_util(m, v_func, output, this->computeBroadcastInfo(m));
}

void Tensor::operator_util(
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
    operator_util(m, v_func, output, e, cur_axis, next_offset, next_m_offset);
  }
}

int Tensor::operator_i_util(
  Tensor const &m,
  std::function<void(const BroadcastInfo &e, float *, const float *)> v_func,
  const BroadcastInfo &e, int cur_axis, unsigned int offset,
  unsigned int m_offset) {

  float *buf = this->getData();
  const float *m_buf = m.getData();
  int status = ML_ERROR_NONE;

  if (e.buffer_axis == cur_axis) {
    v_func(e, buf + offset, m_buf + m_offset);
    return ML_ERROR_NONE;
  }

  cur_axis++;
  for (unsigned int i = 0; i < dim.getTensorDim(cur_axis); ++i) {
    unsigned int next_offset = offset + i * strides[cur_axis];
    unsigned int next_m_offset = m_offset + i * e.strides[cur_axis];
    status =
      operator_i_util(m, v_func, e, cur_axis, next_offset, next_m_offset);
    if (status != ML_ERROR_NONE) {
      ml_loge("[operator_i] failed: %d", status);
      return status;
    }
  }

  return status;
}

int Tensor::multiply_i(Tensor const &m) {
  auto f = [&](const BroadcastInfo &e, float *buf, const float *m_buf) {
    for (unsigned int i = 0; i < e.buffer_size; ++i) {
      *buf *= *m_buf;
      buf += strides[3];
      m_buf += e.strides[3];
    }
  };

  return operator_i(m, f);
}

Tensor Tensor::multiply(Tensor const &m) const { CLONE_OP_I(multiply_i, m); }

Tensor &Tensor::multiply(Tensor const &m, Tensor &output) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *output) {
    for (unsigned int i = 0; i < e.buffer_size; ++i) {
      *output = *buf * *m_buf;
      buf += strides[3];
      m_buf += e.strides[3];
      output += strides[3];
    }
  };

  CREATE_IF_EMPTY_DIMS(output, dim);
  operator_(m, f, output);

  return output;
}

int Tensor::divide_i(Tensor const &m) {
  auto f = [&](const BroadcastInfo &e, float *buf, const float *m_buf) {
    for (unsigned int i = 0; i < e.buffer_size; ++i) {
      *buf /= *m_buf;
      buf += strides[3];
      m_buf += e.strides[3];
    }
  };

  return operator_i(m, f);
}

Tensor Tensor::divide(Tensor const &m) const { CLONE_OP_I(divide_i, m); }

Tensor &Tensor::divide(Tensor const &m, Tensor &output) const {
  auto f = [&](const BroadcastInfo &e, const float *buf, const float *m_buf,
               float *output) {
    for (unsigned int i = 0; i < e.buffer_size; ++i) {
      *output = *buf / *m_buf;
      buf += strides[3];
      m_buf += e.strides[3];
      output += strides[3];
    }
  };

  CREATE_IF_EMPTY_DIMS(output, dim);
  operator_(m, f, output);

  return output;
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
  return sum(ret, axis, alpha);
}
Tensor &Tensor::sum(Tensor &ret, unsigned int axis, float alpha) const {
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
  if (axes.empty())
    throw std::invalid_argument("empty axes given");

  Tensor ret = this->sum(axes[0], alpha);

  for (unsigned int i = 1; i < axes.size(); ++i)
    ret = ret.sum(axes[i]);

  return ret;
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
  if (trans && dim.rank() > 2) {
    throw exception::not_supported("Error: support only for rank of dot "
                                   "matrix <= 2 with trans");
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
    // However, result is not initialized properly. There might include garbage
    // like nan. When we have to use this value as in C = alpha*A*B + beta*C,
    // then have to check gargabe data of C is not effect or not.

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

Tensor Tensor::transpose(std::string direction) const {
  unsigned int SL, SI, SJ, SK;
  int dir[MAXDIM - 1];
  unsigned int fromDim[4];
  const float *inptr;
  float *outptr;

  fromDim[0] = dim.batch();
  fromDim[1] = dim.channel();
  fromDim[2] = dim.height();
  fromDim[3] = dim.width();

  getValues(3, direction, dir);
  Tensor result(dim.batch(), fromDim[dir[0] + 1], fromDim[dir[1] + 1],
                fromDim[dir[2] + 1]);

  int indexI = dir[0];
  int indexJ = dir[1];

  SL = fromDim[0], SI = fromDim[1], SJ = fromDim[2], SK = fromDim[3];

  inptr = getData();
  outptr = result.getData();

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

  return result;
}

Tensor Tensor::apply(std::function<float(float)> f) const {
  Tensor result;
  return apply(f, result);
}

int Tensor::apply_i(std::function<float(float)> f) {
  float *data = getData();

  std::transform(data, data + length(), data, f);

  return ML_ERROR_NONE;
}

Tensor &Tensor::apply(std::function<float(float)> f, Tensor &output) const {
  CREATE_IF_EMPTY_DIMS(output, dim);

  const float *data = getData();
  float *rdata = output.getData();

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

void Tensor::copy(const float *buf) { scopy(length(), buf, 1, getData(), 1); }

void Tensor::copy(const Tensor &from) {
  // todo: enable copy to non-contiguous tensor
  if (!is_contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  if (length() == from.length()) {
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
  if (d.getDataLen() != dim.getDataLen()) {
    throw std::invalid_argument("Error: reshape cannot change the tensor size");
  }
  dim = d;
  strides = d.computeStrides();
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

    /// If given dimension is 1, it could be reuesed, the stride remaining 0
    /// Need to check if dim[i] == 1 && m_dim[i] == 1 first though
    /// If so, strides should not change
    if (m_dim.getTensorDim(i) == 1) {
      continue;
    }

    std::stringstream ss;
    ss << "[computeBroadcastInfo] broadcasting only allowed for"
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
