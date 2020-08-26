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
#include <cstring>
#include <iomanip>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <tensor.h>
#include <util_func.h>

#include <lazy_tensor.h>

#ifdef USE_CUBLAS
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

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

namespace nntrainer {

static auto rng = [] {
  std::mt19937 rng;
  rng.seed(getSeed());
  return rng;
}();

Tensor::Tensor(const TensorDim &d, const float *buf) :
  dim(d),
  strides{{1, 2, 3}},
  is_contiguous(true),
  data(d.getDataLen() == 0
         ? nullptr
         : std::shared_ptr<float>(new float[d.getDataLen()],
                                  std::default_delete<float[]>())) {
  if (d.getDataLen() == 0) {
    return;
  }

  // todo: initialize appropriate strides
  if (buf != nullptr) {
    float *data = getData();
    unsigned int len = length();

#ifdef USE_BLAS
    cblas_scopy(len, buf, 1, data, 1);
#else
    for (unsigned int i = 0; i < len; ++i) {
      data[i] = buf[i];
    }
#endif
  }
}

void Tensor::swap(Tensor &lhs, Tensor &rhs) noexcept {
  std::swap(lhs.dim, rhs.dim);
  std::swap(lhs.data, rhs.data);
  std::swap(lhs.strides, rhs.strides);
  std::swap(lhs.is_contiguous, rhs.is_contiguous);
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

float Tensor::getValue(unsigned int batch, unsigned int c, unsigned int h,
                       unsigned int w) const {
  return getData()[batch * dim.getFeatureLen() +
                   c * dim.height() * dim.width() + h * dim.width() + w];
}

void Tensor::setValue(unsigned int batch, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  if (!is_contiguous) {
    throw std::runtime_error("cannot set value of non-contiguous tensor");
  }

  getData()[batch * dim.getFeatureLen() + c * dim.height() * dim.width() +
            h * dim.width() + w] = value;
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
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d) :
  strides{{1, 2, 3}} {

  if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
    throw std::out_of_range(
      "[Tensor] trying to initialize Tensor from empty vector");
  }

  dim.batch(d.size());
  dim.channel(d[0].size());
  dim.height(d[0][0].size());
  dim.width(d[0][0][0].size());
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

  float *data = getData();
  unsigned int len = length();

#ifdef USE_BLAS
  cblas_sscal(len, value, data, 1);
#else
  for (unsigned int k = 0; k < len; ++k) {
    data[k] *= value;
  }
#endif
  return ML_ERROR_NONE;
}

Tensor Tensor::multiply(float const &value) {
  Tensor result = this->clone();
  result.multiply_i(value);

  return result;
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
  Tensor result;
  result.copy(*this);
  result.divide_i(value);

  return result;
}

int Tensor::add_i(float const &value) {
  float *data = getData();
  unsigned int len = length();
#ifdef USE_BLAS
  Tensor tmp(dim);
  tmp.setValue(value);
  cblas_saxpy(len, 1, tmp.getData(), 1, data, 1);
#else
  for (unsigned int k = 0; k < len; ++k) {
    data[k] += value;
  }
#endif

  return ML_ERROR_NONE;
}

Tensor Tensor::add(float const &value) {
  Tensor result = this->clone();
  result.add_i(value);

  return result;
}

/**
 * @brief Add Tensor Element by Element without mem copy
 * @param[in] m Tensor to be added
 * #retval #ML_ERROR_NONE  Successful
 * #retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
 */
int Tensor::add_i(Tensor const &m, float const alpha) {
  if ((dim.height() != m.dim.height()) || (dim.width() != m.dim.width())) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  float *data = getData();
  const float *mdata = m.getData();
  unsigned int len = length();

#ifdef USE_BLAS
  unsigned int size = dim.getFeatureLen();
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      cblas_saxpy(size, alpha, mdata, 1, &(data[k * size]), 1);
    }
  } else {
    if (dim.batch() != m.dim.batch()) {
      return ML_ERROR_INVALID_PARAMETER;
    }
    cblas_saxpy(len, alpha, mdata, 1, data, 1);
  }
#else
  unsigned int i, j, k;
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < m.dim.getFeatureLen(); ++i) {
        j = k * m.dim.getFeatureLen();
        data[j + i] += alpha * mdata[i];
      }
    }
  } else {
    if (dim.batch() != m.dim.batch()) {
      return ML_ERROR_INVALID_PARAMETER;
    }
    for (k = 0; k < len; ++k) {
      data[k] += alpha * mdata[k];
    }
  }
#endif

  return ML_ERROR_NONE;
}

Tensor Tensor::add(Tensor const &m, float const alpha) const {
  Tensor result;

  result.copy(*this);

  if (result.add_i(m, alpha) != ML_ERROR_NONE)
    throw std::runtime_error("Error: Dimension must be equal each other");

  return result;
}

int Tensor::subtract_i(Tensor const &m) {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  if (dim.batch() != m.dim.batch() && m.dim.batch() != 1) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  float *data = getData();
  const float *mdata = m.getData();
  unsigned int len = length();

#ifdef USE_BLAS
  unsigned int size =
    this->dim.channel() * this->dim.width() * this->dim.height();
  float alpha = -1.0f;

  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      cblas_saxpy(size, alpha, mdata, 1, &(data[k * size]), 1);
    }
  } else {
    cblas_saxpy(len, alpha, mdata, 1, data, 1);
  }
#else
  unsigned int i, j, k;
  if (m.dim.batch() == 1) {
    len = m.dim.getFeatureLen();
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < len; ++i) {
        j = k * len;
        data[j + i] -= mdata[i];
      }
    }
  } else {
    for (k = 0; k < len; ++k) {
      data[k] -= mdata[k];
    }
  }
#endif

  return ML_ERROR_NONE;
}

Tensor Tensor::subtract(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result = this->clone();
  result.subtract_i(m);

  return result;
}

int Tensor::subtract_i(float const &value) { return this->add_i(-value); }

Tensor Tensor::subtract(float const &value) {
  Tensor result = this->clone();

  if (result.subtract_i(value) != ML_ERROR_NONE) {
    throw std::runtime_error("Error: there was an error on subtraction");
  }

  return result;
}

int Tensor::multiply_i(Tensor const &m) {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  float *data = getData();
  const float *mdata = m.getData();

  unsigned int len = length();
  unsigned int end = len / 4;
  unsigned int e = dim.getFeatureLen() / 4;
  unsigned int i;
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      int b = k * dim.getFeatureLen();
      for (i = 0; i < e * 4; i += 4) {
        data[b + i + 0] *= mdata[i + 0];
        data[b + i + 1] *= mdata[i + 1];
        data[b + i + 2] *= mdata[i + 2];
        data[b + i + 3] *= mdata[i + 3];
      }
      for (unsigned int j = i; j < dim.getFeatureLen(); j++)
        data[b + j] *= mdata[j];
    }
  } else {
    for (i = 0; i < end * 4; i += 4) {
      data[i + 0] *= mdata[i + 0];
      data[i + 1] *= mdata[i + 1];
      data[i + 2] *= mdata[i + 2];
      data[i + 3] *= mdata[i + 3];
    }
    for (unsigned int j = i; j < len; ++j)
      data[j] *= mdata[j];
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::multiply(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result;
  result.copy(*this);
  result.multiply_i(m);

  return result;
}

int Tensor::divide_i(Tensor const &m) {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  float *data = getData();
  const float *mdata = m.getData();

  unsigned int len = length();
  unsigned int end = len / 4;
  unsigned int e = dim.getFeatureLen() / 4;
  unsigned int i, j, k;

  // todo: effectively check if m.data[index] is 0
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      unsigned int b = k * dim.getFeatureLen();
      for (i = 0; i < e * 4; i += 4) {
        data[b + i + 0] /= mdata[i + 0];
        data[b + i + 1] /= mdata[i + 1];
        data[b + i + 2] /= mdata[i + 2];
        data[b + i + 3] /= mdata[i + 3];
      }
      for (unsigned int j = i; j < dim.getFeatureLen(); ++j)
        data[b + j] /= mdata[j];
    }
  } else {
    for (i = 0; i < end * 4; i += 4) {
      data[i + 0] /= mdata[i + 0];
      data[i + 1] /= mdata[i + 1];
      data[i + 2] /= mdata[i + 2];
      data[i + 3] /= mdata[i + 3];
    }
    for (j = i; j < len; ++j)
      data[j] /= mdata[j];
  }

  return ML_ERROR_NONE;
}

Tensor Tensor::divide(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result = this->clone();
  result.divide_i(m);

  return result;
}

/**
 * This is to sum the Tensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1) dimension.
 */
Tensor Tensor::sum_by_batch() const {
  unsigned int k;
  Tensor ret(dim.batch(), 1, 1, 1);

  const float *data = getData();
  float *rdata = ret.getData();

#ifdef USE_BLAS
  for (k = 0; k < dim.batch(); ++k)
    rdata[k] =
      cblas_sasum(dim.getFeatureLen(), &(data[k * dim.getFeatureLen()]), 1);
#else
  unsigned int i;
  for (k = 0; k < dim.batch(); ++k) {
    unsigned int id = k * dim.getFeatureLen();
    rdata[k] = 0.0f;
    for (i = 0; i < dim.getFeatureLen(); ++i) {
      rdata[k] += data[id + i];
    }
  }
#endif

  return ret;
}

/**
 * @brief Calculate sum according to the axis.
 */
Tensor Tensor::sum(int axis) const {
  Tensor ret;

  const float *data = getData();

  switch (axis) {
  case 0: {
    ret = Tensor(1, dim.channel(), dim.height(), dim.width());
    ret.setZero();
    float *rdata = ret.getData();
    for (unsigned int l = 0; l < dim.channel(); ++l) {
      unsigned int L = l * dim.width() * dim.height();
      for (unsigned int i = 0; i < dim.height(); ++i) {
        unsigned int I = i * dim.width();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          for (unsigned int k = 0; k < dim.batch(); ++k) {
            unsigned int K = k * dim.getFeatureLen();
            rdata[L + I + j] += data[K + L + I + j];
          }
        }
      }
    }
  } break;
  case 1: {
    ret = Tensor(dim.batch(), 1, dim.height(), dim.width());
    ret.setZero();
    float *rdata = ret.getData();
    for (unsigned int l = 0; l < dim.batch(); ++l) {
      unsigned int L = dim.width() * dim.height() * l;
      unsigned int LL = l * dim.getFeatureLen();
      for (unsigned int j = 0; j < dim.height(); ++j) {
        unsigned int J = j * dim.width();
        for (unsigned int i = 0; i < dim.width(); ++i) {
          for (unsigned int k = 0; k < dim.channel(); ++k) {
            unsigned int K = k * dim.width() * dim.height();
            rdata[L + J + i] += data[LL + K + J + i];
          }
        }
      }
    }
  } break;
  case 2: {
    ret = Tensor(dim.batch(), dim.channel(), 1, dim.width());
    ret.setZero();
    float *rdata = ret.getData();
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      unsigned int K = k * dim.channel() * dim.width();
      unsigned int KK = k * dim.getFeatureLen();
      for (unsigned int l = 0; l < dim.channel(); ++l) {
        unsigned int L = l * dim.width();
        unsigned int LL = l * dim.width() * dim.height();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          for (unsigned int i = 0; i < dim.height(); ++i) {
            unsigned int I = i * dim.width();
            rdata[K + L + j] += data[KK + LL + j + I];
          }
        }
      }
    }
  } break;
  case 3: {
    ret = Tensor(dim.batch(), dim.channel(), dim.height(), 1);
    ret.setZero();
    float *rdata = ret.getData();
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      unsigned int K = k * dim.channel() * dim.height();
      unsigned int KK = k * dim.getFeatureLen();
      for (unsigned int l = 0; l < dim.channel(); ++l) {
        unsigned int L = l * dim.height();
        unsigned int LL = l * dim.height() * dim.width();
        for (unsigned int i = 0; i < dim.height(); ++i) {
          unsigned int II = i * dim.width();
          for (unsigned int j = 0; j < dim.width(); ++j) {
            rdata[K + L + i] += data[KK + LL + II + j];
          }
        }
      }
    }
  } break;
  default:
    throw std::out_of_range("Error: Cannot exceede 3");
    break;
  }
  return ret;
}

/**
 * @note: This dot product flattens the fist 3 axis for the purpose of
 * computation. So, while performing, these matrices are behaving as 2-D
 * matrices. The dimensions are restored while returning back the tensor.
 */
Tensor Tensor::dot(Tensor const &m, bool trans, bool trans_m) const {
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
  Tensor result;

  unsigned int M, N, K, lda, ldb, ldc;

  if (!trans && !trans_m) {
    if (dim2 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim2 */
    N = mdim2;
    M = dim1;
    lda = K;
    ldb = N;
    result = Tensor(batch(), channel(), height(), mdim2);
  } else if (!trans && trans_m) {
    if (dim2 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim2 */
    N = mdim1;
    M = dim1;
    lda = K;
    ldb = K;
    result = Tensor(batch(), channel(), height(), mdim1);
  } else if (trans && !trans_m) {
    if (dim1 != mdim1)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim1; /** == dim1 */
    N = mdim2;
    M = dim2;
    lda = M;
    ldb = N;
    result = Tensor(1, 1, dim2, mdim2);
  } else {
    if (dim1 != mdim2)
      throw std::runtime_error(
        "Error: incompatible dimensions for dot product");
    K = mdim2; /** == dim1 */
    N = mdim1;
    M = dim2;
    lda = M;
    ldb = K;
    result = Tensor(1, 1, dim2, mdim1);
  }
  ldc = N;

  const float *data = getData();
  const float *mdata = m.getData();
  float *rdata = result.getData();
  const float alpha = 1.0f;
  const float beta = 0.0f;
#ifdef USE_BLAS
  enum CBLAS_TRANSPOSE transA = trans ? CblasTrans : CblasNoTrans;
  enum CBLAS_TRANSPOSE transB = trans_m ? CblasTrans : CblasNoTrans;
  cblas_sgemm(CblasRowMajor, transA, transB, M, N, K, alpha, data, lda, mdata,
              ldb, beta, rdata, ldc);
#elif USE_CUBLAS
  int devID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  float *d_A, *d_B, *d_C;

  unsigned int size_A = this->length() * sizeof(float);
  unsigned int size_B = m.length() * sizeof(float);
  unsigned int size_C = result.length() * sizeof(float);

  cudaMalloc((void **)&d_A, size_A);
  cudaMalloc((void **)&d_B, size_B);
  cudaMemcpy(d_A, data, size_A, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, mdata, size_B, cudaMemcpyHostToDevice);
  cudaMalloc((void **)&d_C, size_C);

  cublasHandle_t handle;
  cublasCreate(&handle);

  cublasOperation_t transA = trans ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasOperation_t transB = trans_m ? CUBLAS_OP_T : CUBLAS_OP_N;
  cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_B, N, d_A, K,
              &beta, d_C, N);

  cudaMemcpy(rdata, d_C, size_C, cudaMemcpyDeviceToHost);
  cublasDestroy(handle);
#else
  float w = 0.0f;
  unsigned int i, j, k, h;
  Tensor this_t, m_t;

  if (trans) {
    this_t = this->transpose("0:2:1");
    data = this_t.getData();
  }

  if (trans_m) {
    m_t = this->transpose("0:2:1");
    mdata = m_t.getData();
  }

  result.setZero();
  for (unsigned int n = 0; n < N; n++) {
    for (unsigned int m = 0; m < M; m++) {
      for (unsigned int k = 0; k < K; k++) {
        rdata[n * ldc + m] += data[m * lda + k] * mdata[k * ldb + n];
      }
    }
  }
#endif

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
  Tensor result(dim.batch(), dim.channel(), dim.height(), dim.width());
  unsigned int i;

  const float *data = getData();
  float *rdata = result.getData();
  unsigned int len = length();

  for (i = 0; i < len; ++i)
    rdata[i] = f(data[i]);

  return result;
}

Tensor Tensor::apply(std::function<Tensor(Tensor)> f) const { return f(*this); }

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

void Tensor::copy(const Tensor &from) {
  // todo: enable copy to non-contiguous tensor
  if (!is_contiguous) {
    throw std::runtime_error("Cannot copy non-contiguous tensor");
  }

  Tensor t = Tensor(from.getDim(), from.getData());
  swap(t, *this);
}

Tensor Tensor::clone() const {
  Tensor t;
  t.copy(*this);
  return t;
}

void Tensor::reshape(TensorDim d) {
  if (d.getDataLen() != dim.getDataLen()) {
    throw std::invalid_argument("Error: reshape cannot change the tensor size");
  }
  dim = d;
}

void Tensor::save(std::ofstream &file) {
  file.write((char *)getData(), getSize());
}

void Tensor::read(std::ifstream &file) {
  file.read((char *)getData(), getSize());
}

/**
 * @brief Calculate average value according to the axis.
 */
Tensor Tensor::average(int axis) const {
  const unsigned int *dim_arr = dim.getDim();
  if (axis >= MAXDIM || dim_arr[axis] == 1)
    return *this;

  TensorDim out_dim = dim;
  out_dim.setTensorDim(axis, 1);

  Tensor result;
  result = std::move(this->sum(axis));
  result.divide_i(dim.getDim()[axis]);

  return result;
}

/**
 * @brief Calculate average value according to the axis.
 */
Tensor Tensor::average() const {
  LazyTensor lazy_result = this->chain();

  for (unsigned int axis = 0; axis < dim.getNumDim(); ++axis)
    lazy_result = lazy_result.average(axis);

  return lazy_result.run();
}

void Tensor::setValue(float val) {
  float *data = getData();
  std::fill(data, data + length(), val);
}

void Tensor::setZero() { setValue(0); }

int Tensor::argmax() const {
  int index = 0;
  float maximum = min_limits;
  const float *data = getData();
  unsigned int len = length();

  for (unsigned int i = 0; i < len; i++) {
    if (data[i] > maximum) {
      maximum = data[i];
      index = i;
    }
  }
  return index;
}

float Tensor::l2norm() const {
  unsigned int len = length();
  const float *data = getData();

#ifdef USE_BLAS
  return cblas_snrm2(len, data, 1);
#else
  // fix me: to the version that does not allow overflow
  float sum = 0.0f;
  float tmp;
#pragma omp parallel for private(tmp) reduction(+ : sum)
  for (unsigned int i = 0; i < len; i++) {
    tmp = data[i];
    sum += tmp * tmp;
  }
  return sqrt(sum);
#endif
}

Tensor Tensor::normalization() const {
  Tensor results;
  float Min = max_limits;
  float Max = min_limits;

  const float *data = getData();

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    for (unsigned int l = 0; l < dim.channel(); ++l) {
      for (unsigned int i = 0; i < dim.height(); ++i) {
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int id = k * dim.getFeatureLen() +
                            l * dim.height() * dim.width() + i * dim.width() +
                            j;
          if (data[id] < Min)
            Min = data[id];
          if (data[id] > Max)
            Max = data[id];
        }
      }
    }
  }

  float dif = Max - Min;

  results = this->chain().subtract_i(Min).divide_i(dif).run();

  return results;
}

LazyTensor Tensor::chain() const { return LazyTensor(*this); }

Tensor Tensor::standardization() const {
  Tensor result(dim);

  const float *data = getData();
  float *rdata = result.getData();

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    int K = k * dim.getFeatureLen();
    float mean;
    float mean_tmp = 0.0f;
    float std_tmp = 0.0f;
    float std_dev = 0.0f;

    for (unsigned int l = 0; l < dim.channel(); ++l) {
      unsigned int L = K + l * dim.height() * dim.width();
      for (unsigned int i = 0; i < dim.height(); ++i) {
        unsigned int I = L + i * dim.width();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int J = I + j;
          mean_tmp += data[J];
        }
      }
    }

    mean = mean_tmp / (this->dim.getFeatureLen());

    for (unsigned int l = 0; l < dim.channel(); ++l) {
      unsigned int L = K + l * dim.height() * dim.width();
      for (unsigned int i = 0; i < dim.height(); ++i) {
        unsigned int I = L + i * dim.width();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int J = I + j;
          std_tmp += (data[J] - mean) * (data[J] - mean);
        }
      }
    }
    std_dev = sqrt(std_tmp) / (this->dim.getFeatureLen());

    for (unsigned int l = 0; l < dim.channel(); ++l) {
      unsigned int L = K + l * dim.height() * dim.width();
      for (unsigned int i = 0; i < dim.height(); ++i) {
        unsigned int I = L + i * dim.width();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int J = I + j;
          rdata[J] = (data[J] - mean) / std_dev;
        }
      }
    }
  }

  return result;
}
} /* namespace nntrainer */
