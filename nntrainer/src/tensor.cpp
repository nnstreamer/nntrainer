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
#include <iterator>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <random>
#include <sstream>
#include <stdio.h>
#include <tensor.h>
#include <util_func.h>

#include <lazy_tensor.h>

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

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
  return getData()[getIndex(batch, c, h, w)];
}

void Tensor::setValue(unsigned int batch, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  if (!is_contiguous) {
    throw std::runtime_error("cannot set value of non-contiguous tensor");
  }

  getData()[getIndex(batch, c, h, w)] = value;
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

Tensor Tensor::multiply(float const &value) { CLONE_OP_I(multiply_i, value); }

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

Tensor Tensor::add(float const &value) { CLONE_OP_I(add_i, value); }

void Tensor::saxpy(const unsigned int N, const float alpha, const float *X,
                   const int incX, float *Y, const int incY) {
#ifdef USE_BLAS
  cblas_saxpy(N, alpha, X, incX, Y, incY);
#else
  unsigned int xi, yi;
  if (incX <= 0 or incY <= 0)
    throw std::invalid_argument(
      "Error: negative inc not supported without cblas");
  for (unsigned int i = 0; i < N; i++)
    Y[i] = Y[i * incY] + X[i * incX] * alpha;
#endif
}

/**
 * @brief Add Tensor Element by Element without mem copy
 * @param[in] m Tensor to be added
 * #retval #ML_ERROR_NONE  Successful
 * #retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
 * TODO: add axis rather doing add over the last two dimensions always
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
  CLONE_OP_I(add_i, m, alpha);
}

int Tensor::subtract_i(Tensor const &m) { return add_i(m, -1); }

Tensor Tensor::subtract(Tensor const &m) const { return add(m, -1); }

int Tensor::subtract_i(float const &value) { return this->add_i(-value); }

Tensor Tensor::subtract(float const &value) { return this->add(-value); }

int Tensor::operator_i(Tensor const &m,
                       std::function<float(float const, float const)> op) {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    return ML_ERROR_INVALID_PARAMETER;
  }

  float *data = getData();
  const float *mdata = m.getData();

  unsigned int feat_len = dim.getFeatureLen();
  unsigned int len = length();
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      int b = k * feat_len;
      std::transform(data + b, data + b + feat_len, mdata, data + b, op);
    }
  } else {
    if (batch() != m.batch()) {
      return ML_ERROR_INVALID_PARAMETER;
    }
    std::transform(data, data + len, mdata, data, op);
  }

  return ML_ERROR_NONE;
}

int Tensor::multiply_i(Tensor const &m) {
  return operator_i(m, std::multiplies<float>());
}

Tensor Tensor::multiply(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  CLONE_OP_I(multiply_i, m);
}

int Tensor::divide_i(Tensor const &m) {
  return operator_i(m, std::divides<float>());
}

Tensor Tensor::divide(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  CLONE_OP_I(divide_i, m);
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

#ifdef USE_BLAS
  Tensor ones(1, 1, 1, feat_len);
  ones.setValue(1.0);
  cblas_sgemv(CblasRowMajor, CblasNoTrans, batch, feat_len, 1, data, feat_len,
              ones.getData(), 1, 0.0, rdata, 1);
#else
  unsigned int i, k;
  for (k = 0; k < batch; ++k) {
    unsigned int id = k * feat_len;
    rdata[k] = 0.0f;
    for (i = 0; i < feat_len; ++i) {
      rdata[k] += data[id + i];
    }
  }
#endif

  return ret;
}

/**
 * @brief Calculate sum according to the axis.
 */
Tensor Tensor::sum(int axis, float alpha) const {
  Tensor ret;

  const float *data = getData();

  if (dim.getDim()[axis] == 1 and alpha == 1.0)
    return this->clone();

  switch (axis) {
  case 0: {
    ret = Tensor(1, dim.channel(), dim.height(), dim.width());
#ifdef USE_BLAS
    unsigned int feat_len = dim.getFeatureLen();
    unsigned int batch = dim.batch();
    Tensor ones(1, 1, 1, feat_len);
    ones.setValue(alpha);
    cblas_sgemv(CblasRowMajor, CblasTrans, batch, feat_len, 1, data, feat_len,
                ones.getData(), 1, 0.0, ret.getData(), 1);
#else
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
#endif
  } break;
  case 1: {
    ret = Tensor(dim.batch(), 1, dim.height(), dim.width());
#ifdef USE_BLAS
    unsigned int feat_len = dim.height() * dim.width();
    unsigned int channel = dim.channel();
    Tensor ones(1, 1, 1, channel);
    ones.setValue(alpha);
    float *rdata = ret.getData();
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      cblas_sgemv(CblasRowMajor, CblasTrans, channel, feat_len, 1,
                  &data[k * dim.getFeatureLen()], feat_len, ones.getData(), 1,
                  0.0, &rdata[k * feat_len], 1);
    }
#else
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
#endif
  } break;
  case 2: {
    ret = Tensor(dim.batch(), dim.channel(), 1, dim.width());
#ifdef USE_BLAS
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
        cblas_sgemv(CblasRowMajor, CblasTrans, height, width, 1, &data[idx],
                    width, ones.getData(), 1, 0.0, &rdata[ridx], 1);
      }
    }
#else
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
#endif
  } break;
  case 3: {
    ret = Tensor(dim.batch(), dim.channel(), dim.height(), 1);
#ifdef USE_BLAS
    unsigned int m = ret.dim.getDataLen();
    unsigned int n = dim.width();
    Tensor ones(1, 1, 1, n);
    ones.setValue(alpha);
    cblas_sgemv(CblasRowMajor, CblasNoTrans, m, n, 1, data, n, ones.getData(),
                1, 0.0, ret.getData(), 1);
#else
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
#endif
  } break;
  default:
    throw std::out_of_range("Error: Cannot exceede 3");
    break;
  }
  return ret;
}

/**
 * If the dim.batch() size of m is one, the it is reused for
 * every calculation along with dim.batch().
 * TODO: support dim.channel() > 1.
 */
Tensor Tensor::dot(Tensor const &m) const {
  if (dim.width() != m.dim.height()) {
    throw std::runtime_error("Error dim.width() != m.dim.height()");
  }

  if (dim.channel() != 1 || m.dim.channel() != 1) {
    throw std::runtime_error("Error channel() != 1");
  }

  int mwidth = m.dim.width();
  Tensor result(dim.batch(), 1, dim.height(), mwidth);

  const float *data = getData();
  const float *mdata = m.getData();
  float *rdata = result.getData();

#ifdef USE_BLAS
  float alpha_dgemm = 1.0f;
  float beta_dgemm = 0.0f;
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); k++) {
      unsigned int i = k * dim.width() * dim.height();
      unsigned int ii = k * dim.height() * m.dim.width();
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim.height(),
                  m.dim.width(), dim.width(), alpha_dgemm, &(data[i]),
                  dim.width(), mdata, m.dim.width(), beta_dgemm, &(rdata[ii]),
                  m.dim.width());
    }
  } else {
    for (unsigned int k = 0; k < dim.batch(); k++) {
      unsigned int i = k * dim.width() * dim.height();
      unsigned int j = k * m.dim.width() * m.dim.height();
      unsigned int ii = k * dim.height() * m.dim.width();

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim.height(),
                  m.dim.width(), dim.width(), alpha_dgemm, &(data[i]),
                  dim.width(), &(mdata[j]), m.dim.width(), beta_dgemm,
                  &(rdata[ii]), m.dim.width());
    }
  }
#elif USE_CUBLAS
  int devID = 0;
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, devID);
  float *d_A, *d_B, *d_C;

  unsigned int size_A = this->dim.width() * dim.height() * sizeof(float);
  unsigned int size_B = m.dim.width() * m.dim.height() * sizeof(float);
  unsigned int size_C =
    result.dim.width() * result.dim.height() * sizeof(float);

  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); k++) {
      unsigned int i = k * dim.width() * dim.height();
      unsigned int ii = k * dim.height() * m.dim.width();

      cudaMalloc((void **)&d_A, size_A);
      cudaMalloc((void **)&d_B, size_B);
      cudaMemcpy(d_A, &data[i], size_A, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, mdata, size_B, cudaMemcpyHostToDevice);
      cudaMalloc((void **)&d_C, size_C);

      {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle;

        (cublasCreate(&handle));

        (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.dim.width(),
                     dim.height(), dim.width(), &alpha, d_B, m.dim.width(), d_A,
                     dim.width(), &beta, d_C, m.dim.width()));

        (cudaMemcpy(&rdata[ii], d_C, size_C, cudaMemcpyDeviceToHost));
        (cublasDestroy(handle));
      }
    }
  } else {
    for (unsigned int k = 0; k < dim.batch(); k++) {
      unsigned int i = k * dim.width() * dim.height();
      unsigned int j = k * m.dim.width() * m.dim.height();
      unsigned int ii = k * dim.height() * m.dim.width();

      (cudaMalloc((void **)&d_A, size_A));
      (cudaMalloc((void **)&d_B, size_B));
      (cudaMemcpy(d_A, &data[i], size_A, cudaMemcpyHostToDevice));
      (cudaMemcpy(d_B, &mdata[j], size_B, cudaMemcpyHostToDevice));
      (cudaMalloc((void **)&d_C, size_C));

      {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle;

        (cublasCreate(&handle));

        (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.dim.width(),
                     dim.height(), dim.width(), &alpha, d_B, m.dim.width(), d_A,
                     dim.width(), &beta, d_C, m.dim.width()));

        (cudaMemcpy(&rdata[ii], d_C, size_C, cudaMemcpyDeviceToHost));
        (cublasDestroy(handle));
      }
    }
  }
#else
  float w = 0.0f;
  unsigned int i, j, k, h;
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < dim.height(); ++i) {
        for (j = 0; j < m.dim.width(); ++j) {
          for (h = 0; h < dim.width(); ++h) {
            w += data[k * dim.height() * dim.width() + i * dim.width() + h] *
                 mdata[h * m.dim.width() + j];
          }
          rdata[k * dim.height() * m.dim.width() + i * m.dim.width() + j] = w;
          w = 0.0f;
        }
      }
    }
  } else {
    for (k = 0; k < dim.batch(); k++) {
      for (i = 0; i < dim.height(); i++) {
        for (j = 0; j < m.dim.width(); j++) {
          for (h = 0; h < dim.width(); h++) {
            w += data[k * dim.height() * dim.width() + i * dim.width() + h] *
                 mdata[k * dim.width() * m.dim.width() + h * m.dim.width() + j];
          }
          rdata[k * dim.height() * m.dim.width() + i * m.dim.width() + j] = w;
          w = 0.0f;
        }
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
  const float *data = getData();
  float *rdata = result.getData();

  std::transform(data, data + length(), rdata, f);

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
  unsigned int axis_size = dim.getDim()[axis];
  if (axis_size == 1)
    return this->clone();

  return this->sum(axis, 1.0 / ((float)axis_size));
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

void Tensor::setZero() { setValue(0); }

unsigned int Tensor::argmax() const {
  const float *data = getData();
  auto max_iter = std::max_element(data, data + length());
  return std::distance(data, max_iter);
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
  const float *data = getData();

  auto bounds = std::minmax_element(data, data + length());
  const float min = *bounds.first;
  const float max = *bounds.second;

  return this->chain().subtract_i(min).divide_i(max - min).run();
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
