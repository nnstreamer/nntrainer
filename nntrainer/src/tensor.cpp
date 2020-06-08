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
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <sstream>
#include <stdio.h>
#include <tensor.h>

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

Tensor::Tensor(TensorDim d) {
  dim = d;
  this->data = std::vector<float>(dim.getDataLen());
  setZero();
}

Tensor::Tensor(int height, int width) {
  dim.height(height);
  dim.width(width);
  this->data = std::vector<float>(dim.getDataLen());
  setZero();
}

Tensor::Tensor(int channel, int height, int width) {
  dim.height(height);
  dim.width(width);
  dim.batch(channel);
  this->data = std::vector<float>(dim.getDataLen());
  setZero();
}

Tensor::Tensor(int batch, int channel, int height, int width) {
  dim.height(height);
  dim.width(width);
  dim.batch(batch);
  dim.channel(channel);
  this->data = std::vector<float>(dim.getDataLen());
  setZero();
}

float Tensor::getValue(unsigned int batch, unsigned int c, unsigned int h,
                       unsigned int w) {
  return this->data[batch * dim.channel() * dim.height() * dim.width() +
                    c * dim.height() * dim.width() + h * dim.width() + w];
}

void Tensor::setValue(unsigned int batch, unsigned int c, unsigned int h,
                      unsigned int w, float value) {
  this->data[batch * dim.channel() * dim.height() * dim.width() +
             c * dim.height() * dim.width() + h * dim.width() + w] = value;
}

Tensor::Tensor(std::vector<std::vector<float>> const &d) {

  dim.height(d.size());
  dim.width(d[0].size());
  this->data = std::vector<float>(dim.getDataLen());

  for (unsigned int j = 0; j < dim.height(); ++j)
    for (unsigned int k = 0; k < dim.width(); ++k)
      this->setValue(0, 0, j, k, d[j][k]);
}

Tensor::Tensor(std::vector<std::vector<std::vector<float>>> const &d) {
  dim.channel(d.size());
  dim.height(d[0].size());
  dim.width(d[0][0].size());
  this->data = std::vector<float>(dim.getDataLen());

  for (unsigned int j = 0; j < dim.channel(); ++j)
    for (unsigned int k = 0; k < dim.height(); ++k)
      for (unsigned int l = 0; l < dim.width(); ++l)
        this->setValue(0, j, k, l, d[j][k][l]);
}

Tensor::Tensor(
  std::vector<std::vector<std::vector<std::vector<float>>>> const &d) {

  dim.batch(d.size());
  dim.channel(d[0].size());
  dim.height(d[0][0].size());
  dim.width(d[0][0][0].size());
  this->data = std::vector<float>(dim.getDataLen());

  for (unsigned int i = 0; i < dim.batch(); ++i)
    for (unsigned int j = 0; j < dim.channel(); ++j)
      for (unsigned int k = 0; k < dim.height(); ++k)
        for (unsigned int l = 0; l < dim.width(); ++l)
          this->setValue(i, j, k, l, d[i][j][k][l]);
}

Tensor Tensor::multiply(float const &value) {
  Tensor result(dim);
#ifdef USE_BLAS
  memset(result.data.data(), 0, sizeof(float) * result.dim.getDataLen());
  cblas_saxpy(dim.getDataLen(), value, this->data.data(), 1, result.data.data(),
              1);
#else
  for (unsigned int k = 0; k < dim.getDataLen(); ++k) {
    result.data[k] = data[k] * value;
  }
#endif
  return result;
}

Tensor Tensor::divide(float const &value) {
  Tensor result(dim);
  if (value == 0.0) {
    throw std::runtime_error("Error: Divide by zero");
  }
#ifdef USE_BLAS
  memset(result.data.data(), 0, sizeof(float) * result.dim.getDataLen());
  cblas_saxpy(dim.getDataLen(), 1.0 / value, this->data.data(), 1,
              result.data.data(), 1);
#else
  for (unsigned int k = 0; k < dim.getDataLen(); ++k) {
    result.data[k] = data[k] / value;
  }
#endif
  return result;
}

Tensor Tensor::add(float const &value) {
  Tensor result(dim);
#ifdef USE_BLAS
  cblas_scopy(dim.getDataLen(), this->data.data(), 1, result.data.data(), 1);
  Tensor tmp(dim);
  for (unsigned int i = 0; i < tmp.dim.getDataLen(); ++i)
    tmp.data[i] = 1.0;
  cblas_saxpy(dim.getDataLen(), value, tmp.data.data(), 1, result.data.data(),
              1);
#else
  for (unsigned int k = 0; k < dim.getDataLen(); ++k) {
    result.data[k] = data[k] + value;
  }
#endif

  return result;
}

/**
 * @brief Add Tensor Element by Element without mem copy
 * @param[in] m Tensor to be added
 * #retval #ML_ERROR_NONE  Successful
 * #retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
 */
int Tensor::add_i(Tensor const &m) {
  if ((dim.height() != m.dim.height()) || (dim.width() != m.dim.width())) {
    return ML_ERROR_INVALID_PARAMETER;
  }

#ifdef USE_BLAS
  unsigned int size = dim.width() * dim.height() * dim.channel();
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      cblas_saxpy(size, 1.0, m.data.data(), 1, &(this->data.data()[k * size]),
                  1);
    }
  } else {
    cblas_saxpy(dim.getDataLen(), 1.0, m.data.data(), 1, this->data.data(), 1);
  }
#else
  unsigned int i, j, k;
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < m.dim.getFeatureLen(); ++i) {
        j = k * m.dim.getFeatureLen();
        this->data[j + i] += m.data[i];
      }
    }
  } else {
    for (k = 0; k < dim.getDataLen(); ++k) {
      this->data[k] = this->data[k] + m.data[k];
    }
  }
#endif

  return ML_ERROR_NONE;
}

Tensor Tensor::add(Tensor const &m) const {
  if ((dim.height() != m.dim.height()) || (dim.width() != m.dim.width())) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim);
  result.copy(*this);
  result.add_i(m);

  return result;
}

Tensor Tensor::subtract(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim);

#ifdef USE_BLAS
  cblas_scopy(dim.getDataLen(), this->data.data(), 1, result.data.data(), 1);

  unsigned int size =
    this->dim.channel() * this->dim.width() * this->dim.height();
  float alpha = -1.0;

  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      cblas_saxpy(size, alpha, m.data.data(), 1,
                  &(result.data.data()[k * size]), 1);
    }
  } else {
    assert(dim.batch() == m.dim.batch());
    cblas_saxpy(dim.getDataLen(), alpha, m.data.data(), 1, result.data.data(),
                1);
  }
#else
  unsigned int i, j, k, len;
  len = m.dim.getFeatureLen();
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < len; ++i) {
        j = k * len;
        result.data[j + i] = data[j + i] - m.data[i];
      }
    }
  } else {
    for (k = 0; k < m.dim.getDataLen(); ++k) {
      result.data[k] = data[k] - m.data[k];
    }
  }
#endif
  return result;
}

Tensor Tensor::subtract(float const &value) {
  Tensor result(dim);
#ifdef USE_BLAS
  cblas_scopy(dim.getDataLen(), this->data.data(), 1, result.data.data(), 1);
  Tensor tmp(dim);
  for (unsigned int i = 0; i < tmp.dim.getDataLen(); ++i)
    tmp.data[i] = -1.0;
  cblas_saxpy(dim.getDataLen(), value, tmp.data.data(), 1, result.data.data(),
              1);
#else
  for (unsigned int k = 0; k < dim.getDataLen(); ++k) {
    result.data[k] = data[k] - value;
  }
#endif

  return result;
}

Tensor Tensor::multiply(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim);

  int end = dim.getDataLen() / 4;
  int e = dim.getFeatureLen() / 4;
  int i;
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      int b = k * dim.getFeatureLen();
      for (i = 0; i < e * 4; i += 4) {
        result.data[b + i + 0] = this->data[b + i + 0] * m.data[i + 0];
        result.data[b + i + 1] = this->data[b + i + 1] * m.data[i + 1];
        result.data[b + i + 2] = this->data[b + i + 2] * m.data[i + 2];
        result.data[b + i + 3] = this->data[b + i + 3] * m.data[i + 3];
      }
      for (unsigned int j = i; j < dim.getFeatureLen(); j++)
        result.data[b + j] = this->data[b + j] * m.data[j];
    }
  } else {
    for (i = 0; i < end * 4; i += 4) {
      result.data[i + 0] = this->data[i + 0] * m.data[i + 0];
      result.data[i + 1] = this->data[i + 1] * m.data[i + 1];
      result.data[i + 2] = this->data[i + 2] * m.data[i + 2];
      result.data[i + 3] = this->data[i + 3] * m.data[i + 3];
    }
    for (unsigned int j = i; j < dim.getDataLen(); ++j)
      result.data[j] = this->data[j] * m.data[j];
  }

  return result;
}

Tensor Tensor::divide(Tensor const &m) const {
  if (dim.channel() != m.dim.channel() || dim.height() != m.dim.height() ||
      dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim.batch(), dim.channel(), dim.height(), dim.width());

  unsigned int end = dim.getDataLen() / 4;
  unsigned int e = dim.getFeatureLen() / 4;
  unsigned int i, j, k;

  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      unsigned int b = k * dim.getFeatureLen();
      for (i = 0; i < e * 4; i += 4) {
        result.data[b + i + 0] = this->data[b + i + 0] / m.data[i + 0];
        result.data[b + i + 1] = this->data[b + i + 1] / m.data[i + 1];
        result.data[b + i + 2] = this->data[b + i + 2] / m.data[i + 2];
        result.data[b + i + 3] = this->data[b + i + 3] / m.data[i + 3];
      }
      for (unsigned int j = i; j < dim.getFeatureLen(); ++j)
        result.data[b + j] = this->data[b + j] / m.data[j];
    }
  } else {
    for (i = 0; i < end * 4; i += 4) {
      result.data[i + 0] = this->data[i + 0] / m.data[i + 0];
      result.data[i + 1] = this->data[i + 1] / m.data[i + 1];
      result.data[i + 2] = this->data[i + 2] / m.data[i + 2];
      result.data[i + 3] = this->data[i + 3] / m.data[i + 3];
    }
    for (j = i; j < dim.getDataLen(); ++j)
      result.data[j] = this->data[j] / m.data[j];
  }

  return result;
}

/**
 * This is to sum the Tensor data according to the dim.batch().
 * Therefore the result has M(dim.batch(), 1, 1) dimension.
 */
Tensor Tensor::sum() const {
  unsigned int k;
  Tensor ret(dim.batch(), 1, 1, 1);
#ifdef USE_BLAS
  for (k = 0; k < dim.batch(); ++k)
    ret.data[k] = cblas_sasum(dim.getFeatureLen(),
                              &(data.data()[k * dim.getFeatureLen()]), 1);
#else
  unsigned int i;
  for (k = 0; k < dim.batch(); ++k) {

    unsigned int id = k * dim.getFeatureLen();
    ret.data[id] = 0.0;
    for (i = 0; i < dim.getFeatureLen(); ++i) {
      ret.data[id] += data[id + i];
    }
  }
#endif

  return ret;
}

Tensor Tensor::sum(int axis) const {
  Tensor ret;
  switch (axis) {
  case 0: {

    ret = Tensor(1, dim.channel(), dim.height(), dim.width());
    for (unsigned int l = 0; l < dim.channel(); ++l) {
      unsigned int L = l * dim.width() * dim.height();
      for (unsigned int i = 0; i < dim.height(); ++i) {
        unsigned int I = i * dim.width();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          for (unsigned int k = 0; k < dim.batch(); ++k) {
            unsigned int K = k * dim.getFeatureLen();
            ret.data[L + I + j] += data[K + L + I + j];
          }
        }
      }
    }
  } break;
  case 1: {
    ret = Tensor(dim.batch(), 1, dim.height(), dim.width());
    for (unsigned int l = 0; l < dim.batch(); ++l) {
      unsigned int L = dim.width() * dim.height() * l;
      unsigned int LL = l * dim.getFeatureLen();
      for (unsigned int j = 0; j < dim.height(); ++j) {
        unsigned int J = j * dim.width();
        for (unsigned int i = 0; i < dim.width(); ++i) {
          for (unsigned int k = 0; k < dim.channel(); ++k) {
            unsigned int K = k * dim.width() * dim.height();
            ret.data[(L + J + i)] += data[LL + K + J + i];
          }
        }
      }
    }
  } break;
  case 2: {
    ret = Tensor(dim.batch(), dim.channel(), 1, dim.width());
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      unsigned int K = k * dim.channel() * dim.width();
      unsigned int KK = k * dim.getFeatureLen();
      for (unsigned int l = 0; l < dim.channel(); ++l) {
        unsigned int L = l * dim.width();
        unsigned int LL = l * dim.width() * dim.height();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          for (unsigned int i = 0; i < dim.height(); ++i) {
            unsigned int I = i * dim.width();
            ret.data[K + L + j] += data[KK + LL + j + I];
          }
        }
      }
    }
  } break;
  case 3: {
    ret = Tensor(dim.batch(), dim.channel(), dim.height(), 1);
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      unsigned int K = k * dim.channel() * dim.height();
      unsigned int KK = k * dim.getFeatureLen();
      for (unsigned int l = 0; l < dim.channel(); ++l) {
        unsigned int L = l * dim.height();
        unsigned int LL = l * dim.height() * dim.width();
        for (unsigned int i = 0; i < dim.height(); ++i) {
          unsigned int II = i * dim.width();
          for (unsigned int j = 0; j < dim.width(); ++j) {
            ret.data[K + L + i] += data[KK + LL + II + j];
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
 * If the dim.batch() size of m is one, the it is reused for
 * every calculation along with dim.batch().
 * Currently dot function only supports the case which dim.channel() == 1.
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

#ifdef USE_BLAS
  float alpha_dgemm = 1.0;
  float beta_dgemm = 1.0;
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); k++) {
      unsigned int i = k * dim.width() * dim.height();
      unsigned int ii = k * dim.height() * m.dim.width();
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim.height(),
                  m.dim.width(), dim.width(), alpha_dgemm, &(data.data()[i]),
                  dim.width(), m.data.data(), m.dim.width(), beta_dgemm,
                  &(result.data.data()[ii]), m.dim.width());
    }
  } else {
    for (unsigned int k = 0; k < dim.batch(); k++) {
      unsigned int i = k * dim.width() * dim.height();
      unsigned int j = k * m.dim.width() * m.dim.height();
      unsigned int ii = k * dim.height() * m.dim.width();

      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, dim.height(),
                  m.dim.width(), dim.width(), alpha_dgemm, &(data.data()[i]),
                  dim.width(), &(m.data.data()[j]), m.dim.width(), beta_dgemm,
                  &(result.data.data()[ii]), m.dim.width());
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
      cudaMemcpy(d_A, &data.data()[i], size_A, cudaMemcpyHostToDevice);
      cudaMemcpy(d_B, m.data.data(), size_B, cudaMemcpyHostToDevice);
      cudaMalloc((void **)&d_C, size_C);

      {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle;

        (cublasCreate(&handle));

        (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.dim.width(),
                     dim.height(), dim.width(), &alpha, d_B, m.dim.width(), d_A,
                     dim.width(), &beta, d_C, m.dim.width()));

        (cudaMemcpy(&result.data.data()[ii], d_C, size_C,
                    cudaMemcpyDeviceToHost));
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
      (cudaMemcpy(d_A, &data.data()[i], size_A, cudaMemcpyHostToDevice));
      (cudaMemcpy(d_B, &m.data.data()[j], size_B, cudaMemcpyHostToDevice));
      (cudaMalloc((void **)&d_C, size_C));

      {
        const float alpha = 1.0f;
        const float beta = 0.0f;
        cublasHandle_t handle;

        (cublasCreate(&handle));

        (cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m.dim.width(),
                     dim.height(), dim.width(), &alpha, d_B, m.dim.width(), d_A,
                     dim.width(), &beta, d_C, m.dim.width()));

        (cudaMemcpy(&result.data.data()[ii], d_C, size_C,
                    cudaMemcpyDeviceToHost));
        (cublasDestroy(handle));
      }
    }
  }
#else
  float w = 0.0;
  unsigned int i, j, k, h;
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < dim.height(); ++i) {
        for (j = 0; j < m.dim.width(); ++j) {
          for (h = 0; h < dim.width(); ++h) {
            w += data[k * dim.height() * dim.width() + i * dim.width() + h] *
                 m.data[h * m.dim.width() + j];
          }
          result
            .data[k * dim.height() * m.dim.width() + i * m.dim.width() + j] = w;
          w = 0.0;
        }
      }
    }
  } else {
    for (k = 0; k < dim.batch(); k++) {
      for (i = 0; i < dim.height(); i++) {
        for (j = 0; j < m.dim.width(); j++) {
          for (h = 0; h < dim.width(); h++) {
            w +=
              data[k * dim.height() * dim.width() + i * dim.width() + h] *
              m.data[k * dim.width() * m.dim.width() + h * m.dim.width() + j];
          }
          result
            .data[k * dim.height() * m.dim.width() + i * m.dim.width() + j] = w;
          w = 0.0;
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

  inptr = data.data();
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

Tensor Tensor::apply(float (*function)(float)) const {
  Tensor result(dim.batch(), dim.channel(), dim.height(), dim.width());
  unsigned int i;

  for (i = 0; i < dim.getDataLen(); ++i)
    result.data[i] = (*function)(data[i]);

  return result;
}

Tensor Tensor::apply(Tensor (*function)(Tensor)) const {
  return (*function)(*this);
}

void Tensor::print(std::ostream &out) const {
  unsigned int i, j, k, l;
  std::stringstream ss;
  for (k = 0; k < dim.batch(); k++) {
    for (l = 0; l < dim.channel(); l++) {
      for (i = 0; i < dim.height(); i++) {
        for (j = 0; j < dim.width(); j++) {
          out << data[k * dim.getFeatureLen() + l * dim.width() * dim.height() +
                      i * dim.width() + j]
              << " ";
        }
        out << std::endl;
      }
      out << std::endl;
    }
    out << "-------" << std::endl;
  }
}

std::ostream &operator<<(std::ostream &out, Tensor const &m) {
  m.print(out);
  return out;
}

Tensor &Tensor::copy(const Tensor &from) {
  if (this != &from && from.dim.getDataLen() != 0) {
    dim.channel(from.dim.channel());
    dim.height(from.dim.height());
    dim.width(from.dim.width());
    dim.batch(from.dim.batch());
    if (this->data.empty()) {
      this->data.resize(from.data.size());
    }
#ifdef USE_BLAS
    cblas_scopy(dim.getDataLen(), from.data.data(), 1, this->data.data(), 1);
#else
    for (int i = 0; i < dim.getDataLen(); ++i)
      data[i] = from.data[i];
#endif
  }

  return *this;
}

/**
 * This generate one dimension vector has the every element in Tensor
 */
std::vector<float> Tensor::mat2vec() {
  std::vector<float> ret;

  for (unsigned int i = 0; i < dim.getDataLen(); i++)
    ret.push_back(data[i]);

  return ret;
}

void Tensor::save(std::ofstream &file) {
  for (unsigned int i = 0; i < dim.getDataLen(); i++)
    file.write((char *)&data[i], sizeof(float));
}

void Tensor::read(std::ifstream &file) {
  for (unsigned int i = 0; i < dim.getDataLen(); i++)
    file.read((char *)&data[i], sizeof(float));
}

/**
 * This calculates average value according to the dim.batch() direction.
 * That is the why it has (1, dim.height(), dim.width()) dimension.
 */
Tensor Tensor::average() const {
  if (dim.batch() == 1)
    return *this;

  Tensor result(1, dim.channel(), dim.height(), dim.width());

  result = this->sum(0);
  result.divide(dim.batch());

  return result;
}

void Tensor::setZero() {
  memset(this->data.data(), 0, sizeof(float) * dim.getDataLen());
}

int Tensor::argmax() {
  int index = 0;
  float maximum = 0.0;
  for (unsigned int i = 0; i < dim.getDataLen(); i++) {
    if (this->data[i] > maximum) {
      maximum = this->data[i];
      index = i;
    }
  }
  return index;
}

float Tensor::l2norm() const {
  float sum = 0.0;
  float tmp;
  unsigned int len = dim.getDataLen();

  for (unsigned int i = 0; i < len; i++) {
    tmp = this->data[i];
    sum += tmp * tmp;
  }

  return sqrt(sum);
}

Tensor Tensor::normalization() const {
  Tensor results(dim);
  float Min = 1000000.0;
  float Max = 0.0;

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    for (unsigned int l = 0; l < dim.channel(); ++l) {
      for (unsigned int i = 0; i < dim.height(); ++i) {
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int id = k * dim.getFeatureLen() +
                            l * dim.height() * dim.width() + i * dim.width() +
                            j;
          if (this->data[id] < Min)
            Min = this->data[id];
          if (this->data[id] > Max)
            Max = this->data[id];
        }
      }
    }
  }
  float dif = Max - Min;

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    for (unsigned int l = 0; l < dim.channel(); ++l) {
      for (unsigned int i = 0; i < dim.height(); ++i) {
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int id = k * dim.getFeatureLen() +
                            l * dim.height() * dim.width() + i * dim.width() +
                            j;
          results.data[id] = (this->data[id] - Min) / dif;
        }
      }
    }
  }
  return results;
}

Tensor Tensor::standardization() const {
  Tensor result(dim);

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    int K = k * dim.getFeatureLen();
    float mean;
    float mean_tmp = 0.0;
    float std_tmp = 0.0;
    float std_dev = 0.0;

    for (unsigned int l = 0; l < dim.channel(); ++l) {
      unsigned int L = K + l * dim.height() * dim.width();
      for (unsigned int i = 0; i < dim.height(); ++i) {
        unsigned int I = L + i * dim.width();
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int J = I + j;
          mean_tmp += this->data[J];
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
          std_tmp += (this->data[J] - mean) * (this->data[J] - mean);
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
          result.data[J] = (this->data[J] - mean) / std_dev;
        }
      }
    }
  }

  return result;
}
} /* namespace nntrainer */
