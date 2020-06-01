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
#include <sstream>
#include <stdio.h>
#include <tensor.h>

#ifdef USE_CUBLAS
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

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

Tensor::Tensor(int batch, int height, int width) {
  dim.height(height);
  dim.width(width);
  dim.batch(batch);
  this->data = std::vector<float>(dim.getDataLen());
  setZero();
}

float Tensor::getValue(unsigned int batch, unsigned int h, unsigned int w) {
  return this->data[batch * dim.height() * dim.width() + h * dim.width() + w];
}

void Tensor::setValue(unsigned int batch, unsigned int h, unsigned int w,
                      float value) {
  this->data[batch * dim.height() * dim.width() + h * dim.width() + w] = value;
}

Tensor::Tensor(std::vector<std::vector<float>> const &d) {

  dim.height(d.size());
  dim.width(d[0].size());
  this->data = std::vector<float>(dim.getDataLen());

  for (unsigned int j = 0; j < dim.height(); ++j)
    for (unsigned int k = 0; k < dim.width(); ++k)
      this->setValue(0, j, k, d[j][k]);
}

Tensor::Tensor(std::vector<std::vector<std::vector<float>>> const &d) {

  dim.batch(d.size());
  dim.height(d[0].size());
  dim.width(d[0][0].size());
  this->data = std::vector<float>(dim.getDataLen());

  for (unsigned int i = 0; i < dim.batch(); ++i)
    for (unsigned int j = 0; j < dim.height(); ++j)
      for (unsigned int k = 0; k < dim.width(); ++k)
        this->setValue(i, j, k, d[i][j][k]);
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

Tensor Tensor::add(Tensor const &m) const {
  if ((dim.height() != m.dim.height()) || (dim.width() != m.dim.width())) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim);
#ifdef USE_BLAS
  cblas_scopy(dim.getDataLen(), this->data.data(), 1, result.data.data(), 1);
  unsigned int size = dim.width() * dim.height();
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      cblas_saxpy(size, 1.0, m.data.data(), 1, &(result.data.data()[k * size]),
                  1);
    }
  } else {
    cblas_saxpy(dim.getDataLen(), 1.0, m.data.data(), 1, result.data.data(), 1);
  }
#else
  unsigned int i, j, k;
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < m.dim.getDataLen(); ++i) {
        j = k * m.dim.getDataLen();
        result.data[j + i] = data[j + i] + m.data[i];
      }
    }
  } else {
    for (k = 0; k < dim.getDataLen(); ++k) {
      result.data[k] = data[k] + m.data[k];
    }
  }
#endif

  return result;
}

Tensor Tensor::subtract(Tensor const &m) const {
  if (dim.height() != m.dim.height() || dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim);

#ifdef USE_BLAS
  cblas_scopy(dim.getDataLen(), this->data.data(), 1, result.data.data(), 1);
  unsigned int size = this->dim.width() * this->dim.height();
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
  len = m.dim.getDataLen();
  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      for (i = 0; i < len; ++i) {
        j = k * len;
        result.data[j + i] = data[j + i] - m.data[i];
      }
    }
  } else {
    for (k = 0; k < len; ++k) {
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
  if (dim.height() != m.dim.height() || dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim);

  int end = dim.getDataLen() / 4;
  int e = dim.width() * dim.height() / 4;
  int i;
  if (m.dim.batch() == 1) {
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      int b = k * dim.width() * dim.height();
      for (i = 0; i < e * 4; i += 4) {
        result.data[b + i + 0] = this->data[b + i + 0] * m.data[i + 0];
        result.data[b + i + 1] = this->data[b + i + 1] * m.data[i + 1];
        result.data[b + i + 2] = this->data[b + i + 2] * m.data[i + 2];
        result.data[b + i + 3] = this->data[b + i + 3] * m.data[i + 3];
      }
      for (unsigned int j = i; j < dim.width() * dim.height(); j++)
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
  if (dim.height() != m.dim.height() || dim.width() != m.dim.width()) {
    throw std::runtime_error("Error: Dimension must be equal each other");
  }

  Tensor result(dim.batch(), dim.height(), dim.width());

  unsigned int end = dim.getDataLen() / 4;
  unsigned int e = dim.width() * dim.height() / 4;
  unsigned int i, j, k;

  if (m.dim.batch() == 1) {
    for (k = 0; k < dim.batch(); ++k) {
      unsigned int b = k * dim.width() * dim.height();
      for (i = 0; i < e * 4; i += 4) {
        result.data[b + i + 0] = this->data[b + i + 0] / m.data[i + 0];
        result.data[b + i + 1] = this->data[b + i + 1] / m.data[i + 1];
        result.data[b + i + 2] = this->data[b + i + 2] / m.data[i + 2];
        result.data[b + i + 3] = this->data[b + i + 3] / m.data[i + 3];
      }
      for (unsigned int j = i; j < dim.width() * dim.height(); ++j)
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
  Tensor ret(dim.batch(), 1, 1);
#ifdef USE_BLAS
  for (k = 0; k < dim.batch(); ++k)
    ret.data[k] =
      cblas_sasum(dim.width() * dim.height(),
                  &(data.data()[k * dim.width() * dim.height()]), 1);
#else
  unsigned int i;
  for (k = 0; k < dim.batch(); ++k) {
    unsigned int id = k * dim.width() * dim.height();
    ret.data[id] = 0.0;
    for (i = 0; i < dim.height() * dim.width(); ++i) {
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
    ret = Tensor(1, dim.height(), dim.width());
    for (unsigned int i = 0; i < dim.height(); ++i) {
      unsigned int I = i * dim.width();
      for (unsigned int j = 0; j < dim.width(); ++j) {
        for (unsigned int k = 0; k < dim.batch(); ++k) {
          unsigned int K = k * dim.width() * dim.height();
          ret.data[I + j] += data[K + I + j];
        }
      }
    }
  } break;
  case 1: {
    ret = Tensor(dim.batch(), 1, dim.width());
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      unsigned int K = k * dim.width();
      for (unsigned int j = 0; j < dim.width(); ++j) {
        for (unsigned int i = 0; i < dim.height(); ++i) {
          unsigned int I = i * dim.width() * dim.batch();
          ret.data[K + j] += data[K + I + j];
        }
      }
    }
  } break;
  case 2: {
    ret = Tensor(dim.batch(), dim.height(), 1);
    for (unsigned int k = 0; k < dim.batch(); ++k) {
      unsigned int K = k * dim.height();
      for (unsigned int i = 0; i < dim.height(); ++i) {
        for (unsigned int j = 0; j < dim.width(); ++j) {
          unsigned int J = j * dim.height() * dim.batch();
          ret.data[K + i] += data[K + J + i];
        }
      }
    }
  } break;
  default:
    throw std::out_of_range("Error: Cannot exceede 2");
    break;
  }
  return ret;
}

/**
 * If the dim.batch() sizeo of m is one, the it is reused for
 * every calculation along with dim.batch()
 */
Tensor Tensor::dot(Tensor const &m) const {
  if (dim.width() != m.dim.height()) {
    throw std::runtime_error("Error dim.width() != m.dim.height()");
  }
  int mwidth = m.dim.width();
  Tensor result(dim.batch(), dim.height(), mwidth);

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

Tensor Tensor::transpose() const {
  Tensor result(dim.batch(), dim.width(), dim.height());
  unsigned int i, j, k;
  for (k = 0; k < dim.batch(); ++k) {
    unsigned int b = k * dim.width() * dim.height();
    for (i = 0; i < dim.width(); ++i) {
      for (j = 0; j < dim.height(); ++j) {
        result.data[b + i * dim.height() + j] = data[b + j * dim.width() + i];
      }
    }
  }
  return result;
}

Tensor Tensor::apply(float (*function)(float)) const {
  Tensor result(dim.batch(), dim.height(), dim.width());
  unsigned int i;

  for (i = 0; i < dim.getDataLen(); ++i)
    result.data[i] = (*function)(data[i]);

  return result;
}

Tensor Tensor::apply(Tensor (*function)(Tensor)) const {
  return (*function)(*this);
}

void Tensor::print(std::ostream &out) const {
  unsigned int i, j, k;
  std::stringstream ss;
  for (k = 0; k < dim.batch(); k++) {
    for (i = 0; i < dim.height(); i++) {
      for (j = 0; j < dim.width(); j++) {
        out << data[k * dim.width() * dim.height() + i * dim.width() + j]
            << " ";
      }
      out << std::endl;
    }
    out << std::endl;
  }
}

std::ostream &operator<<(std::ostream &out, Tensor const &m) {
  m.print(out);
  return out;
}

Tensor &Tensor::copy(const Tensor &from) {
  if (this != &from && from.dim.getDataLen() != 0) {
    dim.height(from.dim.height());
    dim.width(from.dim.width());
    dim.batch(from.dim.batch());
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

  Tensor result(1, dim.height(), dim.width());
  for (unsigned int i = 0; i < dim.height(); i++) {
    for (unsigned int j = 0; j < dim.width(); j++) {
      result.data[i * dim.width() + j] = 0.0;
      for (unsigned int k = 0; k < dim.batch(); k++) {
        result.data[i * dim.width() + j] +=
          data[k * dim.width() * dim.height() + i * dim.width() + j];
      }
      result.data[i * dim.width() + j] =
        result.data[i * dim.width() + j] / (float)dim.batch();
    }
  }
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
  for (unsigned int i = 0; i < dim.getDataLen(); i++) {
    sum += this->data[i] * this->data[i];
  }

  return sqrt(sum);
}

Tensor Tensor::normalization() const {
  Tensor results(dim.batch(), dim.height(), dim.width());
  float Min = 1000000.0;
  float Max = 0.0;

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    for (unsigned int i = 0; i < dim.height(); ++i) {
      for (unsigned int j = 0; j < dim.width(); ++j) {
        unsigned int id = k * dim.height() * dim.width() + i * dim.width() + j;
        if (this->data[id] < Min)
          Min = this->data[id];
        if (this->data[id] > Max)
          Max = this->data[id];
      }
    }
  }
  float dif = Max - Min;

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    for (unsigned int i = 0; i < dim.height(); ++i) {
      for (unsigned int j = 0; j < dim.width(); ++j) {
        unsigned int id = k * dim.height() * dim.width() + i * dim.width() + j;
        results.data[id] = (this->data[id] - Min) / dif;
      }
    }
  }

  return results;
}

Tensor Tensor::standardization() const {
  Tensor result(dim.batch(), dim.height(), dim.width());

  for (unsigned int k = 0; k < dim.batch(); ++k) {
    int K = k * dim.height() * dim.width();
    float mean;
    float mean_tmp = 0.0;
    float std_tmp = 0.0;
    float std_dev = 0.0;

    for (unsigned int i = 0; i < dim.height(); ++i) {
      unsigned int I = K + i * dim.width();
      for (unsigned int j = 0; j < dim.width(); ++j) {
        unsigned int J = I + j;
        mean_tmp += this->data[J];
      }
    }

    mean = mean_tmp / (this->dim.width() * this->dim.height());

    for (unsigned int i = 0; i < dim.height(); ++i) {
      unsigned int I = K + i * dim.width();
      for (unsigned int j = 0; j < dim.width(); ++j) {
        unsigned int J = I + j;
        std_tmp += (this->data[J] - mean) * (this->data[J] - mean);
      }
    }

    std_dev = sqrt(std_tmp) / (this->dim.height() * this->dim.width());

    for (unsigned int i = 0; i < dim.height(); ++i) {
      unsigned int I = K + i * dim.width();
      for (unsigned int j = 0; j < dim.width(); ++j) {
        unsigned int J = I + j;
        result.data[J] = (this->data[J] - mean) / std_dev;
      }
    }
  }

  return result;
}
} /* namespace nntrainer */
