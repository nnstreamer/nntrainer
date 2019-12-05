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
 * @file	matrix.cpp
 * @date	04 December 2019
 * @brief	This is Matrix class for calculation
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "include/matrix.h"
#include <assert.h>
#include <stdio.h>
#include <sstream>

Matrix::Matrix(int height, int width) {
  this->height = height;
  this->width = width;
  this->batch = 1;
  this->dim = 2;
  this->data.push_back(std::vector<std::vector<double>>(height, std::vector<double>(width)));
}

Matrix::Matrix(int batch, int height, int width) {
  this->height = height;
  this->width = width;
  this->batch = batch;
  this->dim = 3;

  for (int i = 0; i < batch; i++) {
    this->data.push_back(std::vector<std::vector<double>>(height, std::vector<double>(width)));
  }
}

Matrix::Matrix(std::vector<std::vector<double>> const &data) {
  assert(data.size() != 0);
  this->height = data.size();
  this->width = data[0].size();
  this->batch = 1;
  this->dim = 2;
  this->data.push_back(data);
}

Matrix::Matrix(std::vector<std::vector<std::vector<double>>> const &data) {
  assert(data.size() != 0 && data[0].size() != 0);
  this->batch = data.size();
  this->height = data[0].size();
  this->width = data[0][0].size();
  this->dim = 3;
  this->data = data;
}

Matrix Matrix::multiply(double const &value) {
  Matrix result(batch, height, width);
  int i, j, k;

  for (k = 0; k < batch; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        result.data[k][i][j] = data[k][i][j] * value;
      }
    }
  }

  return result;
}

Matrix Matrix::divide(double const &value) {
  Matrix result(batch, height, width);
  int i, j, k;

  for (k = 0; k < batch; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        result.data[k][i][j] = data[k][i][j] / value;
      }
    }
  }

  return result;
}

Matrix Matrix::add(double const &value) {
  Matrix result(batch, height, width);
  int i, j, k;

  for (k = 0; k < batch; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        result.data[k][i][j] = data[k][i][j] + value;
      }
    }
  }

  return result;
}

Matrix Matrix::add(Matrix const &m) const {
  assert(height == m.height && width == m.width);

  Matrix result(batch, height, width);
  int i, j, k;
  if (m.batch == 1) {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] + m.data[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] + m.data[k][i][j];
        }
      }
    }
  }
  return result;
}

Matrix Matrix::subtract(Matrix const &m) const {
  assert(height == m.height && width == m.width);
  Matrix result(batch, height, width);
  int i, j, k;

  if (m.batch == 1) {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] - m.data[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] - m.data[k][i][j];
        }
      }
    }
  }
  return result;
}

Matrix Matrix::multiply(Matrix const &m) const {
  assert(height == m.height && width == m.width);
  Matrix result(batch, height, width);

  int i, j, k;

  if (m.batch == 1) {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] * m.data[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] * m.data[k][i][j];
        }
      }
    }
  }

  return result;
}

Matrix Matrix::divide(Matrix const &m) const {
  assert(height == m.height && width == m.width);
  Matrix result(batch, height, width);

  int i, j, k;

  if (m.batch == 1) {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] / m.data[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.data[k][i][j] = data[k][i][j] / m.data[k][i][j];
        }
      }
    }
  }

  return result;
}

/**
 * This is to sum the Matrix data according to the batch.
 * Therefore the result has M(batch, 1, 1) dimension.
 */
Matrix Matrix::sum() const {
  int i, j, k;
  Matrix ret(batch, 1, 1);

  for (k = 0; k < batch; k++) {
    ret.data[k][0][0] = 0.0;
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        ret.data[k][0][0] += data[k][i][j];
      }
    }
  }

  return ret;
}

/**
 * If the batch sizeo of m is one, the it is reused for
 * every calculation along with batch
 */
Matrix Matrix::dot(Matrix const &m) const {
  assert(width == m.height);
  int i, j, h, k, mwidth = m.width;
  double w = 0;

  Matrix result(batch, height, mwidth);
  if (m.batch == 1) {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < mwidth; j++) {
          for (h = 0; h < width; h++) {
            w += data[k][i][h] * m.data[0][h][j];
          }
          result.data[k][i][j] = w;
          w = 0;
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < mwidth; j++) {
          for (h = 0; h < width; h++) {
            w += data[k][i][h] * m.data[k][h][j];
          }
          result.data[k][i][j] = w;
          w = 0;
        }
      }
    }
  }

  return result;
}

Matrix Matrix::transpose() const {
  Matrix result(batch, width, height);
  int i, j, k;
  for (k = 0; k < batch; k++) {
    for (i = 0; i < width; i++) {
      for (j = 0; j < height; j++) {
        result.data[k][i][j] = data[k][j][i];
      }
    }
  }
  return result;
}

Matrix Matrix::applyFunction(double (*function)(double)) const {
  Matrix result(batch, height, width);
  int i, j, k;

  for (k = 0; k < batch; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        result.data[k][i][j] = (*function)(data[k][i][j]);
      }
    }
  }
  return result;
}

void Matrix::print(std::ostream &out) const {
  int i, j, k;
  std::stringstream ss;
  for (k = 0; k < batch; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        out << data[k][i][j] << " ";
      }
      out << std::endl;
    }
    out << std::endl;
  }
}

std::ostream &operator<<(std::ostream &out, Matrix const &m) {
  m.print(out);
  return out;
}

Matrix &Matrix::copy(const Matrix &from) {
  if (this != &from && from.data.size() != 0) {
    height = from.height;
    width = from.width;
    batch = from.batch;
    for (int k = 0; k < batch; k++) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          data[k][i][j] = from.data[k][i][j];
        }
      }
    }
  }
  return *this;
}

/**
 * This generate one dimension vector has the every element in Matrix
 */
std::vector<double> Matrix::Mat2Vec() {
  std::vector<double> ret;
  for (int k = 0; k < batch; k++)
    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        ret.push_back(data[k][i][j]);

  return ret;
}

void Matrix::save(std::ofstream &file) {
  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        file.write((char *)&data[k][i][j], sizeof(double));
      }
    }
  }
}

void Matrix::read(std::ifstream &file) {
  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        file.read((char *)&data[k][i][j], sizeof(double));
      }
    }
  }
}

/**
 * This calculates average value according to the batch direction.
 * That is the why it has (1, height, width) dimension.
 */
Matrix Matrix::average() const {
  if (batch == 1)
    return *this;

  Matrix result(1, height, width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.data[0][i][j] = 0.0;
      for (int k = 0; k < batch; k++) {
        result.data[0][i][j] += data[k][i][j];
      }
      result.data[0][i][j] = result.data[0][i][j] / (double)batch;
    }
  }
  return result;
}

void Matrix::setZero() {
  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        this->data[k][i][j] = 0.0;
      }
    }
  }
}

Matrix Matrix::softmax() const {
  Matrix result(batch, height, width);
  Matrix divisor(batch, height, 1);

  divisor.setZero();

  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        divisor.data[k][i][0] += exp(this->data[k][i][j]);
      }
    }
  }

  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        result.data[k][i][j] = exp(this->data[k][i][j]) / divisor.data[k][i][0];
      }
    }
  }
  return result;
}

void Matrix::setValue(int batch, int height, int width, double value) { this->data[batch][height][width] = value; }
