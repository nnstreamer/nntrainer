#include "include/matrix.h"
#include <assert.h>
#include <sstream>
#include <stdio.h>

Matrix::Matrix() {}

Matrix::Matrix(int height, int width) {
  this->height = height;
  this->width = width;
  this->batch = 1;
  this->dim = 2;
  this->array.push_back(
      std::vector<std::vector<double>>(height, std::vector<double>(width)));
}

Matrix::Matrix(int batch, int height, int width) {
  this->height = height;
  this->width = width;
  this->batch = batch;
  this->dim = 3;
  for (int i = 0; i < batch; i++) {
    this->array.push_back(
        std::vector<std::vector<double>>(height, std::vector<double>(width)));
  }
}

Matrix::Matrix(std::vector<std::vector<double>> const &array) {
  assert(array.size() != 0);
  this->height = array.size();
  this->width = array[0].size();
  this->batch = 1;
  this->dim = 2;
  this->array.push_back(array);
}

Matrix::Matrix(std::vector<std::vector<std::vector<double>>> const &array) {
  assert(array.size() != 0 && array[0].size() != 0);
  this->batch = array.size();
  this->height = array[0].size();
  this->width = array[0][0].size();
  this->dim = 3;
  this->array = array;
}

Matrix Matrix::multiply(double const &value) {
  Matrix result(batch, height, width);
  int i, j, k;

  for (k = 0; k < batch; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        result.array[k][i][j] = array[k][i][j] * value;
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
        result.array[k][i][j] = array[k][i][j] / value;
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
        result.array[k][i][j] = array[k][i][j] + value;
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
          result.array[k][i][j] = array[k][i][j] + m.array[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.array[k][i][j] = array[k][i][j] + m.array[k][i][j];
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
          result.array[k][i][j] = array[k][i][j] - m.array[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.array[k][i][j] = array[k][i][j] - m.array[k][i][j];
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
          result.array[k][i][j] = array[k][i][j] * m.array[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.array[k][i][j] = array[k][i][j] * m.array[k][i][j];
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
          result.array[k][i][j] = array[k][i][j] / m.array[0][i][j];
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
          result.array[k][i][j] = array[k][i][j] / m.array[k][i][j];
        }
      }
    }
  }

  return result;
}

Matrix Matrix::sum() const {
  int i, j, k;
  Matrix ret(batch, 1, 1);

  for (k = 0; k < batch; k++) {
    ret.array[k][0][0] = 0.0;
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        ret.array[k][0][0] += array[k][i][j];
      }
    }
  }

  return ret;
}

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
            w += array[k][i][h] * m.array[0][h][j];
          }
          result.array[k][i][j] = w;
          w = 0;
        }
      }
    }
  } else {
    for (k = 0; k < batch; k++) {
      for (i = 0; i < height; i++) {
        for (j = 0; j < mwidth; j++) {
          for (h = 0; h < width; h++) {
            w += array[k][i][h] * m.array[k][h][j];
          }
          result.array[k][i][j] = w;
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
        result.array[k][i][j] = array[k][j][i];
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
        result.array[k][i][j] = (*function)(array[k][i][j]);
      }
    }
  }

  return result;
}

void Matrix::print(std::ostream &flux) const {
  int i, j, k, l;
  int maxLength[batch][width];
  std::stringstream ss;

  for (k = 0; k < batch; k++) {
    for (i = 0; i < width; i++) {
      maxLength[k][i] = 0;
    }
  }

  for (k = 0; k < batch; k++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        ss << array[k][i][j];
        if (maxLength[k][j] < (int)(ss.str().size())) {
          maxLength[k][j] = ss.str().size();
        }
        ss.str(std::string());
      }
    }
  }

  for (l = 0; l < batch; l++) {
    for (i = 0; i < height; i++) {
      for (j = 0; j < width; j++) {
        flux << array[l][i][j];
        ss << array[l][i][j];

        for (int k = 0; k < (int)(maxLength[l][j] - ss.str().size() + 1); k++) {
          flux << " ";
        }
        ss.str(std::string());
      }
      flux << std::endl;
    }
  }
}

std::ostream &operator<<(std::ostream &flux, Matrix const &m) {
  m.print(flux);
  return flux;
}

Matrix &Matrix::copy(const Matrix &from) {
  if (this != &from) {
    height = from.height;
    width = from.width;
    batch = from.batch;
    for (int k = 0; k < batch; k++) {
      for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
          array[k][i][j] = from.array[k][i][j];
        }
      }
    }
  }
  return *this;
}

std::vector<double> Matrix::Mat2Vec() {
  std::vector<double> ret;
  for (int k = 0; k < batch; k++)
    for (int i = 0; i < height; i++)
      for (int j = 0; j < width; j++)
        ret.push_back(array[k][i][j]);

  return ret;
}

void Matrix::save(std::ofstream &file) {
  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        file.write((char *)&array[k][i][j], sizeof(double));
      }
    }
  }
}

void Matrix::read(std::ifstream &file) {
  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        file.read((char *)&array[k][i][j], sizeof(double));
      }
    }
  }
}

Matrix Matrix::average() const {
  if (batch == 1)
    return *this;

  Matrix result(1, height, width);
  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      result.array[0][i][j] = 0.0;
      for (int k = 0; k < batch; k++) {
        result.array[0][i][j] += array[k][i][j];
      }
      result.array[0][i][j] = result.array[0][i][j] / (double)batch;
    }
  }
  return result;
}

void Matrix::setZero() {
  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        this->array[k][i][j] = 0.0;
      }
    }
  }
}

Matrix Matrix::softmax() const {
  Matrix result(batch, height, width);
  Matrix mother(batch, height, 1);

  mother.setZero();

  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        mother.array[k][i][0] += exp(this->array[k][i][j]);
      }
    }
  }

  for (int k = 0; k < batch; k++) {
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        result.array[k][i][j] =
            exp(this->array[k][i][j]) / mother.array[k][i][0];
      }
    }
  }
  return result;
}

void Matrix::setValue(int batch, int height, int width, double value) {
  this->array[batch][height][width] = value;
}
