#include "matrix.h"
#include <assert.h>
#include <sstream>
#include <stdio.h>

Matrix::Matrix() {}

Matrix::Matrix(int height, int width) {
  this->height = height;
  this->width = width;
  this->array =
      std::vector<std::vector<double>>(height, std::vector<double>(width));
}

Matrix::Matrix(std::vector<std::vector<double>> const &array) {
  assert(array.size() != 0);
  this->height = array.size();
  this->width = array[0].size();
  this->array = array;
}

Matrix Matrix::multiply(double const &value) {
  Matrix result(height, width);
  int i, j;

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      result.array[i][j] = array[i][j] * value;
    }
  }

  return result;
}

Matrix Matrix::add(Matrix const &m) const {
  assert(height = m.height &&width = m.width);

  Matrix result(height, width);
  int i, j;
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      result.array[i][j] = array[i][j] + m.array[i][j];
    }
  }
  return result;
}

Matrix Matrix::subtract(Matrix const &m) const {
  assert(height = m.height &&width = m.width);
  Matrix result(height, width);
  int i, j;

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      result.array[i][j] = array[i][j] - m.array[i][j];
    }
  }

  return result;
}

Matrix Matrix::multiply(Matrix const &m) const {
  assert(height = m.height &&width = m.width);
  Matrix result(height, width);

  int i, j;

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      result.array[i][j] = array[i][j] * m.array[i][j];
    }
  }

  return result;
}

double Matrix::sum() const {
  int i, j;
  double ret = 0.0;
  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      ret += array[i][j];
    }
  }
  return ret;
}

Matrix Matrix::dot(Matrix const &m) const {
  assert(width = m.height);
  int i, j, h, mwidth = m.width;
  double w = 0;

  Matrix result(height, mwidth);

  for (i = 0; i < height; i++) {
    for (j = 0; j < mwidth; j++) {
      for (h = 0; h < width; h++) {
        w += array[i][h] * m.array[h][j];
      }
      result.array[i][j] = w;
      w = 0;
    }
  }

  return result;
}

Matrix Matrix::transpose() const {
  Matrix result(width, height);
  int i, j;
  for (i = 0; i < width; i++) {
    for (j = 0; j < height; j++) {
      result.array[i][j] = array[j][i];
    }
  }
  return result;
}

Matrix Matrix::applyFunction(double (*function)(double)) const {
  Matrix result(height, width);
  int i, j;

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      result.array[i][j] = (*function)(array[i][j]);
    }
  }

  return result;
}

void Matrix::print(std::ostream &flux) const {
  int i, j;
  int maxLength[width];
  std::stringstream ss;

  for (i = 0; i < width; i++) {
    maxLength[i] = 0;
  }

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      ss << array[i][j];
      if (maxLength[j] < ss.str().size()) {
        maxLength[j] = ss.str().size();
      }
      ss.str(std::string());
    }
  }

  for (i = 0; i < height; i++) {
    for (j = 0; j < width; j++) {
      flux << array[i][j];
      ss << array[i][j];

      for (int k = 0; k < maxLength[j] - ss.str().size() + 1; k++) {
        flux << " ";
      }
      ss.str(std::string());
    }
    flux << std::endl;
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
    printf("%d %d\n", height, width);
    for (int i = 0; i < height; i++) {
      for (int j = 0; j < width; j++) {
        array[i][j] = from.array[i][j];
      }
    }
  }
  return *this;
}
