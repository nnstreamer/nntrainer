#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

class Matrix {
public:
  Matrix();
  Matrix(int height, int width);
  Matrix(int batch, int height, int width);
  Matrix(std::vector<std::vector<double>> const &array);
  Matrix(std::vector<std::vector<std::vector<double>>> const &array);

  Matrix multiply(double const &value);
  Matrix divide(double const &value);
  Matrix add(Matrix const &m) const;
  Matrix add(double const &value);
  Matrix subtract(Matrix const &m) const;
  Matrix multiply(Matrix const &m) const;
  Matrix divide(Matrix const &m) const;

  Matrix dot(Matrix const &m) const;
  Matrix transpose() const;
  Matrix sum() const;
  Matrix average() const;
  Matrix softmax() const;
  void setZero();

  std::vector<double> Mat2Vec();

  Matrix applyFunction(double (*function)(double)) const;

  void print(std::ostream &flux) const;

  int getWidth() { return width; };
  int getHeight() { return height; };
  int getBatch() { return batch; };
  void setValue(int batch, int i, int j, double value);

  Matrix &copy(Matrix const &from);

  void save(std::ofstream &file);
  void read(std::ifstream &file);

private:
  std::vector<std::vector<std::vector<double>>> array;
  int height;
  int width;
  int batch;
  int dim;
};

std::ostream &operator<<(std::ostream &flux, Matrix const &m);

#endif
