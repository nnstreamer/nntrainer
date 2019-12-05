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
 * @file	matrix.h
 * @date	04 December 2019
 * @brief	This is Matrix class for calculation
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <cmath>
#include <fstream>
#include <iostream>
#include <vector>

/**
 * @class   Matrix Class for Calculation
 * @brief   Matrix Class for Calculation
 */
class Matrix {
 public:
  /**
   * @brief     Constructor of Matrix
   */
  Matrix(){};
  
  /**
   * @brief     Constructor of Matrix with batch size one
   * @param[in] heihgt Height of Matrix
   * @param[in] width Width of Matrix
   */  
  Matrix(int height, int width);

  /**
   * @brief     Constructor of Matrix
   * @param[in] batch Batch of Matrix
   * @param[in] heihgt Height of Matrix
   * @param[in] width Width of Matrix
   */
  Matrix(int batch, int height, int width);

  /**
   * @brief   Constructor of Matrix
   * @param[in] data data for the Matrix with batch size one
   */
  Matrix(std::vector<std::vector<double>> const &data);

  /**
   * @brief     Constructor of Matrix
   * @param[in] data data for the Matrix
   */
  Matrix(std::vector<std::vector<std::vector<double>>> const &data);

  /**
   * @brief     Multiply value element by element
   * @param[in] value multiplier
   * @retval    Calculated Matrix
   */  
  Matrix multiply(double const &value);

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @retval    Calculated Matrix
   */  
  Matrix divide(double const &value);

  /**
   * @brief     Add Matrix Element by Element
   * @param[in] m Matrix to be added
   * @retval    Calculated Matrix
   */  
  Matrix add(Matrix const &m) const;

  /**
   * @brief     Add value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated Matrix
   */  
  Matrix add(double const &value);

  /**
   * @brief     Substract Matrix Element by Element
   * @param[in] m Matrix to be added
   * @retval    Calculated Matrix
   */  
  Matrix subtract(Matrix const &m) const;

  /**
   * @brief     Multiply Matrix Element by Element ( Not the MxM )
   * @param[in] m Matrix to be multiplied
   * @retval    Calculated Matrix
   */  
  Matrix multiply(Matrix const &m) const;

  /**
   * @brief     Divide Matrix Element by Element
   * @param[in] m Divisor Matrix
   * @retval    Calculated Matrix
   */  
  Matrix divide(Matrix const &m) const;

  /**
   * @brief     Dot Product of Matrix ( equal MxM )
   * @param[in] m Matrix
   * @retval    Calculated Matrix
   */  
  Matrix dot(Matrix const &m) const;

  /**
   * @brief     Transpose Matrix
   * @retval    Calculated Matrix
   */  
  Matrix transpose() const;

  /**
   * @brief     sum all the Matrix elements according to the batch
   * @retval    Calculated Matrix(batch, 1, 1)
   */  
  Matrix sum() const;

  /**
   * @brief     Averaging the Matrix elements according to the batch
   * @retval    Calculated Matrix(1, height, width)
   */  
  Matrix average() const;

  /**
   * @brief     Softmax the Matrix elements
   * @retval    Calculated Matrix
   */  
  Matrix softmax() const;

  /**
   * @brief     Fill the Matrix elements with zero
   */  
  void setZero();

  /**
   * @brief     Reduce Rank ( Matrix to Vector )
   * @retval    Saved vector
   */  
  std::vector<double> Mat2Vec();
  
  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    Matrix
   */  
  Matrix applyFunction(double (*function)(double)) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @retval    Matrix
   */    
  void print(std::ostream &out) const;

  /**
   * @brief     Get Width of Matrix
   * @retval    int Width
   */  
  int getWidth() { return width; };

  /**
   * @brief     Get Height of Matrix
   * @retval    int Height
   */  
  int getHeight() { return height; };

  /**
   * @brief     Get Batch of Matrix
   * @retval    int Batch
   */  
  int getBatch() { return batch; };

  /**
   * @brief     Set the elelemnt value
   * @param[in] batch batch location
   * @param[in] i height location
   * @param[in] j width location
   * @param[in] value value to be stored
   */  
  void setValue(int batch, int i, int j, double value);

  /**
   * @brief     Copy the Matrix
   * @param[in] from Matrix to be Copyed
   * @retval    Matix
   */  
  Matrix &copy(Matrix const &from);
  
  /**
   * @brief     Save the Matrix into file
   * @param[in] file output file stream
   */  
  void save(std::ofstream &file);

  /**
   * @brief     Read the Matrix from file
   * @param[in] file input file stream
   */  
  void read(std::ifstream &file);

private:
/**< handle the data as a std::vector type */
  std::vector<std::vector<std::vector<double>>> data;
  int height;
  int width;
  int batch;
  int dim;
};

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, Matrix const &m);

#endif
