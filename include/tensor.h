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
 * @file	tensor.h
 * @date	04 December 2019
 * @brief	This is Tensor class for calculation
 * @see		https://github.sec.samsung.net/jijoong-moon/Transfer-Learning.git
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__

#ifdef USE_BLAS
extern "C" {
#include <cblas.h>
}
#endif

#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include <regex>

#define MAXDIM 4

/**
 * @Namespace   Namespace of Tensor
 * @brief       Namespace for Tensor
 */
namespace Tensors {

class TensorDim {
 public:
  TensorDim() {
    for (int i = 0; i < MAXDIM; ++i) {
      Dim[i] = 1;
    }
  }
  ~TensorDim(){};
  unsigned int batch() { return Dim[0]; };
  unsigned int channel() { return Dim[1]; };
  unsigned int height() { return Dim[2]; };
  unsigned int width() { return Dim[3]; };

  void batch(unsigned int b) { Dim[0] = b; };
  void channel(unsigned int c) { Dim[1] = c; };
  void height(unsigned int h) { Dim[2] = h; };
  void width(unsigned int w) { Dim[3] = w; };

  unsigned int *getDim() { return Dim; }

  void setTensorDim(std::string input_shape);

 private:
  unsigned int Dim[4];
};

/**
 * @class   Tensor Class for Calculation
 * @brief   Tensor Class for Calculation
 */
class Tensor {
  
 public:
  /**
   * @brief     Constructor of Tensor
   */
  Tensor() : height(0), width(0), batch(0), dim(0), len(0){};

  /**
   * @brief     Constructor of Tensor with batch size one
   * @param[in] heihgt Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(int height, int width);

  /**
   * @brief     Constructor of Tensor
   * @param[in] batch Batch of Tensor
   * @param[in] heihgt Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(int batch, int height, int width);

  /**
   * @brief   Constructor of Tensor
   * @param[in] d data for the Tensor with batch size one
   */
  Tensor(std::vector<std::vector<float>> const &d);

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor
   */
  Tensor(std::vector<std::vector<std::vector<float>>> const &d);

  /**
   * @brief     return value at specific location
   * @param[in] batch batch location
   * @param[in] h height location
   * @param[in] w width location
   */
  float getValue(int batch, int h, int w);

  /**
   * @brief     Multiply value element by element
   * @param[in] value multiplier
   * @retval    Calculated Tensor
   */
  Tensor multiply(float const &value);

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @retval    Calculated Tensor
   */
  Tensor divide(float const &value);

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] m Tensor to be added
   * @retval    Calculated Tensor
   */
  Tensor add(Tensor const &m) const;

  /**
   * @brief     Add value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated Tensor
   */
  Tensor add(float const &value);

  /**
   * @brief     Substract Tensor Element by Element
   * @param[in] m Tensor to be added
   * @retval    Calculated Tensor
   */
  Tensor subtract(Tensor const &m) const;

  /**
   * @brief     subtract value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated Tensor
   */
  Tensor subtract(float const &value);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @retval    Calculated Tensor
   */
  Tensor multiply(Tensor const &m) const;

  /**
   * @brief     Divide Tensor Element by Element
   * @param[in] m Divisor Tensor
   * @retval    Calculated Tensor
   */
  Tensor divide(Tensor const &m) const;

  /**
   * @brief     Dot Product of Tensor ( equal MxM )
   * @param[in] m Tensor
   * @retval    Calculated Tensor
   */
  Tensor dot(Tensor const &m) const;

  /**
   * @brief     Transpose Tensor
   * @retval    Calculated Tensor
   */
  Tensor transpose() const;

  /**
   * @brief     sum all the Tensor elements according to the batch
   * @retval    Calculated Tensor(batch, 1, 1)
   */
  Tensor sum() const;

  /**
   * @brief     sum all the Tensor elements according to the axis
   * @retval    Calculated Tensor
   */
  Tensor sum(int axis) const;

  /**
   * @brief     Averaging the Tensor elements according to the batch
   * @retval    Calculated Tensor(1, height, width)
   */
  Tensor average() const;

  /**
   * @brief     Softmax the Tensor elements
   * @retval    Calculated Tensor
   */
  Tensor softmax() const;

  /**
   * @brief     l2norm the Tensor elements
   * @retval    Calculated l2norm
   */
  float l2norm() const;

  /**
   * @brief     Normalize the Tensor elements
   * @retval    Calculated Tensor
   */
  Tensor normalization() const;

  /**
   * @brief     Standardize the Tensor elements
   * @retval    Calculated Tensor
   */
  Tensor standardization() const;

  /**
   * @brief     Fill the Tensor elements with zero
   */
  void setZero();

  /**
   * @brief     Reduce Rank ( Tensor to Vector )
   * @retval    Saved vector
   */
  std::vector<float> Mat2Vec();

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  Tensor applyFunction(float (*function)(float)) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @retval    Tensor
   */
  void print(std::ostream &out) const;

  /**
   * @brief     Get Width of Tensor
   * @retval    int Width
   */
  int getWidth() { return width; };

  /**
   * @brief     Get Height of Tensor
   * @retval    int Height
   */
  int getHeight() { return height; };

  /**
   * @brief     Get Batch of Tensor
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
  void setValue(int batch, int i, int j, float value);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be Copyed
   * @retval    Matix
   */
  Tensor &copy(Tensor const &from);

  /**
   * @brief     Save the Tensor into file
   * @param[in] file output file stream
   */
  void save(std::ofstream &file);

  /**
   * @brief     Read the Tensor from file
   * @param[in] file input file stream
   */
  void read(std::ifstream &file);

  /**
   * @brief     return argument index which value is max
   * @retval    int argument index
   */
  int argmax();

 private:
  /**< handle the data as a std::vector type */
  std::vector<float> data;
  int height;
  int width;
  int batch;
  int dim;
  int len;
};

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, Tensor const &m);
}

#endif
