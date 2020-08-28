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
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__
#ifdef __cplusplus

#include <array>
#include <cmath>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <tensor_dim.h>

#define MAKE_SHARED_TENSOR(x) std::make_shared<nntrainer::Tensor>(x)

namespace nntrainer {

class LazyTensor;
/**
 * @class   Tensor Class for Calculation
 * @brief   Tensor Class for Calculation
 */
class Tensor {
public:
  Tensor(const TensorDim &d, const float *buf = nullptr);

  /**
   * @brief     Basic Constructor of Tensor
   */
  Tensor() : Tensor(TensorDim()){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] batch Batch of Tensor
   * @param[in] channel Channel of Tensor
   * @param[in] height Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(int batch, int channel, int height, int width) :
    Tensor(TensorDim(batch, channel, height, width)){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] channel Channel of Tensor
   * @param[in] height Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(int channel, int height, int width) :
    Tensor(1, channel, height, width){};

  /**
   * @brief     Constructor of Tensor with batch size one and channel size one
   * @param[in] height Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(int height, int width) : Tensor(1, 1, height, width){};

  /**
   * @brief     Constructor of Tensor with just width
   * @param[in] width Width of Tensor
   */
  Tensor(int width) : Tensor(1, 1, 1, width){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<float>>>> const &d);

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor
   */
  Tensor(std::vector<std::vector<std::vector<float>>> const &d) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   */
  Tensor(std::vector<std::vector<float>> const &d) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}){};

  /**
   *  @brief  Copy constructor of Tensor.
   *  @param[in] Tensor &
   */
  Tensor(const Tensor &rhs) = default;

  /**
   *  @brief  Move constructor of Tensor.
   *  @param[in] Tensor &&
   */
  Tensor(Tensor &&rhs) noexcept = default;

  /**
   * @brief  Copy assignment operator.
   * @param[in] rhs Tensor to be copied.
   */
  Tensor &operator=(const Tensor &rhs) = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Tensor to be moved.
   */
  Tensor &operator=(Tensor &&rhs) noexcept = default;

  void swap(Tensor &lhs, Tensor &rhs) noexcept;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator==(const Tensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator!=(const Tensor &rhs) const { return !(*this == rhs); }

  /**
   * @brief     return value at specific location
   * @param[in] batch batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  float getValue(unsigned int batch, unsigned int c, unsigned int h,
                 unsigned int w) const;

  /**
   * @brief     Multiply value element by element immediately
   * @param[in] value multiplier
   * @retval    #ML_ERROR_INVALID_PARAMETER Tensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int multiply_i(float const &value);

  /**
   * @brief     Multiply value element by element
   * @param[in] value multiplier
   * @retval    Calculated Tensor
   */
  Tensor multiply(float const &value);

  /**
   * @brief     Divide value element by element immediately
   * @param[in] value divisor
   * @retval    #ML_ERROR_INVALID_PARAMETER Tensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int divide_i(float const &value);

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @retval    Calculated Tensor
   */
  Tensor divide(float const &value);

  /**
   * @brief Add Tensor Element by Element without mem copy
   * @param[in] m Tensor to be added
   * @param[out] alpha Values to be scaled
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(Tensor const &m, float const alpha = 1);

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] m Tensor to be added
   * @retval    Calculated Tensor
   */
  Tensor add(Tensor const &m, float const alpha = 1) const;

  /**
   * @brief Add Tensor Element immediately to target tensor without mem copy
   * @param[in] value value to be added
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(float const &value);

  /**
   * @brief     Add value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated Tensor
   */
  Tensor add(float const &value);

  /**
   * @brief     memcpyless version of subtract
   * @param[in] m Tensor to be subtracted
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int subtract_i(Tensor const &m);

  /**
   * @brief     Substract Tensor Element by Element
   * @param[in] m Tensor to be subtracted
   * @retval    Calculated Tensor
   */
  Tensor subtract(Tensor const &m) const;

  /**
   * @brief     memcpyless version of subtract
   * @param[in] value value to subtract
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int subtract_i(float const &value);

  /**
   * @brief     subtract value Element by Element
   * @param[in] value value to be subtracted
   * @retval    Calculated Tensor
   */
  Tensor subtract(float const &value);

  /**
   * @brief     Multiply Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @retval    #ML_ERROR_NONE successful
   */
  int multiply_i(Tensor const &m);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @retval    Calculated Tensor
   */
  Tensor multiply(Tensor const &m) const;

  /**
   * @brief     divide Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @retval    #ML_ERROR_NONE successful
   */
  int divide_i(Tensor const &m);

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
   * @param[in] direction to transpose ex) 0:2:1
   * @retval    Calculated Tensor
   */
  Tensor transpose(std::string direction) const;

  /**
   * @brief     sum all the Tensor elements according to the batch
   * @retval    Calculated Tensor(batch, 1, 1, 1)
   */
  Tensor sum_by_batch();

  /**
   * @brief     sum all the Tensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated Tensor
   */
  Tensor sum(int axis, float alpha = 1.0) const;

  /**
   * @brief     Averaging the Tensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @retval    Calculated Tensor
   */
  Tensor average(int axis) const;

  /**
   * @brief     Averaging the Tensor elements by all axis
   * @retval    Calculated Tensor
   */
  Tensor average() const;

  /**
   * @brief     Anchor a starting point to defer following evaluation
   * @retval    LazyTensor class that can be used with run();
   */
  LazyTensor chain() const;

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
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  Tensor apply(std::function<float(float)> f) const;

  /**
   * @brief     Apply function to Tensor
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  Tensor apply(std::function<Tensor(Tensor)> f) const;

  Tensor apply_i(std::function<int(const Tensor &)> f) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @retval    Tensor
   */
  void print(std::ostream &out) const;

  /**
   * @brief     Get length of current _data
   * @retval    unsigned int length of the current _data
   */
  unsigned int length() const { return dim.getDataLen(); }

  /**
   * @brief     Get size of the data
   * @retval    size_t Size in bytes
   */
  size_t getSize() const { return length() * sizeof(float); }

  /**
   * @brief     Set the element value
   * @param[in] batch batch location
   * @param[in] c channel location
   * @param[in] i height location
   * @param[in] j width location
   * @param[in] value value to be stored
   */
  void setValue(unsigned int batch, unsigned int c, unsigned int i,
                unsigned int j, float value);

  /**
   * @brief     Fill the Tensor elements with value
   * @param[in] value value to be stored
   */
  void setValue(float value);

  /**
   * @brief     Set the tensor with random normal distribution
   * @param[in] mean mean of the distribution
   * @param[in] std standard deviation of the distribution
   */
  void setRandNormal(float mean = 0.0f, float std = 0.05f);

  /**
   * @brief     Set the tensor with random uniform distribution
   * @param[in] min minimum value for the distribution
   * @param[in] max maximum value for the distribution
   */
  void setRandUniform(float min = -0.05f, float max = 0.05f);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   */
  void copy(const Tensor &from);

  /**
   * @brief     Convient wrapper for inplace copy of @a this.
   * @retval    Copied version of this
   */
  Tensor clone() const;

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
   * @retval    unsigned int argument index
   */
  unsigned int argmax() const;

  /**
   * @brief     return a copy of the Tensor Dim
   * @retval    TensorDim
   */
  TensorDim getDim() const { return TensorDim(dim); }

  /**
   * @brief     return Tensor Dim for a given axis
   * @retval    dimension
   */
  unsigned int getTensorDim(unsigned int axis);

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  unsigned int batch() const { return dim.batch(); }

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  unsigned int channel() const { return dim.channel(); }

  /**
   * @brief     return Tensor height size
   * @retval    height size
   */
  unsigned int height() const { return dim.height(); }

  /**
   * @brief     return Tensor batch size
   * @retval    width size
   */
  unsigned int width() const { return dim.width(); }

  /**
   * @brief     return Data pointer of Tensor
   * @retval    float pointer
   */
  float *getData() { return data.get(); }

  const float *getData() const { return data.get(); }

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  float *getAddress(unsigned int i);

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  const float *getAddress(unsigned int i) const;

  /**
   * @brief     set Tensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  void reshape(const TensorDim &d);

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  const std::array<int, MAXDIM> getStrides() const noexcept { return strides; }

  /**
   * @brief     perform saxpy operation.
   */
  static void saxpy(const unsigned int N, const float alpha, const float *X,
                    const int incX, float *Y, const int incY);

private:
  /**
   * @brief Get linear index given the n-d index
   */
  inline unsigned int getIndex(unsigned int b, unsigned int c, unsigned int h,
                               unsigned int w) const {
    return (b * dim.getFeatureLen() + c * dim.height() * dim.width() +
            h * dim.width() + w);
  }

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   * @param[in] m Tensor
   * @param[in] op operation to apply
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int operator_i(Tensor const &m,
                 std::function<float(float const, float const)> op);

  /**< handle the data as a std::shared_ptr<float> type */
  TensorDim dim;
  std::array<int, MAXDIM> strides;
  bool is_contiguous;
  std::shared_ptr<float> data;

  static constexpr float min_limits = std::numeric_limits<float>::min();
  static constexpr float max_limits = std::numeric_limits<float>::max();
  template <typename T> void setDist(T dist);
  static constexpr float epsilon = 1e-5;
};

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, Tensor const &m);

typedef std::shared_ptr<Tensor> sharedTensor;

typedef std::shared_ptr<const Tensor> sharedConstTensor;

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_H__ */
