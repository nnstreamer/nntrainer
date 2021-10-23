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
 * @todo deprecate new tensor allocation for out of place operations.
 */

#ifndef __TENSOR_H__
#define __TENSOR_H__
#ifdef __cplusplus

#include <array>
#include <functional>
#include <memory>
#include <vector>

#include <tensor_dim.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

#define MAKE_SHARED_TENSOR(...) std::make_shared<nntrainer::Tensor>(__VA_ARGS__)

namespace nntrainer {

using TensorDim = ml::train::TensorDim;

class LazyTensor;
class SrcSharedTensor;

/**
 * @class   Tensor Class for Calculation
 * @brief   Tensor Class for Calculation
 */
class Tensor {
public:
  /**
   * @brief     Enumeration of Weight Initialization Type
   * @todo      support intialization from file
   */
  enum class Initializer {
    ZEROS,          /** Zero initialization */
    ONES,           /** One initialization */
    LECUN_NORMAL,   /** LeCun normal initialization */
    LECUN_UNIFORM,  /** uniform initialization */
    XAVIER_NORMAL,  /** Xavier normal initialization */
    XAVIER_UNIFORM, /** Xavier uniform initialization */
    HE_NORMAL,      /** He normal initialization */
    HE_UNIFORM,     /** He uniform initialization */
    NONE            /** No initialization */
  };

  /**
   * @brief     Basic Constructor of Tensor
   */
  Tensor(std::string name_ = "") :
    dim(TensorDim()),
    strides(dim.computeStrides()),
    contiguous(true),
    initializer(Initializer::NONE),
    name(name_),
    data(nullptr),
    src_tensor() {}

  /**
   * @brief     Constructor of Tensor with dimension, possibly lazily
   * @param d Tensor dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  Tensor(const TensorDim &d, bool alloc_now,
         Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief     Constructor of Tensor with dimension/buf
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @note Memory for this tensor is instantaneously allocated
   */
  Tensor(const TensorDim &d, const float *buf = nullptr);

  /**
   * @brief     Constructor of Tensor
   * @param[in] batch Batch of Tensor
   * @param[in] channel Channel of Tensor
   * @param[in] height Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(unsigned int batch, unsigned int channel, unsigned int height,
         unsigned int width) :
    Tensor(TensorDim(batch, channel, height, width)){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] channel Channel of Tensor
   * @param[in] height Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(unsigned int channel, unsigned int height, unsigned int width) :
    Tensor(1, channel, height, width){};

  /**
   * @brief     Constructor of Tensor with batch size one and channel size one
   * @param[in] height Height of Tensor
   * @param[in] width Width of Tensor
   */
  Tensor(unsigned int height, unsigned int width) :
    Tensor(1, 1, height, width){};

  /**
   * @brief     Constructor of Tensor with just width
   * @param[in] width Width of Tensor
   */
  explicit Tensor(unsigned int width) : Tensor(1, 1, 1, width){};

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

  /**
   * @brief Construct a new Tensor object from a buffer
   * This will not copy buffer to a new tensor but directly uses it
   *
   * @param buf buffer
   * @param bytes buffer size in bytes
   * @param d tensor dim
   * @param offset offset to be used from current
   * @return Tensor object
   * @throws std::invalid_argument if buf is null
   */
  static Tensor Map(float *buf, unsigned int bytes, const TensorDim &d,
                    int offset = 0);

  /**
   * @brief Construct a new Tensor object from a buffer
   * This will shared the buf
   *
   * @param buf buffer
   * @param size buffer size in bytes
   * @param d tensor dim
   * @param offset offset to be used
   * @return Tensor object
   * @throws std::invalid_argument if buf is null
   */
  static Tensor Map(std::shared_ptr<float> buf, unsigned int size,
                    const TensorDim &d, int offset = 0);

  friend void swap(Tensor &lhs, Tensor &rhs) noexcept {
    std::swap(lhs.dim, rhs.dim);
    std::swap(lhs.strides, rhs.strides);
    std::swap(lhs.contiguous, rhs.contiguous);
    std::swap(lhs.initializer, rhs.initializer);
    std::swap(lhs.data, rhs.data);
    std::swap(lhs.name, rhs.name);
  }

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
   * @brief    Allocate memory for this tensor
   */
  void allocate();

  /**
   * @brief    Deallocate memory for this tensor
   * @note     This will not necessary free the memory as tensors share memory
   */
  void deallocate() { data = nullptr; }

  /**
   * @brief    Check if the tensor has memory allocated/assigned/associated
   */
  bool isAllocated() const { return data != nullptr; }

  /**
   * @brief     return value at specific location
   * @param[in] batch batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  float getValue(unsigned int batch, unsigned int c, unsigned int h,
                 unsigned int w) const noexcept {
    return getData()[getIndex(batch, c, h, w)];
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  float getValue(unsigned int idx) const noexcept { return getData()[idx]; }

  /**
   * @brief Get the Value thinking that it is padded
   * for example, for the tensor (virtually padded) below,
   * getValue(0, 0, 2, 2, 1, 1, .0f) will return 5
   * padding available for height and width axis for now
   * 0 0 0 0 0
   * 0 1 2 3 0
   * 0 4 5 6 0
   * 0 7 8 9 0
   * 0 0 0 0 0
   * @param b batch index
   * @param c channel index
   * @param h height index
   * @param w width index
   * @param ph padding height
   * @param pw padding width
   * @return float value
   */
  float getValuePaddedVirtual(unsigned int b, unsigned int c, unsigned int h,
                              unsigned int w, unsigned int ph, unsigned int pw,
                              float pad_value = 0) const EXCEPT_WHEN_DEBUG {
#if DEBUG
    unsigned int padded_h = 2 * ph + h;
    unsigned int padded_w = 2 * pw + w;
    if (h > padded_h && w > padded_w) {
      throw std::out_of_range(
        "[Tensor::getValuePadded] trying to access out of range");
    }
#endif

    if (ph <= h && h < ph + height() && pw <= w && w < pw + width()) {
      return getValue(b, c, h - ph, w - pw);
    }

    return pad_value;
  }

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
  Tensor multiply(float const &value) const;

  /**
   * @brief     multiply value element by element
   * @param[in] value multiplier
   * @param[out] out out tensor to store the result
   * @retval    Calculated Tensor
   */
  Tensor &multiply(float const &value, Tensor &out) const;

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
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @retval    Calculated Tensor
   */
  Tensor &multiply(Tensor const &m, Tensor &output) const;

  /**
   * @brief     Multiply Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply_i
   */
  int multiply_i_strided(Tensor const &m);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  Tensor multiply_strided(Tensor const &m) const;

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  Tensor &multiply_strided(Tensor const &m, Tensor &output) const;

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
  Tensor divide(float const &value) const;

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @param[out] out out parameter to store the result
   * @retval    Calculated Tensor
   */
  Tensor &divide(float const &value, Tensor &out) const;

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
   * @brief     divide Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @retval    Calculated Tensor
   */
  Tensor &divide(Tensor const &m, Tensor &output) const;

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
  Tensor add(float const &value) const;

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] value value to be added
   * @param[out] out Tensor to save output without allocating new memory
   * @retval    Calculated Tensor
   */
  Tensor &add(float const &value, Tensor &out) const;

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
   * @brief     Add Tensor Element by Element
   * @param[in] m Tensor to be added
   * @param[out] m Tensor to be out
   * @retval    Calculated Tensor
   */
  Tensor &add(Tensor const &m, Tensor &out, float const alpha = 1) const;

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
  Tensor subtract(float const &value) const;

  /**
   * @brief     Subtract Tensor Element by Element
   * @param[in] value value to be added
   * @param[out] out Tensor to save output without allocating new memory
   * @retval    Calculated Tensor
   */
  Tensor &subtract(float const &value, Tensor &out) const;

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
   * @brief     Subtract Tensor Element by Element
   * @param[in] m Tensor to be added
   * @param[out] m Tensor to be out
   * @retval    Calculated Tensor
   */
  Tensor &subtract(Tensor const &m, Tensor &out) const;

  /**
   * @brief Tensor power elementwise
   *
   * @param exponent exponent
   * @return int ML_ERROR_NONE if successful
   */
  int pow_i(float exponent);

  /**
   * @brief    Tensor power Element by Element
   * @param[in] exponent exponent
   * @retval Calculated Tensor
   */
  Tensor pow(float exponent) const;

  /**
   * @brief    Tensor power Element by Element
   * @param[in] exponent exponent
   * @param[out] out out to store the result
   * @retval Calculated Tensor
   */
  Tensor &pow(float exponent, Tensor &out) const;

  /**
   * @brief     Dot Product of Tensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and second-last
   * dimension of passed tensor m.
   * @param[in] m Tensor
   * @param[in] trans Transpose
   * @param[in] trans_m Transpose m
   * @retval    Calculated Tensor
   */
  Tensor dot(Tensor const &m, bool trans = false, bool trans_m = false) const;

  /**
   * @brief     Dot Product of Tensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and second-last
   * dimension of passed tensor m.
   * @param[in] m Tensor
   * @param[in] output output Tensor
   * @param[in] trans Transpose
   * @param[in] trans_m Transpose m
   * @param[in] beta beta
   * @retval    Calculated Tensor
   */
  Tensor &dot(Tensor const &m, Tensor &output, bool trans = false,
              bool trans_m = false, float beta = 0.0f) const;

  /**
   * @brief     Transpose Tensor
   * @param[in] direction to transpose ex) 0:2:1
   * @retval    Calculated Tensor
   */
  Tensor transpose(const std::string &direction) const;

  /**
   * @brief Transpose Tensor
   * @param direction to transpose ex) 0:2:1
   * @param[out] Tensor to save to, dimension is always reshaped.
   * @retval Tensor& reference to the out
   */
  Tensor &transpose(const std::string &direction, Tensor &out) const;

  /**
   * @brief Calculate Drop Out Mask : x * 1.0/(1.0-rate)
   * @param dropout drop out rate
   * @retval Tensor& reference of drop out mask
   */
  Tensor dropout_mask(float dropout) const;

  /**
   * @brief     sum all the Tensor elements according to the batch
   * @retval    Calculated Tensor(batch, 1, 1, 1)
   */
  Tensor sum_by_batch() const;

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
  Tensor sum(unsigned int axis, float alpha = 1.0) const;

  /**
   * @brief     sum all the Tensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[out] output output tensor
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated Tensor
   */
  Tensor &sum(unsigned int axis, Tensor &output, float alpha = 1.0) const;

  /**
   * @brief sum all the Tensor by multiple axes
   *
   * @param axes axes to sum along
   * @param alpha Scale the sum by this value
   * @return Tensor
   */
  Tensor sum(const std::vector<unsigned int> &axes, float alpha = 1.0) const;

  /**
   * @brief sum all the Tensor by multiple axes
   *
   * @param axes axes to sum along
   * @param[out] output output tensor
   * @param alpha Scale the sum by this value
   * @return Tensor
   */
  Tensor &sum(const std::vector<unsigned int> &axes, Tensor &output,
              float alpha = 1.0) const;

  /**
   * @brief     Averaging the Tensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @retval    Calculated Tensor
   */
  Tensor average(unsigned int axis) const;
  /**
   * @brief     Averaging the Tensor elements according to the axis
   *
   * @retval    Calculated Tensor
   */
  Tensor &average(unsigned int axis, Tensor &output) const;

  /**
   * @brief average all the Tensor by multiple axes
   *
   * @param axes axes to sum along
   * @return Tensor
   */
  Tensor average(const std::vector<unsigned int> &axes) const;

  /**
   * @brief average all the Tensor by multiple axes
   *
   * @param axes axes to sum along
   * @param output output tensor
   * @return Tensor
   */
  Tensor &average(const std::vector<unsigned int> &axes, Tensor &output) const;

  /**
   * @brief     Averaging the Tensor elements by all axis
   * @retval    Calculated Tensor
   */
  Tensor average() const;

  /**
   * @brief     Averaging the Tensor elements by all axis
   * @retval    Calculated Tensor
   */
  Tensor &average(Tensor &output) const;

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
  Tensor &normalization(Tensor &output) const;

  /**
   * @brief     Standardize the Tensor elements
   * @retval    Calculated Tensor
   */
  Tensor &standardization(Tensor &output) const;

  /**
   * @brief     Normalize the Tensor elements in-place
   * @retval    Calculated Tensor
   */
  void normalization_i();

  /**
   * @brief     Standardize the Tensor elements in-place
   * @retval    Calculated Tensor
   */
  void standardization_i();

  /**
   * @brief Apply instantly to the element
   *
   * @param f function to apply
   * @return int ML_ERROR_NONE if successful
   */
  int apply_i(std::function<float(float)> f);

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  Tensor apply(std::function<float(float)> f) const;

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    Tensor
   */
  Tensor &apply(std::function<float(float)> f, Tensor &output) const;

  /**
   * @brief     Apply function to Tensor
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  Tensor apply(std::function<Tensor(Tensor)> f) const;

  /**
   * @brief     Apply function to Tensor
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    Tensor
   */
  Tensor &apply(std::function<Tensor &(Tensor, Tensor &)> f,
                Tensor &output) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @retval    Tensor
   */
  void print(std::ostream &out) const;

  /**
   * @brief     Get size of current tensor
   * @retval    unsigned int size of the current tensor
   */
  unsigned int size() const { return dim.getDataLen(); }

  /**
   * @brief     Get if the tensor is empty
   * @retval    true if the tensor is empty
   */
  bool empty() const { return size() == 0; }

  /**
   * @brief     Get size of the data in bytes
   * @retval    size_t Size in bytes
   */
  size_t bytes() const { return size() * sizeof(float); }

  /**
   * @brief     Set the element value
   * @param[in] batch batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   * @param[in] value value to be stored
   */
  void setValue(unsigned int batch, unsigned int c, unsigned int h,
                unsigned int w, float value) noexcept {
    data.get()[getIndex(batch, c, h, w)] = value;
  }

  /**
   * @brief     Set the element value
   * @param[in] offset offset from start location
   * @param[in] value value to be stored
   *
   * @todo      This is a temporary workout. Remove this once multiple datatypes
   * are supported.
   */
  void setValueInt(unsigned int offset, int value) noexcept {
    int *data_int = (int *)data.get();
    data_int[offset] = value;
  }

  /**
   * @brief     Get int interpretable tensor
   *
   * @todo      This is a temporary workout. Remove this once multiple datatypes
   * are supported.
   */
  template <typename T> const int *getData() const { return (int *)data.get(); }

  /**
   * @brief     Fill the Tensor elements with value
   * @param[in] value value to be stored
   */
  void setValue(float value);

  /**
   * @brief     Fill the Tensor elements with zero
   */
  void setZero();

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
   * @brief     Initialize the memory of the given tensor
   */
  void initialize();

  /**
   * @brief     Initialize the memory of the given tensor
   * @param     init Initiailizer to use for the initialization
   */
  void initialize(Initializer init) {
    initializer = init;
    initialize();
  }

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   *
   * @note copy can reshape the tensor to match the shape
   */
  void copy(const Tensor &from);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   */
  void copyData(const Tensor &from);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   */
  void copy_with_stride(const Tensor &from);

  /**
   * @brief Get slice of the tensor, sliced by batch
   * @param[in] offset offset in batch to start the slice
   * @param[in] size size of the slice
   * @retval slice of this tensor
   * @note This function provides a slice of this tensor, and does not create a
   * copy
   */
  Tensor getBatchSlice(unsigned int offset, unsigned int size) const;

  /**
   * @brief Get new tensor which shares memory with current tensor but different
   * shape
   *
   * @param dim new dimension to be set for this tensor
   * @param offset offset to be used from the start of the data in elements
   * @note The new tensor will share the same data as the current tensor but
   * can have different size.
   * @note New size added with offset must be less than the size of the original
   * tensor.
   */
  Tensor getSharedDataTensor(const TensorDim dim, unsigned int offset,
                             bool reset_stride = true,
                             const std::string &name_ = "") const;

  /**
   * @brief make this tensor share memory with given tensor
   *
   * @param src Source tensor whose memory is to be shared
   * @param offset offset to be used from the start of the data in bytes
   * @note This tensor will share the same data as the current tensor but
   * can have different size.
   * @note This tensor's size added with offset must be less than the size of
   * the source tensor.
   * @note The stride of the source tensor and this tensor must be same.
   */
  void makeSharedDataTensor(const Tensor &src, unsigned int offset = 0);

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
   * @brief     return argument index which value is max by batch
   * @retval    unsigned int argument index
   */
  std::vector<unsigned int> argmax() const;

  /**
   * @brief     return max of the absolute values of the tensor
   * @retval    maximum absolute value
   */
  float max_abs() const;

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
   * @brief     update batch size for this tensor
   * @param     batch size
   * @note      The batchsize of src_tensor need not be related with this
   * tensor's batch size
   *
   * @note      The memory for this tensor will re-allocated/re-assigned if the
   * updated batch size is different than the current batch size.
   *
   * @note      If this tensor is/was the src_tensor for some other, then
   * reduction in batch size can make the dependent tensors allocate fail due to
   * memory smaller. Caller must handle this in their own end.
   *
   * @note      If this tensor is re-allocated, then the memory might not be
   * immediately freed as the tensor already dependeing on this tensor also
   * share the same memory. So, the peak memory consumption in worst case can
   * reach the total memory requirements of a model with old batchsize and the
   * new batch size. It is recommended to first deallocate all the tensors,
   * updateBatch and then allocate again to avoid such issues.
   */
  void updateBatch(unsigned int batch) {
    if (dim.batch() == batch) {
      return;
    }

    if (isAllocated())
      throw std::invalid_argument(
        "Cannot update batch for an allocated tensor");
    dim.batch(batch);
  }

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  template <typename T = float> T *getData() { return (T *)data.get(); }

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  template <typename T = float> const T *getData() const {
    return (T *)data.get();
  }

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
   * @brief    get address of n-d data
   */
  float *getAddress(unsigned int b, unsigned int c, unsigned int h,
                    unsigned int w) {
    return getAddress(getIndex(b, c, h, w));
  }

  /**
   * @brief    get address of n-d data
   */
  const float *getAddress(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const {
    return getAddress(getIndex(b, c, h, w));
  }

  /**
   * @brief     set Tensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  void reshape(const TensorDim &d);

  /**
   * @brief fill tensor data with current value,
   * if dimension is not exactly same, it is a hard error in this function
   * so, only stride is overriden to @a this
   *
   * @param from Tensor to fill the data from
   * @param allocate if unallocated, allocate with from.getDim()
   * @throws std::invalid_argument if dimension and stride does not match
   */
  void fill(const Tensor &from, bool allocate = false);

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  const std::array<unsigned int, TensorDim::MAXDIM> getStrides() const
    noexcept {
    return strides;
  }
  /**
   * @brief Get linear index given the n-d index
   */
  inline unsigned int getIndex(unsigned int b, unsigned int c, unsigned int h,
                               unsigned int w) const noexcept {
    return (b * strides[0] + c * strides[1] + h * strides[2] + w * strides[3]);
  }

  /**
   * @brief   Get name of the tensor
   *
   * @return name of the tensor
   */
  void setName(const std::string &name_) { name = name_; }

  /**
   * @brief   Get name of the tensor
   *
   * @return name of the tensor
   */
  const std::string &getName() const { return name; }

  /**
   * @brief Set the memory buffer for the tensor
   *
   * @param buf the memory buffer
   * @param init intialize the buffer
   */
  void setData(const void *buf, bool init = false) {
    if (buf) {
      data = std::shared_ptr<float>((float *)buf, [](void *) {});
      if (init)
        initialize();
    } else {
      data = nullptr;
    }
  }

  /**
   * @brief Get initializer for the tensor
   *
   * @return initializer of the tensor
   */
  Tensor::Initializer getInitializer() const { return initializer; }
  static constexpr float epsilon = 1e-5;

private:
  /**< handle the data as a std::shared_ptr<float> type */
  TensorDim dim;
  std::array<unsigned int, TensorDim::MAXDIM> strides;
  bool contiguous;
  Tensor::Initializer initializer;
  std::string name; /**< name of the tensor */

  std::shared_ptr<float> data;

  /**<
   * When using shared_data with tensor, this stores the ptr of the source
   * tensor which handles the full memory. If tensor data is already allocated,
   * this does not affect the tensor. If the tensor data is not allocated, and
   * src_ptr is valid, this tensor will use the memory allocated by the src_ptr
   */
  std::shared_ptr<SrcSharedTensor> src_tensor;

  struct BroadcastInfo;

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   * @param[in] m Tensor
   * @param[in] v_func vectorized function to apply
   * @param e broadcast info.
   * @param cur_axis current axis. pass default when calling outside.
   * @param offset offset for this.  pass default when calling outside.
   * @param m_offset offset for m.  pass default when calling outside.
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void
  apply_broadcast_util(Tensor const &m,
                       std::function<void(const BroadcastInfo &e, const float *,
                                          const float *, float *)>
                         v_func,
                       Tensor &output, const BroadcastInfo &e,
                       int cur_axis = -1, unsigned int offset = 0,
                       unsigned int m_offset = 0) const;

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   *
   * @param[in] m Tensor
   * @param[in] v_func vectorized function to apply
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void apply_broadcast(Tensor const &m,
                       std::function<void(const BroadcastInfo &e, const float *,
                                          const float *, float *)>
                         v_func,
                       Tensor &output) const;

  /**
   * @brief compute Loop info for broadcasting and vectorization
   *
   * @param m target tensor to be calculated against.
   * @return BroadcastInfo Loopinfo needed to run external loop
   */
  BroadcastInfo computeBroadcastInfo(const Tensor &m) const;

  /**
   * @brief Set the Dist object
   *
   * @tparam T distrubution engine
   * @param dist distribution engine
   */
  template <typename T> void setDist(T dist);

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const float *buf) noexcept;

  /**
   * @brief Update destination tensor to share memory with source tensor
   *
   * @param src src tensor containing the memory
   * @param dest destination tensor which will share the memory
   * @param offset offset to be used from the start of the data in bytes
   * @note The new tensor will share the same data as the current tensor but
   * can have different size.
   * @note New size added with offset must be less than the size of the original
   * tensor.
   */
  static void createSharedDataTensor(const Tensor &src, Tensor &dest,
                                     unsigned int offset);

  /**
   * @brief    Reallocate memory for this tensor
   * @note     This will not necessary free the memory as tensors share memory
   * @note     This can increase the peak memory consumption when callled on all
   * the tensors of a model sequentially. It is advised to first deallocate all
   * the tensors and then allocate, than reallocate tensors one by one.
   */
  void reallocate() {
    deallocate();
    allocate();
  }

  /**
   * @brief Merge the given two axis for tensor at second axis inplace
   *
   * @param axis1 first axis to merge
   * @param axis2 second axis to merge
   */
  void mergeAxis(unsigned int axis1, unsigned int axis2);
}; // namespace nntrainer

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, Tensor const &m);

typedef std::shared_ptr<Tensor> sharedTensor;

typedef std::shared_ptr<const Tensor> sharedConstTensor;

typedef std::vector<sharedConstTensor> sharedConstTensors;

typedef std::vector<sharedTensor> sharedTensors;

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_H__ */
