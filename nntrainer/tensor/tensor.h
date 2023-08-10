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
#include <random>
#include <stdexcept>
#include <vector>

#include <blas_interface.h>
#include <iostream>
#include <memory_data.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tensor_dim.h>
#include <util_func.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

#define MAKE_SHARED_TENSOR(...) std::make_shared<nntrainer::Tensor>(__VA_ARGS__)

#define CREATE_IF_EMPTY_DIMS(tensor, ...) \
  do {                                    \
    if (tensor.empty())                   \
      tensor = Tensor(__VA_ARGS__);       \
  } while (0);

namespace nntrainer {

using TensorDim = ml::train::TensorDim;
using Tformat = ml::train::TensorDim::Format;
using Tdatatype = ml::train::TensorDim::DataType;

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
  Tensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32) :
    dim(TensorDim(fm, d_type)),
    strides(dim.computeStrides()),
    contiguous(true),
    initializer(Initializer::NONE),
    name(name_),
    data(nullptr),
    offset(0),
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
  Tensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief     Constructor of Tensor
   * @param[in] d0 Batch of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  Tensor(size_t d0, size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32) :
    Tensor(TensorDim(d0, d1, d2, d3, fm, d_type), nullptr){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  Tensor(size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32) :
    Tensor(1, d1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of Tensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  Tensor(size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
         Tdatatype d_type = Tdatatype::FP32) :
    Tensor(1, 1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of Tensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  explicit Tensor(size_t d3, Tformat fm = Tformat::NCHW,
                  Tdatatype d_type = Tdatatype::FP32) :
    Tensor(1, 1, 1, d3, fm, d_type){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d0 Batch of Tensor
   * @param[in] d1 Channel (NCHW) or Height (NHWC)
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  Tensor(size_t d0, size_t d1, size_t d2, size_t d3,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(TensorDim(d0, d1, d2, d3, t_type), nullptr){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  Tensor(size_t d1, size_t d2, size_t d3,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(1, d1, d2, d3, t_type){};

  /**
   * @brief     Constructor of Tensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  Tensor(size_t d2, size_t d3, ml::train::TensorDim::TensorType t_type) :
    Tensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3,
           (t_type.format == Tformat::NCHW) ? d2 : 1,
           (t_type.format == Tformat::NCHW) ? d3 : d2, t_type){};
  /**
   * @brief     Constructor of Tensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  explicit Tensor(size_t d3, ml::train::TensorDim::TensorType t_type) :
    Tensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3, 1,
           (t_type.format == Tformat::NCHW) ? d3 : 1, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor. It needs to set format properly.
   */

  Tensor(std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
         ml::train::TensorDim::TensorType t_type) {
    if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
      throw std::out_of_range(
        "[Tensor] trying to initialize Tensor from empty vector");
    }
    // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
    // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
    // dim[1] == height, dim[2] == width, dim[3] == channel
    dim.setTensorDim(0, d.size());
    if (t_type.format == Tformat::NCHW) {
      dim.setTensorDim(1, d[0].size());
      dim.setTensorDim(2, d[0][0].size());
      dim.setTensorDim(3, d[0][0][0].size());
    } else {
      dim.setTensorDim(2, d[0].size());
      dim.setTensorDim(3, d[0][0].size());
      dim.setTensorDim(1, d[0][0][0].size());
    }

    setTensorType(t_type);

    strides = dim.computeStrides();

    MemoryData *mem_data =
      new MemoryData((void *)(new float[dim.getDataLen()]()));
    data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
      delete[] mem_data->getAddr<float>();
    });
    offset = 0;
    contiguous = true;
    initializer = Initializer::NONE;

    // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
    // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
    // dim[1] == height, dim[2] == width, dim[3] == channel
    if (t_type.format == Tformat::NCHW) {
      for (unsigned int i = 0; i < batch(); ++i)
        for (unsigned int j = 0; j < channel(); ++j)
          for (unsigned int k = 0; k < height(); ++k)
            for (unsigned int l = 0; l < width(); ++l)
              this->setValue(i, j, k, l, d[i][j][k][l]);
    } else {
      for (unsigned int i = 0; i < batch(); ++i)
        for (unsigned int j = 0; j < height(); ++j)
          for (unsigned int k = 0; k < width(); ++k)
            for (unsigned int l = 0; l < channel(); ++l)
              this->setValue(i, l, j, k, d[i][j][k][l]);
    }
  };

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   */
  Tensor(std::vector<std::vector<std::vector<float>>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   */
  Tensor(std::vector<std::vector<float>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

#ifdef ENABLE_FP16
  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   */
  Tensor(std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
         ml::train::TensorDim::TensorType t_type) {

    if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
      throw std::out_of_range(
        "[Tensor] trying to initialize Tensor from empty vector");
    }

    dim.setTensorDim(0, d.size());
    if (t_type.format == Tformat::NCHW) {
      dim.setTensorDim(1, d[0].size());
      dim.setTensorDim(2, d[0][0].size());
      dim.setTensorDim(3, d[0][0][0].size());
    } else {
      dim.setTensorDim(2, d[0].size());
      dim.setTensorDim(3, d[0][0].size());
      dim.setTensorDim(1, d[0][0][0].size());
    }

    setTensorType(t_type);

    strides = dim.computeStrides();

    MemoryData *mem_data =
      new MemoryData((void *)(new _FP16[dim.getDataLen()]()));
    data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
      delete[] mem_data->getAddr<_FP16>();
    });
    offset = 0;
    contiguous = true;
    initializer = Initializer::NONE;

    setDataType(Tdatatype::FP16);

    // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
    // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
    // dim[1] == height, dim[2] == width, dim[3] == channel
    if (t_type.format == Tformat::NCHW) {
      for (unsigned int i = 0; i < batch(); ++i)
        for (unsigned int j = 0; j < channel(); ++j)
          for (unsigned int k = 0; k < height(); ++k)
            for (unsigned int l = 0; l < width(); ++l)
              this->setValue(i, j, k, l, d[i][j][k][l]);
    } else {
      for (unsigned int i = 0; i < batch(); ++i)
        for (unsigned int j = 0; j < height(); ++j)
          for (unsigned int k = 0; k < width(); ++k)
            for (unsigned int l = 0; l < channel(); ++l)
              this->setValue(i, l, j, k, d[i][j][k][l]);
    }
  };

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   */
  Tensor(std::vector<std::vector<std::vector<_FP16>>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   */
  Tensor(std::vector<std::vector<_FP16>> const &d,
         ml::train::TensorDim::TensorType t_type) :
    Tensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

#endif

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
  template <typename T = float>
  static Tensor Map(T *buf, unsigned int bytes, const TensorDim &d,
                    size_t offset = 0) {
    if (d.getDataLen() == 0 || buf == nullptr) {
      throw std::invalid_argument(
        "[Tensor::Map] empty tensor dim is not allowed");
    }

    if (d.getDataLen() * sizeof(T) + offset > bytes) {
      throw std::invalid_argument(
        "Creating shared tensor of size bigger than tensor memory.");
    }

    Tensor tmp;
    tmp.dim = d;
    tmp.strides = d.computeStrides();
    /// Tensor does not own the memory
    tmp.data = std::shared_ptr<MemoryData>(new MemoryData((void *)buf),
                                           std::default_delete<MemoryData>());
    tmp.offset = offset;

    return tmp;
  };

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
  void deallocate() {
    data = nullptr;
    offset = 0;
  }

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
  template <typename T = float>
  const T &getValue(unsigned int batch, unsigned int c, unsigned int h,
                    unsigned int w) const noexcept {
    return getValue<T>(getIndex(batch, c, h, w));
  }

  template <typename T = float>
  T &getValue(unsigned int batch, unsigned int c, unsigned int h,
              unsigned int w) noexcept {
    return getValue<T>(getIndex(batch, c, h, w));
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  template <typename T = float>
  const T &getValue(unsigned int idx) const noexcept {
    return getData<T>()[idx];
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  template <typename T = float> T &getValue(unsigned int idx) noexcept {
    return getData<T>()[idx];
  }

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
  template <typename T = float>
  const T getValuePaddedVirtual(unsigned int b, unsigned int c, unsigned int h,
                                unsigned int w, unsigned int ph,
                                unsigned int pw,
                                T pad_value = 0) const EXCEPT_WHEN_DEBUG {
#if DEBUG
    unsigned int padded_h = 2 * ph + h;
    unsigned int padded_w = 2 * pw + w;
    if (h > padded_h && w > padded_w) {
      throw std::out_of_range(
        "[Tensor::getValuePadded] trying to access out of range");
    }
#endif

    if (ph <= h && h < ph + height() && pw <= w && w < pw + width()) {
      return getValue<T>(b, c, h - ph, w - pw);
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
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   */
  int multiply_i(Tensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   */
  Tensor multiply(Tensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   */
  Tensor &multiply(Tensor const &m, Tensor &output,
                   const float beta = 0.0) const;

  /**
   * @brief     Multiply Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply_i
   */
  int multiply_i_strided(Tensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  Tensor multiply_strided(Tensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  Tensor &multiply_strided(Tensor const &m, Tensor &output,
                           const float beta = 0.0) const;

  /**
   * @brief     Add Tensor Elementwise
   * @param[in] m Tensor to be added
   * @param[in] beta scalar to add output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add_i
   */
  int add_i_strided(Tensor const &m, const float beta = 0.0);

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] m Tensor to be added
   * @param[in] beta Value to be scale the added tensor
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  Tensor add_strided(Tensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] m Tensor to be added
   * @param[out] output Tensor to store the result
   * @param[in] beta Value to be scale the added tensor
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  Tensor &add_strided(Tensor const &m, Tensor &output,
                      const float beta = 0.0) const;

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
   * @brief  gaussian error function
   * @return int ML_ERROR_NONE if successful
   */
  int erf_i();

  /**
   * @brief    gaussian error function
   * @retval Calculated Tensor
   */
  Tensor erf() const;

  /**
   * @brief    gaussian error function
   * @param[out] out out to store the result
   * @retval Calculated Tensor
   */
  Tensor &erf(Tensor &out) const;

  /**
   * @brief  getter of size of data
   * @retval size of data
   */
  unsigned int sizeofData() { return dim.getDataTypeSize(); }

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
   * @brief compute the derivative of this in the current tensor
   * @param m same as given to the dot()
   * @param output_deriv the derivative of the output
   * @param[in] trans same as given to the dot()
   * @param[in] trans_m same as given to the dot()
   * @param[in] beta same as given to the dot()
   * @note This will compute the derivative in-place and will overwrite existing
   * data in the tensor
   */
  Tensor &dot_deriv_wrt_1(Tensor const &m, Tensor const &output_deriv,
                          bool trans = false, bool trans_m = false,
                          float beta = 0.0f);

  /**
   * @brief compute the derivative wrt m in the m tensor
   * @param m_deriv tensor where derivative wrt m will be stored
   * @param output_deriv the derivative of the output
   * @param[in] trans same as given to the dot()
   * @param[in] trans_m same as given to the dot()
   * @param[in] beta same as given to the dot()
   * @note The caller tensor must be the same tensor as the one which called the
   * dot() product.
   */
  Tensor &dot_deriv_wrt_2(Tensor &m_deriv, Tensor const &output_deriv,
                          bool trans = false, bool trans_m = false,
                          float beta = 0.0f) const;

  /**
   * @copydoc Tensor::dot(Tensor const &m, Tensor &output, bool trans,
              bool trans_m, float beta) const
   * @details performs dot operation over a batch of inputs
   */
  Tensor &dotBatched(Tensor const &m, Tensor &result, bool trans = false,
                     bool trans_m = false, float beta = 0.0f) const;

  /**
   * @copydoc Tensor::dot_deriv_wrt_1(Tensor const &m, Tensor const
   &output_deriv, bool trans, bool trans_m, float beta)
   */
  Tensor &dot_batched_deriv_wrt_1(Tensor const &m, Tensor const &output_deriv,
                                  bool trans = false, bool trans_m = false,
                                  float beta = 0.0f);

  /**
   * @brief Tensor::dot_deriv_wrt_2(Tensor const &m_deriv, Tensor const
   &output_deriv, bool trans, bool trans_m, float beta) const
   */
  Tensor &dot_batched_deriv_wrt_2(Tensor &m_deriv, Tensor const &output_deriv,
                                  bool trans = false, bool trans_m = false,
                                  float beta = 0.0f) const;

  /**
   * @brief Transpose Tensor
   *
   * @param direction to transpose ex) 0:2:1
   * @return Tensor
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
   * @brief Calculate Drop Out Mask : x * 1.0/(1.0-rate) inplace
   * @param dropout drop out rate
   */
  void dropout_mask(float dropout);

  /**
   * @brief Calculate filter mask
   * @param mask_len length of each mask along the last axis
   * @param invert invert the mask
   */
  void filter_mask(const Tensor &mask_len, bool reverse = false);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param zoneout zone out rate
   * @retval Tensor zone out mask for opposite tensor
   */
  Tensor zoneout_mask(float zoneout);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param opposite opposite zone out mask
   * @param zoneout zone out rate
   */
  void zoneout_mask(Tensor &opposite, float zoneout);

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
  Tensor &sum(unsigned int axis, Tensor &output, float alpha = 1.0,
              float beta = 0.0) const;

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
   * @brief     i data index
   * @retval    address of ith data
   */
  template <typename T = float> T *getAddress(unsigned int i) {
    size_t index = getIndex(batch(), channel(), height(), width());
    if (i > index) {
      return nullptr;
    }
    return &getData<T>()[i];
  }

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  template <typename T = float> const T *getAddress(unsigned int i) const {
    size_t index = getIndex(batch(), channel(), height(), width());
    if (i > index) {
      return nullptr;
    }

    return &getData<T>()[i];
  }

  /**
   * @brief    get address of n-d data
   */
  template <typename T = float>
  T *getAddress(unsigned int b, unsigned int c, unsigned int h,
                unsigned int w) {
    return getAddress<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief    get address of n-d data
   */
  template <typename T = float>
  const T *getAddress(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w) const {
    return getAddress<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief Apply instantly to the element
   *
   * @param f function to apply
   * @return int ML_ERROR_NONE if successful
   */
  template <typename T = float> int apply_i(std::function<T(T)> f) {
    Tensor result = *this;
    apply<T>(f, result);

    return ML_ERROR_NONE;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    Tensor
   */
  template <typename T = float>
  Tensor &apply(std::function<T(T)> f, Tensor &output) const {
    CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

    if (dim != output.dim) {
      /// @todo add unittest
      throw std::invalid_argument(
        "[Tensor::apply] output dimension does not match");
    }

    if (contiguous && output.contiguous) {
      const T *data = (getData<T>());
      T *rdata = (output.getData<T>());

      std::transform(data, data + size(), rdata, f);
    } else if (strides[3] == 1 && output.strides[3] == 1) {
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            T *out_data = output.getAddress<T>(b, c, h, 0);
            const T *in_data = getAddress<T>(b, c, h, 0);
            std::transform(in_data, in_data + width(), out_data, f);
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            for (unsigned int w = 0; w < width(); ++w) {
              output.setValue(b, c, h, w, f(getValue<T>(b, c, h, w)));
            }
          }
        }
      }
    }

    // if (dim.getDataType() == Tdatatype::FP32) {
    //   if (contiguous && output.contiguous) {
    //     const float *data = (getData<float>());
    //     float *rdata = (output.getData<float>());

    //     std::transform(data, data + size(), rdata, f);
    //   } else if (strides[3] == 1 && output.strides[3] == 1) {
    //     /** @todo optimize this with combining these loops where stride is 1
    //     */ for (unsigned int b = 0; b < batch(); ++b) {
    //       for (unsigned int c = 0; c < channel(); ++c) {
    //         for (unsigned int h = 0; h < height(); ++h) {
    //           float *out_data = output.getAddress<float>(b, c, h, 0);
    //           const float *in_data = getAddress<float>(b, c, h, 0);
    //           std::transform(in_data, in_data + width(), out_data, f);
    //         }
    //       }
    //     }
    //   } else {
    //     for (unsigned int b = 0; b < batch(); ++b) {
    //       for (unsigned int c = 0; c < channel(); ++c) {
    //         for (unsigned int h = 0; h < height(); ++h) {
    //           for (unsigned int w = 0; w < width(); ++w) {
    //             output.setValue(b, c, h, w, f(getValue<float>(b, c, h, w)));
    //           }
    //         }
    //       }
    //     }
    //   }
    // } else if (dim.getDataType() == Tdatatype::FP16) {

    //   auto f_16 = [f](_FP16 x) -> _FP16 {
    //     return static_cast<_FP16>(f(static_cast<float>(x)));
    //   };

    //   // std::function<_FP16(_FP16)> f_16 =
    //   //   static_cast<std::function<_FP16(_FP16)>>(f);

    //   if (contiguous && output.contiguous) {
    //     const _FP16 *data = (getData<_FP16>());
    //     _FP16 *rdata = (output.getData<_FP16>());

    //     std::transform(data, data + size(), rdata, f_16);
    //   } else if (strides[3] == 1 && output.strides[3] == 1) {
    //     /** @todo optimize this with combining these loops where stride is 1
    //     */ for (unsigned int b = 0; b < batch(); ++b) {
    //       for (unsigned int c = 0; c < channel(); ++c) {
    //         for (unsigned int h = 0; h < height(); ++h) {
    //           _FP16 *out_data = output.getAddress<_FP16>(b, c, h, 0);
    //           const _FP16 *in_data = getAddress<_FP16>(b, c, h, 0);
    //           std::transform(in_data, in_data + width(), out_data, f_16);
    //         }
    //       }
    //     }
    //   } else {
    //     for (unsigned int b = 0; b < batch(); ++b) {
    //       for (unsigned int c = 0; c < channel(); ++c) {
    //         for (unsigned int h = 0; h < height(); ++h) {
    //           for (unsigned int w = 0; w < width(); ++w) {
    //             output.setValue(b, c, h, w, f_16(getValue<_FP16>(b, c, h,
    //             w)));
    //           }
    //         }
    //       }
    //     }
    //   }
    // }
    return output;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  template <typename T = float> Tensor apply(std::function<T(T)> f) const {
    Tensor result;
    apply<T>(f, result);

    return result;
  };

  // /**
  //  * @brief Apply instantly to the element
  //  *
  //  * @param f function to apply
  //  * @return int ML_ERROR_NONE if successful
  //  */
  // int apply_i(std::function<_FP16(_FP16)> f) {
  //   Tensor result = *this;
  //   apply(f, result);

  //   return ML_ERROR_NONE;
  // };

  // /**
  //  * @brief     Apply function element by element
  //  * @param[in] *function function pointer applied
  //  * @retval    Tensor
  //  */
  // Tensor apply(std::function<_FP16(_FP16)> f) const {
  //   Tensor result;
  //   return apply(f, result);
  // };

  // /**
  //  * @brief     Apply function element by element
  //  * @param[in] *function function pointer applied
  //  * @retval    Tensor
  //  */
  // Tensor apply(std::function<float(float)> f) const {
  //   Tensor result;
  //   return apply(f, result);
  // };

  // /**
  //  * @brief     Apply function element by element
  //  * @param[in] *function function pointer applied
  //  * @param[out] output output tensor
  //  * @retval    Tensor
  //  */
  // Tensor &apply(std::function<_FP16(_FP16)> f, Tensor &output) const {
  //   CREATE_IF_EMPTY_DIMS(output, dim, nullptr);

  //   if (dim != output.dim) {
  //     /// @todo add unittest
  //     throw std::invalid_argument(
  //       "[Tensor::apply] output dimension does not match");
  //   }

  //   #ifdef ENABLE_FP16
  //     if (contiguous && output.contiguous) {
  //       const _FP16 *data = (getData<_FP16>());
  //       _FP16 *rdata = (output.getData<_FP16>());

  //       std::transform(data, data + size(), rdata, f);
  //     } else if (strides[3] == 1 && output.strides[3] == 1) {
  //       /** @todo optimize this with combining these loops where stride is 1
  //       */ for (unsigned int b = 0; b < batch(); ++b) {
  //         for (unsigned int c = 0; c < channel(); ++c) {
  //           for (unsigned int h = 0; h < height(); ++h) {
  //             _FP16 *out_data = (_FP16 *)output.getAddress(b, c, h, 0);
  //             const _FP16 *in_data = (_FP16 *)getAddress(b, c, h, 0);
  //             std::transform(in_data, in_data + width(), out_data, f);
  //           }
  //         }
  //       }
  //     } else {
  //       for (unsigned int b = 0; b < batch(); ++b) {
  //         for (unsigned int c = 0; c < channel(); ++c) {
  //           for (unsigned int h = 0; h < height(); ++h) {
  //             for (unsigned int w = 0; w < width(); ++w) {
  //               output.setValue(b, c, h, w,
  //                               f((_FP16)((_FP16)getValue(b, c, h, w))));
  //             }
  //           }
  //         }
  //       }
  //     }
  //   #else
  //     throw std::invalid_argument("Error: enable-fp16 is not enabled");
  //   #endif

  //   return output;
  // };

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
   * @brief     Print element
   * @param[in] out out stream
   * @param[in] opt print formatting option. opt=0 would pretty print the data,
   * else it would print the raw data.
   * @retval    Tensor
   */
  void print_(std::ostream &out, uint opt = 0) const;

  /**
   * @brief     Get size of current tensor
   * @retval    unsigned int size of the current tensor
   */
  size_t size() const { return dim.getDataLen(); }

  /**
   * @brief     Get if the tensor is empty
   * @retval    true if the tensor is empty
   */
  bool empty() const { return size() == 0; }

  /**
   * @brief     Get size of the data in bytes
   * @retval    size_t Size in bytes
   */
  size_t bytes() const { return size() * dim.getDataTypeSize(); }

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
    if (getDataType() == Tdatatype::FP32) {
      getData<float>()[getIndex(batch, c, h, w)] = value;
    } else if (getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
      getData<_FP16>()[getIndex(batch, c, h, w)] = static_cast<_FP16>(value);
#else
      ml_loge("%s", "Error: enable-fp16 is not enabled");
#endif
    }
  }

  /**
   * @brief     add the element value to the location
   * @param[in] batch batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   * @param[in] value value to be stored
   * @param[in] beta scalar to multiply output with and add
   */
  void addValue(unsigned int batch, unsigned int c, unsigned int h,
                unsigned int w, float value, float beta) noexcept {
    auto const &idx = getIndex(batch, c, h, w);
    if (dim.getDataType() == Tdatatype::FP32) {
      getData<float>()[idx] *= beta;
      getData<float>()[idx] += value;
    } else if (dim.getDataType() == Tdatatype::FP16) {
#ifdef ENABLE_FP16
      getData<_FP16>()[idx] *= static_cast<_FP16>(beta);
      getData<_FP16>()[idx] += static_cast<_FP16>(value);
#else
      ml_loge("%s", "Error: enable-fp16 is not enabled");
#endif
    }
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
    int *data_int = (int *)getData();
    data_int[offset] = value;
  }

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
   * @brief Set the Dist object
   *
   * @tparam T distrubution engine
   * @param dist distribution engine
   */
  template <typename T, typename Engine> void setDist(Engine dist) {
    NNTR_THROW_IF(!contiguous, std::invalid_argument)
      << getName() << " Tensor is not contiguous, cannot set distribution";

    T *data_ = getData<T>();
    unsigned int len = size();
    for (unsigned int i = 0; i < len; ++i) {
      data_[i] = (T)dist(rng);
    }
  };

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
   * @brief     Set the tensor with random bernoulli distribution
   * @param[in] probability probability value for the distribution
   */
  void setRandBernoulli(float probability = 0.5f);

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
   * @brief     set the memory format
   * @param     fm format of Tensor
   */
  void convertFormat(TensorDim::Format fm) {
    if (getFormat() != fm) {
      transpose("2:1:0");
    }

    dim.setFormat(fm);
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
  Tensor getBatchSlice(size_t offset, unsigned int size) const;

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
  Tensor getSharedDataTensor(const TensorDim dim, size_t offset,
                             bool reset_stride = true,
                             const std::string &name_ = "") const;
  /**
   * @brief split tensor along axis.
   *
   * @param num_size num_size
   * @param axis axis
   * @return Tensor splitted tensor
   */
  std::vector<Tensor> split(unsigned num_size, int axis = 0);

  /**
   * @brief split tensor along axis.
   *
   * @param sizes sizes
   * @param axis axis
   * @return Tensor splitted tensor
   * @note if the given array sizes is just a 1 unsigned int value, assumes that
   * it divide tensor by given size evenly
   */
  std::vector<Tensor> split(std::vector<size_t> sizes, int axis = 0);

  /**
   * @brief concatenate tensors along axis
   *
   * @param tensors tensors to be concatenated to the first tensor
   * @param axis axis
   * @return Tensor concatenated tensor
   */
  static Tensor cat(const std::vector<Tensor> &tensors, int axis = 0);

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
  void makeSharedDataTensor(const Tensor &src, size_t offset = 0);

  /**
   * @brief     Convient wrapper for inplace copy of @a this.
   * @retval    Copied version of this
   */
  Tensor clone() const;

  /**
   * @brief     Save the Tensor into file
   * @param[in] file output file stream
   */
  void save(std::ostream &file);

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
  size_t getTensorDim(unsigned int axis);

  /**
   * @brief     return Tensor Type
   */
  TensorDim::TensorType getTensorType() const { return dim.getTensorType(); };

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  size_t batch() const { return dim.batch(); }

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  size_t channel() const { return dim.channel(); }

  /**
   * @brief     return Tensor height size
   * @retval    height size
   */
  size_t height() const { return dim.height(); }

  /**
   * @brief     return Tensor batch size
   * @retval    width size
   */
  size_t width() const { return dim.width(); }

  /**
   * @brief     return Tensor Data Type Size
   * @retval    data type size
   */
  uint getDataTypeSize() const { return dim.getDataTypeSize(); }

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
   * immediately freed as the tensor already depending on this tensor also
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
  template <typename T = float> T *getData() {
    if (!data)
      return nullptr;

    data->validate();
    return data->getAddr<T>() + offset;
  }

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  template <typename T = float> const T *getData() const {
    if (!data)
      return nullptr;

    data->validate();
    return data->getAddr<T>() + offset;
  }

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  template <typename T = float> T *getData(size_t idx) const {
    if (!data)
      return nullptr;

    size_t index = idx;

    data->validate();
    return data->getAddr<T>() + offset + index;
  }

  /**
   * @brief     setter data type
   * @param[in] Data Type
   */
  void setDataType(Tdatatype d_type) { dim.setDataType(d_type); }

  /**
   * @brief     setter tensor type
   * @param[in] tensor Type
   */
  void setTensorType(ml::train::TensorDim::TensorType t_type) {
    dim.setTensorType(t_type);
  }

  /**
   * @brief     put data of Tensor
   *
   * @note      It is only effective when memory_swap is used
   */
  void putData() const {
    if (!data)
      return;

    data->invalidate();
  }

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  const std::shared_ptr<MemoryData> getMemoryData() const { return data; }

  /**
   * @brief     return offset
   */
  size_t getOffset() const { return offset; }

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
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
  const std::array<size_t, TensorDim::MAXDIM> getStrides() const noexcept {
    return strides;
  }
  /**
   * @brief Get linear index given the n-d index
   */
  inline size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w) const noexcept {
    if (getFormat() == Tformat::NCHW) {
      return (b * strides[0] + c * strides[1] + h * strides[2] +
              w * strides[3]);
    } else {
      return (b * strides[0] + h * strides[1] + w * strides[2] +
              c * strides[3]);
    }
  }

  /**
   * @brief Check if two given axes are contiguous
   */
  bool checkContinuous(unsigned int n, unsigned int np1) const {
    std::vector<unsigned int> continuous_order_nhwc = {0, 3, 1, 2};
    bool continuous = false;
    if (getFormat() == Tformat::NHWC) {
      if (continuous_order_nhwc[np1] == continuous_order_nhwc[n] + 1)
        continuous = true;
    } else {
      if (n + 1 == np1)
        continuous = true;
    }
    return continuous;
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
  void setData(const std::shared_ptr<MemoryData> buf, size_t off = 0,
               bool init = false) {
    if (buf) {
      data = buf;
      offset = off;
      if (init)
        initialize();
    } else {
      data = nullptr;
      offset = 0;
    }
  }

  /**
   * @brief Get initializer for the tensor
   *
   * @return initializer of the tensor
   */
  Tensor::Initializer getInitializer() const { return initializer; }

  /**
   * @brief Get format for the tensor
   *
   * @return format of the tensor
   */
  TensorDim::Format getFormat() const { return dim.getFormat(); }

  /**
   * @brief Get data type for the tensor
   *
   * @return data type of the tensor
   */
  Tdatatype getDataType() const { return dim.getDataType(); }

  static constexpr float epsilon = 1e-5;

private:
  /**< handle the data as a std::shared_ptr<float> type */
  TensorDim dim;
  std::array<size_t, TensorDim::MAXDIM> strides;
  bool contiguous;
  Tensor::Initializer initializer;
  std::string name; /**< name of the tensor */
  std::shared_ptr<MemoryData> data;
  size_t offset;

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
                       int cur_axis = -1, size_t offset = 0,
                       size_t m_offset = 0) const;

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
#ifdef ENABLE_FP16
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
                       std::function<void(const BroadcastInfo &e, const _FP16 *,
                                          const _FP16 *, _FP16 *)>
                         v_func,
                       Tensor &output, const BroadcastInfo &e,
                       int cur_axis = -1, size_t offset = 0,
                       size_t m_offset = 0) const;
  /**
   * @brief Applies the given operator to the tensor with the passed argument
   *
   * @param[in] m Tensor
   * @param[in] v_func vectorized function to apply
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void apply_broadcast(Tensor const &m,
                       std::function<void(const BroadcastInfo &e, const _FP16 *,
                                          const _FP16 *, _FP16 *)>
                         v_func,
                       Tensor &output) const;
#endif
  /**
   * @brief compute Loop info for broadcasting and vectorization
   *
   * @param m target tensor to be calculated against.
   * @return BroadcastInfo Loopinfo needed to run external loop
   */
  BroadcastInfo computeBroadcastInfo(const Tensor &m) const;

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);

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
                                     size_t offset);

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

  /**
   * @brief     rotate 180 dgree
   * @param[in] in input Tensor
   * @retVal Tensor rotated tensor (180 degree)
   */
  Tensor rotate_180(Tensor in);

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
