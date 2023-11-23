// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	float_tensor.h
 * @date	09 November 2023
 * @brief	This is FloatTensor class for 32-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __FLOAT_TENSOR_H__
#define __FLOAT_TENSOR_H__
#ifdef __cplusplus

#include <array>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <blas_interface.h>
#include <memory_data.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <src_shared_tensor.h>
#include <tensor_dim.h>
#include <util_func.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

#define MAKE_SHARED_FLOAT_TENSOR(...) \
  std::make_shared<nntrainer::FloatTensor>(__VA_ARGS__)

#define CREATE_IF_EMPTY_DIMS_FLOAT(tensor, ...) \
  do {                                          \
    if (tensor.empty())                         \
      tensor = FloatTensor(__VA_ARGS__);        \
  } while (0);

namespace nntrainer {

using TensorDim = ml::train::TensorDim;
using Tformat = ml::train::TensorDim::Format;
using Tdatatype = ml::train::TensorDim::DataType;

class LazyTensor;

/**
 * @class   FloatTensor Class for Calculation
 * @brief   FloatTensor Class for Calculation
 */
class FloatTensor {
public:
  /**
   * @brief     Basic Constructor of FloatTensor
   */
  FloatTensor(std::string name_ = "", Tformat fm = Tformat::NCHW) :
    dim(TensorDim(fm, Tdatatype::FP32)),
    strides(dim.computeStrides()),
    contiguous(true),
    initializer(Initializer::NONE),
    name(name_),
    data(nullptr),
    offset(0),
    src_tensor() {}

  /**
   * @brief     Constructor of FloatTensor with dimension, possibly lazily
   * @param d FloatTensor dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  FloatTensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief     Constructor of FloatTensor with dimension/buf
   * @param d FloatTensor dim for this tensor
   * @param buf buffer
   * @note Memory for this tensor is instantaneously allocated
   */
  FloatTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief     Constructor of FloatTensor
   * @param[in] d0 Batch of FloatTensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  FloatTensor(size_t d0, size_t d1, size_t d2, size_t d3,
              Tformat fm = Tformat::NCHW) :
    FloatTensor(TensorDim(d0, d1, d2, d3, fm, Tdatatype::FP32), nullptr){};

  /**
   * @brief     Constructor of FloatTensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  FloatTensor(size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW) :
    FloatTensor(1, d1, d2, d3, fm){};

  /**
   * @brief     Constructor of FloatTensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  FloatTensor(size_t d2, size_t d3, Tformat fm = Tformat::NCHW) :
    FloatTensor(1, 1, d2, d3, fm){};

  /**
   * @brief     Constructor of FloatTensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  explicit FloatTensor(size_t d3, Tformat fm = Tformat::NCHW) :
    FloatTensor(1, 1, 1, d3, fm){};

  /**
   * @brief     Constructor of FloatTensor
   * @param[in] d0 Batch of FloatTensor
   * @param[in] d1 Channel (NCHW) or Height (NHWC)
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  FloatTensor(size_t d0, size_t d1, size_t d2, size_t d3,
              ml::train::TensorDim::TensorType t_type) :
    FloatTensor(TensorDim(d0, d1, d2, d3, t_type), nullptr){};

  /**
   * @brief     Constructor of FloatTensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  FloatTensor(size_t d1, size_t d2, size_t d3,
              ml::train::TensorDim::TensorType t_type) :
    FloatTensor(1, d1, d2, d3, t_type){};

  /**
   * @brief     Constructor of FloatTensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  FloatTensor(size_t d2, size_t d3, ml::train::TensorDim::TensorType t_type) :
    FloatTensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3,
                (t_type.format == Tformat::NCHW) ? d2 : 1,
                (t_type.format == Tformat::NCHW) ? d3 : d2, t_type){};
  /**
   * @brief     Constructor of FloatTensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  explicit FloatTensor(size_t d3, ml::train::TensorDim::TensorType t_type) :
    FloatTensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3, 1,
                (t_type.format == Tformat::NCHW) ? d3 : 1, t_type){};

  /**
   * @brief     Constructor of FloatTensor
   * @param[in] d data for the FloatTensor. It needs to set format properly.
   */

  FloatTensor(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
    ml::train::TensorDim::TensorType t_type) {
    if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
      throw std::out_of_range(
        "[FloatTensor] trying to initialize FloatTensor from empty vector");
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
   * @brief     Constructor of FloatTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the FloatTensor. It needs to set format properly.
   */
  FloatTensor(std::vector<std::vector<std::vector<float>>> const &d,
              ml::train::TensorDim::TensorType t_type) :
    FloatTensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of FloatTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the FloatTensor with batch size one
   */
  FloatTensor(std::vector<std::vector<float>> const &d,
              ml::train::TensorDim::TensorType t_type) :
    FloatTensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   *  @brief  Copy constructor of FloatTensor.
   *  @param[in] FloatTensor &
   */
  FloatTensor(const FloatTensor &rhs) = default;

  /**
   *  @brief  Move constructor of FloatTensor.
   *  @param[in] FloatTensor &&
   */
  FloatTensor(FloatTensor &&rhs) noexcept = default;

  /**
   * @brief  Copy assignment operator.
   * @param[in] rhs FloatTensor to be copied.
   */
  FloatTensor &operator=(const FloatTensor &rhs) = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FloatTensor to be moved.
   */
  FloatTensor &operator=(FloatTensor &&rhs) noexcept = default;

  /**
   * @brief Construct a new FloatTensor object from a buffer
   * This will not copy buffer to a new tensor but directly uses it
   *
   * @param buf buffer
   * @param bytes buffer size in bytes
   * @param d tensor dim
   * @param offset offset to be used from current
   * @return FloatTensor object
   * @throws std::invalid_argument if buf is null
   */
  static FloatTensor Map(float *buf, unsigned int bytes, const TensorDim &d,
                         size_t offset = 0) {
    if (d.getDataLen() == 0 || buf == nullptr) {
      throw std::invalid_argument(
        "[FloatTensor::Map] empty tensor dim is not allowed");
    }

    if (d.getDataLen() * sizeof(float) + offset > bytes) {
      throw std::invalid_argument(
        "Creating shared tensor of size bigger than tensor memory.");
    }

    FloatTensor tmp;
    tmp.dim = d;
    tmp.strides = d.computeStrides();
    /// FloatTensor does not own the memory
    tmp.data = std::shared_ptr<MemoryData>(new MemoryData((void *)buf),
                                           std::default_delete<MemoryData>());
    tmp.offset = offset;

    return tmp;
  };

  friend void swap(FloatTensor &lhs, FloatTensor &rhs) noexcept {
    std::swap(lhs.dim, rhs.dim);
    std::swap(lhs.strides, rhs.strides);
    std::swap(lhs.contiguous, rhs.contiguous);
    std::swap(lhs.initializer, rhs.initializer);
    std::swap(lhs.data, rhs.data);
    std::swap(lhs.name, rhs.name);
  }

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs FloatTensor to be compared with
   */
  bool operator==(const FloatTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs FloatTensor to be compared with
   */
  bool operator!=(const FloatTensor &rhs) const { return !(*this == rhs); }

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
  const float &getValue(unsigned int batch, unsigned int c, unsigned int h,
                        unsigned int w) const noexcept {
    return getValue(getIndex(batch, c, h, w));
  }

  float &getValue(unsigned int batch, unsigned int c, unsigned int h,
                  unsigned int w) noexcept {
    return getValue(getIndex(batch, c, h, w));
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  const float &getValue(unsigned int idx) const noexcept {
    return getData()[idx];
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  float &getValue(unsigned int idx) noexcept { return getData()[idx]; }

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
  const float
  getValuePaddedVirtual(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, unsigned int ph, unsigned int pw,
                        float pad_value = 0) const EXCEPT_WHEN_DEBUG {
#if DEBUG
    unsigned int padded_h = 2 * ph + h;
    unsigned int padded_w = 2 * pw + w;
    if (h > padded_h && w > padded_w) {
      throw std::out_of_range(
        "[FloatTensor::getValuePadded] trying to access out of range");
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
   * @retval    #ML_ERROR_INVALID_PARAMETER FloatTensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int multiply_i(float const &value);

  /**
   * @brief     Multiply value element by element
   * @param[in] value multiplier
   * @retval    Calculated FloatTensor
   */
  FloatTensor multiply(float const &value) const;

  /**
   * @brief     multiply value element by element
   * @param[in] value multiplier
   * @param[out] out out tensor to store the result
   * @retval    Calculated FloatTensor
   */
  FloatTensor &multiply(float const &value, FloatTensor &out) const;

  /**
   * @brief     Multiply FloatTensor Elementwise
   * @param[in] m FloatTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   */
  int multiply_i(FloatTensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply FloatTensor Element by Element ( Not the MxM )
   * @param[in] m FloatTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated FloatTensor
   */
  FloatTensor multiply(FloatTensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Multiply FloatTensor Element by Element ( Not the MxM )
   * @param[in] m FloatTensor to be multiplied
   * @param[out] output FloatTensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated FloatTensor
   */
  FloatTensor &multiply(FloatTensor const &m, FloatTensor &output,
                        const float beta = 0.0) const;

  /**
   * @brief     Multiply FloatTensor Elementwise
   * @param[in] m FloatTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply_i
   */
  int multiply_i_strided(FloatTensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply FloatTensor Element by Element ( Not the MxM )
   * @param[in] m FloatTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated FloatTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  FloatTensor multiply_strided(FloatTensor const &m,
                               const float beta = 0.0) const;

  /**
   * @brief     Multiply FloatTensor Element by Element ( Not the MxM )
   * @param[in] m FloatTensor to be multiplied
   * @param[out] output FloatTensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated FloatTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  FloatTensor &multiply_strided(FloatTensor const &m, FloatTensor &output,
                                const float beta = 0.0) const;

  /**
   * @brief     Add FloatTensor Elementwise
   * @param[in] m FloatTensor to be added
   * @param[in] beta scalar to add output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add_i
   */
  int add_i_strided(FloatTensor const &m, const float beta = 0.0);

  /**
   * @brief     Add FloatTensor Element by Element
   * @param[in] m FloatTensor to be added
   * @param[in] beta Value to be scale the added tensor
   * @retval    Calculated FloatTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  FloatTensor add_strided(FloatTensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Add FloatTensor Element by Element
   * @param[in] m FloatTensor to be added
   * @param[out] output FloatTensor to store the result
   * @param[in] beta Value to be scale the added tensor
   * @retval    Calculated FloatTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  FloatTensor &add_strided(FloatTensor const &m, FloatTensor &output,
                           const float beta = 0.0) const;

  /**
   * @brief     Divide value element by element immediately
   * @param[in] value divisor
   * @retval    #ML_ERROR_INVALID_PARAMETER FloatTensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int divide_i(float const &value);

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @retval    Calculated FloatTensor
   */
  FloatTensor divide(float const &value) const;

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @param[out] out out parameter to store the result
   * @retval    Calculated FloatTensor
   */
  FloatTensor &divide(float const &value, FloatTensor &out) const;

  /**
   * @brief     divide FloatTensor Elementwise
   * @param[in] m FloatTensor to be multiplied
   * @retval    #ML_ERROR_NONE successful
   */
  int divide_i(FloatTensor const &m);

  /**
   * @brief     Divide FloatTensor Element by Element
   * @param[in] m Divisor FloatTensor
   * @retval    Calculated FloatTensor
   */
  FloatTensor divide(FloatTensor const &m) const;

  /**
   * @brief     divide FloatTensor Elementwise
   * @param[in] m FloatTensor to be multiplied
   * @param[out] output FloatTensor to store the result
   * @retval    Calculated FloatTensor
   */
  FloatTensor &divide(FloatTensor const &m, FloatTensor &output) const;

  /**
   * @brief Add FloatTensor Element immediately to target tensor without mem
   * copy
   * @param[in] value value to be added
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(float const &value);

  /**
   * @brief     Add value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated FloatTensor
   */
  FloatTensor add(float const &value) const;

  /**
   * @brief     Add FloatTensor Element by Element
   * @param[in] value value to be added
   * @param[out] out FloatTensor to save output without allocating new memory
   * @retval    Calculated FloatTensor
   */
  FloatTensor &add(float const &value, FloatTensor &out) const;

  /**
   * @brief Add FloatTensor Element by Element without mem copy
   * @param[in] m FloatTensor to be added
   * @param[out] alpha Values to be scaled
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(FloatTensor const &m, float const alpha = 1);

  /**
   * @brief     Add FloatTensor Element by Element
   * @param[in] m FloatTensor to be added
   * @retval    Calculated FloatTensor
   */
  FloatTensor add(FloatTensor const &m, float const alpha = 1) const;

  /**
   * @brief     Add FloatTensor Element by Element
   * @param[in] m FloatTensor to be added
   * @param[out] m FloatTensor to be out
   * @retval    Calculated FloatTensor
   */
  FloatTensor &add(FloatTensor const &m, FloatTensor &out,
                   float const alpha = 1) const;

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
   * @retval    Calculated FloatTensor
   */
  FloatTensor subtract(float const &value) const;

  /**
   * @brief     Subtract FloatTensor Element by Element
   * @param[in] value value to be added
   * @param[out] out FloatTensor to save output without allocating new memory
   * @retval    Calculated FloatTensor
   */
  FloatTensor &subtract(float const &value, FloatTensor &out) const;

  /**
   * @brief     memcpyless version of subtract
   * @param[in] m FloatTensor to be subtracted
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int subtract_i(FloatTensor const &m);

  /**
   * @brief     Substract FloatTensor Element by Element
   * @param[in] m FloatTensor to be subtracted
   * @retval    Calculated FloatTensor
   */
  FloatTensor subtract(FloatTensor const &m) const;

  /**
   * @brief     Subtract FloatTensor Element by Element
   * @param[in] m FloatTensor to be added
   * @param[out] m FloatTensor to be out
   * @retval    Calculated FloatTensor
   */
  FloatTensor &subtract(FloatTensor const &m, FloatTensor &out) const;

  /**
   * @brief FloatTensor power elementwise
   *
   * @param exponent exponent
   * @return int ML_ERROR_NONE if successful
   */
  int pow_i(float exponent);

  /**
   * @brief    FloatTensor power Element by Element
   * @param[in] exponent exponent
   * @retval Calculated FloatTensor
   */
  FloatTensor pow(float exponent) const;

  /**
   * @brief    FloatTensor power Element by Element
   * @param[in] exponent exponent
   * @param[out] out out to store the result
   * @retval Calculated FloatTensor
   */
  FloatTensor &pow(float exponent, FloatTensor &out) const;

  /**
   * @brief  gaussian error function
   * @return int ML_ERROR_NONE if successful
   */
  int erf_i();

  /**
   * @brief    gaussian error function
   * @retval Calculated FloatTensor
   */
  FloatTensor erf() const;

  /**
   * @brief    gaussian error function
   * @param[out] out out to store the result
   * @retval Calculated FloatTensor
   */
  FloatTensor &erf(FloatTensor &out) const;

  /**
   * @brief  getter of size of data
   * @retval size of data
   */
  unsigned int sizeofData() const { return dim.getDataTypeSize(); }

  /**
   * @brief     Dot Product of FloatTensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and second-last
   * dimension of passed tensor m.
   * @param[in] m FloatTensor
   * @param[in] trans Transpose
   * @param[in] trans_m Transpose m
   * @retval    Calculated FloatTensor
   */
  FloatTensor dot(FloatTensor const &m, bool trans = false,
                  bool trans_m = false) const;

  /**
   * @brief     Dot Product of FloatTensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and second-last
   * dimension of passed tensor m.
   * @param[in] m FloatTensor
   * @param[in] output output FloatTensor
   * @param[in] trans Transpose
   * @param[in] trans_m Transpose m
   * @param[in] beta beta
   * @retval    Calculated FloatTensor
   */
  FloatTensor &dot(FloatTensor const &m, FloatTensor &output,
                   bool trans = false, bool trans_m = false,
                   float beta = 0.0f) const;

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
  FloatTensor &dot_deriv_wrt_1(FloatTensor const &m,
                               FloatTensor const &output_deriv,
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
  FloatTensor &dot_deriv_wrt_2(FloatTensor &m_deriv,
                               FloatTensor const &output_deriv,
                               bool trans = false, bool trans_m = false,
                               float beta = 0.0f) const;

  /**
   * @copydoc FloatTensor::dot(FloatTensor const &m, FloatTensor &output, bool
   trans, bool trans_m, float beta) const
   * @details performs dot operation over a batch of inputs
   */
  FloatTensor &dotBatched(FloatTensor const &m, FloatTensor &result,
                          bool trans = false, bool trans_m = false,
                          float beta = 0.0f) const;

  /**
   * @copydoc FloatTensor::dot_deriv_wrt_1(FloatTensor const &m, FloatTensor
   const &output_deriv, bool trans, bool trans_m, float beta)
   */
  FloatTensor &dot_batched_deriv_wrt_1(FloatTensor const &m,
                                       FloatTensor const &output_deriv,
                                       bool trans = false, bool trans_m = false,
                                       float beta = 0.0f);

  /**
   * @brief FloatTensor::dot_deriv_wrt_2(FloatTensor const &m_deriv, FloatTensor
   const &output_deriv, bool trans, bool trans_m, float beta) const
   */
  FloatTensor &dot_batched_deriv_wrt_2(FloatTensor &m_deriv,
                                       FloatTensor const &output_deriv,
                                       bool trans = false, bool trans_m = false,
                                       float beta = 0.0f) const;

  /**
   * @brief Transpose FloatTensor
   *
   * @param direction to transpose ex) 0:2:1
   * @return FloatTensor
   */
  FloatTensor transpose(const std::string &direction) const;

  /**
   * @brief Transpose FloatTensor
   * @param direction to transpose ex) 0:2:1
   * @param[out] FloatTensor to save to, dimension is always reshaped.
   * @retval FloatTensor& reference to the out
   */
  FloatTensor &transpose(const std::string &direction, FloatTensor &out) const;

  /**
   * @brief Calculate Drop Out Mask : x * 1.0/(1.0-rate)
   * @param dropout drop out rate
   * @retval FloatTensor& reference of drop out mask
   */
  FloatTensor dropout_mask(float dropout) const;

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
  void filter_mask(const FloatTensor &mask_len, bool reverse = false);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param zoneout zone out rate
   * @retval FloatTensor zone out mask for opposite tensor
   */
  FloatTensor zoneout_mask(float zoneout);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param opposite opposite zone out mask
   * @param zoneout zone out rate
   */
  void zoneout_mask(FloatTensor &opposite, float zoneout);

  /**
   * @brief     sum all the FloatTensor elements according to the batch
   * @retval    Calculated FloatTensor(batch, 1, 1, 1)
   */
  FloatTensor sum_by_batch() const;

  /**
   * @brief     sum all the FloatTensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated FloatTensor
   */
  FloatTensor sum(unsigned int axis, float alpha = 1.0) const;

  /**
   * @brief     sum all the FloatTensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[out] output output tensor
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated FloatTensor
   */
  FloatTensor &sum(unsigned int axis, FloatTensor &output, float alpha = 1.0,
                   float beta = 0.0) const;

  /**
   * @brief sum all the FloatTensor by multiple axes
   *
   * @param axes axes to sum along
   * @param alpha Scale the sum by this value
   * @return FloatTensor
   */
  FloatTensor sum(const std::vector<unsigned int> &axes,
                  float alpha = 1.0) const;

  /**
   * @brief sum all the FloatTensor by multiple axes
   *
   * @param axes axes to sum along
   * @param[out] output output tensor
   * @param alpha Scale the sum by this value
   * @return FloatTensor
   */
  FloatTensor &sum(const std::vector<unsigned int> &axes, FloatTensor &output,
                   float alpha = 1.0) const;

  /**
   * @brief     Averaging the FloatTensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @retval    Calculated FloatTensor
   */
  FloatTensor average(unsigned int axis) const;
  /**
   * @brief     Averaging the FloatTensor elements according to the axis
   *
   * @retval    Calculated FloatTensor
   */
  FloatTensor &average(unsigned int axis, FloatTensor &output) const;

  /**
   * @brief average all the FloatTensor by multiple axes
   *
   * @param axes axes to sum along
   * @return FloatTensor
   */
  FloatTensor average(const std::vector<unsigned int> &axes) const;

  /**
   * @brief average all the FloatTensor by multiple axes
   *
   * @param axes axes to sum along
   * @param output output tensor
   * @return FloatTensor
   */
  FloatTensor &average(const std::vector<unsigned int> &axes,
                       FloatTensor &output) const;

  /**
   * @brief     Averaging the FloatTensor elements by all axis
   * @retval    Calculated FloatTensor
   */
  FloatTensor average() const;

  /**
   * @brief     Averaging the FloatTensor elements by all axis
   * @retval    Calculated FloatTensor
   */
  FloatTensor &average(FloatTensor &output) const;

  /**
   * @brief     Anchor a starting point to defer following evaluation
   * @retval    LazyTensor class that can be used with run();
   */
  LazyTensor chain() const;

  /**
   * @brief     Softmax the FloatTensor elements
   * @retval    Calculated FloatTensor
   */
  FloatTensor softmax() const;

  /**
   * @brief     l2norm the FloatTensor elements
   * @retval    Calculated l2norm
   */
  float l2norm() const;

  /**
   * @brief     Normalize the FloatTensor elements
   * @retval    Calculated FloatTensor
   */
  FloatTensor &normalization(FloatTensor &output) const;

  /**
   * @brief     Standardize the FloatTensor elements
   * @retval    Calculated FloatTensor
   */
  FloatTensor &standardization(FloatTensor &output) const;

  /**
   * @brief     Normalize the FloatTensor elements in-place
   * @retval    Calculated FloatTensor
   */
  void normalization_i();

  /**
   * @brief     Standardize the FloatTensor elements in-place
   * @retval    Calculated FloatTensor
   */
  void standardization_i();

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  float *getAddress(unsigned int i) {
    size_t index = getIndex(batch(), channel(), height(), width());
    if (i > index) {
      return nullptr;
    }
    return &getData()[i];
  }

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  const float *getAddress(unsigned int i) const {
    size_t index = getIndex(batch(), channel(), height(), width());
    if (i > index) {
      return nullptr;
    }

    return &getData()[i];
  }

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
   * @brief Apply instantly to the element
   *
   * @param f function to apply
   * @return int ML_ERROR_NONE if successful
   */
  int apply_i(std::function<float(float)> f) {
    FloatTensor result = *this;
    apply(f, result);

    return ML_ERROR_NONE;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    FloatTensor
   */
  FloatTensor &apply(std::function<float(float)> f, FloatTensor &output) const {
    CREATE_IF_EMPTY_DIMS_FLOAT(output, dim, nullptr);

    if (dim != output.dim) {
      /// @todo add unittest
      throw std::invalid_argument(
        "[FloatTensor::apply] output dimension does not match");
    }

    if (contiguous && output.contiguous) {
      const float *data = getData();
      float *rdata = output.getData();

      std::transform(data, data + size(), rdata, f);
    } else if (strides[3] == 1 && output.strides[3] == 1) {
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            float *out_data = output.getAddress(b, c, h, 0);
            const float *in_data = getAddress(b, c, h, 0);
            std::transform(in_data, in_data + width(), out_data, f);
          }
        }
      }
    } else {
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            for (unsigned int w = 0; w < width(); ++w) {
              output.setValue(b, c, h, w, f(getValue(b, c, h, w)));
            }
          }
        }
      }
    }

    return output;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    FloatTensor
   */
  FloatTensor apply(std::function<float(float)> f) const {
    FloatTensor result;
    apply(f, result);

    return result;
  };

  /**
   * @brief     Apply function to FloatTensor
   * @param[in] *function function pointer applied
   * @retval    FloatTensor
   */
  FloatTensor apply(std::function<FloatTensor(FloatTensor)> f) const;

  /**
   * @brief     Apply function to FloatTensor
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    FloatTensor
   */
  FloatTensor &apply(std::function<FloatTensor &(FloatTensor, FloatTensor &)> f,
                     FloatTensor &output) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @retval    FloatTensor
   */
  void print(std::ostream &out) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @param[in] opt print formatting option. opt=0 would pretty print the data,
   * else it would print the raw data.
   * @retval    FloatTensor
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
    getData()[getIndex(batch, c, h, w)] = value;
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
    getData()[idx] *= beta;
    getData()[idx] += value;
  }

  /**
   * @brief     Fill the FloatTensor elements with value
   * @param[in] value value to be stored
   */
  void setValue(float value);

  /**
   * @brief     Fill the FloatTensor elements with zero
   */
  void setZero();

  /**
   * @brief Set the Dist object
   *
   * @tparam T distrubution engine
   * @param dist distribution engine
   */
  template <typename Engine> void setDist(Engine dist) {
    NNTR_THROW_IF(!contiguous, std::invalid_argument)
      << getName() << " FloatTensor is not contiguous, cannot set distribution";

    float *data_ = getData();
    unsigned int len = size();
    for (unsigned int i = 0; i < len; ++i) {
      data_[i] = (float)dist(rng);
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
   * @param     fm format of FloatTensor
   */
  void convertFormat(TensorDim::Format fm) {
    if (getFormat() != fm) {
      transpose("2:1:0");
    }

    dim.setFormat(fm);
  }

  /**
   * @brief     Copy the FloatTensor
   * @param[in] from FloatTensor to be copied
   *
   * @note copy can reshape the tensor to match the shape
   */
  void copy(const FloatTensor &from);

  /**
   * @brief     Copy the FloatTensor
   * @param[in] from FloatTensor to be copied
   */
  void copyData(const FloatTensor &from);

  /**
   * @brief     Copy the FloatTensor
   * @param[in] from FloatTensor to be copied
   */
  void copy_with_stride(const FloatTensor &from);

  /**
   * @brief Get slice of the tensor, sliced by batch
   * @param[in] offset offset in batch to start the slice
   * @param[in] size size of the slice
   * @retval slice of this tensor
   * @note This function provides a slice of this tensor, and does not create a
   * copy
   */
  FloatTensor getBatchSlice(size_t offset, unsigned int size) const;

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
  FloatTensor getSharedDataTensor(const TensorDim dim, size_t offset,
                                  bool reset_stride = true,
                                  const std::string &name_ = "") const;
  /**
   * @brief split tensor along axis.
   *
   * @param num_size num_size
   * @param axis axis
   * @return FloatTensor splitted tensor
   */
  std::vector<FloatTensor> split(unsigned num_size, int axis = 0);

  /**
   * @brief split tensor along axis.
   *
   * @param sizes sizes
   * @param axis axis
   * @return FloatTensor splitted tensor
   * @note if the given array sizes is just a 1 unsigned int value, assumes that
   * it divide tensor by given size evenly
   */
  std::vector<FloatTensor> split(std::vector<size_t> sizes, int axis = 0);

  /**
   * @brief concatenate tensors along axis
   *
   * @param tensors tensors to be concatenated to the first tensor
   * @param axis axis
   * @return FloatTensor concatenated tensor
   */
  static FloatTensor cat(const std::vector<FloatTensor> &tensors, int axis = 0);

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
  void makeSharedDataTensor(const FloatTensor &src, size_t offset = 0);

  /**
   * @brief     Convient wrapper for inplace copy of @a this.
   * @retval    Copied version of this
   */
  FloatTensor clone() const;

  /**
   * @brief     Save the FloatTensor into file
   * @param[in] file output file stream
   */
  void save(std::ostream &file);

  /**
   * @brief     Read the FloatTensor from file
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
   * @brief     return a copy of the FloatTensor Dim
   * @retval    TensorDim
   */
  TensorDim getDim() const { return TensorDim(dim); }

  /**
   * @brief     return FloatTensor Dim for a given axis
   * @retval    dimension
   */
  size_t getTensorDim(unsigned int axis);

  /**
   * @brief     return FloatTensor Type
   */
  TensorDim::TensorType getTensorType() const { return dim.getTensorType(); };

  /**
   * @brief     return FloatTensor batch size
   * @retval    batch size
   */
  size_t batch() const { return dim.batch(); }

  /**
   * @brief     return FloatTensor batch size
   * @retval    batch size
   */
  size_t channel() const { return dim.channel(); }

  /**
   * @brief     return FloatTensor height size
   * @retval    height size
   */
  size_t height() const { return dim.height(); }

  /**
   * @brief     return FloatTensor batch size
   * @retval    width size
   */
  size_t width() const { return dim.width(); }

  /**
   * @brief     return FloatTensor Data Type Size
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
   * @brief     return Data pointer of FloatTensor
   * @retval    template T pointer (float pointer as default)
   */
  float *getData() {
    if (!data)
      return nullptr;

    data->validate();
    return data->getAddr<float>() + offset;
  }

  /**
   * @brief     return Data pointer of FloatTensor
   * @retval    template T pointer (float pointer as default)
   */
  const float *getData() const {
    if (!data)
      return nullptr;

    data->validate();
    return data->getAddr<float>() + offset;
  }

  /**
   * @brief     return Data pointer of FloatTensor
   * @retval    template T pointer (float pointer as default)
   */
  float *getData(size_t idx) const {
    if (!data)
      return nullptr;

    size_t index = idx;

    data->validate();
    return data->getAddr<float>() + offset + index;
  }

  /**
   * @brief     setter tensor type
   * @param[in] tensor Type
   */
  void setTensorType(ml::train::TensorDim::TensorType t_type) {
    dim.setTensorType(t_type);
  }

  /**
   * @brief     put data of FloatTensor
   *
   * @note      It is only effective when memory_swap is used
   */
  void putData() const {
    if (!data)
      return;

    data->invalidate();
  }

  /**
   * @brief     return Data pointer of FloatTensor
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
   * @brief     set FloatTensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  void reshape(const TensorDim &d);

  /**
   * @brief fill tensor data with current value,
   * if dimension is not exactly same, it is a hard error in this function
   * so, only stride is overriden to @a this
   *
   * @param from FloatTensor to fill the data from
   * @param allocate if unallocated, allocate with from.getDim()
   * @throws std::invalid_argument if dimension and stride does not match
   */
  void fill(const FloatTensor &from, bool allocate = false);

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  const std::array<size_t, TensorDim::MAXDIM> getStrides() const noexcept {
    return strides;
  }

  /**
   * @brief     return contiguous state of tensor.
   * @retval    bool contiguous
   */
  bool getContiguous() const { return contiguous; }

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
  Initializer getInitializer() const { return initializer; }

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

protected:
  /**< handle the data as a std::shared_ptr<float> type */
  TensorDim dim;
  std::array<size_t, TensorDim::MAXDIM> strides;
  bool contiguous;
  Initializer initializer;
  std::string name; /**< name of the tensor */
  std::shared_ptr<MemoryData> data;
  size_t offset;

  /**<
   * When using shared_data with tensor, this stores the ptr of the source
   * tensor which handles the full memory. If tensor data is already allocated,
   * this does not affect the tensor. If the tensor data is not allocated, and
   * src_ptr is valid, this tensor will use the memory allocated by the src_ptr
   */
  std::shared_ptr<SrcSharedTensorV2<FloatTensor>> src_tensor;

  struct BroadcastInfo;

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   * @param[in] m FloatTensor
   * @param[in] v_func vectorized function to apply
   * @param e broadcast info.
   * @param cur_axis current axis. pass default when calling outside.
   * @param offset offset for this.  pass default when calling outside.
   * @param m_offset offset for m.  pass default when calling outside.
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void
  apply_broadcast_util(FloatTensor const &m,
                       std::function<void(const BroadcastInfo &e, const float *,
                                          const float *, float *)>
                         v_func,
                       FloatTensor &output, const BroadcastInfo &e,
                       int cur_axis = -1, size_t offset = 0,
                       size_t m_offset = 0) const;

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   *
   * @param[in] m FloatTensor
   * @param[in] v_func vectorized function to apply
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void apply_broadcast(FloatTensor const &m,
                       std::function<void(const BroadcastInfo &e, const float *,
                                          const float *, float *)>
                         v_func,
                       FloatTensor &output) const;

  /**
   * @brief compute Loop info for broadcasting and vectorization
   *
   * @param m target tensor to be calculated against.
   * @return BroadcastInfo Loopinfo needed to run external loop
   */
  BroadcastInfo computeBroadcastInfo(const FloatTensor &m) const;

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
  static void createSharedDataTensor(const FloatTensor &src, FloatTensor &dest,
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
   * @param[in] in input FloatTensor
   * @retVal FloatTensor rotated tensor (180 degree)
   */
  FloatTensor rotate_180(FloatTensor in);

}; // namespace nntrainer

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, FloatTensor const &m);

typedef std::shared_ptr<FloatTensor> sharedFloatTensor;

typedef std::shared_ptr<const FloatTensor> sharedConstFloatTensor;

typedef std::vector<sharedConstFloatTensor> sharedConstFloatTensors;

typedef std::vector<sharedFloatTensor> sharedFloatTensors;

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __FLOAT_TENSOR_H__ */
