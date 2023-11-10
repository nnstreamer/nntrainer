// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	 half_tensor.h
 * @date	 09 November 2023
 * @brief	 This is HalfTensor class for calculation
 * @see		 https://github.com/nnstreamer/nntrainer
 * @author	 Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		 No known bugs except for NYI items
 */

#ifndef __HALF_TENSOR_H__
#define __HALF_TENSOR_H__
#ifdef __cplusplus

#ifdef ENABLE_FP16

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
#include <tensor_dim.h>
#include <util_func.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

#define MAKE_SHARED_HALF_TENSOR(...) \
  std::make_shared<nntrainer::HalfTensor>(__VA_ARGS__)

#define CREATE_IF_EMPTY_DIMS_HALF(tensor, ...) \
  do {                                         \
    if (tensor.empty())                        \
      tensor = HalfTensor(__VA_ARGS__);        \
  } while (0);

namespace nntrainer {

using TensorDim = ml::train::TensorDim;
using Tformat = ml::train::TensorDim::Format;
using Tdatatype = ml::train::TensorDim::DataType;

class LazyTensor;
class SrcSharedHalfTensor;

/**
 * @class   HalfTensor Class for Calculation
 * @brief   HalfTensor Class for Calculation
 */
class HalfTensor {
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
   * @brief     Basic Constructor of HalfTensor
   */
  HalfTensor(std::string name_ = "", Tformat fm = Tformat::NCHW,
             Tdatatype d_type = Tdatatype::FP16) :
    dim(TensorDim(fm, d_type)),
    strides(dim.computeStrides()),
    contiguous(true),
    initializer(Initializer::NONE),
    name(name_),
    data(nullptr),
    offset(0),
    src_tensor() {}

  /**
   * @brief     Constructor of HalfTensor with dimension, possibly lazily
   * @param d HalfTensor dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  HalfTensor(const TensorDim &d, bool alloc_now,
             Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief     Constructor of HalfTensor with dimension/buf
   * @param d HalfTensor dim for this tensor
   * @param buf buffer
   * @note Memory for this tensor is instantaneously allocated
   */
  HalfTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief     Constructor of HalfTensor
   * @param[in] d0 Batch of HalfTensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  HalfTensor(size_t d0, size_t d1, size_t d2, size_t d3,
             Tformat fm = Tformat::NCHW, Tdatatype d_type = Tdatatype::FP16) :
    HalfTensor(TensorDim(d0, d1, d2, d3, fm, d_type), nullptr){};

  /**
   * @brief     Constructor of HalfTensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  HalfTensor(size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
             Tdatatype d_type = Tdatatype::FP16) :
    HalfTensor(1, d1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of HalfTensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  HalfTensor(size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
             Tdatatype d_type = Tdatatype::FP16) :
    HalfTensor(1, 1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of HalfTensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  explicit HalfTensor(size_t d3, Tformat fm = Tformat::NCHW,
                      Tdatatype d_type = Tdatatype::FP16) :
    HalfTensor(1, 1, 1, d3, fm, d_type){};

  /**
   * @brief     Constructor of HalfTensor
   * @param[in] d0 Batch of HalfTensor
   * @param[in] d1 Channel (NCHW) or Height (NHWC)
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  HalfTensor(size_t d0, size_t d1, size_t d2, size_t d3,
             ml::train::TensorDim::TensorType t_type) :
    HalfTensor(TensorDim(d0, d1, d2, d3, t_type), nullptr){};

  /**
   * @brief     Constructor of HalfTensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  HalfTensor(size_t d1, size_t d2, size_t d3,
             ml::train::TensorDim::TensorType t_type) :
    HalfTensor(1, d1, d2, d3, t_type){};

  /**
   * @brief     Constructor of HalfTensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  HalfTensor(size_t d2, size_t d3, ml::train::TensorDim::TensorType t_type) :
    HalfTensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3,
               (t_type.format == Tformat::NCHW) ? d2 : 1,
               (t_type.format == Tformat::NCHW) ? d3 : d2, t_type){};
  /**
   * @brief     Constructor of HalfTensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  explicit HalfTensor(size_t d3, ml::train::TensorDim::TensorType t_type) :
    HalfTensor(1, (t_type.format == Tformat::NCHW) ? 1 : d3, 1,
               (t_type.format == Tformat::NCHW) ? d3 : 1, t_type){};

  /**
   * @brief     Constructor of HalfTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the HalfTensor with batch size one
   */
  HalfTensor(std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
             ml::train::TensorDim::TensorType t_type) {

    if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
      throw std::out_of_range(
        "[HalfTensor] trying to initialize HalfTensor from empty vector");
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
   * @brief     Constructor of HalfTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the HalfTensor. It needs to set format properly.
   */
  HalfTensor(std::vector<std::vector<std::vector<_FP16>>> const &d,
             ml::train::TensorDim::TensorType t_type) :
    HalfTensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of HalfTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the HalfTensor with batch size one
   */
  HalfTensor(std::vector<std::vector<_FP16>> const &d,
             ml::train::TensorDim::TensorType t_type) :
    HalfTensor(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   *  @brief  Copy constructor of HalfTensor.
   *  @param[in] HalfTensor &
   */
  HalfTensor(const HalfTensor &rhs) = default;

  /**
   *  @brief  Move constructor of HalfTensor.
   *  @param[in] HalfTensor &&
   */
  HalfTensor(HalfTensor &&rhs) noexcept = default;

  /**
   * @brief  Copy assignment operator.
   * @param[in] rhs HalfTensor to be copied.
   */
  HalfTensor &operator=(const HalfTensor &rhs) = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs HalfTensor to be moved.
   */
  HalfTensor &operator=(HalfTensor &&rhs) noexcept = default;

  /**
   * @brief Construct a new HalfTensor object from a buffer
   * This will not copy buffer to a new tensor but directly uses it
   *
   * @param buf buffer
   * @param bytes buffer size in bytes
   * @param d tensor dim
   * @param offset offset to be used from current
   * @return HalfTensor object
   * @throws std::invalid_argument if buf is null
   */
  static HalfTensor Map(_FP16 *buf, unsigned int bytes, const TensorDim &d,
                        size_t offset = 0) {
    if (d.getDataLen() == 0 || buf == nullptr) {
      throw std::invalid_argument(
        "[HalfTensor::Map] empty tensor dim is not allowed");
    }

    if (d.getDataLen() * sizeof(_FP16) + offset > bytes) {
      throw std::invalid_argument(
        "Creating shared tensor of size bigger than tensor memory.");
    }

    HalfTensor tmp;
    tmp.dim = d;
    tmp.strides = d.computeStrides();
    /// HalfTensor does not own the memory
    tmp.data = std::shared_ptr<MemoryData>(new MemoryData((void *)buf),
                                           std::default_delete<MemoryData>());
    tmp.offset = offset;

    return tmp;
  };

  friend void swap(HalfTensor &lhs, HalfTensor &rhs) noexcept {
    std::swap(lhs.dim, rhs.dim);
    std::swap(lhs.strides, rhs.strides);
    std::swap(lhs.contiguous, rhs.contiguous);
    std::swap(lhs.initializer, rhs.initializer);
    std::swap(lhs.data, rhs.data);
    std::swap(lhs.name, rhs.name);
  }

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs HalfTensor to be compared with
   */
  bool operator==(const HalfTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs HalfTensor to be compared with
   */
  bool operator!=(const HalfTensor &rhs) const { return !(*this == rhs); }

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
  const _FP16 &getValue(unsigned int batch, unsigned int c, unsigned int h,
                        unsigned int w) const noexcept {
    return getValue(getIndex(batch, c, h, w));
  }

  _FP16 &getValue(unsigned int batch, unsigned int c, unsigned int h,
                  unsigned int w) noexcept {
    return getValue(getIndex(batch, c, h, w));
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  const _FP16 &getValue(unsigned int idx) const noexcept {
    if (getDataType() == Tdatatype::QINT4) {
      return getData()[idx / 2];
    }
    return getData()[idx];
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  _FP16 &getValue(unsigned int idx) noexcept {
    if (getDataType() == Tdatatype::QINT4) {
      return getData()[idx / 2];
    }
    return getData()[idx];
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
  const _FP16
  getValuePaddedVirtual(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w, unsigned int ph, unsigned int pw,
                        _FP16 pad_value = 0) const EXCEPT_WHEN_DEBUG {
#if DEBUG
    unsigned int padded_h = 2 * ph + h;
    unsigned int padded_w = 2 * pw + w;
    if (h > padded_h && w > padded_w) {
      throw std::out_of_range(
        "[HalfTensor::getValuePadded] trying to access out of range");
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
   * @retval    #ML_ERROR_INVALID_PARAMETER HalfTensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int multiply_i(float const &value);

  /**
   * @brief     Multiply value element by element
   * @param[in] value multiplier
   * @retval    Calculated HalfTensor
   */
  HalfTensor multiply(float const &value) const;

  /**
   * @brief     multiply value element by element
   * @param[in] value multiplier
   * @param[out] out out tensor to store the result
   * @retval    Calculated HalfTensor
   */
  HalfTensor &multiply(float const &value, HalfTensor &out) const;

  /**
   * @brief     Multiply HalfTensor Elementwise
   * @param[in] m HalfTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   */
  int multiply_i(HalfTensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply HalfTensor Element by Element ( Not the MxM )
   * @param[in] m HalfTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated HalfTensor
   */
  HalfTensor multiply(HalfTensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Multiply HalfTensor Element by Element ( Not the MxM )
   * @param[in] m HalfTensor to be multiplied
   * @param[out] output HalfTensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated HalfTensor
   */
  HalfTensor &multiply(HalfTensor const &m, HalfTensor &output,
                       const float beta = 0.0) const;

  /**
   * @brief     Multiply HalfTensor Elementwise
   * @param[in] m HalfTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply_i
   */
  int multiply_i_strided(HalfTensor const &m, const float beta = 0.0);

  /**
   * @brief     Multiply HalfTensor Element by Element ( Not the MxM )
   * @param[in] m HalfTensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated HalfTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  HalfTensor multiply_strided(HalfTensor const &m,
                              const float beta = 0.0) const;

  /**
   * @brief     Multiply HalfTensor Element by Element ( Not the MxM )
   * @param[in] m HalfTensor to be multiplied
   * @param[out] output HalfTensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated HalfTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  HalfTensor &multiply_strided(HalfTensor const &m, HalfTensor &output,
                               const float beta = 0.0) const;

  /**
   * @brief     Add HalfTensor Elementwise
   * @param[in] m HalfTensor to be added
   * @param[in] beta scalar to add output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add_i
   */
  int add_i_strided(HalfTensor const &m, const float beta = 0.0);

  /**
   * @brief     Add HalfTensor Element by Element
   * @param[in] m HalfTensor to be added
   * @param[in] beta Value to be scale the added tensor
   * @retval    Calculated HalfTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  HalfTensor add_strided(HalfTensor const &m, const float beta = 0.0) const;

  /**
   * @brief     Add HalfTensor Element by Element
   * @param[in] m HalfTensor to be added
   * @param[out] output HalfTensor to store the result
   * @param[in] beta Value to be scale the added tensor
   * @retval    Calculated HalfTensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to add
   */
  HalfTensor &add_strided(HalfTensor const &m, HalfTensor &output,
                          const float beta = 0.0) const;

  /**
   * @brief     Divide value element by element immediately
   * @param[in] value divisor
   * @retval    #ML_ERROR_INVALID_PARAMETER HalfTensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int divide_i(float const &value);

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @retval    Calculated HalfTensor
   */
  HalfTensor divide(float const &value) const;

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @param[out] out out parameter to store the result
   * @retval    Calculated HalfTensor
   */
  HalfTensor &divide(float const &value, HalfTensor &out) const;

  /**
   * @brief     divide HalfTensor Elementwise
   * @param[in] m HalfTensor to be multiplied
   * @retval    #ML_ERROR_NONE successful
   */
  int divide_i(HalfTensor const &m);

  /**
   * @brief     Divide HalfTensor Element by Element
   * @param[in] m Divisor HalfTensor
   * @retval    Calculated HalfTensor
   */
  HalfTensor divide(HalfTensor const &m) const;

  /**
   * @brief     divide HalfTensor Elementwise
   * @param[in] m HalfTensor to be multiplied
   * @param[out] output HalfTensor to store the result
   * @retval    Calculated HalfTensor
   */
  HalfTensor &divide(HalfTensor const &m, HalfTensor &output) const;

  /**
   * @brief Add HalfTensor Element immediately to target tensor without mem copy
   * @param[in] value value to be added
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(float const &value);

  /**
   * @brief     Add value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated HalfTensor
   */
  HalfTensor add(float const &value) const;

  /**
   * @brief     Add HalfTensor Element by Element
   * @param[in] value value to be added
   * @param[out] out HalfTensor to save output without allocating new memory
   * @retval    Calculated HalfTensor
   */
  HalfTensor &add(float const &value, HalfTensor &out) const;

  /**
   * @brief Add HalfTensor Element by Element without mem copy
   * @param[in] m HalfTensor to be added
   * @param[out] alpha Values to be scaled
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(HalfTensor const &m, float const alpha = 1);

  /**
   * @brief     Add HalfTensor Element by Element
   * @param[in] m HalfTensor to be added
   * @retval    Calculated HalfTensor
   */
  HalfTensor add(HalfTensor const &m, float const alpha = 1) const;

  /**
   * @brief     Add HalfTensor Element by Element
   * @param[in] m HalfTensor to be added
   * @param[out] m HalfTensor to be out
   * @retval    Calculated HalfTensor
   */
  HalfTensor &add(HalfTensor const &m, HalfTensor &out,
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
   * @retval    Calculated HalfTensor
   */
  HalfTensor subtract(float const &value) const;

  /**
   * @brief     Subtract HalfTensor Element by Element
   * @param[in] value value to be added
   * @param[out] out HalfTensor to save output without allocating new memory
   * @retval    Calculated HalfTensor
   */
  HalfTensor &subtract(float const &value, HalfTensor &out) const;

  /**
   * @brief     memcpyless version of subtract
   * @param[in] m HalfTensor to be subtracted
   * @retval #ML_ERROR_NONE  Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int subtract_i(HalfTensor const &m);

  /**
   * @brief     Substract HalfTensor Element by Element
   * @param[in] m HalfTensor to be subtracted
   * @retval    Calculated HalfTensor
   */
  HalfTensor subtract(HalfTensor const &m) const;

  /**
   * @brief     Subtract HalfTensor Element by Element
   * @param[in] m HalfTensor to be added
   * @param[out] m HalfTensor to be out
   * @retval    Calculated HalfTensor
   */
  HalfTensor &subtract(HalfTensor const &m, HalfTensor &out) const;

  /**
   * @brief HalfTensor power elementwise
   *
   * @param exponent exponent
   * @return int ML_ERROR_NONE if successful
   */
  int pow_i(float exponent);

  /**
   * @brief    HalfTensor power Element by Element
   * @param[in] exponent exponent
   * @retval Calculated HalfTensor
   */
  HalfTensor pow(float exponent) const;

  /**
   * @brief    HalfTensor power Element by Element
   * @param[in] exponent exponent
   * @param[out] out out to store the result
   * @retval Calculated HalfTensor
   */
  HalfTensor &pow(float exponent, HalfTensor &out) const;

  /**
   * @brief  gaussian error function
   * @return int ML_ERROR_NONE if successful
   */
  int erf_i();

  /**
   * @brief    gaussian error function
   * @retval Calculated HalfTensor
   */
  HalfTensor erf() const;

  /**
   * @brief    gaussian error function
   * @param[out] out out to store the result
   * @retval Calculated HalfTensor
   */
  HalfTensor &erf(HalfTensor &out) const;

  /**
   * @brief  getter of size of data
   * @retval size of data
   */
  unsigned int sizeofData() { return dim.getDataTypeSize(); }

  /**
   * @brief     Dot Product of HalfTensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and second-last
   * dimension of passed tensor m.
   * @param[in] m HalfTensor
   * @param[in] trans Transpose
   * @param[in] trans_m Transpose m
   * @retval    Calculated HalfTensor
   */
  HalfTensor dot(HalfTensor const &m, bool trans = false,
                 bool trans_m = false) const;

  /**
   * @brief     Dot Product of HalfTensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and second-last
   * dimension of passed tensor m.
   * @param[in] m HalfTensor
   * @param[in] output output HalfTensor
   * @param[in] trans Transpose
   * @param[in] trans_m Transpose m
   * @param[in] beta beta
   * @retval    Calculated HalfTensor
   */
  HalfTensor &dot(HalfTensor const &m, HalfTensor &output, bool trans = false,
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
  HalfTensor &dot_deriv_wrt_1(HalfTensor const &m,
                              HalfTensor const &output_deriv,
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
  HalfTensor &dot_deriv_wrt_2(HalfTensor &m_deriv,
                              HalfTensor const &output_deriv,
                              bool trans = false, bool trans_m = false,
                              float beta = 0.0f) const;

  /**
   * @copydoc HalfTensor::dot(HalfTensor const &m, HalfTensor &output, bool
   trans, bool trans_m, float beta) const
   * @details performs dot operation over a batch of inputs
   */
  HalfTensor &dotBatched(HalfTensor const &m, HalfTensor &result,
                         bool trans = false, bool trans_m = false,
                         float beta = 0.0f) const;

  /**
   * @copydoc HalfTensor::dot_deriv_wrt_1(HalfTensor const &m, HalfTensor const
   &output_deriv, bool trans, bool trans_m, float beta)
   */
  HalfTensor &dot_batched_deriv_wrt_1(HalfTensor const &m,
                                      HalfTensor const &output_deriv,
                                      bool trans = false, bool trans_m = false,
                                      float beta = 0.0f);

  /**
   * @brief HalfTensor::dot_deriv_wrt_2(HalfTensor const &m_deriv, HalfTensor
   const &output_deriv, bool trans, bool trans_m, float beta) const
   */
  HalfTensor &dot_batched_deriv_wrt_2(HalfTensor &m_deriv,
                                      HalfTensor const &output_deriv,
                                      bool trans = false, bool trans_m = false,
                                      float beta = 0.0f) const;

  /**
   * @brief Transpose HalfTensor
   *
   * @param direction to transpose ex) 0:2:1
   * @return HalfTensor
   */
  HalfTensor transpose(const std::string &direction) const;

  /**
   * @brief Transpose HalfTensor
   * @param direction to transpose ex) 0:2:1
   * @param[out] HalfTensor to save to, dimension is always reshaped.
   * @retval HalfTensor& reference to the out
   */
  HalfTensor &transpose(const std::string &direction, HalfTensor &out) const;

  /**
   * @brief Calculate Drop Out Mask : x * 1.0/(1.0-rate)
   * @param dropout drop out rate
   * @retval HalfTensor& reference of drop out mask
   */
  HalfTensor dropout_mask(float dropout) const;

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
  void filter_mask(const HalfTensor &mask_len, bool reverse = false);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param zoneout zone out rate
   * @retval HalfTensor zone out mask for opposite tensor
   */
  HalfTensor zoneout_mask(float zoneout);

  /**
   * @brief Calculate 2 Zone Out Mask
   * @details Calculate zone out mask according to the bernoulli distribution.
   * Zone out mask with rate @a zoneout for inplace and the other zone out mask
   * with rate @a (1-zoneout).
   * @param opposite opposite zone out mask
   * @param zoneout zone out rate
   */
  void zoneout_mask(HalfTensor &opposite, float zoneout);

  /**
   * @brief     sum all the HalfTensor elements according to the batch
   * @retval    Calculated HalfTensor(batch, 1, 1, 1)
   */
  HalfTensor sum_by_batch() const;

  /**
   * @brief     sum all the HalfTensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated HalfTensor
   */
  HalfTensor sum(unsigned int axis, float alpha = 1.0) const;

  /**
   * @brief     sum all the HalfTensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @param[in] axis Axis to calculate sum along
   * @param[out] output output tensor
   * @param[in] alpha Scale the sum by this value
   * @retval    Calculated HalfTensor
   */
  HalfTensor &sum(unsigned int axis, HalfTensor &output, float alpha = 1.0,
                  float beta = 0.0) const;

  /**
   * @brief sum all the HalfTensor by multiple axes
   *
   * @param axes axes to sum along
   * @param alpha Scale the sum by this value
   * @return HalfTensor
   */
  HalfTensor sum(const std::vector<unsigned int> &axes,
                 float alpha = 1.0) const;

  /**
   * @brief sum all the HalfTensor by multiple axes
   *
   * @param axes axes to sum along
   * @param[out] output output tensor
   * @param alpha Scale the sum by this value
   * @return HalfTensor
   */
  HalfTensor &sum(const std::vector<unsigned int> &axes, HalfTensor &output,
                  float alpha = 1.0) const;

  /**
   * @brief     Averaging the HalfTensor elements according to the axis
   *            0 : batch direction
   *            1 : channel direction
   *            2 : height direction
   *            3 : width direction
   * @retval    Calculated HalfTensor
   */
  HalfTensor average(unsigned int axis) const;
  /**
   * @brief     Averaging the HalfTensor elements according to the axis
   *
   * @retval    Calculated HalfTensor
   */
  HalfTensor &average(unsigned int axis, HalfTensor &output) const;

  /**
   * @brief average all the HalfTensor by multiple axes
   *
   * @param axes axes to sum along
   * @return HalfTensor
   */
  HalfTensor average(const std::vector<unsigned int> &axes) const;

  /**
   * @brief average all the HalfTensor by multiple axes
   *
   * @param axes axes to sum along
   * @param output output tensor
   * @return HalfTensor
   */
  HalfTensor &average(const std::vector<unsigned int> &axes,
                      HalfTensor &output) const;

  /**
   * @brief     Averaging the HalfTensor elements by all axis
   * @retval    Calculated HalfTensor
   */
  HalfTensor average() const;

  /**
   * @brief     Averaging the HalfTensor elements by all axis
   * @retval    Calculated HalfTensor
   */
  HalfTensor &average(HalfTensor &output) const;

  /**
   * @brief     Anchor a starting point to defer following evaluation
   * @retval    LazyTensor class that can be used with run();
   */
  LazyTensor chain() const;

  /**
   * @brief     Softmax the HalfTensor elements
   * @retval    Calculated HalfTensor
   */
  HalfTensor softmax() const;

  /**
   * @brief     l2norm the HalfTensor elements
   * @retval    Calculated l2norm
   */
  float l2norm() const;

  /**
   * @brief     Normalize the HalfTensor elements
   * @retval    Calculated HalfTensor
   */
  HalfTensor &normalization(HalfTensor &output) const;

  /**
   * @brief     Standardize the HalfTensor elements
   * @retval    Calculated HalfTensor
   */
  HalfTensor &standardization(HalfTensor &output) const;

  /**
   * @brief     Normalize the HalfTensor elements in-place
   * @retval    Calculated HalfTensor
   */
  void normalization_i();

  /**
   * @brief     Standardize the HalfTensor elements in-place
   * @retval    Calculated HalfTensor
   */
  void standardization_i();

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  _FP16 *getAddress(unsigned int i) {
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
  const _FP16 *getAddress(unsigned int i) const {
    size_t index = getIndex(batch(), channel(), height(), width());
    if (i > index) {
      return nullptr;
    }

    return &getData()[i];
  }

  /**
   * @brief    get address of n-d data
   */
  _FP16 *getAddress(unsigned int b, unsigned int c, unsigned int h,
                    unsigned int w) {
    return getAddress(getIndex(b, c, h, w));
  }

  /**
   * @brief    get address of n-d data
   */
  const _FP16 *getAddress(unsigned int b, unsigned int c, unsigned int h,
                          unsigned int w) const {
    return getAddress(getIndex(b, c, h, w));
  }

  /**
   * @brief Apply instantly to the element
   *
   * @param f function to apply
   * @return int ML_ERROR_NONE if successful
   */
  int apply_i(std::function<_FP16(_FP16)> f) {
    HalfTensor result = *this;
    apply(f, result);

    return ML_ERROR_NONE;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    HalfTensor
   */
  HalfTensor &apply(std::function<_FP16(_FP16)> f, HalfTensor &output) const {
    CREATE_IF_EMPTY_DIMS_HALF(output, dim, nullptr);

    if (dim != output.dim) {
      /// @todo add unittest
      throw std::invalid_argument(
        "[HalfTensor::apply] output dimension does not match");
    }

    if (contiguous && output.contiguous) {
      const _FP16 *data = getData();
      _FP16 *rdata = output.getData();

      std::transform(data, data + size(), rdata, f);
    } else if (strides[3] == 1 && output.strides[3] == 1) {
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            _FP16 *out_data = output.getAddress(b, c, h, 0);
            const _FP16 *in_data = getAddress(b, c, h, 0);
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
   * @retval    HalfTensor
   */
  HalfTensor apply(std::function<_FP16(_FP16)> f) const {
    HalfTensor result;
    apply(f, result);

    return result;
  };

  /**
   * @brief     Apply function to HalfTensor
   * @param[in] *function function pointer applied
   * @retval    HalfTensor
   */
  HalfTensor apply(std::function<HalfTensor(HalfTensor)> f) const;

  /**
   * @brief     Apply function to HalfTensor
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    HalfTensor
   */
  HalfTensor &apply(std::function<HalfTensor &(HalfTensor, HalfTensor &)> f,
                    HalfTensor &output) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @retval    HalfTensor
   */
  void print(std::ostream &out) const;

  /**
   * @brief     Print element
   * @param[in] out out stream
   * @param[in] opt print formatting option. opt=0 would pretty print the data,
   * else it would print the raw data.
   * @retval    HalfTensor
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
  size_t bytes() const {
    if (getDataType() == Tdatatype::QINT4) {
      return (size() * dim.getDataTypeSize() + 1) / 2;
    }
    return size() * dim.getDataTypeSize();
  }

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
    getData()[getIndex(batch, c, h, w)] = static_cast<_FP16>(value);
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

    getData()[idx] *= static_cast<_FP16>(beta);
    getData()[idx] += static_cast<_FP16>(value);
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
   * @brief     Fill the HalfTensor elements with value
   * @param[in] value value to be stored
   */
  void setValue(float value);

  /**
   * @brief     Fill the HalfTensor elements with zero
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
      << getName() << " HalfTensor is not contiguous, cannot set distribution";

    _FP16 *data_ = getData();
    unsigned int len = size();
    for (unsigned int i = 0; i < len; ++i) {
      data_[i] = static_cast<_FP16>(dist(rng));
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
   * @param     fm format of HalfTensor
   */
  void convertFormat(TensorDim::Format fm) {
    if (getFormat() != fm) {
      transpose("2:1:0");
    }

    dim.setFormat(fm);
  }

  /**
   * @brief     Copy the HalfTensor
   * @param[in] from HalfTensor to be copied
   *
   * @note copy can reshape the tensor to match the shape
   */
  void copy(const HalfTensor &from);

  /**
   * @brief     Copy the HalfTensor
   * @param[in] from HalfTensor to be copied
   */
  void copyData(const HalfTensor &from);

  /**
   * @brief     Copy the HalfTensor
   * @param[in] from HalfTensor to be copied
   */
  void copy_with_stride(const HalfTensor &from);

  /**
   * @brief Get slice of the tensor, sliced by batch
   * @param[in] offset offset in batch to start the slice
   * @param[in] size size of the slice
   * @retval slice of this tensor
   * @note This function provides a slice of this tensor, and does not create a
   * copy
   */
  HalfTensor getBatchSlice(size_t offset, unsigned int size) const;

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
  HalfTensor getSharedDataTensor(const TensorDim dim, size_t offset,
                                 bool reset_stride = true,
                                 const std::string &name_ = "") const;
  /**
   * @brief split tensor along axis.
   *
   * @param num_size num_size
   * @param axis axis
   * @return HalfTensor splitted tensor
   */
  std::vector<HalfTensor> split(unsigned num_size, int axis = 0);

  /**
   * @brief split tensor along axis.
   *
   * @param sizes sizes
   * @param axis axis
   * @return HalfTensor splitted tensor
   * @note if the given array sizes is just a 1 unsigned int value, assumes that
   * it divide tensor by given size evenly
   */
  std::vector<HalfTensor> split(std::vector<size_t> sizes, int axis = 0);

  /**
   * @brief concatenate tensors along axis
   *
   * @param tensors tensors to be concatenated to the first tensor
   * @param axis axis
   * @return HalfTensor concatenated tensor
   */
  static HalfTensor cat(const std::vector<HalfTensor> &tensors, int axis = 0);

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
  void makeSharedDataTensor(const HalfTensor &src, size_t offset = 0);

  /**
   * @brief     Convient wrapper for inplace copy of @a this.
   * @retval    Copied version of this
   */
  HalfTensor clone() const;

  /**
   * @brief     Save the HalfTensor into file
   * @param[in] file output file stream
   */
  void save(std::ostream &file);

  /**
   * @brief     Read the HalfTensor from file
   * @param[in] file input file stream
   * @param[in] s_type scale factor data type
   */
  void read(std::ifstream &file, Tdatatype s_type = Tdatatype::FP32);

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
   * @brief     return a copy of the HalfTensor Dim
   * @retval    TensorDim
   */
  TensorDim getDim() const { return TensorDim(dim); }

  /**
   * @brief     return HalfTensor Dim for a given axis
   * @retval    dimension
   */
  size_t getTensorDim(unsigned int axis);

  /**
   * @brief     return HalfTensor Type
   */
  TensorDim::TensorType getTensorType() const { return dim.getTensorType(); };

  /**
   * @brief     return HalfTensor batch size
   * @retval    batch size
   */
  size_t batch() const { return dim.batch(); }

  /**
   * @brief     return HalfTensor batch size
   * @retval    batch size
   */
  size_t channel() const { return dim.channel(); }

  /**
   * @brief     return HalfTensor height size
   * @retval    height size
   */
  size_t height() const { return dim.height(); }

  /**
   * @brief     return HalfTensor batch size
   * @retval    width size
   */
  size_t width() const { return dim.width(); }

  /**
   * @brief     return HalfTensor Data Type Size
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
   * @brief     return Data pointer of HalfTensor
   * @retval    template T pointer (float pointer as default)
   */
  _FP16 *getData() {
    if (!data)
      return nullptr;

    data->validate();
    return data->getAddr<_FP16>() + offset;
  }

  /**
   * @brief     return Data pointer of HalfTensor
   * @retval    template T pointer (float pointer as default)
   */
  const _FP16 *getData() const {
    if (!data)
      return nullptr;

    data->validate();
    return data->getAddr<_FP16>() + offset;
  }

  /**
   * @brief     return Data pointer of HalfTensor
   * @retval    template T pointer (float pointer as default)
   */
  _FP16 *getData(size_t idx) const {
    if (!data)
      return nullptr;

    size_t index = idx;

    data->validate();
    return data->getAddr<_FP16>() + offset + index;
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
   * @brief     put data of HalfTensor
   *
   * @note      It is only effective when memory_swap is used
   */
  void putData() const {
    if (!data)
      return;

    data->invalidate();
  }

  /**
   * @brief     return Data pointer of HalfTensor
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
   * @brief     set HalfTensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  void reshape(const TensorDim &d);

  /**
   * @brief fill tensor data with current value,
   * if dimension is not exactly same, it is a hard error in this function
   * so, only stride is overriden to @a this
   *
   * @param from HalfTensor to fill the data from
   * @param allocate if unallocated, allocate with from.getDim()
   * @throws std::invalid_argument if dimension and stride does not match
   */
  void fill(const HalfTensor &from, bool allocate = false);

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
  HalfTensor::Initializer getInitializer() const { return initializer; }

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
  HalfTensor::Initializer initializer;
  std::string name; /**< name of the tensor */
  std::shared_ptr<MemoryData> data;
  size_t offset;

  /**<
   * When using shared_data with tensor, this stores the ptr of the source
   * tensor which handles the full memory. If tensor data is already allocated,
   * this does not affect the tensor. If the tensor data is not allocated, and
   * src_ptr is valid, this tensor will use the memory allocated by the src_ptr
   */
  std::shared_ptr<SrcSharedHalfTensor> src_tensor;

  struct BroadcastInfo;

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   * @param[in] m HalfTensor
   * @param[in] v_func vectorized function to apply
   * @param e broadcast info.
   * @param cur_axis current axis. pass default when calling outside.
   * @param offset offset for this.  pass default when calling outside.
   * @param m_offset offset for m.  pass default when calling outside.
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void
  apply_broadcast_util(HalfTensor const &m,
                       std::function<void(const BroadcastInfo &e, const _FP16 *,
                                          const _FP16 *, _FP16 *)>
                         v_func,
                       HalfTensor &output, const BroadcastInfo &e,
                       int cur_axis = -1, size_t offset = 0,
                       size_t m_offset = 0) const;
  /**
   * @brief Applies the given operator to the tensor with the passed argument
   *
   * @param[in] m HalfTensor
   * @param[in] v_func vectorized function to apply
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void apply_broadcast(HalfTensor const &m,
                       std::function<void(const BroadcastInfo &e, const _FP16 *,
                                          const _FP16 *, _FP16 *)>
                         v_func,
                       HalfTensor &output) const;
  /**
   * @brief compute Loop info for broadcasting and vectorization
   *
   * @param m target tensor to be calculated against.
   * @return BroadcastInfo Loopinfo needed to run external loop
   */
  BroadcastInfo computeBroadcastInfo(const HalfTensor &m) const;

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
  static void createSharedDataTensor(const HalfTensor &src, HalfTensor &dest,
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
   * @param[in] in input HalfTensor
   * @retVal HalfTensor rotated tensor (180 degree)
   */
  HalfTensor rotate_180(HalfTensor in);

}; // namespace nntrainer

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, HalfTensor const &m);

typedef std::shared_ptr<HalfTensor> sharedHalfTensor;

typedef std::shared_ptr<const HalfTensor> sharedConstHalfTensor;

typedef std::vector<sharedConstHalfTensor> sharedConstHalfTensors;

typedef std::vector<sharedHalfTensor> sharedHalfTensors;

} /* namespace nntrainer */

#endif /* ENABLE_FP16 */

#endif /* __cplusplus */
#endif /* __HALF_TENSOR_H__ */
