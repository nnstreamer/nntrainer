// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file	tensor_v2.h
 * @date	16 November 2023
 * @brief	This is Tensor class with Type erause
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __TENSOR_V2_H__
#define __TENSOR_V2_H__
#ifdef __cplusplus

#include <array>
#include <functional>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

#include <blas_interface.h>
#include <float_tensor.h>
#include <half_tensor.h>
#include <iostream>
#include <memory_data.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <src_shared_tensor.h>
#include <tensor_base.h>
#include <tensor_dim.h>
#include <util_func.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

#define MAKE_SHARED_TENSOR_V2(...) \
  std::make_shared<nntrainer::TensorV2>(__VA_ARGS__)

#define CREATE_IF_EMPTY_DIMS_V2(tensor, ...) \
  do {                                       \
    if (tensor.empty())                      \
      tensor = TensorV2(__VA_ARGS__);        \
  } while (0);

namespace nntrainer {

using TensorDim = ml::train::TensorDim;
using Tformat = ml::train::TensorDim::Format;
using Tdatatype = ml::train::TensorDim::DataType;

/**
 * @class   TensorV2 Class
 * @brief   TensorV2 Class
 */
class TensorV2 {
public:
  /**
   * @brief     Basic Constructor of TensorV2
   */
  TensorV2(std::string name_ = "", Tformat fm = Tformat::NCHW,
           Tdatatype d_type = Tdatatype::FP32);

  /**
   * @brief     Constructor of TensorV2 with dimension, possibly lazily
   * @param d TensorV2 dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  TensorV2(const TensorDim &d, bool alloc_now,
           Initializer init = Initializer::NONE, std::string name = "");
  /**
   * @brief     Constructor of TensorV2 with dimension/buf
   * @param d TensorV2 dim for this tensor
   * @param buf buffer
   * @note Memory for this tensor is instantaneously allocated
   */
  TensorV2(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief     Constructor of TensorV2
   * @param[in] d0 Batch of TensorV2
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  TensorV2(size_t d0, size_t d1, size_t d2, size_t d3,
           Tformat fm = Tformat::NCHW, Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(TensorDim(d0, d1, d2, d3, fm, d_type), nullptr){};

  /**
   * @brief     Constructor of TensorV2
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   */
  TensorV2(size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
           Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(1, d1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of TensorV2 with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  TensorV2(size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
           Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(1, 1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of TensorV2 with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  explicit TensorV2(size_t d3, Tformat fm = Tformat::NCHW,
                    Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(1, 1, 1, d3, fm, d_type){};

  /**
   * @brief     Constructor of TensorV2
   * @param[in] d0 Batch of TensorV2
   * @param[in] d1 Channel (NCHW) or Height (NHWC)
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   */
  TensorV2(size_t d0, size_t d1, size_t d2, size_t d3,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(TensorDim(d0, d1, d2, d3, t_type), nullptr){};

  /**
   * @brief     Constructor of TensorV2 using FloatTensor
   * @param[in] d data for the TensorV2. It needs to set format properly.
   */
  TensorV2(std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
           ml::train::TensorDim::TensorType t_type);

  /**
   * @brief     Constructor of TensorV2 using FloatTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the TensorV2. It needs to set format properly.
   */
  TensorV2(std::vector<std::vector<std::vector<float>>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of TensorV2 using FloatTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the TensorV2 with batch size one
   */
  TensorV2(std::vector<std::vector<float>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

#ifdef ENABLE_FP16
  /**
   * @brief     Constructor of TensorV2 using HalfTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the TensorV2 with batch size one
   */
  TensorV2(std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
           ml::train::TensorDim::TensorType t_type);

  /**
   * @brief     Constructor of TensorV2 using HalfTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the TensorV2. It needs to set format properly.
   */
  TensorV2(std::vector<std::vector<std::vector<_FP16>>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of TensorV2 using HalfTensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the TensorV2 with batch size one
   */
  TensorV2(std::vector<std::vector<_FP16>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};
#endif

  /**
   *  @brief  Copy constructor of Tensor.
   *  @param[in] TensorV2 &
   */
  TensorV2(const TensorV2 &rhs) = default;

  /**
   *  @brief  Move constructor of Tensor.
   *  @param[in] TensorV2 &&
   */
  TensorV2(TensorV2 &&rhs) noexcept = default;

  /**
   * @brief  Copy assignment operator.
   * @param[in] rhs Tensor to be copied.
   */
  TensorV2 &operator=(const TensorV2 &rhs) = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs Tensor to be moved.
   */
  TensorV2 &operator=(TensorV2 &&rhs) noexcept = default;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator==(const TensorV2 &rhs) const;
  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   */
  bool operator!=(const TensorV2 &rhs) const;

  /**
   * @brief     Swap tensor
   * @param[in] lhs Tensor to be swapped
   * @param[in] rhs Tensor to be swapped
   */
  friend void swap(TensorV2 &lhs, TensorV2 &rhs) noexcept {
    swap(lhs.object, rhs.object);
  }

  /**
   * @brief    Allocate memory for this tensor
   */
  void allocate();

  /**
   * @brief    Deallocate memory for this tensor
   * @note     This will not necessary free the memory as tensors share memory
   */
  void deallocate();

  /**
   * @brief    Check if the tensor has memory allocated/assigned/associated
   */
  bool isAllocated() const;

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

  /**
   * @brief     return value at specific location
   * @param[in] batch batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
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
    return ((T *)getData())[idx];
  }

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  template <typename T = float> T &getValue(unsigned int idx) noexcept {
    return ((T *)getData())[idx];
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
   * @brief Set the memory buffer for the tensor
   *
   * @param buf the memory buffer
   * @param init intialize the buffer
   */
  void setData(const std::shared_ptr<MemoryData> buf, size_t off = 0,
               bool init = false);

  /**
   * @brief     return Data pointer of Tensor
   * @retval    void pointer
   * @note      this function will be removed
   */
  const void *getData() const;

  /**
   * @brief     return Data pointer of Tensor
   * @retval    void pointer
   * @note      this function will be removed
   */
  void *getData(size_t idx) const;

  /**
   * @brief  getter of size of data
   * @retval size of data
   */
  unsigned int sizeofData() const;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  void *getAddress(unsigned int i);

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  const void *getAddress(unsigned int i) const;

  /**
   * @brief    get address of n-d data
   */
  void *getAddress(unsigned int b, unsigned int c, unsigned int h,
                   unsigned int w);

  /**
   * @brief    get address of n-d data
   */
  const void *getAddress(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w) const;

  /**
   * @brief     Fill the Tensor elements with value
   * @param[in] value value to be stored
   */
  void setValue(float value);

  /**
   * @brief     Set the element value
   * @param[in] batch batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   * @param[in] value value to be stored
   */
  void setValue(unsigned int batch, unsigned int c, unsigned int h,
                unsigned int w, float value) noexcept;

  /**
   * @brief     add the element value to the location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   * @param[in] value value to be stored
   * @param[in] beta scalar to multiply output with and add
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) noexcept;

  /**
   * @brief     Fill the Tensor elements with zero
   */
  void setZero();

  /**
   * @brief     Set the tensor with random normal distribution
   * @param[in] mean mean of the distribution
   * @param[in] std standard deviation of the distribution
   */
  void setRandNormal(float mean, float std);

  /**
   * @brief     Set the tensor with random uniform distribution
   * @param[in] min minimum value for the distribution
   * @param[in] max maximum value for the distribution
   */
  void setRandUniform(float min, float max);

  /**
   * @brief     Set the tensor with random bernoulli distribution
   * @param[in] probability probability value for the distribution
   */
  void setRandBernoulli(float probability);

  /**
   * @brief     Initialize the memory of the given tensor
   */
  void initialize();

  /**
   * @brief     Initialize the memory of the given tensor
   * @param     init Initiailizer to use for the initialization
   */
  void initialize(Initializer init);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   *
   * @note copy can reshape the tensor to match the shape
   */
  void copy(const TensorV2 &from);

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
  size_t size() const;

  /**
   * @brief     Get if the tensor is empty
   * @retval    true if the tensor is empty
   */
  bool empty() const;

  /**
   * @brief     Get size of the data in bytes
   * @retval    size_t Size in bytes
   */
  size_t bytes() const;

  /**
   * @brief Get linear index given the n-d index
   */
  size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                  unsigned int w) const noexcept;

  /**
   * @brief Check if two given axes are contiguous
   */
  bool checkContinuous(unsigned int n, unsigned int np1) const;

  /**
   * @brief   Get name of the tensor
   *
   * @return name of the tensor
   */
  void setName(const std::string &name_);

  /**
   * @brief   Get name of the tensor
   *
   * @return name of the tensor
   */
  const std::string &getName() const;

  /**
   * @brief Get initializer for the tensor
   * @retval initializer of the tensor
   */
  Initializer getInitializer() const;

  /**
   * @brief     return a copy of the Tensor Dim
   * @retval    TensorDim
   */
  TensorDim getDim() const;

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  const std::array<size_t, TensorDim::MAXDIM> getStrides() const noexcept;

  /**
   * @brief     return contiguous state of tensor.
   * @retval    bool contiguous
   */
  bool getContiguous() const;

  /**
   * @brief     return Tensor Type
   */
  TensorDim::TensorType getTensorType() const;

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  size_t batch() const;

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  size_t channel() const;

  /**
   * @brief     return Tensor height size
   * @retval    height size
   */
  size_t height() const;

  /**
   * @brief     return Tensor batch size
   * @retval    width size
   */
  size_t width() const;

  /**
   * @brief     return Tensor Data Type Size
   * @retval    data type size
   */
  uint getDataTypeSize() const;

  /**
   * @brief Get format for the tensor
   * @retval format of the tensor
   */
  TensorDim::Format getFormat() const;

  /**
   * @brief Get data type for the tensor
   * @retval data type of the tensor
   */
  Tdatatype getDataType() const;

  /**
   * @brief     set Tensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  void reshape(const TensorDim &d);

  /**
   * @brief Apply instantly to the element
   *
   * @param f function to apply
   * @return int ML_ERROR_NONE if successful
   */
  template <typename T = float> int apply_i(std::function<T(T)> f) {
    TensorV2 result = *this;
    apply<T>(f, result);

    return ML_ERROR_NONE;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @retval    TensorV2
   */
  template <typename T = float> TensorV2 apply(std::function<T(T)> f) const {
    TensorV2 result;
    apply<T>(f, result);

    return result;
  };

  /**
   * @brief     Apply function element by element
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    Tensor
   */
  template <typename T = float>
  TensorV2 &apply(std::function<T(T)> f, TensorV2 &output) const {
    TensorDim dim = getDim();
    CREATE_IF_EMPTY_DIMS_V2(output, dim, nullptr);

    if (dim != output.getDim()) {
      /// @todo add unittest
      throw std::invalid_argument(
        "[Tensor::apply] output dimension does not match");
    }

    if (getContiguous() && output.getContiguous()) {
      const T *data = (T *)getData();
      T *rdata = (T *)output.getData();

      std::transform(data, data + size(), rdata, f);
    } else if (getStrides()[3] == 1 && output.getStrides()[3] == 1) {
      /** @todo optimize this with combining these loops where stride is 1 */
      for (unsigned int b = 0; b < batch(); ++b) {
        for (unsigned int c = 0; c < channel(); ++c) {
          for (unsigned int h = 0; h < height(); ++h) {
            T *out_data = (T *)output.getAddress(b, c, h, 0);
            const T *in_data = (T *)getAddress(b, c, h, 0);
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

    return output;
  };

  /**
   * @brief     Apply function to TensorV2
   * @param[in] *function function pointer applied
   * @retval    TensorV2
   */
  TensorV2 apply(std::function<TensorV2(TensorV2)> f) const;

  /**
   * @brief     Apply function to TensorV2
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    TensorV2
   */
  TensorV2 &apply(std::function<TensorV2 &(TensorV2, TensorV2 &)> f,
                  TensorV2 &output) const;

private:
  std::shared_ptr<TensorConcept> object;

  struct BroadcastInfoV2;
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
  void apply_broadcast_util(
    TensorV2 const &m,
    std::function<void(const BroadcastInfoV2 &e, const float *, const float *,
                       float *)>
      v_func,
    TensorV2 &output, const BroadcastInfoV2 &e, int cur_axis = -1,
    size_t offset = 0, size_t m_offset = 0) const;

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   *
   * @param[in] m Tensor
   * @param[in] v_func vectorized function to apply
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void
  apply_broadcast(TensorV2 const &m,
                  std::function<void(const BroadcastInfoV2 &e, const float *,
                                     const float *, float *)>
                    v_func,
                  TensorV2 &output) const;

  /**
   * @brief compute Loop info for broadcasting and vectorization
   *
   * @param m target tensor to be calculated against.
   * @return BroadcastInfo Loopinfo needed to run external loop
   */
  BroadcastInfoV2 computeBroadcastInfo(const TensorV2 &m) const;

  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);

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

}; // namespace nntrainer

/**
 * @brief   Overriding output stream
 */
std::ostream &operator<<(std::ostream &out, TensorV2 const &m);

typedef std::shared_ptr<TensorV2> sharedTestTensor;

typedef std::shared_ptr<const TensorV2> sharedConstTestTensor;

typedef std::vector<sharedConstTestTensor> sharedConstTestTensors;

typedef std::vector<sharedTestTensor> sharedTestTensors;

} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __TENSOR_V2_H__ */
