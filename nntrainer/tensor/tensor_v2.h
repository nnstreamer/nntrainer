// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_v2.h
 * @date	01 December 2023
 * @brief	This is a TensorV2 class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __TENSOR_V2_H__
#define __TENSOR_V2_H__
#ifdef __cplusplus

#include <cstddef>

#include <tensor_base.h>

namespace nntrainer {

/**
 * @class   TensorV2 Class
 * @brief   TensorV2 Class
 */
class TensorV2 {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  TensorV2(std::string name_ = "", Tformat fm = Tformat::NCHW,
           Tdatatype d_type = Tdatatype::FP32);

  /**
   * @brief     Constructor of Tensor with dimension, possibly lazily
   * @param d Tensor dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  TensorV2(const TensorDim &d, bool alloc_now,
           Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief     Constructor of Tensor with dimension/buf
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @note Memory for this tensor is instantaneously allocated
   */
  TensorV2(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief     Constructor of Tensor
   * @param[in] d0 Batch of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   */
  TensorV2(size_t d0, size_t d1, size_t d2, size_t d3,
           Tformat fm = Tformat::NCHW, Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(TensorDim(d0, d1, d2, d3, fm, d_type), nullptr){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   */
  TensorV2(size_t d1, size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
           Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(1, d1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of Tensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   */
  TensorV2(size_t d2, size_t d3, Tformat fm = Tformat::NCHW,
           Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(1, 1, d2, d3, fm, d_type){};

  /**
   * @brief     Constructor of Tensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] fm Tensor Format
   * @param[in] d_type Tensor Data Type
   */
  explicit TensorV2(size_t d3, Tformat fm = Tformat::NCHW,
                    Tdatatype d_type = Tdatatype::FP32) :
    TensorV2(1, 1, 1, d3, fm, d_type){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d0 Batch of Tensor
   * @param[in] d1 Channel (NCHW) or Height (NHWC)
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] t_type Tensor Type
   */
  TensorV2(size_t d0, size_t d1, size_t d2, size_t d3,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(TensorDim(d0, d1, d2, d3, t_type), nullptr){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d1 Channel
   * @param[in] d2 Height
   * @param[in] d3 Width
   * @param[in] t_type Tensor Type
   */
  TensorV2(size_t d1, size_t d2, size_t d3,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(1, d1, d2, d3, t_type){};

  /**
   * @brief     Constructor of Tensor with batch size one and d1 size one
   * @param[in] d2 Height (NCHW) or Width (NHWC)
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] t_type Tensor Type
   */
  TensorV2(size_t d2, size_t d3, ml::train::TensorDim::TensorType t_type) :
    TensorV2(1, (t_type.format == Tformat::NCHW) ? 1 : d3,
             (t_type.format == Tformat::NCHW) ? d2 : 1,
             (t_type.format == Tformat::NCHW) ? d3 : d2, t_type){};
  /**
   * @brief     Constructor of Tensor with just Width or Channel
   * @param[in] d3 Width (NCHW) or Channel (NHWC)
   * @param[in] t_type Tensor Type
   */
  explicit TensorV2(size_t d3, ml::train::TensorDim::TensorType t_type) :
    TensorV2(1, (t_type.format == Tformat::NCHW) ? 1 : d3, 1,
             (t_type.format == Tformat::NCHW) ? d3 : 1, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  TensorV2(std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
           ml::train::TensorDim::TensorType t_type);

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  TensorV2(std::vector<std::vector<std::vector<float>>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  TensorV2(std::vector<std::vector<float>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

#ifdef ENABLE_FP16
  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  TensorV2(std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
           ml::train::TensorDim::TensorType t_type);

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor. It needs to set format properly.
   * @param[in] t_type Tensor Type
   */
  TensorV2(std::vector<std::vector<std::vector<_FP16>>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

  /**
   * @brief     Constructor of Tensor
   * @note      This constructor copies vector again. needs refactoring
   * @param[in] d data for the Tensor with batch size one
   * @param[in] t_type Tensor Type
   */
  TensorV2(std::vector<std::vector<_FP16>> const &d,
           ml::train::TensorDim::TensorType t_type) :
    TensorV2(std::vector<std::decay<decltype(d)>::type>{d}, t_type){};

#endif

  /**
   * @brief Basic Destructor
   */
  ~TensorV2() { free(itensor); }

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
  bool isAllocated();

  /**
   * @brief     return Data pointer of TensorV2
   * @retval    template T pointer
   */
  template <typename T> T *getData() const { return (T *)itensor->getData(); }

  /**
   * @brief     return Data pointer of TensorV2
   * @retval    template T pointer
   */
  template <typename T> T *getData(size_t idx) const {
    return (T *)itensor->getData(idx);
  }

  /**
   * @brief     i data index
   * @retval    template T pointer (address of ith data)
   */
  template <typename T> T *getAddress(unsigned int i) {
    return (T *)itensor->getAddress(i);
  }

  /**
   * @brief     i data index
   * @retval    template T pointer (address of ith data)
   */
  template <typename T> const T *getAddress(unsigned int i) const {
    return (T *)itensor->getAddress(i);
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
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  template <typename T = float>
  const T &getValue(unsigned int b, unsigned int c, unsigned int h,
                    unsigned int w) const noexcept {
    return getValue<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  template <typename T = float>
  T &getValue(unsigned int b, unsigned int c, unsigned int h,
              unsigned int w) noexcept {
    return getValue<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief     Fill the Tensor elements with value
   * @param[in] value value to be stored
   */
  void setValue(float value);

  /**
   * @brief     Set the element value
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   * @param[in] value value to be stored
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value);

  /**
   * @brief     Fill the Tensor elements with zero
   */
  void setZero();

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
   * @brief     Print element
   * @param[in] out out stream
   */
  void print(std::ostream &out) const;

  /**
   * @brief     put data of Tensor
   * @note      It is only effective when memory_swap is used
   */
  void putData() const;

  /**
   * @brief Get initializer for the tensor
   *
   * @return initializer of the tensor
   */
  Initializer getInitializer() const;

  /**
   * @brief Get format for the tensor
   * @return format of the tensor
   */
  TensorDim::Format getFormat() const;

  /**
   * @brief Get data type for the tensor
   *
   * @return data type of the tensor
   */
  Tdatatype getDataType() const;

  /**
   * @brief Get linear index given the n-d index
   */
  size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                  unsigned int w) const noexcept;
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
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  size_t batch() const;

  /**
   * @brief     return Tensor channel size
   * @retval    channel size
   */
  size_t channel() const;

  /**
   * @brief     return Tensor height size
   * @retval    height size
   */
  size_t height() const;

  /**
   * @brief     return Tensor width size
   * @retval    width size
   */
  size_t width() const;

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
  void createSharedDataTensor(const TensorV2 &src, TensorV2 &dest,
                              size_t offset) const;

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
  TensorV2 getSharedDataTensor(const TensorDim dim_, size_t offset,
                               bool reset_stride,
                               const std::string &name_) const;

private:
  TensorBase *itensor;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_V2_H__ */
