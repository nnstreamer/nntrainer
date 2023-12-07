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
