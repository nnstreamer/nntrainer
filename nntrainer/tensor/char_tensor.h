// SPDX-License-Identifier: Apache-2.0
/**
 * @file	char_tensor.h
 * @date	02 April 2024
 * @brief	This is CharTensor class for 8-bit integer calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __CHAR_TENSOR_H__
#define __CHAR_TENSOR_H__
#ifdef __cplusplus

#include <tensor_base.h>

namespace nntrainer {

/**
 * @class CharTensor class
 * @brief CharTensor class for 8-bit integer calculation
 */
class CharTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  CharTensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new CharTensor object
   *
   * @param d Tensor dim for this float tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  CharTensor(const TensorDim &d, bool alloc_now,
             Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new CharTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  CharTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new CharTensor object
   *
   * @param d data for the Tensor
   * @param fm format for the Tensor
   */
  CharTensor(
    std::vector<std::vector<std::vector<std::vector<int8_t>>>> const &d,
    Tformat fm);

  /**
   * @brief Construct a new CharTensor object
   * @param rhs TensorBase object to copy
   */
  CharTensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @brief Basic Destructor
   */
  ~CharTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator==(const CharTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator!=(const CharTensor &rhs) const { return !(*this == rhs); }

  /**
   * @copydoc Tensor::allocate()
   */
  void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  void deallocate() override;

  /**
   * @copydoc Tensor::getData()
   */
  void *getData() const override;

  /**
   * @copydoc Tensor::getData(size_t idx)
   */
  void *getData(size_t idx) const override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  void *getAddress(unsigned int i) override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  const void *getAddress(unsigned int i) const override;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  const int8_t &getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  int8_t &getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const int8_t &getValue(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  int8_t &getValue(unsigned int b, unsigned int c, unsigned int h,
                   unsigned int w);

  /**
   * @copydoc Tensor::setValue(float value)
   */
  void setValue(float value) override;

  /**
   * @copydoc Tensor::setValue(b, c, h, w, value)
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override;

  /**
   * @copydoc Tensor::addValue(b, c, h, w, value, beta)
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override;

  /**
   * @copydoc Tensor::setZero()
   */
  void setZero() override;

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::initialize(Initializer init)
   */
  void initialize(Initializer init) override;

  /**
   * @copydoc Tensor::copy(const Tensor &from)
   */
  void copy(const Tensor &from) override;

  /**
   * @copydoc Tensor::copyData(const Tensor &from)
   */
  void copyData(const Tensor &from) override;

  /**
   * @copydoc Tensor::copy_with_stride()
   */
  void copy_with_stride(const Tensor &input, Tensor &output) override;

  /**
   * @copydoc Tensor::argmax()
   */
  std::vector<unsigned int> argmax() const override;

  /**
   * @copydoc Tensor::max_abs()
   */
  float max_abs() const override;

  /**
   * @copydoc Tensor::maxValue()
   */
  float maxValue() const override;

  /**
   * @copydoc Tensor::minValue()
   */
  float minValue() const override;

  /**
   * @copydoc Tensor::print(std::ostream &out)
   */
  void print(std::ostream &out) const override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  void copy(const void *buf);

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (QINT8)
   */
  std::string getStringDataType() const override { return "QINT8"; }

  /**
   * @copydoc Tensor::isValid()
   */
  bool isValid() const override { return true; }; // NYI
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CHAR_TENSOR_H__ */
