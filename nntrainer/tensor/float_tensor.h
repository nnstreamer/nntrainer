// SPDX-License-Identifier: Apache-2.0
/**
 * @file	float_tensor.h
 * @date	01 December 2023
 * @brief	This is FloatTensor class for 32-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __FLOAT_TENSOR_H__
#define __FLOAT_TENSOR_H__
#ifdef __cplusplus

#include <tensor_base.h>
#include <tensor_v2.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

namespace nntrainer {

/**
 * @class FloatTensor class
 * @brief FloatTensor class for 32-bit floating point calculation
 */
class FloatTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  FloatTensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Basic Destructor
   */
  ~FloatTensor() {}

  /**
   * @copydoc TensorV2::allocate()
   */
  void allocate() override;

  /**
   * @copydoc TensorV2::deallocate()
   */
  void deallocate() override;

  /**
   * @copydoc TensorV2::getData()
   */
  void *getData() const override;

  /**
   * @copydoc TensorV2::getData(size_t idx)
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
   * @copydoc TensorV2::setValue(float value)
   */
  void setValue(float value) override;

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  void setValue(unsigned int batch, unsigned int c, unsigned int h,
                unsigned int w, float value) override;

  /**
   * @copydoc TensorV2::setZero()
   */
  void setZero() override;

  /**
   * @copydoc TensorV2::initialize()
   */
  void initialize() override;

  /**
   * @copydoc TensorV2::initialize(Initializer init)
   */
  void initialize(Initializer init) override;

  /**
   * @copydoc TensorV2::print(std::ostream &out)
   */
  void print(std::ostream &out) const override;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FLOAT_TENSOR_H__ */
