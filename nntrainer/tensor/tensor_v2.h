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
#include <tensor_dim.h>
#include <util_func.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

#define MAKE_SHARED_TEST_TENSOR(...) \
  std::make_shared<nntrainer::TensorV2>(__VA_ARGS__)

#define CREATE_IF_EMPTY_DIMS_TEST(tensor, ...) \
  do {                                         \
    if (tensor.empty())                        \
      tensor = TensorV2(__VA_ARGS__);          \
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
private:
  class TensorConcept {
  public:
    virtual ~TensorConcept() {}
    /**
     * @copydoc TensorV2::allocate()
     */
    virtual void allocate() = 0;

    /**
     * @copydoc TensorV2::deallocate()
     */
    virtual void deallocate() = 0;

    /**
     * @copydoc TensorV2::isAllocated()
     */
    virtual bool isAllocated() = 0;

    /**
     * @copydoc TensorV2::getData()
     */
    virtual const void *getData() const = 0;

    /**
     * @copydoc TensorV2::getData(size_t idx)
     */
    virtual void *getData(size_t idx) const = 0;

    /**
     * @brief     i data index
     * @retval    address of ith data
     */
    virtual void *getAddress(unsigned int i) = 0;

    /**
     * @brief     i data index
     * @retval    address of ith data
     */
    virtual const void *getAddress(unsigned int i) const = 0;

    /**
     * @copydoc TensorV2::setValue(float value)
     */
    virtual void setValue(float value) = 0;

    /**
     * @copydoc TensorV2::setZero()
     */
    virtual void setZero() = 0;

    /**
     * @copydoc TensorV2::setRandNormal(float mean, float std)
     */
    virtual void setRandNormal(float mean, float std) = 0;

    /**
     * @copydoc TensorV2::setRandUniform(float min, float max)
     */
    virtual void setRandUniform(float min, float max) = 0;

    /**
     * @copydoc TensorV2::setRandBernoulli(float probability)
     */
    virtual void setRandBernoulli(float probability) = 0;

    /**
     * @copydoc TensorV2::initialize()
     */
    virtual void initialize() = 0;

    /**
     * @copydoc TensorV2::initialize(Initializer init)
     */
    virtual void initialize(Initializer init) = 0;

    /**
     * @copydoc TensorV2::print(std::ostream &out)
     */
    virtual void print(std::ostream &out) const = 0;

    /**
     * @copydoc TensorV2::getIndex()
     */
    virtual size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w) const noexcept = 0;

    /**
     * @copydoc TensorV2::getInitializer()
     */
    virtual Initializer getInitializer() const = 0;

    /**
     * @copydoc TensorV2::getFormat()
     */
    virtual TensorDim::Format getFormat() const = 0;

    /**
     * @copydoc TensorV2::getDataType()
     */
    virtual Tdatatype getDataType() const = 0;
  };

  /**
   * @brief TensorBase class
   * @note TensorClass : FloatTensor, HalfTensor, etc.
   */
  template <typename TensorClass> class TensorBase : public TensorConcept {
  public:
    /**
     * @brief     Basic Constructor of TensorBase
     */
    TensorBase(TensorClass t) : object(t) {}

    /**
     * @brief     Basic Destructor of TensorBase
     */
    virtual ~TensorBase() {}

    /**
     * @copydoc TensorV2::allocate()
     */
    virtual void allocate() { object.allocate(); }

    /**
     * @copydoc TensorV2::deallocate()
     */
    virtual void deallocate() { object.deallocate(); }

    /**
     * @copydoc TensorV2::isAllocated()
     */
    virtual bool isAllocated() { return object.isAllocated(); }

    /**
     * @copydoc TensorV2::getData()
     */
    virtual const void *getData() const { return object.getData(); }

    /**
     * @copydoc TensorV2::getData(size_t idx)
     */
    virtual void *getData(size_t idx) const { return object.getData(idx); }

    /**
     * @brief     i data index
     * @retval    address of ith data
     */
    virtual void *getAddress(unsigned int i) { return object.getAddress(i); }

    /**
     * @brief     i data index
     * @retval    address of ith data
     */
    virtual const void *getAddress(unsigned int i) const {
      return object.getAddress(i);
    }

    /**
     * @copydoc TensorV2::setValue(float value)
     */
    virtual void setValue(float value) { object.setValue(value); }

    /**
     * @copydoc TensorV2::setZero()
     */
    virtual void setZero() { object.setZero(); }

    /**
     * @copydoc TensorV2::setRandNormal(float mean, float std)
     */
    virtual void setRandNormal(float mean, float std) {
      object.setRandNormal(mean, std);
    }

    /**
     * @copydoc TensorV2::setRandUniform(float min, float max)
     */
    virtual void setRandUniform(float min, float max) {
      object.setRandUniform(min, max);
    }

    /**
     * @copydoc TensorV2::setRandBernoulli(float probability)
     */
    virtual void setRandBernoulli(float probability) {
      object.setRandBernoulli(probability);
    }

    /**
     * @copydoc TensorV2::initialize()
     */
    virtual void initialize() { object.initialize(); }

    /**
     * @copydoc TensorV2::initialize(Initializer init)
     */
    virtual void initialize(Initializer init) { object.initialize(init); }

    /**
     * @copydoc TensorV2::print(std::ostream &out)
     */
    virtual void print(std::ostream &out) const { object.print(out); }

    /**
     * @copydoc TensorV2::getIndex()
     */
    virtual size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                            unsigned int w) const noexcept {
      return object.getIndex(b, c, h, w);
    }

    /**
     * @copydoc TensorV2::getInitializer()
     */
    virtual Initializer getInitializer() const {
      return object.getInitializer();
    }

    /**
     * @copydoc TensorV2::getFormat()
     */
    virtual TensorDim::Format getFormat() const { return object.getFormat(); }

    /**
     * @copydoc TensorV2::getDataType()
     */
    virtual Tdatatype getDataType() const { return object.getDataType(); }

  private:
    TensorClass object;
  };

  std::shared_ptr<TensorConcept> object;

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
  template <typename TensorClass>
  TensorClass &operator=(const TensorClass &rhs);

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Tensor to be moved.
   */
  template <typename TensorClass>
  TensorClass &operator=(TensorClass &&rhs) noexcept;

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
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  const void *getData() const;

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  void *getData(size_t idx) const;

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
   * @brief     Print element
   * @param[in] out out stream
   * @retval    Tensor
   */
  void print(std::ostream &out) const;

  /**
   * @brief Get linear index given the n-d index
   */
  size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                  unsigned int w) const noexcept;

  /**
   * @brief Get initializer for the tensor
   * @retval initializer of the tensor
   */
  Initializer getInitializer() const;

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