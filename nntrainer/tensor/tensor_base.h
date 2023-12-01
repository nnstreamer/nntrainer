// SPDX-License-Identifier: Apache-2.0
/**
 * @file	tensor_base.h
 * @date	01 December 2023
 * @brief	This is Tensor base class
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __TENSOR_BASE_H__
#define __TENSOR_BASE_H__
#ifdef __cplusplus

#include <memory>
#include <stdexcept>

#include <memory_data.h>
#include <tensor_dim.h>

namespace nntrainer {

using TensorDim = ml::train::TensorDim;
using Tformat = ml::train::TensorDim::Format;
using Tdatatype = ml::train::TensorDim::DataType;

class TensorV2;

/**
 * @class SrcSharedTensorV2
 * @brief Source of the shared tensor
 */
class SrcSharedTensorV2 {
public:
  /**
   * @brief   Constructor for the class
   */
  SrcSharedTensorV2() : src(nullptr), off(0) {}

  /**
   * @brief   Constructor for the class
   */
  SrcSharedTensorV2(const TensorV2 *tensor, size_t offset) :
    src(tensor),
    off(offset) {}

  /**
   * @brief   Get the allocated src tensor
   */
  const TensorV2 *tensor() const {
    if (!src)
      throw std::runtime_error("Accessing empty src tensor");

    return src;
  }

  /**
   * @brief   Get the offset from the source tensor
   */
  size_t offset() const { return off; }

private:
  const TensorV2 *src; /**< Tensor of the source */
  size_t off;          /**< offset from the source data ptr */
};

/**
 * @class TensorBase class
 * @brief TensorBase is an abstract class
 */
class TensorBase {
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
  TensorBase(std::string name_ = "", Tformat fm = Tformat::NCHW,
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
   * @brief Basic Destructor
   */
  virtual ~TensorBase() {}

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
  bool isAllocated() { return false; }

  /**
   * @copydoc TensorV2::getData()
   */
  virtual void *getData() const = 0;

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
   * @copydoc TensorV2::getIndex()
   */
  size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                  unsigned int w) const noexcept {
    return 0;
  }

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  virtual void setValue(float value) = 0;

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  virtual void setValue(unsigned int batch, unsigned int c, unsigned int h,
                        unsigned int w, float value) = 0;

  /**
   * @copydoc TensorV2::setZero()
   */
  virtual void setZero() = 0;

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

protected:
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
  std::shared_ptr<SrcSharedTensorV2> src_tensor;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_BASE_H__ */
