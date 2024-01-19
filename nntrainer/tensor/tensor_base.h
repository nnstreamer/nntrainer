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
#include <nntrainer_error.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace nntrainer {

using TensorDim = ml::train::TensorDim;
using Tformat = ml::train::TensorDim::Format;
using Tdatatype = ml::train::TensorDim::DataType;

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

class TensorV2;
class SrcSharedTensorBase;

/**
 * @class TensorBase class
 * @brief TensorBase is an abstract class
 */
class TensorBase {
public:
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
   * @brief     Constructor of Tensor with dimension, possibly lazily
   * @param d Tensor dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  TensorBase(const TensorDim &d, bool alloc_now,
             Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief     Constructor of Tensor with dimension/buf
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @note Memory for this tensor is instantaneously allocated
   */
  TensorBase(const TensorDim &d, const void *buf = nullptr) :
    TensorBase(d, true) {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor information
   */
  bool operator==(const TensorBase &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor information
   */
  bool operator!=(const TensorBase &rhs) const { return !(*this == rhs); }

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
  bool isAllocated() { return data != nullptr; }

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
   * @copydoc TensorV2::setRandNormal()
   */
  virtual void setRandNormal(float mean, float stddev) = 0;

  /**
   * @copydoc TensorV2::setRandBernoulli()
   */
  virtual void setRandUniform(float min, float max) = 0;

  /**
   * @copydoc TensorV2::setRandBernoulli()
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
   * @copydoc TensorV2::multiply_i(float const &value)
   */
  virtual int multiply_i(float const &value) = 0;

  /**
   * @copydoc TensorV2::multiply(float const &value, TensorV2 &out)
   */
  virtual TensorV2 &multiply(float const &value, TensorV2 &out) const = 0;

  /**
   * @copydoc TensorV2::multiply(TensorV2 const &m, TensorV2 &output, const
   * float beta = 0.0)
   */
  virtual TensorV2 &multiply(TensorV2 const &m, TensorV2 &output,
                             const float beta = 0.0) const = 0;

  /**
   * @copydoc TensorV2::print(std::ostream &out)
   */
  virtual void print(std::ostream &out) const = 0;

  /**
   * @copydoc TensorV2::apply(std::function<T(T)> f, TensorV2 &output)
   */
  virtual TensorV2 &apply(std::function<float(float)> f,
                          TensorV2 &output) const {
    return output;
  }

#ifdef ENABLE_FP16
  /**
   * @copydoc TensorV2::apply(std::function<T(T)> f, TensorV2 &output)
   */
  virtual TensorV2 &apply(std::function<_FP16(_FP16)> f,
                          TensorV2 &output) const {
    return output;
  }
#endif

  /**
   * @brief     put data of Tensor
   * @note      It is only effective when memory_swap is used
   */
  void putData() const;

  /**
   * @brief     return a copy of the Tensor Dim
   * @retval    TensorDim
   */
  TensorDim getDim() const { return TensorDim(dim); }

  /**
   * @brief     return Tensor Type
   */
  TensorDim::TensorType getTensorType() const { return dim.getTensorType(); }

  /**
   * @brief Get initializer for the tensor
   * @retval initializer of the tensor
   */
  Initializer getInitializer() const { return initializer; }

  /**
   * @brief Get format for the tensor
   * @retval format of the tensor
   */
  TensorDim::Format getFormat() const { return dim.getFormat(); }

  /**
   * @brief Get data type for the tensor
   * @retval data type of the tensor
   */
  Tdatatype getDataType() const { return dim.getDataType(); }

  /**
   * @brief     return whether tensor is contiguous or not.
   * @retval    bool contiguous
   */
  const bool getContiguous() const noexcept { return contiguous; }

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  const std::array<size_t, TensorDim::MAXDIM> getStrides() const noexcept {
    return strides;
  }

  /**
   * @brief     Set name of the tensor
   */
  void setName(const std::string &name_) { name = name_; }

  /**
   * @brief     Get name of the tensor
   * @retval    string name
   */
  const std::string &getName() const { return name; }

  /**
   * @brief Get linear index given the n-d index
   */
  size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                  unsigned int w) const noexcept;

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
   * @brief Allocate data based on the source tensor
   * @note As this memory is shared, do NOT initialize
   */
  void allocateSrcTensor();

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
  void createSharedDataTensor(const TensorBase *src, TensorBase *dest,
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
  TensorBase *getSharedDataTensor(const TensorDim dim_, size_t offset,
                                  bool reset_stride, const std::string &name_);

  static constexpr float epsilon = 1e-5;

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
  std::shared_ptr<SrcSharedTensorBase> src_tensor;

  /**
   * @struct External Loop Info for broadcasted info
   * @brief External Loop Info for broadcasted iteration. Please refer to
   * DISABLED_private_external_loop_n in unittest_nntrainer_tensor.
   * @note This should better be implemented in iterator fashion before used
   * extensively.
   */
  struct BroadcastInfoV2 {

    /**
     * @brief Construct a new External Loop Info object
     */
    BroadcastInfoV2() :
      buffer_size(0),
      buffer_axis(-1),
      strides{0, 0, 0, 0},
      tensor_type({Tformat::NCHW, Tdatatype::FP32}) {}

    unsigned int buffer_size; /**< virtual size of the buffer */
    int buffer_axis;          /**< the smallest axis that should be looped.
                                   -1 means no loop needed*/
    std::array<unsigned int, TensorDim::MAXDIM>
      strides; /**< modified strides for the loop */
    nntrainer::TensorDim::TensorType tensor_type;
  };

  /**
   * @brief compute Loop info for broadcasting and vectorization
   *
   * @param m target tensor to be calculated against.
   * @return BroadcastInfo Loopinfo needed to run external loop
   */
  BroadcastInfoV2 computeBroadcastInfo(const TensorV2 &m) const;
};

/**
 * @class SrcSharedTensorBase
 * @brief Source of the shared tensor
 */
class SrcSharedTensorBase {
public:
  /**
   * @brief   Constructor for the class
   */
  SrcSharedTensorBase() : src(nullptr), off(0) {}

  /**
   * @brief   Constructor for the class
   */
  SrcSharedTensorBase(const TensorBase *tensor, size_t offset) :
    src(tensor),
    off(offset) {}

  /**
   * @brief   Get the allocated src tensor
   */
  const TensorBase *tensor() const {
    if (!src)
      throw std::runtime_error("Accessing empty src tensor");

    return src;
  }

  /**
   * @brief   Get the offset from the source tensor
   */
  size_t offset() const { return off; }

private:
  const TensorBase *src; /**< Tensor of the source */
  size_t off;            /**< offset from the source data ptr */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_BASE_H__ */
