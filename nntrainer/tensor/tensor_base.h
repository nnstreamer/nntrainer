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
#include <quantizer.h>
#include <tensor_dim.h>
#include <util_func.h>

#define transposeloop(cl, ci, cj, ck, sl, si, sj, sk)                          \
  do {                                                                         \
    unsigned int i, j, k, l;                                                   \
    int inidx = 0, outidx = 0;                                                 \
    for (cl = 0; cl < sl; cl++)                                                \
      for (ci = 0; ci < si; ci++)                                              \
        for (cj = 0; cj < sj; cj++)                                            \
          for (ck = 0; ck < sk; ck++) {                                        \
            outidx = si * sj * sk * cl + sj * sk * ci + sk * cj + ck;          \
            inidx = l * SI * SJ * SK + i * SJ * SK + j * SK + k;               \
            outptr[outidx] = inptr[inidx];                                     \
          }                                                                    \
  } while (0);

#define transposeloop_nhwc(cl, ci, cj, ck, sl, si, sj, sk)                     \
  do {                                                                         \
    unsigned int i, j, k, l;                                                   \
    int inidx = 0, outidx = 0;                                                 \
    for (cl = 0; cl < sl; cl++)                                                \
      for (ci = 0; ci < si; ci++)                                              \
        for (cj = 0; cj < sj; cj++)                                            \
          for (ck = 0; ck < sk; ck++) {                                        \
            outidx = si * sj * sk * cl + sj * sk * ci + sk * cj + ck;          \
            inidx = l * SJ * SK * SI + j * SK * SI + k * SI + i;               \
            outptr[outidx] = inptr[inidx];                                     \
          }                                                                    \
  } while (0);

namespace nntrainer {

using TensorDim = ml::train::TensorDim;
using Tformat = ml::train::TensorDim::Format;
using Tdatatype = ml::train::TensorDim::DataType;
using TStorageOrder = ml::train::TensorDim::StorageOrder;

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

class Tensor;
class SrcSharedTensorBase;

/**
 * @class TensorBase class
 * @brief TensorBase is an abstract class that provides a base for various
 * tensor classes with different data types such as FloatTensor to extend and
 * implement abstract methods.
 *
 * @note Basic functions required for tensor memory allocation and data
 * modification, such as allocate(), getData(), and setValue(), are necessary
 * when creating subclasses (new tensor class).
 *
 * The remaining operations that are used for mathematical operations are not
 * essential to create a new tensor class but later should be implemented in a
 * child class in order to utilize its tensor operations fully.
 */
class TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  NNTR_API TensorBase(std::string name_ = "", Tformat fm = Tformat::NCHW,
                      Tdatatype d_type = Tdatatype::FP32) :
    dim(TensorDim(fm, d_type)),
    strides(dim.computeStrides()),
    contiguous(true),
    initializer(Initializer::NONE),
    name(name_),
    data(nullptr),
    offset(0),
    file_offset(0),
    src_tensor() {}

  /**
   * @brief     Constructor of Tensor with dimension, possibly lazily
   * @param d Tensor dim for this tensor
   * @param alloc_now If the memory of the tensor must be allocated
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  NNTR_API TensorBase(const TensorDim &d, bool alloc_now,
                      Initializer init = Initializer::NONE,
                      std::string name = "");

  /**
   * @brief     Constructor of Tensor with dimension/buf
   * @param d Tensor dim for this tensor
   * @param buf buffer
   * @note Memory for this tensor is instantaneously allocated
   */
  NNTR_API TensorBase(const TensorDim &d, const void *buf = nullptr) :
    TensorBase(d, true) {}

  /**
   *  @brief  Copy constructor of TensorBase.
   *  @param[in] Tensor &
   */
  NNTR_API TensorBase(const TensorBase &rhs) {
    dim = rhs.dim;
    strides = rhs.strides;
    contiguous = rhs.contiguous;
    initializer = rhs.initializer;
    name = rhs.name;
    data = rhs.data;
    offset = rhs.offset;
    src_tensor = rhs.src_tensor;
  }

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor information
   */
  NNTR_API bool operator==(const TensorBase &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor information
   */
  NNTR_API bool operator!=(const TensorBase &rhs) const {
    return !(*this == rhs);
  }

  /**
   * @copydoc Tensor::setTensorVar(TensorDim d, void *buf, size_t offset)
   */
  NNTR_API void setTensorVar(TensorDim d, void *buf, size_t offset);

  /**
   * @brief Basic Destructor
   */
  NNTR_API virtual ~TensorBase() {}

  /**
   * @copydoc Tensor::allocate()
   */
  NNTR_API virtual void allocate() = 0;

  /**
   * @copydoc Tensor::deallocate()
   */
  NNTR_API virtual void deallocate() = 0;

  /**
   * @copydoc Tensor::isAllocated()
   */
  NNTR_API bool isAllocated() { return data != nullptr; }

  /**
   * @copydoc Tensor::getData()
   */
  NNTR_API virtual void *getData() const = 0;

  /**
   * @copydoc Tensor::getData(size_t idx)
   */
  NNTR_API virtual void *getData(size_t idx) const = 0;

  /**
   * @copydoc Tensor::getScale()
   */
  NNTR_API virtual void *getScale() const {
    throw std::invalid_argument(
      "Tensor::getScale() is not supported in tensor data type " +
      getStringDataType());
  }

  /**
   * @copydoc Tensor::getScale(size_t idx)
   */
  NNTR_API virtual void *getScale(size_t idx) const {
    throw std::invalid_argument(
      "Tensor::getScale() is not supported in tensor data type " +
      getStringDataType());
  }

  /**
   * @copydoc Tensor::getZeroPoint()
   */
  NNTR_API virtual unsigned int *getZeroPoint() const {
    throw std::invalid_argument(
      "Tensor::getZeroPoint() is not supported in tensor data type " +
      getStringDataType());
  }

  /**
   * @copydoc Tensor::getZeroPoint(size_t idx)
   */
  NNTR_API virtual unsigned int *getZeroPoint(size_t idx) const {
    throw std::invalid_argument(
      "Tensor::getZeroPoint() is not supported in tensor data type " +
      getStringDataType());
  }

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  NNTR_API virtual void *getAddress(unsigned int i) = 0;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  NNTR_API virtual const void *getAddress(unsigned int i) const = 0;

  /**
   * @copydoc Tensor::setValue(float value)
   */
  NNTR_API virtual void setValue(float value) = 0;

  /**
   * @copydoc Tensor::setValue(b, c, h, w, value)
   */
  NNTR_API virtual void setValue(unsigned int b, unsigned int c, unsigned int h,
                                 unsigned int w, float value) = 0;

  /**
   * @copydoc Tensor::addValue()
   */
  NNTR_API virtual void addValue(unsigned int b, unsigned int c, unsigned int h,
                                 unsigned int w, float value, float beta) = 0;

  /**
   * @copydoc Tensor::setZero()
   */
  NNTR_API virtual void setZero() = 0;

  /**
   * @copydoc Tensor::setRandNormal()
   */
  NNTR_API virtual void setRandNormal(float mean, float stddev);

  /**
   * @copydoc Tensor::setRandBernoulli()
   */
  NNTR_API virtual void setRandUniform(float min, float max);

  /**
   * @copydoc Tensor::setRandBernoulli()
   */
  NNTR_API virtual void setRandBernoulli(float probability);

  /**
   * @copydoc Tensor::initialize()
   */
  NNTR_API virtual void initialize() = 0;

  /**
   * @copydoc Tensor::initialize(Initializer init)
   */
  NNTR_API virtual void initialize(Initializer init) = 0;

  /**
   * @copydoc Tensor::multiply_strided(Tensor const &m, Tensor &output,
   * const float beta)
   */
  NNTR_API virtual Tensor multiply_strided(Tensor const &m, Tensor &output,
                                           const float beta) const;

  /**
   * @copydoc Tensor::multiply_i(float const &value)
   */
  NNTR_API virtual int multiply_i(float const &value);

  /**
   * @copydoc Tensor::multiply(float const &value, Tensor &output)
   */
  NNTR_API virtual Tensor &multiply(float const &value, Tensor &output) const;

  /**
   * @copydoc Tensor::multiply(Tensor const &m, Tensor &output, const
   * float beta = 0.0)
   */
  NNTR_API virtual Tensor &multiply(Tensor const &m, Tensor &output,
                                    const float beta = 0.0) const;

  /**
   * @copydoc Tensor::divide(float const &value, Tensor &output)
   */
  NNTR_API virtual Tensor &divide(float const &value, Tensor &output) const;

  /**
   * @copydoc Tensor::divide(Tensor const &m, Tensor &output)
   */
  NNTR_API virtual Tensor &divide(Tensor const &m, Tensor &output) const;

  /**
   * @copydoc Tensor::add_strided(Tensor const &input, Tensor &output,
   * const float beta)
   */
  NNTR_API virtual Tensor &add_strided(Tensor const &input, Tensor &output,
                                       const float beta) const;

  /**
   * @copydoc Tensor::add_i_partial()
   */
  NNTR_API virtual int add_i_partial(unsigned int len, unsigned int addr_idx,
                                     Tensor &m, unsigned int incX,
                                     unsigned int incY, const Tensor alphas,
                                     unsigned int alpha_idx);

  /**
   * @copydoc Tensor::add(float const &value, Tensor &output)
   */
  NNTR_API virtual Tensor &add(float const &value, Tensor &output) const;

  /**
   * @copydoc Tensor::add(Tensor const &m, Tensor &output, float const
   * alpha)
   */
  NNTR_API virtual Tensor &add(Tensor const &m, Tensor &output,
                               float const alpha) const;

  /**
   * @copydoc Tensor::subtract(float const &value, Tensor &output)
   */
  NNTR_API virtual Tensor &subtract(float const &value, Tensor &output) const;

  /**
   * @brief      Sum all the Tensor elements according to the batch
   * @param[out] output Tensor(batch, 1, 1, 1)
   */
  NNTR_API virtual void sum_by_batch(Tensor &output) const;

  /**
   * @copydoc Tensor::sum(unsigned int axis, Tensor &output, float alpha,
   * float beta) const
   */
  NNTR_API virtual Tensor &sum(unsigned int axis, Tensor &output, float alpha,
                               float beta) const;

  /**
   * @copydoc Tensor::abs()
   */
  NNTR_API virtual Tensor &abs(Tensor &output) const;

  /**
   * @copydoc Tensor::l2norm
   */
  NNTR_API virtual float l2norm() const;

  /**
   * @copydoc Tensor::pow(float exponent, Tensor &output)
   */
  NNTR_API virtual Tensor &pow(float exponent, Tensor &output) const;

  /**
   * @copydoc Tensor::sqrt(Tensor &output)
   */
  NNTR_API virtual Tensor &sqrt(Tensor &output) const;

  /**
   * @copydoc Tensor::erf(Tensor &output)
   */
  NNTR_API virtual Tensor &erf(Tensor &output) const;

  /**
   * @brief    sin transform function
   * @param[out] out out to store the result
   */
  NNTR_API virtual void sin(Tensor &out, float alpha = 1.0);

  /**
   * @brief    cos transform function
   * @param[out] out out to store the result
   */
  NNTR_API virtual void cos(Tensor &out, float alpha = 1.0);

  /**
   * @brief    tangent transform function
   * @param[out] output output to store the result
   */
  NNTR_API virtual void tan(Tensor &output, float alpha = 1.0);

  /**
   * @brief      inverse squared root function
   * @param[out] out out to store the result
   */
  NNTR_API virtual void inv_sqrt(Tensor &out);

  /**
   * @brief     Dot Product of Tensor ( equal MxM )
   * @details   This applies dot of the last dimension of this and
   * second-last dimension of passed tensor m.
   * @param[in] input Tensor
   * @param[in] output output Tensor
   * @param[in] trans Transpose
   * @param[in] trans_in Transpose input
   * @param[in] beta beta
   * @retval    Calculated Tensor
   */
  NNTR_API virtual Tensor &dot(Tensor const &input, Tensor &output, bool trans,
                               bool trans_in, float beta) const;

  /**
   * @copydoc Tensor::dropout_mask(float dropout)
   */
  NNTR_API virtual void dropout_mask(float dropout);

  /**
   * @copydoc Tensor::filter_mask(const Tensor &mask_len, bool reverse)
   */
  NNTR_API virtual void filter_mask(const Tensor &mask_len, bool reverse);

  /**
   * @copydoc Tensor::zoneout_mask(Tensor &opposite, float zoneout)
   */
  NNTR_API virtual void zoneout_mask(Tensor &opposite, float zoneout);

  /**
   * @copydoc Tensor::split(std::vector<size_t> sizes, int axis)
   */
  NNTR_API virtual std::vector<Tensor> split(std::vector<size_t> sizes,
                                             int axis);

  /**
   * @copydoc Tensor::concat()
   */
  NNTR_API virtual Tensor concat(const std::vector<Tensor> &tensors, int axis,
                                 Tensor &output);

  /**
   * @copydoc Tensor::print(std::ostream &out)
   */
  NNTR_API virtual void print(std::ostream &out) const = 0;

  /**
   * @copydoc Tensor::apply(std::function<T(T)> f, Tensor &output)
   * @note    This will be only used in FloatTensor.
   */
  NNTR_API virtual Tensor &apply(std::function<float(float)> f,
                                 Tensor &output) const;

#ifdef ENABLE_FP16
  /**
   * @copydoc Tensor::apply(std::function<T(T)> f, Tensor &output)
   * @note    This will be only used in HalfTensor.
   */
  NNTR_API virtual Tensor &apply(std::function<_FP16(_FP16)> f,
                                 Tensor &output) const;
#endif

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   *
   * @note copy can reshape the tensor to match the shape
   */
  NNTR_API virtual void copy(const Tensor &from) = 0;

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   */
  NNTR_API virtual void copyData(const Tensor &from) = 0;

  /**
   * @brief      Copy the Tensor
   * @param[in]  input Tensor to be copied
   * @param[out] output output Tensor
   */
  NNTR_API virtual void copy_with_stride(const Tensor &input,
                                         Tensor &output) = 0;

  /**
   * @brief     Save the Tensor into file
   * @param[in] file input file stream
   */
  NNTR_API virtual void save(std::ostream &file);

  /**
   * @brief     Read the Tensor from file
   * @param[in] file input file stream
   */
  NNTR_API virtual void read(std::ifstream &file, size_t start_offset = 0,
                             bool read_from_offset = false);

  /**
   * @copydoc Tensor::readFSU()
   */
  NNTR_API virtual void readFSU();

  /**
   * @copydoc Tensor::argmax()
   */
  NNTR_API virtual std::vector<unsigned int> argmax() const;

  /**
   * @copydoc Tensor::argmin()
   */
  NNTR_API virtual std::vector<unsigned int> argmin() const;

  /**
   * @copydoc Tensor::max_abs()
   */
  NNTR_API virtual float max_abs() const = 0;

  /**
   * @copydoc Tensor::maxValue()
   */
  NNTR_API virtual float maxValue() const = 0;

  /**
   * @copydoc Tensor::minValue()
   */
  NNTR_API virtual float minValue() const = 0;

  /**
   * @copydoc Tensor::transpose(const std::string &direction, Tensor &out)
   */
  NNTR_API virtual Tensor &transpose(const std::string &direction,
                                     Tensor &out) const;

  /**
   * @brief     put data of Tensor
   * @note      It is only effective when fsu is used
   */
  NNTR_API void putData() const;

  /**
   * @brief Set the memory buffer for the tensor
   * @param buf the memory buffer
   * @param off offset
   */
  NNTR_API void setMemoryData(const std::shared_ptr<MemoryData> buf,
                              size_t off);

  /**
   * @brief     return Data pointer of Tensor
   * @retval    template T pointer (float pointer as default)
   */
  NNTR_API const std::shared_ptr<MemoryData> getMemoryData() const;

  /**
   * @brief     return offset
   */
  NNTR_API size_t getOffset() const;

  /**
   * @brief     get FileOffset of Tensor
   * @return    size_t fileOffset
   */
  NNTR_API size_t getFileOffset() const;

  /**
   * @brief     set FileOffset to Tensor
   * @param     off FileOffset
   */
  NNTR_API void setFileOffset(size_t off);

  /**
   * @brief     set Tensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  NNTR_API void reshape(const TensorDim &d);

  /**
   * @brief     return a copy of the Tensor Dim
   * @retval    TensorDim
   */
  NNTR_API TensorDim getDim() const { return TensorDim(dim); }

  /**
   * @brief     return Tensor Type
   */
  NNTR_API TensorDim::TensorType getTensorType() const {
    return dim.getTensorType();
  }

  /**
   * @brief Get initializer for the tensor
   * @retval initializer of the tensor
   */
  NNTR_API Initializer getInitializer() const { return initializer; }

  /**
   * @brief Get format for the tensor
   * @retval format of the tensor
   */
  NNTR_API TensorDim::Format getFormat() const { return dim.getFormat(); }

  /**
   * @brief Get data type for the tensor
   * @retval data type of the tensor
   */
  NNTR_API Tdatatype getDataType() const { return dim.getDataType(); }

  /**
   * @brief     update batch size for this tensor
   * @param     batch size
   */
  NNTR_API void updateBatch(unsigned int batch);

  /**
   * @brief     update the dimension for this tensor
   * @param     dimension dimension to be updated
   */
  NNTR_API void updateDimension(TensorDim dimension);

  /**
   * @brief     return whether tensor is contiguous or not.
   * @retval    bool contiguous
   */
  NNTR_API const bool getContiguous() const noexcept { return contiguous; }

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  NNTR_API const std::array<size_t, TensorDim::MAXDIM>
  getStrides() const noexcept {
    return strides;
  }

  /**
   * @brief     Set name of the tensor
   */
  NNTR_API void setName(const std::string &name_) { name = name_; }

  /**
   * @brief     Get name of the tensor
   * @retval    string name
   */
  NNTR_API const std::string &getName() const { return name; }

  /**
   * @brief Get linear index given the n-d index
   */
  NNTR_API size_t getIndex(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w) const noexcept;

  /**
   * @brief     Save quantization information
   */
  NNTR_API virtual void save_quantization_info(std::ostream &file) {}

  /**
   * @brief     Read quantization information
   */
  NNTR_API virtual void read_quantization_info(std::ifstream &file,
                                               size_t start_offset = 0,
                                               bool read_from_offset = false) {}

  /**
   * @brief     Get size of current tensor
   * @retval    unsigned int size of the current tensor
   */
  NNTR_API virtual size_t size() const { return dim.getDataLen(); }

  /**
   * @brief     Get if the tensor is empty
   * @retval    true if the tensor is empty
   */
  NNTR_API bool empty() const { return size() == 0; }

  /**
   * @brief     Get size of the data in bytes
   * @retval    size_t Size in bytes
   */
  NNTR_API size_t bytes() const { return size() * dim.getDataTypeSize(); }

  /**
   * @brief     Get a total size of the memory data in bytes
   * @retval    size_t Size in bytes
   */
  NNTR_API virtual size_t getMemoryBytes() const {
    return size() * dim.getDataTypeSize();
  }

  /**
   * @brief     return Tensor batch size
   * @retval    batch size
   */
  NNTR_API size_t batch() const { return dim.batch(); }

  /**
   * @brief     return Tensor channel size
   * @retval    channel size
   */
  NNTR_API size_t channel() const { return dim.channel(); }

  /**
   * @brief     return Tensor height size
   * @retval    height size
   */
  NNTR_API size_t height() const { return dim.height(); }

  /**
   * @brief     return Tensor width size
   * @retval    width size
   */
  NNTR_API size_t width() const { return dim.width(); }

  /**
   * @brief     return Tensor scale factor size if exists
   * @retval    scale factor size
   * @note      Override for quantize tensor
   */
  NNTR_API virtual size_t scale_size() const { return 0; }

  /**
   * @brief     return Tensor quantization scheme
   * @retval    Qscheme qscheme
   * @note      Override for quantize tensor
   */
  NNTR_API virtual QScheme q_scheme() const {
    throw std::invalid_argument(
      "Tensor::q_scheme() is not supported in tensor data type " +
      getStringDataType());
  }

  /**
   * @brief Merge the given two axis for tensor at second axis inplace
   *
   * @param axis1 first axis to merge
   * @param axis2 second axis to merge
   */
  NNTR_API void mergeAxis(unsigned int axis1, unsigned int axis2);

  /**
   * @brief Allocate data based on the source tensor
   * @note As this memory is shared, do NOT initialize
   */
  NNTR_API void allocateSrcTensor();

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
  NNTR_API void createSharedDataTensor(const TensorBase *src, TensorBase *dest,
                                       size_t offset) const;

  /**
   * @brief Get new tensor which shares memory with current tensor but different
   * shape
   *
   * @param[in] dim new dimension to be set for this tensor
   * @param[in] offset offset to be used from the start of the data in elements
   * @param[in] reset_stride reset stride
   * @param[in] name_ name of the Tensor
   * @param[out] ret output TensorBase pointer
   * @note The new tensor will share the same data as the current tensor but
   * can have different size.
   * @note New size added with offset must be less than the size of the original
   * tensor.
   */
  NNTR_API void getSharedDataTensor(const TensorDim dim_, size_t offset,
                                    bool reset_stride, const std::string &name_,
                                    TensorBase *ret);

  /**
   * @copydoc Tensor::isValid()
   */
  virtual bool isValid() const = 0;

  static constexpr float epsilon = 1e-5f;

protected:
  TensorDim dim;
  std::array<size_t, TensorDim::MAXDIM> strides;
  bool contiguous;
  Initializer initializer;
  std::string name; /**< name of the tensor */
  std::shared_ptr<MemoryData> data;
  size_t offset;
  size_t file_offset; /**< offset of the tensor in the file */

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
  struct BroadcastInfo {

    /**
     * @brief Construct a new External Loop Info object
     */
    NNTR_API BroadcastInfo() :
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
  NNTR_API BroadcastInfo computeBroadcastInfo(const Tensor &m) const;

  /**
   * @brief Calcuates variables needed to perform tensor flatten dot product
   *
   * @param[in]  input Tensor
   * @param[in]  output output Tensor
   * @param[in]  trans Transpose
   * @param[in]  trans_in Transpose input
   * @param[out] first_three_flat flattened the fist 3 axis
   * @param[out] last_axis last axis
   * @param[out] input_first_three_flat input's flattened the fist 3 axis
   * @param[out] input_last_axis input's last axis
   * @param[out] M number of op(this)'s and output's row
   * @param[out] N number of op(inputs)'s and output's columns
   * @param[out] K number of op(this)'s column and op(input)'s row
   * @param[out] lda leading dimension of this
   * @param[out] ldb leading dimension of input
   * @param[out] ldc leading dimension of output
   *
   * @note op(X) is one of X or X**T
   */
  NNTR_API void calculateFlattenDot(
    Tensor const &input, Tensor &output, bool trans, bool trans_in,
    unsigned int &first_three_flat, unsigned int &last_axis,
    unsigned int &input_first_three_flat, unsigned int &input_last_axis,
    unsigned int &M, unsigned int &N, unsigned int &K, unsigned int &lda,
    unsigned int &ldb, unsigned int &ldc) const;

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type
   * @note   TensorBase::getStringDataType() should not be called. Please define
   * this function in the derived class to the corresponding data type.
   */
  NNTR_API virtual std::string getStringDataType() const {
    return "Undefined type";
  }
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
  NNTR_API SrcSharedTensorBase() : src(nullptr), off(0) {}

  /**
   * @brief   Constructor for the class
   */
  NNTR_API SrcSharedTensorBase(const TensorBase *tensor, size_t offset) :
    src(tensor), off(offset) {}

  /**
   * @brief   Get the allocated src tensor
   */
  NNTR_API const TensorBase *tensor() const {
    if (!src)
      throw std::runtime_error("Accessing empty src tensor");

    return src;
  }

  /**
   * @brief   Get the offset from the source tensor
   */
  NNTR_API size_t offset() const { return off; }

private:
  const TensorBase *src; /**< Tensor of the source */
  size_t off;            /**< offset from the source data ptr */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_BASE_H__ */
