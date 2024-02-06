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

#define CREATE_V2_IF_EMPTY_DIMS(tensor, ...) \
  do {                                       \
    if (tensor.empty())                      \
      tensor = TensorV2(__VA_ARGS__);        \
  } while (0);

#include <cstddef>

#include <nntrainer_log.h>
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
  ~TensorV2() = default;

  /**
   *  @brief  Copy constructor of Tensor.
   *  @param[in] Tensor &
   */
  TensorV2(const TensorV2 &rhs) = default;

  /**
   *  @brief  Move constructor of Tensor.
   *  @param[in] Tensor &&
   */
  TensorV2(TensorV2 &&rhs) noexcept = default;

  /**
   * @brief  Copy assignment operator.
   * @param[in] rhs Tensor to be copied.
   */
  TensorV2 &operator=(const TensorV2 &rhs) = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Tensor to be moved.
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
  bool operator!=(const TensorV2 &rhs) const { return !(*this == rhs); }

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
   * @brief    get address of n-d data
   */
  template <typename T = float>
  T *getAddress(unsigned int b, unsigned int c, unsigned int h,
                unsigned int w) {
    return getAddress<T>(getIndex(b, c, h, w));
  }

  /**
   * @brief    get address of n-d data
   */
  template <typename T = float>
  const T *getAddress(unsigned int b, unsigned int c, unsigned int h,
                      unsigned int w) const {
    return getAddress<T>(getIndex(b, c, h, w));
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
  void setRandNormal(float mean = 0.0f, float stddev = 0.05f);

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
  void initialize(Initializer init);

  /**
   * @brief Apply instantly to the element
   * @param[in] *function function pointer applied
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
   * @retval    Tensor
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
    CREATE_V2_IF_EMPTY_DIMS(
      output, {itensor->getFormat(), itensor->getDataType()}, nullptr);

    if (itensor->getFormat() != output.itensor->getFormat() ||
        itensor->getDataType() != itensor->getDataType()) {
      /// @todo add unittest
      throw std::invalid_argument(
        "[Tensor::apply] output dimension does not match");
    }

    itensor->apply(f, output);

    return output;
  }

  /**
   * @brief     Apply function to Tensor
   * @param[in] *function function pointer applied
   * @retval    Tensor
   */
  TensorV2 apply(std::function<TensorV2(TensorV2)> f) const;

  /**
   * @brief     Apply function to Tensor
   * @param[in] *function function pointer applied
   * @param[out] output output tensor
   * @retval    Tensor
   */
  TensorV2 &apply(std::function<TensorV2 &(TensorV2, TensorV2 &)> f,
                  TensorV2 &output) const;

  /**
   * @brief     Multiply Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply_i
   */
  int multiply_i_strided(TensorV2 const &m, const float beta = 0.0);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  TensorV2 multiply_strided(TensorV2 const &m, const float beta = 0.0) const;

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   *
   * @note support different strided inputs and output
   * @note does not support broadcasting
   *
   * @todo merge this to multiply
   */
  TensorV2 &multiply_strided(TensorV2 const &m, TensorV2 &output,
                             const float beta = 0.0) const;

  /**
   * @brief     Multiply value element by element immediately
   * @param[in] value multiplier
   * @retval    #ML_ERROR_INVALID_PARAMETER Tensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int multiply_i(float const &value);

  /**
   * @brief     Multiply value element by element
   * @param[in] value multiplier
   * @retval    Calculated Tensor
   */
  TensorV2 multiply(float const &value) const;

  /**
   * @brief      multiply value element by element
   * @param[in]  value multiplier
   * @param[out] out out tensor to store the result
   * @retval     Calculated Tensor
   */
  TensorV2 &multiply(float const &value, TensorV2 &out) const;

  /**
   * @brief     Multiply Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    #ML_ERROR_NONE successful
   */
  int multiply_i(TensorV2 const &m, const float beta = 0.0);

  /**
   * @brief     Multiply Tensor Element by Element ( Not the MxM )
   * @param[in] m Tensor to be multiplied
   * @param[in] beta scalar to multiply output with and add
   * @retval    Calculated Tensor
   */
  TensorV2 multiply(TensorV2 const &m, const float beta = 0.0) const;

  /**
   * @brief      Multiply Tensor Element by Element ( Not the MxM )
   * @param[in]  m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @param[in]  beta scalar to multiply output with and add
   * @retval     Calculated Tensor
   */
  TensorV2 &multiply(TensorV2 const &m, TensorV2 &output,
                     const float beta = 0.0) const;

  /**
   * @brief     Divide value element by element immediately
   * @param[in] value divisor
   * @retval    #ML_ERROR_INVALID_PARAMETER Tensor dimension is not right
   * @retval    #ML_ERROR_NONE Successful
   */
  int divide_i(float const &value);

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @retval    Calculated Tensor
   */
  TensorV2 divide(float const &value) const;

  /**
   * @brief     Divide value element by element
   * @param[in] value Divisor
   * @param[out] output Tensor to store the result
   * @retval    Calculated Tensor
   */
  TensorV2 &divide(float const &value, TensorV2 &output) const;

  /**
   * @brief     divide Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @retval    #ML_ERROR_NONE successful
   */
  int divide_i(TensorV2 const &m);

  /**
   * @brief     Divide Tensor Element by Element
   * @param[in] m Divisor Tensor
   * @retval    Calculated Tensor
   */
  TensorV2 divide(TensorV2 const &m) const;

  /**
   * @brief     divide Tensor Elementwise
   * @param[in] m Tensor to be multiplied
   * @param[out] output Tensor to store the result
   * @retval    Calculated Tensor
   */
  TensorV2 &divide(TensorV2 const &m, TensorV2 &output) const;

  /**
   * @brief     Add Tensor Element immediately to target tensor without mem copy
   * @param[in] value value to be added
   * @retval    #ML_ERROR_NONE  Successful
   * @retval    #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(float const &value);

  /**
   * @brief     Add value Element by Element
   * @param[in] value value to be added
   * @retval    Calculated Tensor
   */
  TensorV2 add(float const &value) const;

  /**
   * @brief      Add Tensor Element by Element
   * @param[in]  value value to be added
   * @param[out] output Tensor to save output without allocating new memory
   * @retval     Calculated Tensor
   */
  TensorV2 &add(float const &value, TensorV2 &output) const;

  /**
   * @brief     Add Tensor Element by Element without mem copy
   * @param[in] m Tensor to be added
   * @param[in] alpha Values to be scaled
   * @retval    #ML_ERROR_NONE  Successful
   * @retval    #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  int add_i(TensorV2 const &m, float const alpha = 1);

  /**
   * @brief     Add Tensor Element by Element
   * @param[in] m Tensor to be added
   * @param[in] alpha Values to be scaled
   * @retval    Calculated Tensor
   */
  TensorV2 add(TensorV2 const &m, float const alpha = 1) const;

  /**
   * @brief      Add Tensor Element by Element
   * @param[in]  m Tensor to be added
   * @param[out] output Tensor to be out
   * @param[in]  alpha Values to be scaled
   * @retval     Calculated Tensor
   */
  TensorV2 &add(TensorV2 const &m, TensorV2 &output,
                float const alpha = 1) const;

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
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   *
   * @note copy can reshape the tensor to match the shape
   * @note support copying data from multiple data type
   */
  void copy(const TensorV2 &from);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   * @note      support copying data from multiple data type
   */
  void copyData(const TensorV2 &from);

  /**
   * @brief     Copy the Tensor
   * @param[in] from Tensor to be copied
   * @note      only support copying data from tensor with the same data type
   */
  void copy_with_stride(const TensorV2 &from);

  /**
   * @brief     set Tensor Dim
   * @param[in] d TensorDim
   * @note      Throws std::invalid_argument if size mismatch
   */
  void reshape(const TensorDim &d);

  /**
   * @brief     return a copy of the Tensor Dim
   * @retval    TensorDim
   */
  TensorDim getDim() const;

  /**
   * @brief     return Tensor Type
   */
  TensorDim::TensorType getTensorType() const;

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
   * @brief     return whether tensor is contiguous or not.
   * @retval    bool contiguous
   */
  const bool getContiguous() const noexcept;

  /**
   * @brief     return current stride of tensor.
   * @retval    int[MAXDIM] strides
   */
  const std::array<size_t, TensorDim::MAXDIM> getStrides() const noexcept;

  /**
   * @brief     Check if two given axes are contiguous
   * @param[in] np1 first axis
   * @param[in] np2 second axis to compare with first axis
   * @retval    bool continuous
   */
  bool checkContinuous(unsigned int np1, unsigned int np2) const;

  /**
   * @brief     Set name of the tensor
   * @param[in] name_ tensor name
   */
  void setName(const std::string &name_);

  /**
   * @brief     Get name of the tensor
   * @retval    string name
   */
  const std::string &getName() const;

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

  /**
   * @brief    Swaps Tensor lhs and rhs
   * @param[in] lhs Tensor to be swapped
   * @param[in] rhs Tensor to be swapped
   */
  friend void swap(TensorV2 &lhs, TensorV2 &rhs) noexcept {
    std::swap(lhs.itensor, rhs.itensor);
  }

private:
  std::shared_ptr<TensorBase> itensor;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TENSOR_V2_H__ */
