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
   * @brief Construct a new FloatTensor object
   *
   * @param d Tensor dim for this float tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  FloatTensor(const TensorDim &d, bool alloc_now,
              Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  FloatTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d data for the Tensor
   * @param fm format for the Tensor
   */
  FloatTensor(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
    Tformat fm);

  /**
   * @brief Basic Destructor
   */
  ~FloatTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator==(const FloatTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator!=(const FloatTensor &rhs) const { return !(*this == rhs); }

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
   * @brief     return value at specific location
   * @param[in] i index
   */
  const float getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const float getValue(unsigned int b, unsigned int c, unsigned int h,
                       unsigned int w) const;

  /**
   * @copydoc TensorV2::setValue(float value)
   */
  void setValue(float value) override;

  /**
   * @copydoc TensorV2::setValue(b, c, h, w, value)
   */
  void setValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value) override;

  /**
   * @copydoc TensorV2::addValue(b, c, h, w, value, beta)
   */
  void addValue(unsigned int b, unsigned int c, unsigned int h, unsigned int w,
                float value, float beta) override;

  /**
   * @copydoc TensorV2::setZero()
   */
  void setZero() override;

  /**
   * @brief Set the Dist object
   * @param dist distribution engine
   */
  template <typename Engine> void setDist(Engine dist) {
    NNTR_THROW_IF(!contiguous, std::invalid_argument)
      // << getName() << " Tensor is not contiguous, cannot set distribution";
      << " Tensor is not contiguous, cannot set distribution";

    float *data_ = (float *)getData();
    unsigned int len = size();
    for (unsigned int i = 0; i < len; ++i) {
      data_[i] = (float)dist(rng);
    }
  };

  /**
   * @copydoc TensorV2::setRandNormal()
   */
  void setRandNormal(float mean = 0.0f, float stddev = 0.05f);

  /**
   * @copydoc TensorV2::setRandUniform()
   */
  void setRandUniform(float min = -0.05f, float max = 0.05f);

  /**
   * @copydoc TensorV2::setRandBernoulli()
   */
  void setRandBernoulli(float probability = 0.5f);

  /**
   * @copydoc TensorV2::initialize()
   */
  void initialize() override;

  /**
   * @copydoc TensorV2::initialize(Initializer init)
   */
  void initialize(Initializer init) override;

  /**
   * @copydoc TensorV2::apply(std::function<T(T)> f, TensorV2 &output)
   */
  TensorV2 &apply(std::function<float(float)> f,
                  TensorV2 &output) const override;

  /**
   * @copydoc TensorV2::multiply_strided(TensorV2 const &m, TensorV2 &output,
   * const float beta)
   */
  TensorV2 multiply_strided(TensorV2 const &m, TensorV2 &output,
                            const float beta) const override;

  /**
   * @copydoc TensorV2::multiply_i(float const &value)
   */
  int multiply_i(float const &value) override;

  /**
   * @copydoc TensorV2::multiply(float const &value, TensorV2 &out)
   */
  TensorV2 &multiply(float const &value, TensorV2 &out) const override;

  /**
   * @copydoc TensorV2::multiply(TensorV2 const &m, TensorV2 &output, const
   * float beta = 0.0)
   */
  TensorV2 &multiply(TensorV2 const &m, TensorV2 &output,
                     const float beta = 0.0) const override;

  /**
   * @copydoc TensorV2::divide(float const &value, TensorV2 &output)
   */
  TensorV2 &divide(float const &value, TensorV2 &output) const override;

  /**
   * @copydoc TensorV2::divide(TensorV2 const &m, TensorV2 &output)
   */
  TensorV2 &divide(TensorV2 const &m, TensorV2 &output) const override;

  /**
   * @copydoc TensorV2::add(float const &value, TensorV2 &output)
   */
  TensorV2 &add(float const &value, TensorV2 &output) const override;

  /**
   * @copydoc TensorV2::add(TensorV2 const &m, TensorV2 &output, float const
   * alpha)
   */
  TensorV2 &add(TensorV2 const &m, TensorV2 &output,
                float const alpha) const override;

  /**
   * @copydoc TensorV2::copy(const TensorV2 &from)
   */
  void copy(const TensorV2 &from);

  /**
   * @copydoc TensorV2::copyData(const TensorV2 &from)
   */
  void copyData(const TensorV2 &from);

  /**
   * @copydoc TensorV2::print(std::ostream &out)
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
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FLOAT_TENSOR_H__ */
