// SPDX-License-Identifier: Apache-2.0
/**
 * @file	half_tensor.h
 * @date	01 December 2023
 * @brief	This is HalfTensor class for 16-bit floating point calculation
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __HALF_TENSOR_H__
#define __HALF_TENSOR_H__
#ifdef __cplusplus

#include <tensor.h>
#include <tensor_base.h>

#ifdef DEBUG
#define EXCEPT_WHEN_DEBUG
#else
#define EXCEPT_WHEN_DEBUG noexcept
#endif

namespace nntrainer {

/**
 * @class HalfTensor class
 * @brief HalfTensor class for 16-bit floating point calculation
 */
class HalfTensor : public TensorBase {
public:
  /**
   * @brief     Basic Constructor of Tensor
   */
  HalfTensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new HalfTensor object
   *
   * @param d Tensor dim for this float tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  HalfTensor(const TensorDim &d, bool alloc_now,
             Initializer init = Initializer::NONE, std::string name = "");

  /**
   * @brief Construct a new HalfTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  HalfTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new HalfTensor object
   *
   * @param d data for the Tensor
   * @param fm format for the Tensor
   */
  HalfTensor(std::vector<std::vector<std::vector<std::vector<_FP16>>>> const &d,
             Tformat fm);

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param rhs TensorBase object to copy
   */
  HalfTensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @brief Basic Destructor
   */
  ~HalfTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator==(const HalfTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  bool operator!=(const HalfTensor &rhs) const { return !(*this == rhs); }

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
   * @param[in] idx location
   */
  const _FP16 &getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] idx location
   */
  _FP16 &getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  const _FP16 &getValue(unsigned int b, unsigned int c, unsigned int h,
                        unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  _FP16 &getValue(unsigned int b, unsigned int c, unsigned int h,
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
   * @brief Set the Dist object
   * @param dist distribution engine
   */
  template <typename Engine> void setDist(Engine dist) {
    NNTR_THROW_IF(!contiguous, std::invalid_argument)
      // << getName() << " Tensor is not contiguous, cannot set distribution";
      << " Tensor is not contiguous, cannot set distribution";

    _FP16 *data_ = (_FP16 *)getData();
    unsigned int len = size();
    for (unsigned int i = 0; i < len; ++i) {
      data_[i] = (_FP16)dist(rng);
    }
  };

  /**
   * @copydoc Tensor::setRandNormal()
   */
  void setRandNormal(float mean = 0.0f, float stddev = 0.05f);

  /**
   * @copydoc Tensor::setRandUniform()
   */
  void setRandUniform(float min = -0.05f, float max = 0.05f);

  /**
   * @copydoc Tensor::setRandBernoulli()
   */
  void setRandBernoulli(float probability = 0.5f);

  /**
   * @copydoc Tensor::initialize()
   */
  void initialize() override;

  /**
   * @copydoc Tensor::initialize(Initializer init)
   */
  void initialize(Initializer init) override;

  /**
   * @copydoc Tensor::apply(std::function<T(T)> f, Tensor &output)
   */
  Tensor &apply(std::function<_FP16(_FP16)> f, Tensor &output) const override;

  /**
   * @copydoc Tensor::multiply_strided(Tensor const &m, Tensor &output,
   * const float beta)
   */
  Tensor multiply_strided(Tensor const &m, Tensor &output,
                          const float beta) const override;

  /**
   * @copydoc Tensor::multiply_i(float const &value)
   */
  int multiply_i(float const &value) override;

  /**
   * @copydoc Tensor::multiply(float const &value, Tensor &out)
   */
  Tensor &multiply(float const &value, Tensor &out) const override;

  /**
   * @copydoc Tensor::multiply(Tensor const &m, Tensor &output, const
   * float beta = 0.0)
   */
  Tensor &multiply(Tensor const &m, Tensor &output,
                   const float beta = 0.0) const override;

  /**
   * @copydoc Tensor::divide(float const &value, Tensor &output)
   */
  Tensor &divide(float const &value, Tensor &output) const override;

  /**
   * @copydoc Tensor::divide(Tensor const &m, Tensor &output)
   */
  Tensor &divide(Tensor const &m, Tensor &output) const override;

  /**
   * @copydoc Tensor::add_strided(Tensor const &input, Tensor &output,
   * const float beta)
   */
  Tensor &add_strided(Tensor const &input, Tensor &output,
                      const float beta) const override;

  /**
   * @copydoc Tensor::add_i(Tensor const &m, float const alpha)
   */
  int add_i(Tensor const &m, Tensor &output, float const alpha) override;

  /**
   * @copydoc Tensor::add_i_partial()
   */
  int add_i_partial(unsigned int len, unsigned int addr_idx, Tensor &m,
                    unsigned int incX, unsigned int incY, const Tensor alphas,
                    unsigned int alpha_idx) override;

  /**
   * @copydoc Tensor::add(float const &value, Tensor &output)
   */
  Tensor &add(float const &value, Tensor &output) const override;

  /**
   * @copydoc Tensor::add(Tensor const &m, Tensor &output, float const
   * alpha)
   */
  Tensor &add(Tensor const &m, Tensor &output,
              float const alpha) const override;

  /**
   * @copydoc Tensor::subtract(float const &value, Tensor &output)
   */
  Tensor &subtract(float const &value, Tensor &output) const override;

  /**
   *  @copydoc TensorBase::sum_by_batch(Tensor &output)
   */
  void sum_by_batch(Tensor &output) const override;

  /**
   * @copydoc Tensor::sum(unsigned int axis, Tensor &output, float alpha,
   * float beta) const
   */
  Tensor &sum(unsigned int axis, Tensor &output, float alpha,
              float beta) const override;

  /**
   * @copydoc Tensor::l2norm
   */
  float l2norm() const override;

  /**
   * @copydoc Tensor::pow(float exponent, Tensor &output)
   */
  Tensor &pow(float exponent, Tensor &output) const override;

  /**
   * @copydoc Tensor::erf(Tensor &output)
   */
  Tensor &erf(Tensor &output) const override;

  /**
   * @copydoc TensorBase::inv_sqrt(Tensor &out)
   */
  void inv_sqrt(Tensor &out) override;

  /**
   *  @copydoc Tensor::dot(Tensor const &input, Tensor &output, bool
   * trans, bool trans_in, float beta)
   */
  Tensor &dot(Tensor const &input, Tensor &output, bool trans, bool trans_in,
              float beta) const override;

  /**
   * @copydoc Tensor::dropout_mask(float dropout)
   */
  void dropout_mask(float dropout) override;

  /**
   * @copydoc Tensor::filter_mask(const Tensor &mask_len, bool reverse)
   */
  void filter_mask(const Tensor &mask_len, bool reverse) override;

  /**
   * @copydoc Tensor::zoneout_mask(Tensor &opposite, float zoneout)
   */
  void zoneout_mask(Tensor &opposite, float zoneout) override;

  /**
   * @copydoc Tensor::split(std::vector<size_t> sizes, int axis)
   */
  std::vector<Tensor> split(std::vector<size_t> sizes, int axis) override;

  /**
   * @copydoc Tensor::cat(const std::vector<Tensor> &tensors, int axis)
   */
  Tensor concat(const std::vector<Tensor> &tensors, int axis) override;

  /**
   * @copydoc Tensor::copy(const Tensor &from)
   */
  void copy(const Tensor &from);

  /**
   * @copydoc Tensor::copyData(const Tensor &from)
   */
  void copyData(const Tensor &from);

  /**
   * @brief      Copy the Tensor
   * @param[in]  input Tensor to be copied
   * @param[out] output output Tensor
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
   * @copydoc Tensor::transpose(const std::string &direction, Tensor &out)
   */
  Tensor &transpose(const std::string &direction,
                    Tensor &output) const override;

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
  void
  apply_broadcast_util(Tensor const &m,
                       std::function<void(const BroadcastInfo &e, const _FP16 *,
                                          const _FP16 *, _FP16 *)>
                         v_func,
                       Tensor &output, const BroadcastInfo &e,
                       int cur_axis = -1, size_t offset = 0,
                       size_t m_offset = 0) const;

  /**
   * @brief Applies the given operator to the tensor with the passed argument
   *
   * @param[in] m Tensor
   * @param[in] v_func vectorized function to apply
   * @retval #ML_ERROR_NONE Successful
   * @retval #ML_ERROR_INVALID_PARAMETER Invalid Parameter
   */
  void apply_broadcast(Tensor const &m,
                       std::function<void(const BroadcastInfo &e, const _FP16 *,
                                          const _FP16 *, _FP16 *)>
                         v_func,
                       Tensor &output) const;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __HALF_TENSOR_H__ */
