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
  NNTR_EXPORT FloatTensor(std::string name_ = "", Tformat fm = Tformat::NCHW);

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d Tensor dim for this float tensor
   * @param alloc_now Allocate memory to this tensor or not
   * @param init Initializer for the tensor
   * @param name Name of the tensor
   */
  NNTR_EXPORT FloatTensor(const TensorDim &d, bool alloc_now,
                       Initializer init = Initializer::NONE,
                       std::string name = "");

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d Tensor dim for this tensor
   * @param buf buffer
   */
  NNTR_EXPORT FloatTensor(const TensorDim &d, const void *buf = nullptr);

  /**
   * @brief Construct a new FloatTensor object
   *
   * @param d data for the Tensor
   * @param fm format for the Tensor
   */
  NNTR_EXPORT FloatTensor(
    std::vector<std::vector<std::vector<std::vector<float>>>> const &d,
    Tformat fm) {
    if (d.empty() || d[0].empty() || d[0][0].empty() || d[0][0][0].empty()) {
      throw std::out_of_range(
        "[Tensor] trying to initialize FloatTensor from empty vector");
    }

    dim.setTensorDim(0, d.size());
    if (fm == Tformat::NCHW) {
      dim.setTensorDim(1, d[0].size());
      dim.setTensorDim(2, d[0][0].size());
      dim.setTensorDim(3, d[0][0][0].size());
    } else {
      dim.setTensorDim(2, d[0].size());
      dim.setTensorDim(3, d[0][0].size());
      dim.setTensorDim(1, d[0][0][0].size());
    }

    dim.setTensorType({fm, Tdatatype::FP32});

    strides = dim.computeStrides();
    contiguous = true;
    initializer = Initializer::NONE;

    MemoryData *mem_data =
      new MemoryData((void *)(new float[dim.getDataLen()]()));
    data = std::shared_ptr<MemoryData>(mem_data, [](MemoryData *mem_data) {
      delete[] mem_data->getAddr<float>();
      delete mem_data;
    });

    offset = 0;

    // if fm == Tformat::NCHW, then dim[0] == batch , dim[1] == channel, dim[2]
    // == height, dim[3] == width. and if fm == Tformat::NHWC, dim[0] == batch,
    // dim[1] == height, dim[2] == width, dim[3] == channel
    if (fm == Tformat::NCHW) {
      for (unsigned int i = 0; i < batch(); ++i)
        for (unsigned int j = 0; j < channel(); ++j)
          for (unsigned int k = 0; k < height(); ++k)
            for (unsigned int l = 0; l < width(); ++l)
              this->setValue(i, j, k, l, d[i][j][k][l]);
    } else {
      for (unsigned int i = 0; i < batch(); ++i)
        for (unsigned int j = 0; j < height(); ++j)
          for (unsigned int k = 0; k < width(); ++k)
            for (unsigned int l = 0; l < channel(); ++l)
              this->setValue(i, l, j, k, d[i][j][k][l]);
    }
  }

  /**
   * @brief Construct a new FloatTensor object
   * @param rhs TensorBase object to copy
   */
  NNTR_EXPORT FloatTensor(TensorBase &rhs) : TensorBase(rhs) {}

  /**
   * @brief Basic Destructor
   */
  NNTR_EXPORT ~FloatTensor() {}

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  NNTR_EXPORT bool operator==(const FloatTensor &rhs) const;

  /**
   * @brief     Comparison operator overload
   * @param[in] rhs Tensor to be compared with
   * @note      Only compares Tensor data
   */
  NNTR_EXPORT bool operator!=(const FloatTensor &rhs) const {
    return !(*this == rhs);
  }

  /**
   * @copydoc Tensor::allocate()
   */
  NNTR_EXPORT void allocate() override;

  /**
   * @copydoc Tensor::deallocate()
   */
  NNTR_EXPORT void deallocate() override;

  /**
   * @copydoc Tensor::getData()
   */
  NNTR_EXPORT void *getData() const override;

  /**
   * @copydoc Tensor::getData(size_t idx)
   */
  NNTR_EXPORT void *getData(size_t idx) const override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  NNTR_EXPORT void *getAddress(unsigned int i) override;

  /**
   * @brief     i data index
   * @retval    address of ith data
   */
  NNTR_EXPORT const void *getAddress(unsigned int i) const override;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  NNTR_EXPORT const float &getValue(unsigned int i) const;

  /**
   * @brief     return value at specific location
   * @param[in] i index
   */
  NNTR_EXPORT float &getValue(unsigned int i);

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  NNTR_EXPORT const float &getValue(unsigned int b, unsigned int c, unsigned int h,
                                 unsigned int w) const;

  /**
   * @brief     return value at specific location
   * @param[in] b batch location
   * @param[in] c channel location
   * @param[in] h height location
   * @param[in] w width location
   */
  NNTR_EXPORT float &getValue(unsigned int b, unsigned int c, unsigned int h,
                           unsigned int w);

  /**
   * @copydoc Tensor::setValue(float value)
   */
  NNTR_EXPORT void setValue(float value) override;

  /**
   * @copydoc Tensor::setValue(b, c, h, w, value)
   */
  NNTR_EXPORT void setValue(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w, float value) override;

  /**
   * @copydoc Tensor::addValue(b, c, h, w, value, beta)
   */
  NNTR_EXPORT void addValue(unsigned int b, unsigned int c, unsigned int h,
                         unsigned int w, float value, float beta) override;

  /**
   * @copydoc Tensor::setZero()
   */
  NNTR_EXPORT void setZero() override;

  /**
   * @brief Set the Dist object
   * @param dist distribution engine
   */
  template <typename Engine> NNTR_EXPORT void setDist(Engine dist) {
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
   * @copydoc Tensor::setRandNormal()
   */
  NNTR_EXPORT void setRandNormal(float mean = 0.0f, float stddev = 0.05f) override;

  /**
   * @copydoc Tensor::setRandUniform()
   */
  NNTR_EXPORT void setRandUniform(float min = -0.05f, float max = 0.05f) override;

  /**
   * @copydoc Tensor::setRandBernoulli()
   */
  NNTR_EXPORT void setRandBernoulli(float probability = 0.5f) override;

  /**
   * @copydoc Tensor::initialize()
   */
  NNTR_EXPORT void initialize() override;

  /**
   * @copydoc Tensor::initialize(Initializer init)
   */
  NNTR_EXPORT void initialize(Initializer init) override;

  /**
   * @copydoc Tensor::apply(std::function<T(T)> f, Tensor &output)
   */
  NNTR_EXPORT Tensor &apply(std::function<float(float)> f,
                         Tensor &output) const override;

  /**
   * @copydoc Tensor::multiply_strided(Tensor const &m, Tensor &output,
   * const float beta)
   */
  NNTR_EXPORT Tensor multiply_strided(Tensor const &m, Tensor &output,
                                   const float beta) const override;

  /**
   * @copydoc Tensor::multiply_i(float const &value)
   */
  NNTR_EXPORT int multiply_i(float const &value) override;

  /**
   * @copydoc Tensor::multiply(float const &value, Tensor &out)
   */
  NNTR_EXPORT Tensor &multiply(float const &value, Tensor &out) const override;

  /**
   * @copydoc Tensor::multiply(Tensor const &m, Tensor &output, const
   * float beta = 0.0)
   */
  NNTR_EXPORT Tensor &multiply(Tensor const &m, Tensor &output,
                            const float beta = 0.0) const override;

  /**
   * @copydoc Tensor::divide(float const &value, Tensor &output)
   */
  NNTR_EXPORT Tensor &divide(float const &value, Tensor &output) const override;

  /**
   * @copydoc Tensor::divide(Tensor const &m, Tensor &output)
   */
  NNTR_EXPORT Tensor &divide(Tensor const &m, Tensor &output) const override;

  /**
   * @copydoc Tensor::add_strided(Tensor const &input, Tensor &output,
   * const float beta)
   */
  NNTR_EXPORT Tensor &add_strided(Tensor const &input, Tensor &output,
                               const float beta) const override;

  /**
   * @copydoc Tensor::add_i_partial()
   */
  NNTR_EXPORT int add_i_partial(unsigned int len, unsigned int addr_idx, Tensor &m,
                             unsigned int incX, unsigned int incY,
                             const Tensor alphas,
                             unsigned int alpha_idx) override;

  /**
   * @copydoc Tensor::add(float const &value, Tensor &output)
   */
  NNTR_EXPORT Tensor &add(float const &value, Tensor &output) const override;

  /**
   * @copydoc Tensor::add(Tensor const &m, Tensor &output, float const
   * alpha)
   */
  NNTR_EXPORT Tensor &add(Tensor const &m, Tensor &output,
                       float const alpha) const override;

  /**
   *  @copydoc Tensor::subtract(float const &value, Tensor &output)
   */
  NNTR_EXPORT Tensor &subtract(float const &value, Tensor &output) const override;

  /**
   *  @copydoc TensorBase::sum_by_batch(Tensor &output)
   */
  NNTR_EXPORT void sum_by_batch(Tensor &output) const override;

  /**
   * @copydoc Tensor::sum(unsigned int axis, Tensor &output, float alpha,
   * float beta) const
   */
  NNTR_EXPORT Tensor &sum(unsigned int axis, Tensor &output, float alpha,
                       float beta) const override;

  /**
   * @copydoc Tensor::abs()
   */
  NNTR_EXPORT Tensor &abs(Tensor &output) const override;

  /**
   * @copydoc Tensor::l2norm
   */
  NNTR_EXPORT float l2norm() const override;

  /**
   * @copydoc Tensor::pow(float exponent, Tensor &output)
   */
  NNTR_EXPORT Tensor &pow(float exponent, Tensor &output) const override;

  /**
   * @copydoc Tensor::sqrt(&output)
   */
  NNTR_EXPORT Tensor &sqrt(Tensor &output) const override;

  /**
   * @copydoc Tensor::erf(Tensor &output)
   */
  NNTR_EXPORT Tensor &erf(Tensor &output) const override;

  /**
   * @copydoc Tensor::sin(Tensor &out, float alpha)
   */
  NNTR_EXPORT void sin(Tensor &out, float alpha) override;

  /**
   * @copydoc Tensor::cos(Tensor &out, float alpha)
   */
  NNTR_EXPORT void cos(Tensor &out, float alpha) override;

  /**
   * @copydoc Tensor::tan(Tensor &output, float alpha)
   */
  NNTR_EXPORT void tan(Tensor &output, float alpha) override;

  /**
   * @copydoc TensorBase::inv_sqrt(Tensor &out)
   */
  NNTR_EXPORT void inv_sqrt(Tensor &out) override;

  /**
   *  @copydoc Tensor::dot(Tensor const &input, Tensor &output, bool
   * trans, bool trans_in, float beta)
   */
  NNTR_EXPORT Tensor &dot(Tensor const &input, Tensor &output, bool trans,
                       bool trans_in, float beta) const override;

  /**
   * @copydoc Tensor::dropout_mask(float dropout)
   */
  NNTR_EXPORT void dropout_mask(float dropout) override;

  /**
   * @copydoc Tensor::filter_mask(const Tensor &mask_len, bool reverse)
   */
  NNTR_EXPORT void filter_mask(const Tensor &mask_len, bool reverse) override;

  /**
   * @copydoc Tensor::zoneout_mask(Tensor &opposite, float zoneout)
   */
  NNTR_EXPORT void zoneout_mask(Tensor &opposite, float zoneout) override;

  /**
   * @copydoc Tensor::split(std::vector<size_t> sizes, int axis)
   */
  NNTR_EXPORT std::vector<Tensor> split(std::vector<size_t> sizes,
                                     int axis) override;

  /**
   * @copydoc Tensor::concat()
   */
  NNTR_EXPORT Tensor concat(const std::vector<Tensor> &tensors, int axis,
                         Tensor &output) override;

  /**
   * @copydoc Tensor::copy(const Tensor &from)
   */
  NNTR_EXPORT void copy(const Tensor &from) override;

  /**
   * @copydoc Tensor::copyData(const Tensor &from)
   */
  NNTR_EXPORT void copyData(const Tensor &from) override;

  /**
   * @brief      Copy the Tensor
   * @param[in]  input Tensor to be copied
   * @param[out] output output Tensor
   */
  NNTR_EXPORT void copy_with_stride(const Tensor &input, Tensor &output) override;

  /**
   * @copydoc Tensor::argmax()
   */
  NNTR_EXPORT std::vector<unsigned int> argmax() const override;

  /**
   * @copydoc Tensor::argmin()
   */
  NNTR_EXPORT std::vector<unsigned int> argmin() const override;

  /**
   * @copydoc TensorBase::top_k()
   */
  NNTR_EXPORT void topK(unsigned int k, void *output_data,
                     uint32_t *indices) override;

  /**
   * @copydoc Tensor::max_abs()
   */
  NNTR_EXPORT float max_abs() const override;
  /**
   * @copydoc Tensor::maxValue()
   */
  NNTR_EXPORT float maxValue() const override;

  /**
   * @copydoc Tensor::minValue()
   */
  NNTR_EXPORT float minValue() const override;

  /**
   * @copydoc Tensor::transpose(const std::string &direction, Tensor &out)
   */
  NNTR_EXPORT Tensor &transpose(const std::string &direction,
                             Tensor &output) const override;

  /**
   * @copydoc Tensor::print(std::ostream &out)
   */
  NNTR_EXPORT void print(std::ostream &out) const override;

private:
  /**
   * @brief copy a buffer to @a this, the caller has to ensure that @a this is
   * initialized otherwise undefined behavior
   *
   * @param buf buffer to copy from
   */
  NNTR_EXPORT void copy(const void *buf);

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
  NNTR_EXPORT void
  apply_broadcast_util(Tensor const &m,
                       std::function<void(const BroadcastInfo &e, const float *,
                                          const float *, float *)>
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
  NNTR_EXPORT void
  apply_broadcast(Tensor const &m,
                  std::function<void(const BroadcastInfo &e, const float *,
                                     const float *, float *)>
                    v_func,
                  Tensor &output) const;

  /**
   * @brief  Get the Data Type String object
   * @return std::string of tensor data type (FP32)
   */
  NNTR_EXPORT std::string getStringDataType() const override { return "FP32"; }

  /**
   * @copydoc Tensor::isValid()
   */
  NNTR_EXPORT bool isValid() const override;

  /**
   * @brief Float.dot(Float)
   * @return Tensor& reference to the output tensor
   */
  NNTR_EXPORT Tensor &dotFloat(Tensor const &input, Tensor &output, bool trans,
                            bool trans_in, float beta) const;

  /**
   * @brief Float.dot(Q4K/Q6K)
   * @return Tensor& reference to the output tensor
   */
  NNTR_EXPORT Tensor &dotQnK(Tensor const &input, Tensor &output, bool trans,
                          bool trans_in, float beta, Tdatatype dtype) const;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FLOAT_TENSOR_H__ */
