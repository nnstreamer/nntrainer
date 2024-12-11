// SPDX-License-Identifier: Apache-2.0
/**
 * @file	quantizer.h
 * @date	10 December 2024
 * @brief	This defines quantizers for different types of quantization schemes
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#ifndef __QUANTIZER_H__
#define __QUANTIZER_H__
#ifdef __cplusplus

#include <tensor.h>

namespace nntrainer {

/**
 * @brief defines the quantization scheme
 */
enum class QScheme : uint8_t {
  PER_TENSOR_AFFINE = 0,
  PER_CHANNEL_AFFINE = 1,
  BINARY_CODE_BASED = 2,
};

/**
 * @class Quantizer class
 * @brief Quantizer class is a base class for all quantizers.
 */
class Quantizer {
public:
  /**
   * @brief Basic Constructor of a Quantizer
   */
  Quantizer(Tdatatype dtype_) : dtype(dtype_) {
    NNTR_THROW_IF(dtype_ == Tdatatype::FP32, std::invalid_argument)
      << "Error: cannot quantize to full-precision floating point.";
  }

  /**
   * @brief Basic Destructor of a Quantizer
   */
  virtual ~Quantizer() {}

  /**
   * @brief Quantize a tensor into a quantized tensor.
   * @param[in] input Floating point tensor to quantize
   * @return Tensor quantized tensor
   */
  virtual Tensor quantize(const Tensor &input) = 0;

  /**
   * @brief Dequantize a quantized tensor into a tensor.
   * @param[in] input Quantized tensor to dequantize
   * @return Tensor dequantized tensor
   */
  virtual Tensor dequantize(const Tensor &input) = 0;

  /**
   * @brief Get quantization Scheme type.
   * @return Quantization scheme
   */
  virtual QScheme qscheme() const = 0;

protected:
  // data type to quantize
  Tdatatype dtype;
};

/**
 * @class UniformQuantizer class
 * @brief UniformQuantizer class serves as the parent class for various types of
 * uniform quantizers.
 */
class UniformQuantizer : public Quantizer {
public:
  UniformQuantizer(Tdatatype dtype_) : Quantizer(dtype_) {}
};

/**
 * @class NonUniformQuantizer class
 * @brief NonUniformQuantizer class serves as the parent class for various types
 * of non-uniform quantizers.
 */
class NonUniformQuantizer : public Quantizer {
public:
  NonUniformQuantizer(Tdatatype dtype_) : Quantizer(dtype_) {}
};

/**
 * @class PerTensorAffineQuantizer class
 * @brief PerTensorAffineQuantizer class uses affine quantization scheme.
 *
 * Quantization: x_q = clip(round(x / scale + zero_point), min, max)
 * Dequantization: x = scale * (x_q - zero_point)
 *
 * @note Single scale and zero point values are used for the entire tensor.
 */
class PerTensorAffineQuantizer : public UniformQuantizer {
public:
  /**
   * @brief Basic Constructor of a PerTensorAffineQuantizer
   */
  PerTensorAffineQuantizer(Tdatatype dtype_) : UniformQuantizer(dtype_) {}

  /**
   * @copydoc Quantizer::quantize(const Tensor &input)
   */
  Tensor quantize(const Tensor &input) override;

  /**
   * @copydoc Quantizer::dequantize(const Tensor &input)
   */
  Tensor dequantize(const Tensor &input) override;

  /**
   * @copydoc Quantizer::qscheme()
   */
  QScheme qscheme() const override;
};

/**
 * @class PerChannelAffineQuantizer class
 * @brief PerChannelAffineQuantizer class uses affine quantization scheme.
 *
 * @note PerChannelAffineQuantizer is similar to PerTensorAffineQuantizer, but
 * it has separate scale and zero_point parameters for each channel. This allows
 * for more precise quantization of different channels within the same tensor.
 *
 */
class PerChannelAffineQuantizer : public UniformQuantizer {
public:
  /**
   * @brief Basic Constructor of a PerChannelAffineQuantizer
   */
  PerChannelAffineQuantizer(Tdatatype dtype_) : UniformQuantizer(dtype_) {}

  /**
   * @copydoc Quantizer::quantize(const Tensor &input)
   */
  Tensor quantize(const Tensor &input) override;

  /**
   * @copydoc Quantizer::dequantize(const Tensor &input)
   */
  Tensor dequantize(const Tensor &input) override;

  /**
   * @copydoc Quantizer::qscheme()
   */
  QScheme qscheme() const override;
};

/**
 * @class BinaryCodeBasedQuantizer class
 * @brief BinaryCodeBasedQuantizer class uses Binary-code-based quantization
 * (BCQ) scheme.
 *
 */
class BinaryCodeBasedQuantizer : public NonUniformQuantizer {
public:
  BinaryCodeBasedQuantizer(Tdatatype dtype_) : NonUniformQuantizer(dtype_) {}

  /**
   * @copydoc Quantizer::quantize(const Tensor &input)
   */
  Tensor quantize(const Tensor &input) override;

  /**
   * @copydoc Quantizer::dequantize(const Tensor &input)
   */
  Tensor dequantize(const Tensor &input) override;

  /**
   * @copydoc Quantizer::qscheme()
   */
  QScheme qscheme() const override;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __QUANTIZER_H__ */
