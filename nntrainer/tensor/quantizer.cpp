// SPDX-License-Identifier: Apache-2.0
/**
 * @file	quantizer.cpp
 * @date	10 December 2024
 * @brief	This defines quantizers for different types of quantization schemes
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Donghyeon Jeong <dhyeon.jeong@samsung.com>
 * @bug		No known bugs except for NYI items
 */

#include <math.h>
#include <quantizer.h>

namespace nntrainer {

/**
 * @brief Helper function for clipping
 *
 * @tparam T data type
 * @param val value to clip
 * @param lower lower bound
 * @param upper upper bound
 * @return T cliped data
 */
template <typename T> T clip(const T &val, const T &lower, const T &upper) {
  return std::max(lower, std::min(val, upper));
}

/**
 * @brief PerTensorAffineQuantizer class
 */
std::unique_ptr<Quantizer> PerTensorAffineQuantizer::create() {
  return std::make_unique<PerTensorAffineQuantizer>();
}

Tensor PerTensorAffineQuantizer::quantize(const Tensor &input,
                                          Tdatatype qtype) {
  // Currently only full precision floating point is supported
  NNTR_THROW_IF(input.getDataType() != Tdatatype::FP32, std::invalid_argument)
    << "[Quantizer::quantize] Tensor data type is not floating point";

  NNTR_THROW_IF(qtype == Tdatatype::FP32, std::invalid_argument)
    << "[Quantizer::quantize] Cannot quantize to full precision floating point";

  // 1. Calculate quantization parameters
  calculateQParams(input, qtype);

  // 2. Create output tensor with same dimension but different data type
  TensorDim dim = input.getDim();
  dim.setDataType(qtype);
  Tensor output(dim);

  /// @todo this is a naive impl. need optimization
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(
            b, c, h, w,
            clip(std::lround(input.getValue(b, c, h, w) / scale + zero_point),
                 quant_min, quant_max));
        }
      }
    }
  }

  return output;
}

Tensor PerTensorAffineQuantizer::dequantize(const Tensor &input,
                                            Tdatatype dtype) {
  Tensor output = input.clone(dtype);

  /// @todo this is a naive impl. need optimization
  for (unsigned int b = 0; b < output.batch(); ++b) {
    for (unsigned int c = 0; c < output.channel(); ++c) {
      for (unsigned int h = 0; h < output.height(); ++h) {
        for (unsigned int w = 0; w < output.width(); ++w) {
          output.setValue(b, c, h, w,
                          (input.getValue<int8_t>(b, c, h, w) - zero_point) *
                            scale);
        }
      }
    }
  }

  return output;
}

QScheme PerTensorAffineQuantizer::qscheme() const {
  return QScheme::PER_TENSOR_AFFINE;
}

void PerTensorAffineQuantizer::calculateQParams(const Tensor &input,
                                                Tdatatype qtype) {
  unsigned int N;

  if (qtype == Tdatatype::QINT8) {
    N = 8;
  } else if (qtype == Tdatatype::QINT4) {
    N = 4;
  } else {
    throw std::invalid_argument("Error: Unsupported data type.");
  }

  quant_max = std::pow(2, N - 1) - 1;
  quant_min = -std::pow(2, N - 1);

  /// @todo for quint8, zero point calculation should be added
  float max_val = input.max_abs();
  scale = max_val / ((quant_max - quant_min) / 2.0f);
  scale = std::max(scale, std::numeric_limits<float>::epsilon());
}

/**
 * @brief PerChannelAffineQuantizer class
 */
std::unique_ptr<Quantizer> PerChannelAffineQuantizer::create() {
  return std::make_unique<PerChannelAffineQuantizer>();
}

Tensor PerChannelAffineQuantizer::quantize(const Tensor &input,
                                           Tdatatype qtype) {
  /// @todo NYI
  return input;
}

Tensor PerChannelAffineQuantizer::dequantize(const Tensor &input,
                                             Tdatatype dtype) {
  /// @todo NYI
  return input;
}

QScheme PerChannelAffineQuantizer::qscheme() const {
  return QScheme::PER_CHANNEL_AFFINE;
}

/**
 * @brief BinaryCodeBasedQuantizer class
 */
std::unique_ptr<Quantizer> BinaryCodeBasedQuantizer::create() {
  return std::make_unique<BinaryCodeBasedQuantizer>();
}

Tensor BinaryCodeBasedQuantizer::quantize(const Tensor &input,
                                          Tdatatype qtype) {
  /// @todo NYI
  return input;
}

Tensor BinaryCodeBasedQuantizer::dequantize(const Tensor &input,
                                            Tdatatype dtype) {
  /// @todo NYI
  return input;
}

QScheme BinaryCodeBasedQuantizer::qscheme() const {
  return QScheme::BINARY_CODE_BASED;
}

} // namespace nntrainer
