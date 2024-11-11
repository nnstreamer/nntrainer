// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 UGyeong Song <thddnrud@snu.ac.kr>
 *
 * @file   conv2d_transpose_layer.h
 * @date   13 October 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author UGyeong Song <thddnrud@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is Transposed Convolution Layer Class for Neural Network
 *
 */

#ifndef __CONV2D_TRANSPOSE_LAYER_H_
#define __CONV2D_TRANSPOSE_LAYER_H_
#ifdef __cplusplus

#include <memory.h>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

constexpr const unsigned int CONV2D_TRANSPOSE_DIM = 2;

/**
 * @class   Convolution 2D Transpose Layer
 * @brief   Convolution 2D Transpose Layer
 */
class Conv2DTransposeLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Conv 2D Transpose Layer
   */
  Conv2DTransposeLayer(const std::array<unsigned int, CONV2D_TRANSPOSE_DIM * 2>
                         &padding_ = {0, 0, 0, 0});

  /**
   * @brief     Destructor of Conv 2D Transpose Layer
   */
  ~Conv2DTransposeLayer() = default;

  /**
   *  @brief  Move constructor of Conv 2D Transpose Layer.
   *  @param[in] Conv2dTransposeLayer &&
   */
  Conv2DTransposeLayer(Conv2DTransposeLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Conv2DTransposeLayer to be moved.
   */
  Conv2DTransposeLayer &operator=(Conv2DTransposeLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return Conv2DTransposeLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /* TO DO : support keras type of padding */
  /* enum class PaddingType { */
  /*   full = 0, */
  /*   same = 1, */
  /*   valid = 2, */
  /*   unknown = 3, */
  /* }; */

  inline static const std::string type = "conv2dtranspose";

private:
  std::array<unsigned int, CONV2D_TRANSPOSE_DIM * 2> padding;
  std::tuple<props::FilterSize,
             std::array<props::KernelSize, CONV2D_TRANSPOSE_DIM>,
             std::array<props::Stride, CONV2D_TRANSPOSE_DIM>, props::Padding2D,
             std::array<props::Dilation, CONV2D_TRANSPOSE_DIM>>
    conv_props;

  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV2D_TRANSPOSE_LAYER_H__ */
