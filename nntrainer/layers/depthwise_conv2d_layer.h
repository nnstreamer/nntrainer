// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file   depthwise_conv2d_layer.h
 * @date   27 March 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Depthwise Convolution Layer Class for Neural Network
 *
 */

#ifndef __DEPTHWISE_CONV2D_LAYER_H_
#define __DEPTHWISE_CONV2D_LAYER_H_
#ifdef __cplusplus

#include <memory.h>

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

constexpr const unsigned int DEPTHWISE_CONV2D_DIM = 2;

/**
 * @class   Depthwise Convolution 2D Layer
 * @brief   Depthwise Convolution 2D Layer
 */
class DepthwiseConv2DLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Depthwise Convolution 2D Layer
   */
  DepthwiseConv2DLayer(
    const std::array<unsigned int, CONV2D_DIM * 2> &padding_ = {0, 0, 0, 0});

  /**
   * @brief     Destructor of Depthwise Convolution 2D Layer
   */
  ~DepthwiseConv2DLayer() = default;

  /**
   *  @brief  Move constructor of Depthwise Convolution 2D Layer
   *  @param[in] Conv2dLayer &&
   */
  DepthwiseConv2DLayer(DepthwiseConv2DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs DepthwiseConv2DLayer to be moved.
   */
  DepthwiseConv2DLayer &operator=(DepthwiseConv2DLayer &&rhs) = default;

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
    return DepthwiseConv2DLayer::type;
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

  inline static const std::string type = "depthwiseconv2d";

private:
  std::array<unsigned int, DEPTHWISE_CONV2D_DIM * 2> padding;
  std::tuple<props::FilterSize,
             std::array<props::KernelSize, DEPTHWISE_CONV2D_DIM>,
             std::array<props::Stride, DEPTHWISE_CONV2D_DIM>, props::Padding2D,
             std::array<props::Dilation, DEPTHWISE_CONV2D_DIM>>
    depthwise_conv_props;

  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __DEPTHWISE_CONV2D_LAYER_H_ */
