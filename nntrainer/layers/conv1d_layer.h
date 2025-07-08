// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   conv1d_layer.h
 * @date   13 Oct 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Convolution 1D Layer Class for Neural Network
 *
 */

#ifndef __CONV1D_LAYER_H_
#define __CONV1D_LAYER_H_
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>
#include <memory.h>
#include <conv2d_layer.h>

namespace nntrainer {

class Conv2DLayer;

/**
 * @class   Convolution 1D Layer
 * @brief   Convolution 1D Layer
 */
class Conv1DLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Conv 1D Layer
   */
  NNTR_API Conv1DLayer();

  /**
   * @brief     Destructor of Conv 1D Layer
   */
  NNTR_API ~Conv1DLayer();

  /**
   *  @brief  Move constructor of Conv 1D Layer.
   *  @param[in] Conv1dLayer &&
   */
  NNTR_API Conv1DLayer(Conv1DLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs Conv1DLayer to be moved.
   */
  NNTR_API Conv1DLayer &operator=(Conv1DLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_API void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  NNTR_API void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_API void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const override {
    return Conv1DLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  NNTR_API bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "conv1d";

private:
  std::tuple<props::FilterSize, props::KernelSize, props::Stride,
             props::Padding1D, props::Dilation>
    conv_props;

  std::array<unsigned int, 5> wt_idx; /**< indices of the weights and tensors */
  std::unique_ptr<Conv2DLayer> conv2d_layer; /**< conv2d layer instance */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONV1D_LAYER_H__ */
