// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file        output_layer.h
 * @date        05 Nov 2020
 * @see         https://github.com/nnstreamer/nntrainer
 * @author      Jijoong Moon <jijoong.moon@samsung.com>
 * @bug         No known bugs except for NYI items
 * @brief       This is Multi Output Layer Class for Neural Network
 *
 * @todo        Support inplace for this layer
 */

#ifndef __OUTPUT_LAYER_H__
#define __OUTPUT_LAYER_H__
#ifdef __cplusplus

#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Output Layer
 * @brief   Output Layer
 */
class OutputLayer : public Layer {
public:
  /**
   * @brief     Constructor of Output Layer
   */
  OutputLayer() : Layer() {}

  /**
   * @brief     Destructor of Output Layer
   */
  ~OutputLayer() = default;

  /**
   *  @brief  Move constructor of OutputLayer.
   *  @param[in] OutputLayer &&
   */
  OutputLayer(OutputLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs OutputLayer to be moved.
   */
  OutputLayer &operator=(OutputLayer &&rhs) = default;

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return OutputLayer::type; };

  inline static const std::string type = "multiout";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __OUTPUT_LAYER_H__ */
