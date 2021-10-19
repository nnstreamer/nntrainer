// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   reshape_layer.h
 * @date   16 June 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Reshape Layer Class for Neural Network
 *
 */

#ifndef __RESHAPE_LAYER_H__
#define __RESHAPE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Reshape Layer
 * @brief   Reshape Layer
 */
class ReshapeLayer : public Layer {
public:
  /**
   * @brief     Constructor of Reshape Layer
   */
  ReshapeLayer() : Layer() {}

  /**
   * @brief     Destructor of Reshape Layer
   */
  ~ReshapeLayer() = default;

  /**
   *  @brief  Move constructor of ReshapeLayer.
   *  @param[in] ReshapeLayer &&
   */
  ReshapeLayer(ReshapeLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ReshapeLayer to be moved.
   */
  ReshapeLayer &operator=(ReshapeLayer &&rhs) = default;

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
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return true; }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ReshapeLayer::type; };

  inline static const std::string type = "reshape";

private:
  std::tuple<props::TargetShape>
    reshape_props; /**< reshape properties : target_shape after reshape */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RESHAPE_LAYER_H__ */
