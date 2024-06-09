// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 heka1024 <heka1024@gmail.com>
 *
 * @file   upsample2d_layer.h
 * @date   15 June 2024
 * @brief  This is Upsample2d Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author heka1024 <heka1024@gmail.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __UPSAMPLE2D_LAYER_H__
#define __UPSAMPLE2D_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

#include <node_exporter.h>

namespace nntrainer {

constexpr const unsigned int UPSAMPLE2D_DIM = 2;

/**
 * @class   Upsample2dLayer
 * @brief   Upsamle 2d layer
 */
class Upsample2dLayer : public Layer {
public:
  /**
   * @brief Construct a new Upsample layer object
   *
   */
  Upsample2dLayer();

  /**
   * @brief Destroy the Upsample layer object
   *
   */
  ~Upsample2dLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return Upsample2dLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "upsample2d";

private:
  std::tuple<props::UpsampleMode, std::array<props::KernelSize, UPSAMPLE2D_DIM>>
    upsample2d_props; /* mode, size of kernel */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __UPSAMPLE2D_LAYER_H__ */
