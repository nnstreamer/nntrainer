// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rnnt_loss.h
 * @date   22 July 2021
 * @brief  This file contains the rnnt loss
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __RNNT_LOSS_LAYER_H__
#define __RNNT_LOSS_LAYER_H__
#include <string>

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace custom {

/**
 * @brief A rnnt loss layer which calculates rnnt loss
 *
 */
class RNNTLossLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new RNNT Loss Layer object
   *
   */
  RNNTLossLayer() : Layer() {}

  /**
   * @brief Destroy the RNNT Loss Layer object
   *
   */
  ~RNNTLossLayer() = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

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
                const nntrainer::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const { return true; }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return RNNTLossLayer::type; }

  inline static const std::string type = "rnnt_loss";
};

} // namespace custom

#endif /* __RNNT_LOSS_LAYER_H__ */
