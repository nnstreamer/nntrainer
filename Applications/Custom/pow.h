// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   pow.h
 * @date   16 November 2020
 * @brief  This file contains the simple pow2 layer which squares input
 * elements.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __POW_LAYER_H__
#define __POW_LAYER_H__

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace custom {

/**
 * @brief layer class that calculates f(x) = x ^ exponent (exponent is
 * configurable by PowLayer::setProperty)
 *
 */
class PowLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new Pow Layer object that does elementwise power
   *
   * @param exponent_ exponent
   */
  PowLayer(float exponent_ = 1) : Layer(), exponent(exponent_) {}

  /**
   * @brief Destroy the Pow Layer object
   *
   */
  ~PowLayer() {}

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
                const nntrainer::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return PowLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "pow";

private:
  float exponent;
};

} // namespace custom

#endif /* __POW_LAYER_H__ */
