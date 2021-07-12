// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   mae_loss.h
 * @date   10 June 2021
 * @brief  This file contains the mean absoulte error loss as a sample layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __MAE_LOSS_LAYER_H__
#define __MAE_LOSS_LAYER_H__
#include <string>

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace custom {

/**
 * @brief A sample loss layer which calculates mean absolute error from output
 *
 */
class MaeLossLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new MAE Layer object that does elementwise power
   *
   */
  MaeLossLayer() : Layer() {}

  /**
   * @brief Destroy the MAE Layer object
   *
   */
  ~MaeLossLayer() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override {
    context.setOutputDimensions(context.getInputDimensions());
    /** NYI */
  }

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override

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
  const std::string getType() const override { return MaeLossLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override {
    if (!values.empty()) {
      std::string msg = "[MaeLossLayer] Unknown Layer Properties count " +
                        std::to_string(values.size());
      throw std::invalid_argument(msg);
    }
  }

  /**
   * @copydoc Layer::requireLabel()
   */
  bool MaeLossLayer::requireLabel() const { return true; }

  inline static const std::string type = "mae_loss";
};

} // namespace custom

#endif /* __MAE_LOSS_LAYER_H__ */
