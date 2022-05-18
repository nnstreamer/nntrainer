// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   centering.h
 * @date   08 Jan 2021
 * @brief  This file contains the simple centering layer which has hardcoded
 * mean feature vectors from given combinations
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CENTERING_H__
#define __CENTERING_H__
#include <string>

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace simpleshot {
namespace layers {

/**
 * @brief Center Layer Class that does elementwise
 * subtraction from mean feature vector
 *
 */
class CenteringLayer : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new Centering Layer object
   */
  CenteringLayer() : Layer() {}

  /**
   * @brief Construct a new Centering Layer object
   *
   * @param feature_path feature path to read the variable
   */
  CenteringLayer(const std::string &feature_path);

  /**
   *  @brief  Move constructor.
   *  @param[in] CenteringLayer &&
   */
  CenteringLayer(CenteringLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CenteringLayer to be moved.
   */
  CenteringLayer &operator=(CenteringLayer &&rhs) = default;

  /**
   * @brief Destroy the Centering Layer object
   *
   */
  ~CenteringLayer() {}

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
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return CenteringLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "centering";

private:
  std::string feature_path;
  nntrainer::Tensor mean_feature_vector;
};
} // namespace layers
} // namespace simpleshot

#endif /* __CENTERING_H__ */
