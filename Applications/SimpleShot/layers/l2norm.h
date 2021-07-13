// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   l2norm.h
 * @date   09 Jan 2021
 * @brief  This file contains the simple l2norm layer which normalizes
 * the given feature
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __L2NORM_H__
#define __L2NORM_H__
#include <string>

#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace simpleshot {
namespace layers {

/**
 * @brief Layer class that l2normalizes a feature vector
 *
 */
class L2NormLayer : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new L2norm Layer object
   * that normlizes given feature with l2norm
   */
  L2NormLayer() : Layer() {}

  /**
   *  @brief  Move constructor.
   *  @param[in] L2NormLayer &&
   */
  L2NormLayer(L2NormLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs L2NormLayer to be moved.
   */
  L2NormLayer &operator=(L2NormLayer &&rhs) = default;

  /**
   * @brief Destroy the Centering Layer object
   *
   */
  ~L2NormLayer() {}

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
                const nntrainer::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return L2NormLayer::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "l2norm";
};
} // namespace layers
} // namespace simpleshot

#endif /* __L2NORM__H_ */
