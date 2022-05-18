
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   preprocess_l2norm_layer.h
 * @date   09 Jan 2021
 * @brief  This file contains the simple l2norm layer which normalizes
 * the given feature
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __PREPROCESS_L2NORM_LAYER_H__
#define __PREPROCESS_L2NORM_LAYER_H__
#include <string>

#include <layer_devel.h>

namespace nntrainer {

/**
 * @brief Layer class that l2normalizes a feature vector
 *
 */
class PreprocessL2NormLayer : public Layer {
public:
  /**
   * @brief Construct a new L2norm Layer object
   * that normlizes given feature with l2norm
   */
  PreprocessL2NormLayer() : Layer() {}

  /**
   *  @brief  Move constructor.
   *  @param[in] PreprocessL2NormLayer &&
   */
  PreprocessL2NormLayer(PreprocessL2NormLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs PreprocessL2NormLayer to be moved.
   */
  PreprocessL2NormLayer &operator=(PreprocessL2NormLayer &&rhs) = default;

  /**
   * @brief Destroy the Centering Layer object
   *
   */
  ~PreprocessL2NormLayer() {}

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
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return PreprocessL2NormLayer::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "preprocess_l2norm";
};
} // namespace nntrainer

#endif // __PREPROCESS_L2NORM_LAYER_H__
