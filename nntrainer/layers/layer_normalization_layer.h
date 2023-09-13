// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   layer_normalization_layer.h
 * @date   25 July 2022
 * @see    https://github.com/nnstreamer/nntrainer
 *         https://arxiv.org/abs/1607.06450
 * @author hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Layer Normalization Layer Class for Neural Network
 *
 */

#ifndef __LAYER_NORMALIZATION_LAYER_H__
#define __LAYER_NORMALIZATION_LAYER_H__
#ifdef __cplusplus

#include <array>
#include <functional>
#include <vector>

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   LayerNormalizationLayer
 * @brief   Layer Noramlization Layer
 */
class LayerNormalizationLayer : public Layer {
public:
  /**
   * @brief     Constructor of LayerNormalizationLayer
   */
  LayerNormalizationLayer();

  /**
   * @brief     Destructor of LayerNormalizationLayer
   */
  ~LayerNormalizationLayer() {}

  /**
   * @brief  Move constructor of LayerNormalizationLayer
   * @param[in] rhs LayerNormalizationLayer to be moved
   */
  LayerNormalizationLayer(LayerNormalizationLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator
   * @param[in] rhs LayerNormalizationLayer to be moved
   */
  LayerNormalizationLayer &operator=(LayerNormalizationLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, const ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return LayerNormalizationLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return true; }

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  inline static const std::string type = "layer_normalization";

private:
  std::vector<unsigned int> normalize_axes; /**< normalize axes */
  std::vector<unsigned int>
    remain_axes; /**< remained axes (exclusive with normalize axes) */

  std::array<unsigned int, 7> wt_idx;
  std::tuple<std::vector<props::Axis>, props::Epsilon,
             props::BNPARAMS_GAMMA_INIT, props::BNPARAMS_BETA_INIT,
             props::WeightDecay, props::BiasDecay>
    layer_normalization_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __LAYER_NORMALIZATION_LAYER_H__ */
