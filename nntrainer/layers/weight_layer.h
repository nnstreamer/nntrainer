// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   weight_layer.h
 * @date   2 August 2024
 * @brief  This is a layer that simply stores a weight tensor without any
 * operation.
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __WEIGHT_LAYER_H__
#define __WEIGHT_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   Weight Layer
 * @brief   A layer that simply stores a weight tensor
 */
class WeightLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of Weight Layer
   */
  WeightLayer();

  /**
   * @brief     Destructor of Weight Layer
   */
  ~WeightLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] WeightLayer &&
   */
  WeightLayer(WeightLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs WeightLayer to be moved.
   */
  WeightLayer &operator=(WeightLayer &&rhs) = default;

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return WeightLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    is_inplace = true;
    return InPlaceType::NON_RESTRICTING;
  }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "weight";

private:
  std::tuple<std::vector<props::TensorDimension>,
             std::vector<props::TensorDataType>, std::vector<props::WeightName>>
    weight_props;

  std::vector<unsigned int> weight_idx; /**< indices of the weights */
  unsigned int n_weight;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __WEIGHT_LAYER_H__ */
