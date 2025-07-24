// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   sine_layer.h
 * @date   19 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is sine layer class (operation layer)
 *
 */

#ifndef __SINE_LAYER_H__
#define __SINE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Sine Layer
 * @brief Sine Layer
 */
class SineLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Sine Layer
   */
  SineLayer() :
    UnaryOperationLayer(),
    sine_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Sine Layer
   */
  ~SineLayer(){};

  /**
   *  @brief  Move constructor of Sine Layer.
   *  @param[in] SineLayer &&
   */
  SineLayer(SineLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SineLayer to be moved.
   */
  SineLayer &operator=(SineLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for sine
   *
   * @param input input tensor
   * @param hidden tensor to store the result value
   */
  void forwarding_operation(const Tensor &input, Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const final { return support_backwarding; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    auto inplace_prop = std::get<props::InPlaceProp>(sine_props);
    is_inplace = !inplace_prop.empty() && inplace_prop.get();
    support_backwarding = !is_inplace;
    return supportInPlace() ? InPlaceType::NON_RESTRICTING : InPlaceType::NONE;
  }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const final { return SineLayer::type; };

  std::tuple<props::Print, props::InPlaceProp> sine_props;
  bool support_backwarding; /**< support backwarding */

  inline static const std::string type = "sin";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SINE_LAYER_H__ */
