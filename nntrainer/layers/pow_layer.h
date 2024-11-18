// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   pow_layer.h
 * @date   20 Nov 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is pow layer class (operation layer)
 *
 */

#ifndef __POW_LAYER_H__
#define __POW_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Pow Layer
 * @brief Pow Layer
 */
class PowLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Pow Layer
   */
  PowLayer() :
    UnaryOperationLayer(),
    pow_props(props::Print(), props::InPlaceProp(), props::Exponent()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Pow Layer
   */
  ~PowLayer(){};

  /**
   *  @brief  Move constructor of Pow Layer.
   *  @param[in] PowLayer &&
   */
  PowLayer(PowLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs PowLayer to be moved.
   */
  PowLayer &operator=(PowLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for pow
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
    if (std::get<props::InPlaceProp>(pow_props).empty() ||
        !std::get<props::InPlaceProp>(pow_props).get()) {
      is_inplace = false;
      support_backwarding = true;
    } else {
      is_inplace = true;
      support_backwarding = false;
    }

    if (!supportInPlace())
      return InPlaceType::NONE;
    else
      return InPlaceType::NON_RESTRICTING;
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
  const std::string getType() const final { return PowLayer::type; };

  std::tuple<props::Print, props::InPlaceProp, props::Exponent> pow_props;
  bool support_backwarding; /**< support backwarding */

  inline static const std::string type = "pow";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __POW_LAYER_H__ */
