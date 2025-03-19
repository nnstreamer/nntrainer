// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   tangent_layer.h
 * @date   19 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is tangent layer class (operation layer)
 */

#ifndef __TANGENT_LAYER_H__
#define __TANGENT_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Tangent Layer
 * @brief Tangent Layer
 */
class TangentLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Tangent Layer
   */
  TangentLayer() :
    UnaryOperationLayer(),
    tangent_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Tangent Layer
   */
  ~TangentLayer(){};

  /**
   *  @brief  Move constructor of Tangent Layer.
   *  @param[in] TangentLayer &&
   */
  TangentLayer(TangentLayer &&rhs) noexcept = default;

  /**
   * @brief Move assignment operator.
   * @parma[in] rhs TangentLayer to be moved.
   */
  TangentLayer &operator=(TangentLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for tangent
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
    is_inplace = !std::get<props::InPlaceProp>(tangent_props).empty() &&
                 std::get<props::InPlaceProp>(tangent_props).get();
    support_backwarding = !is_inplace;

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
  const std::string getType() const final { return TangentLayer::type; };

  std::tuple<props::Print, props::InPlaceProp> tangent_props;
  bool support_backwarding; /**< support backwarding */

  inline static const std::string type = "tan";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __TANGENT_LAYER_H__ */
