// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   cosine_layer.h
 * @date   19 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is cosine layer class (operation layer)
 */

#ifndef __COSINE_LAYER_H__
#define __COSINE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Cosine Layer
 * @brief Cosine Layer
 */
class CosineLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Cosine Layer
   */
  NNTR_EXPORT CosineLayer() :
    UnaryOperationLayer(),
    cosine_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Cosine Layer
   */
  NNTR_EXPORT ~CosineLayer(){};

  /**
   *  @brief  Move constructor of Cosine Layer.
   *  @param[in] CosineLayer &&
   */
  NNTR_EXPORT CosineLayer(CosineLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs CosineLayer to be moved.
   */
  NNTR_EXPORT CosineLayer &operator=(CosineLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_EXPORT void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for cosine
   *
   * @param input input tensor
   * @param hidden tensor to store the result value
   */
  NNTR_EXPORT void forwarding_operation(const Tensor &input, Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_EXPORT void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_EXPORT bool supportBackwarding() const final {
    return support_backwarding;
  };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  NNTR_EXPORT InPlaceType initializeInPlace() final {
    is_inplace = !std::get<props::InPlaceProp>(cosine_props).empty() &&
                 std::get<props::InPlaceProp>(cosine_props).get();
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
  NNTR_EXPORT void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_EXPORT const std::string getType() const final {
    return CosineLayer::type;
  };

  std::tuple<props::Print, props::InPlaceProp> cosine_props;
  bool support_backwarding; /**< support backwarding */

  inline static const std::string type = "cos";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __COSINE_LAYER_H__ */
