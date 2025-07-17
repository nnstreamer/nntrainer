// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   sqrt_layer.h
 * @date   18 March 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is sqrt layer class (operation layer)
 *
 */

#ifndef __SQRT_LAYER_H__
#define __SQRT_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class SQRT Layer
 * @brief SQRT Layer
 */
class SQRTLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of SQRT Layer
   */
  NNTR_API SQRTLayer() :
    UnaryOperationLayer(),
    sqrt_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of SQRT Layer
   */
  NNTR_API ~SQRTLayer(){};

  /**
   *  @brief  Move constructor of SQRT Layer.
   *  @param[in] SQRTLayer &&
   */
  NNTR_API SQRTLayer(SQRTLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SQRTLayer to be moved.
   */
  NNTR_API SQRTLayer &operator=(SQRTLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for SQRT
   *
   * @param input input tensor
   * @param hidden tensor to store the result value
   */
  NNTR_API void forwarding_operation(const Tensor &input, Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_API bool supportBackwarding() const final {
    return support_backwarding;
  };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  NNTR_API InPlaceType initializeInPlace() final {
    if (std::get<props::InPlaceProp>(sqrt_props).empty() ||
        !std::get<props::InPlaceProp>(sqrt_props).get()) {
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
  NNTR_API void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const final { return SQRTLayer::type; };

  std::tuple<props::Print, props::InPlaceProp> sqrt_props;
  bool support_backwarding; /**< support backwarding */

  inline static const std::string type = "sqrt";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SQRT_LAYER_H__ */
