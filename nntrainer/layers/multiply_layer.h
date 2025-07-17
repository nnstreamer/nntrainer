// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   multiply_layer.h
 * @date   10 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is mul layer class (operation layer)
 *
 */

#ifndef __MULTIPLY_LAYER_H__
#define __MULTIPLY_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Multiply Layer
 * @brief Multiply Layer
 */
class MultiplyLayer : public BinaryOperationLayer {
public:
  /**
   * @brief Constructor of Multiply Layer
   */
  NNTR_API MultiplyLayer() :
    BinaryOperationLayer(),
    multiply_props(props::Print(), props::InPlaceProp(),
                   props::InPlaceDirectionProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Multiply Layer
   */
  NNTR_API ~MultiplyLayer(){};

  /**
   *  @brief  Move constructor of Multiply Layer.
   *  @param[in] MultiplyLayer &&
   */
  NNTR_API MultiplyLayer(MultiplyLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs MultiplyLayer to be moved.
   */
  NNTR_API MultiplyLayer &operator=(MultiplyLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for add
   *
   * @param input0 input tensor 0
   * @param input1 input tensor 1
   * @param hidden tensor to store the result of addition
   */
  NNTR_API void forwarding_operation(const Tensor &input0, const Tensor &input1,
                                     Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding()
   */
  NNTR_API bool supportBackwarding() const final {
    return support_backwarding;
  };

  /**
   * @brief Get the inplace direction for the tensor operation layer
   *
   * @return InPlaceDirection
   */
  NNTR_API InPlaceDirection getInPlaceDirection() override {
    if (!supportInPlace())
      return InPlaceDirection::NONE;
    if (std::get<props::InPlaceDirectionProp>(multiply_props).empty() ||
        (std::get<props::InPlaceDirectionProp>(multiply_props).get() ==
         "left")) {
      return InPlaceDirection::LEFT;
    } else {
      return InPlaceDirection::RIGHT;
    }
  };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  NNTR_API InPlaceType initializeInPlace() final {
    if (std::get<props::InPlaceProp>(multiply_props).empty() ||
        !std::get<props::InPlaceProp>(multiply_props).get()) {
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
  NNTR_API const std::string getType() const final {
    return MultiplyLayer::type;
  };

  std::tuple<props::Print, props::InPlaceProp, props::InPlaceDirectionProp>
    multiply_props;
  bool support_backwarding; /**< support backwarding */

  static constexpr const char *type = "multiply";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __MULTIPLY_LAYER_H__ */
