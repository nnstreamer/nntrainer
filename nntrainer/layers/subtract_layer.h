// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   subtract_layer.h
 * @date   10 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is subtract layer class (operation layer)
 *
 */

#ifndef __SUBTRACT_LAYER_H__
#define __SUBTRACT_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Subtract Layer
 * @brief Subtract Layer
 */
class SubtractLayer : public BinaryOperationLayer {
public:
  /**
   * @brief Constructor of Subtract Layer
   */
  NNTR_EXPORT SubtractLayer() :
    BinaryOperationLayer(),
    subtract_props(props::Print(), props::InPlaceProp(),
                   props::InPlaceDirectionProp()) {}

  /**
   * @brief Destructor of Sub Layer
   */
  NNTR_EXPORT ~SubtractLayer(){};

  /**
   *  @brief  Move constructor of Sub Layer.
   *  @param[in] SubtractLayer &&
   */
  NNTR_EXPORT SubtractLayer(SubtractLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SubtractLayer to be moved.
   */
  NNTR_EXPORT SubtractLayer &operator=(SubtractLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_EXPORT void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for add
   *
   * @param input0 input tensor 0
   * @param input1 input tensor 1
   * @param hidden tensor to store the result of addition
   */
  NNTR_EXPORT void forwarding_operation(const Tensor &input0, const Tensor &input1,
                                     Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_EXPORT void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  NNTR_EXPORT bool supportBackwarding() const final { return true; };

  /**
   * @brief Get the inplace direction for the tensor operation layer
   *
   * @return InPlaceDirection
   */
  NNTR_EXPORT InPlaceDirection getInPlaceDirection() override {
    if (!supportInPlace())
      return InPlaceDirection::NONE;
    if (std::get<props::InPlaceDirectionProp>(subtract_props).empty() ||
        (std::get<props::InPlaceDirectionProp>(subtract_props).get() ==
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
  NNTR_EXPORT InPlaceType initializeInPlace() final {
    if (std::get<props::InPlaceProp>(subtract_props).empty() ||
        std::get<props::InPlaceProp>(subtract_props).get()) {
      is_inplace = true;
    } else {
      is_inplace = false;
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
    return SubtractLayer::type;
  };

  std::tuple<props::Print, props::InPlaceProp, props::InPlaceDirectionProp>
    subtract_props;

  static constexpr const char *type = "subtract";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SUBTRACT_LAYER_H__ */
