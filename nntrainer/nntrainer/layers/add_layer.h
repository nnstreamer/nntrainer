// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   add_layer.h
 * @date   7 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is add layer class (operation layer)
 *
 */

#ifndef __ADD_LAYER_H__
#define __ADD_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Add Layer
 * @brief Add Layer
 */
class AddLayer : public BinaryOperationLayer {
public:
  /**
   * @brief Destructor of Add Layer
   */
  ~AddLayer() {}

  /**
   * @brief Constructor of Add Layer
   */
  AddLayer() :
    BinaryOperationLayer(),
    add_props(props::Print(), props::InPlaceProp(),
              props::InPlaceDirectionProp()) {}

  /**
   *  @brief  Move constructor of Add Layer.
   *  @param[in] AddLayer &&
   */
  AddLayer(AddLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs AddLayer to be moved.
   */
  AddLayer &operator=(AddLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for add
   *
   * @param input0 input tensor 0
   * @param input1 input tensor 1
   * @param hidden tensor to store the result of addition
   */
  void forwarding_operation(const Tensor &input0, const Tensor &input1,
                            Tensor &hidden) final;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) final;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const final { return true; };

  /**
   * @brief Get the inplace direction for the tensor operation layer
   *
   * @return InPlaceDirection
   */
  InPlaceDirection getInPlaceDirection() override {
    if (!supportInPlace())
      return InPlaceDirection::NONE;
    if (std::get<props::InPlaceDirectionProp>(add_props).empty() ||
        (std::get<props::InPlaceDirectionProp>(add_props).get() == "left")) {
      return InPlaceDirection::LEFT;
    } else {
      return InPlaceDirection::RIGHT;
    }
  };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    if (std::get<props::InPlaceProp>(add_props).empty() ||
        std::get<props::InPlaceProp>(add_props).get()) {
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
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const final {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) final;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const final { return AddLayer::type; }

  std::tuple<props::Print, props::InPlaceProp, props::InPlaceDirectionProp>
    add_props;

  static constexpr const char *type = "add";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ADD_LAYER_H__ */
