// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 SeungBaek Hong <sb92.hong@samsung.com>
 *
 * @file   divide_layer.h
 * @date   10 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author SeungBaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is div layer class (operation layer)
 *
 */

#ifndef __DIVIDE_LAYER_H__
#define __DIVIDE_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Divide Layer
 * @brief Divide Layer
 */
class DivideLayer : public BinaryOperationLayer {
public:
  /**
   * @brief Constructor of Divide Layer
   */
  DivideLayer() :
    BinaryOperationLayer(),
    divide_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Divide Layer
   */
  ~DivideLayer(){};

  /**
   *  @brief  Move constructor of Divide Layer.
   *  @param[in] DivideLayer &&
   */
  DivideLayer(DivideLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs DivideLayer to be moved.
   */
  DivideLayer &operator=(DivideLayer &&rhs) = default;

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
  bool supportBackwarding() const final { return support_backwarding; };

  /**
   * @brief Initialize the in-place settings of the layer
   * @return InPlaceType
   */
  InPlaceType initializeInPlace() final {
    if (std::get<props::InPlaceProp>(divide_props).empty() ||
        !std::get<props::InPlaceProp>(divide_props).get()) {
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
  const std::string getType() const final { return DivideLayer::type; };

  std::tuple<props::Print, props::InPlaceProp> divide_props;
  bool support_backwarding; /**< support backwarding */

  inline static const std::string type = "divide";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __DIVIDE_LAYER_H__ */
