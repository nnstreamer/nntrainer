// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file   neg_layer.h
 * @date   3 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is neg layer class (operation layer)
 */

#ifndef __NEG_LAYER_H__
#define __NEG_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <operation_layer.h>

namespace nntrainer {

/**
 * @class Neg Layer
 * @brief Neg Layer
 */
class NegLayer : public UnaryOperationLayer {
public:
  /**
   * @brief Constructor of Neg Layer
   */
  NegLayer() :
    UnaryOperationLayer(),
    neg_props(props::Print(), props::InPlaceProp()),
    support_backwarding(true) {}

  /**
   * @brief Destructor of Neg Layer
   */
  ~NegLayer(){};

  /**
   *  @brief  Move constructor of Neg Layer.
   *  @param[in] NegLayer &&
   */
  NegLayer(NegLayer &&rhs) noexcept = default;

  /**
   * @brief Move assignment operator.
   * @parma[in] rhs NegLayer to be moved.
   */
  NegLayer &operator=(NegLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) final;

  /**
   * @brief forwarding operation for neg
   *
   * @param input first input tensor
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
    is_inplace = !std::get<props::InPlaceProp>(neg_props).empty() &&
                 std::get<props::InPlaceProp>(neg_props).get();
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
  const std::string getType() const final { return NegLayer::type; };

  std::tuple<props::Print, props::InPlaceProp> neg_props;
  bool support_backwarding;

  inline static const std::string type = "neg";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __NEG_LAYER_H__ */
