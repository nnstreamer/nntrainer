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
  SubtractLayer() : BinaryOperationLayer(), subtract_props(props::Print()) {}

  /**
   * @brief Destructor of Sub Layer
   */
  ~SubtractLayer(){};

  /**
   *  @brief  Move constructor of Sub Layer.
   *  @param[in] SubtractLayer &&
   */
  SubtractLayer(SubtractLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs SubtractLayer to be moved.
   */
  SubtractLayer &operator=(SubtractLayer &&rhs) = default;

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
  const std::string getType() const final { return SubtractLayer::type; };

  std::tuple<props::Print> subtract_props;

  inline static const std::string type = "subtract";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __SUBTRACT_LAYER_H__ */
