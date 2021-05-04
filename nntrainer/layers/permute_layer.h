
// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   permute_layer.h
 * @date   06 May 2021
 * @brief  Permute layer to support transpose
 * @see	   https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#ifndef __PERMUTE_LAYER_H__
#define __PERMUTE_LAYER_H__

#include <array>

#include <base_properties.h>
#include <layer_internal.h>
#include <node_exporter.h>
namespace nntrainer {
/**
 * @class   PermuteLayer
 * @brief   Permute layer to transpose a tensor
 */
class PermuteLayer : public Layer {
public:
  /**
   * @brief     Constructor of Permute Layer
   * @param     direction direction to permute
   */
  template <typename... Args> PermuteLayer(Args... args) : Layer(args...) {}

  /**
   * @brief     Destructor of Permute Layer
   */
  ~PermuteLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] PermuteLayer &&
   */
  PermuteLayer(PermuteLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs PermuteLayer to be moved.
   */
  PermuteLayer &operator=(PermuteLayer &&rhs) = default;

  /**
   * @copydoc Layer::forwarding(bool training)
   */
  void forwarding(bool training = true) override;

  /**
   * @copydoc Layer::calcDerivative()
   */
  void calcDerivative() override;

  /**
   * @brief     copy layer
   * @param[in] l layer to copy
   */
  void copy(std::shared_ptr<Layer> l) override;

  /**
   * @brief     initialize layer
   * @retval #ML_ERROR_NONE Successful.
   * @retval #ML_ERROR_INVALID_PARAMETER invalid parameter.
   */
  int initialize(Manager &manager) override;

  /**
   * @copydoc Layer::export_to(Exporter &exporter, ExportMethods method)
   */
  void export_to(
    Exporter &exporter,
    ExportMethods method = ExportMethods::METHOD_STRINGVECTOR) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return PermuteLayer::type; };

  static const std::string type;

  using Layer::setProperty;

  /**
   * @copydoc Layer::setProperty(std::vector<std::string> values);
   */
  int setProperty(std::vector<std::string> values) override;
};

} // namespace nntrainer
#endif // __PERMUTE_LAYER_H__
