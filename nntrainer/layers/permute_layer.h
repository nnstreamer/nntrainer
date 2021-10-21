
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
#include <string>

#include <base_properties.h>
#include <layer_devel.h>
#include <node_exporter.h>
namespace nntrainer {

namespace props {
/**
 * @brief PermuteDims property, direction property describes the axis to be
 * transposed. to be used with array
 *
 */
class PermuteDims : public nntrainer::Property<unsigned int> {
public:
  static constexpr const char *key = "direction"; /**< unique key to access */
  using prop_tag = uint_prop_tag;                 /**< property type */

  /**
   * @brief check if given value is valid
   *
   * @return true if valid
   * @return false if not valid
   */
  bool isValid(const unsigned int &) const override;
};
} // namespace props

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
  PermuteLayer() : Layer(), direction(), reverse_direction() {}

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
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return PermuteLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "permute";

private:
  std::string
    direction_str; /**< transpose representation, @todo deprecate this */
  std::string rdirection_str; /**< transpose representation, @todo
                                        deprecate this */

  std::array<props::PermuteDims, 3> direction;
  std::array<props::PermuteDims, 3> reverse_direction;
};

} // namespace nntrainer
#endif // __PERMUTE_LAYER_H__
