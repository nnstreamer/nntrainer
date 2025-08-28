// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sumon Nath <sumon.nath@samsung.com>
 *
 * @file   reduce_sum_layer.h
 * @date   29 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sumon Nath <sumon.nath@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Reduce Sum Layer Class for Neural Network
 *
 */

#ifndef __REDUCE_SUM_LAYER_H__
#define __REDUCE_SUM_LAYER_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   Reduce Sum Layer
 * @brief   Reduce Sum Layer
 */
class ReduceSumLayer : public Layer {
public:
  /**
   * @brief     Constructor of Reduce Sum Layer
   */
  ReduceSumLayer() : Layer() {}

  /**
   * @brief     Destructor of Reduce Sum Layer
   */
  ~ReduceSumLayer(){};

  /**
   *  @brief  Move constructor of ReduceSumLayer.
   *  @param[in] ReduceSumLayer &&
   */
  ReduceSumLayer(ReduceSumLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ReduceSumLayer to be moved.
   */
  ReduceSumLayer &operator=(ReduceSumLayer &&rhs) = default;

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ReduceSumLayer::type; };

  static constexpr const char *type = "reduce_sum";

private:
  std::tuple<props::ReduceDimension>
    reduce_sum_props; /**< reduce_sum properties : axis to reduce along */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __REDUCE_SUM_LAYER_H__ */
