// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   dropout.h
 * @date   05 July 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is DropOut Layer Class for Neural Network
 *
 */

#ifndef __DROPOUT_H__
#define __DROPOUT_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>

namespace nntrainer {

/**
 * @class   DropOut Layer
 * @brief   DropOut Layer
 */
class DropOutLayer : public Layer {
public:
  /**
   * @brief     Constructor of DropOut Layer
   */
  DropOutLayer(float dropout = 0.0) :
    Layer(),
    dropout_rate(props::DropOutRate(dropout)),
    epsilon(1e-3) {}

  /**
   * @brief     Destructor of DropOut Layer
   */
  ~DropOutLayer() = default;

  /**
   *  @brief  Move constructor of DropOutLayer.
   *  @param[in] DropOutLayer &&
   */
  DropOutLayer(DropOutLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs DropOutLayer to be moved.
   */
  DropOutLayer &operator=(DropOutLayer &&rhs) = default;

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
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return DropOutLayer::type; };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::supportInPlace()
   *
   * @todo Enable in-place support once supported by manager
   */
  bool supportInPlace() const override { return false; }

  inline static const std::string type = "dropout";

private:
  std::tuple<props::DropOutRate> dropout_rate;
  std::vector<unsigned int> mask_idx;
  float epsilon;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __DROPOUT_H__ */
