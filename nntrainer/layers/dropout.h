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
  NNTR_API DropOutLayer(float dropout = 0.0f) :
    Layer(), dropout_rate(props::DropOutRate(dropout)), epsilon(1e-3f) {}

  /**
   * @brief     Destructor of DropOut Layer
   */
  NNTR_API ~DropOutLayer() = default;

  /**
   *  @brief  Move constructor of DropOutLayer.
   *  @param[in] DropOutLayer &&
   */
  NNTR_API DropOutLayer(DropOutLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs DropOutLayer to be moved.
   */
  NNTR_API DropOutLayer &operator=(DropOutLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_API void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_API void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_API void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_API void
  exportTo(Exporter &exporter,
           const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  NNTR_API const std::string getType() const override {
    return DropOutLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  NNTR_API bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  NNTR_API void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "dropout";

private:
  std::tuple<props::DropOutRate> dropout_rate;
  std::vector<unsigned int> mask_idx;
  float epsilon;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __DROPOUT_H__ */
