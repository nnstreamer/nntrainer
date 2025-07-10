// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 hyeonseok lee <hs89.lee@samsung.com>
 *
 * @file   rnncell.h
 * @date   29 Oct 2021
 * @brief  This is Recurrent Layer Cell Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author hyeonseok lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __RNNCELL_H__
#define __RNNCELL_H__
#ifdef __cplusplus

#include <acti_func.h>
#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

/**
 * @class   RNNCellLayer
 * @brief   RNNCellLayer
 */
class RNNCellLayer : public LayerImpl {
public:
  /**
   * @brief     Constructor of RNNCellLayer
   */
  NNTR_EXPORT RNNCellLayer();

  /**
   * @brief     Destructor of RNNCellLayer
   */
  NNTR_EXPORT ~RNNCellLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] RNNCellLayer &&
   */
  NNTR_EXPORT RNNCellLayer(RNNCellLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs RNNCellLayer to be moved.
   */
  NNTR_EXPORT RNNCellLayer &operator=(RNNCellLayer &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_EXPORT void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_EXPORT void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  NNTR_EXPORT void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  NNTR_EXPORT void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  NNTR_EXPORT void exportTo(Exporter &exporter,
                         const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  NNTR_EXPORT const std::string getType() const override {
    return RNNCellLayer::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  NNTR_EXPORT bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  NNTR_EXPORT void setBatch(RunLayerContext &context, unsigned int batch) override;

  static constexpr const char *type = "rnncell";

private:
  enum INOUT_INDEX {
    INPUT = 0,
    INPUT_HIDDEN_STATE = 1,
    OUTPUT_HIDDEN_STATE = 0,
  };

  /**
   * Unit: number of output neurons
   * IntegrateBias: Integrate bias_ih, bias_hh to bias_h
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * DropOutRate: dropout rate
   *
   * */
  std::tuple<props::Unit, props::IntegrateBias, props::HiddenStateActivation,
             props::DropOutRate>
    rnncell_props;
  std::array<unsigned int, 6> wt_idx; /**< indices of the weights */

  /**
   * @brief     activation function for h_t : default is tanh
   */
  ActiFunc acti_func;

  /**
   * @brief     to pretect overflow
   */
  float epsilon;
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RNNCELL_H__ */
