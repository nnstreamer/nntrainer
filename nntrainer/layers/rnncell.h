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
  RNNCellLayer();

  /**
   * @brief     Destructor of RNNCellLayer
   */
  ~RNNCellLayer() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] RNNCellLayer &&
   */
  RNNCellLayer(RNNCellLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs RNNCellLayer to be moved.
   */
  RNNCellLayer &operator=(RNNCellLayer &&rhs) = default;

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
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return RNNCellLayer::type; };

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
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(RunLayerContext &context, unsigned int batch) override;

  inline static const std::string type = "rnncell";

private:
  /**
   * Unit: number of output neurons
   * HiddenStateActivation: activation type for hidden state. default is tanh
   * DropOutRate: dropout rate
   * IntegrateBias: Integrate bias_ih, bias_hh to bias_h
   * MaxTimestep: maximum timestep for rnncell
   * TimeStep: timestep for which rnncell should operate
   *
   * */
  std::tuple<props::Unit, props::HiddenStateActivation, props::DropOutRate,
             props::IntegrateBias, props::MaxTimestep, props::Timestep>
    rnncell_props;
  std::array<unsigned int, 7> wt_idx; /**< indices of the weights */

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
