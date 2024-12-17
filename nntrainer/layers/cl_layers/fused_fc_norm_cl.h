// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Debadri Samaddar <s.debadri@samsung.com>
 *
 * @file   fused_fc_norm_cl.h
 * @date   7 May 2024
 * @brief  This is Fully Connected Layer Class of Neural Network with OpenCl
 * implementation
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __FUSED_FC_RMS_LAYER_CL_H__
#define __FUSED_FC_RMS_LAYER_CL_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>

namespace nntrainer {

namespace props {

/**
 * @brief FUSED_FC_RMS_NORM_GAMMA_INIT_GPU Initialization Enumeration
 * Information
 *
 */
class FUSED_FC_RMS_NORM_GAMMA_INIT_GPU final
  : public ::nntrainer::EnumProperty<::nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a RMS_NORM_GAMMA_INIT object
   */
  FUSED_FC_RMS_NORM_GAMMA_INIT_GPU(
    ::nntrainer::Initializer value = ::nntrainer::Initializer::ONES) {
    set(value);
  };
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};
}; // namespace props

/**
 * @class   Fused Fully Connected Layer with RMS Normalization Layer Class for
 * @brief   Fused Fully Connected Layer with RMS Normalization Layer Class for
 */
class FullyConnectedRMSNormLayerCl : public LayerImpl {
public:
  /**
   * @brief     Constructor of Fused Fully Connected && RMS Norm Layer
   */
  FullyConnectedRMSNormLayerCl();

  /**
   * @brief     Destructor of Fused Fully Connected && RMS Norm Layer
   */
  ~FullyConnectedRMSNormLayerCl() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] FullyConnectedRMSNorm &&
   */
  FullyConnectedRMSNormLayerCl(FullyConnectedRMSNormLayerCl &&rhs) noexcept =
    default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs FullyConnectedLayer to be moved.
   */
  FullyConnectedRMSNormLayerCl &
  operator=(FullyConnectedRMSNormLayerCl &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
￼   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
￼   * int from, unsigned int to, bool training)
￼   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  void calcGradient(RunLayerContext &context) override;

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override {
    return FullyConnectedRMSNormLayerCl::type;
  };

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return true; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "fully_connected_rmsNorm";

private:
  std::tuple<props::Unit, props::FUSED_FC_RMS_NORM_GAMMA_INIT_GPU,
             props::Epsilon>
    fc_rms_props; /**< fc layer properties : unit - number of output neurons */
  std::array<unsigned int, 3>
    weight_idx; /**< indices of the weights for FC layer */
  // std::array<unsigned int, 1> wt_idx; /**< indices of the weights for RMS
  // layer */ std::tuple<props::RMS_NORM_GAMMA_INIT_GPU, props::Epsilon>
  //   rmsnorm_props; /**< rmsnorm layer properties */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __FUSED_FC_RMS_LAYER_CL_H__ */
