// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020
 *
 * @file   rmsnorm_layer.h
 * @date   8 June 2024
 * @brief  This is RMS Norm Layer Class of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Thummala Pallavi <t.pallavi@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __RMSNORM_LAYER_CL_H__
#define __RMSNORM_LAYER_CL_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_impl.h>
#include <nntrainer_log.h>

#include <cl_context.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {

namespace props {

/**
 * @brief RMS_NORM_GAMMA_INIT_GPU Initialization Enumeration Information
 *
 */
class RMS_NORM_GAMMA_INIT_GPU final
  : public ::nntrainer::EnumProperty<::nntrainer::props::InitializerInfo> {
public:
  /**
   * @brief Construct a RMS_NORM_GAMMA_INIT object
   */
  RMS_NORM_GAMMA_INIT_GPU(
    ::nntrainer::Initializer value = ::nntrainer::Initializer::ONES) {
    set(value);
  };
  using prop_tag = enum_class_prop_tag;
  static constexpr const char *key = "gamma_initializer";
};
}; // namespace props

/**
 * @class   RMSNormLayer
 * @brief   RMS Norm layer
 */

class RMSNormLayerCl : public LayerImpl {

private:
  inline static ClContext cl_context_ref;

public:
  /**
   * @brief     Constructor of RMS Norm Layer
   */
  RMSNormLayerCl();

  /**
   * @brief     Destructor of RMS Norm Layer
   */
  ~RMSNormLayerCl() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] RMSNorm &&
   */
  RMSNormLayerCl(RMSNormLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs RMS Norm to be moved.
   */
  RMSNormLayerCl &operator=(RMSNormLayerCl &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
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
  const std::string getType() const override { return RMSNormLayerCl::type; };

  static opencl::Kernel kernel_rmsnorm;
  static opencl::Kernel kernel_rmsnorm_fp16;

  /**
   * @brief Process data and dimensions for rms norm operation
   * @param[in] input Tensor
   * @param[in] result Tensor
   * @param[in] gamma Tensor
   * @param[in] epsilon float
   */

  void rmsnormProcess(Tensor const &input, Tensor &result, Tensor const &gamma,
                      const float epsilon);
#ifdef ENABLE_FP16
  /**
   * @brief Process data and dimensions for FP16 rms norm operation
   * @param[in] input Tensor
   * @param[in] result Tensor
   * @param[in] gamma Tensor
   * @param[in] epsilon float
   */

  void rmsnormProcess_fp16(Tensor const &input, Tensor &result,
                           Tensor const &gamma, const float epsilon);
#endif
  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "rmsnorm";

private:
  std::array<unsigned int, 1> wt_idx;
  std::tuple<props::RMS_NORM_GAMMA_INIT_GPU, props::Epsilon>
    rmsnorm_props; /**< rmsnorm layer properties */
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RMSNORM_LAYER_CL__ */
