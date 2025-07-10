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
#include <layer_impl_cl.h>
#include <nntrainer_log.h>

#include <cl_context.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {

/**
 * @class   RMSNormLayer
 * @brief   RMS Norm layer
 */
class RMSNormLayerCl : public LayerImplCl {

public:
  /**
   * @brief     Constructor of RMS Norm Layer
   */
  NNTR_EXPORT RMSNormLayerCl();

  /**
   * @brief     Destructor of RMS Norm Layer
   */
  NNTR_EXPORT ~RMSNormLayerCl() = default;

  /**
   *  @brief  Move constructor.
   *  @param[in] RMSNorm &&
   */
  NNTR_EXPORT RMSNormLayerCl(RMSNormLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs RMS Norm to be moved.
   */
  NNTR_EXPORT RMSNormLayerCl &operator=(RMSNormLayerCl &&rhs) = default;

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  NNTR_EXPORT void finalize(InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  NNTR_EXPORT void forwarding(RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  NNTR_EXPORT void incremental_forwarding(RunLayerContext &context,
                                       unsigned int from, unsigned int to,
                                       bool training) override;

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
    return RMSNormLayerCl::type;
  };

  /**
   * @brief Process data and dimensions for rms norm operation
   * @param[in] input Tensor
   * @param[in] result Tensor
   * @param[in] gamma Tensor
   * @param[in] epsilon float
   */

  NNTR_EXPORT void rmsnormProcess(Tensor const &input, Tensor &result,
                               Tensor const &gamma, const float epsilon);
#ifdef ENABLE_FP16
  /**
   * @brief Process data and dimensions for FP16 rms norm operation
   * @param[in] input Tensor
   * @param[in] result Tensor
   * @param[in] gamma Tensor
   * @param[in] epsilon float
   */

  NNTR_EXPORT void rmsnormProcess_fp16(Tensor const &input, Tensor &result,
                                    Tensor const &gamma, const float epsilon);
#endif
  /**
   * @copydoc Layer::supportBackwarding()
   */
  NNTR_EXPORT bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief registerClKernels
   */
  NNTR_EXPORT static bool registerClKernels(ClContext *global_cl_context);

  static constexpr const char *type = "rmsnorm";

private:
  std::array<unsigned int, 1> wt_idx;

  std::tuple<props::GammaInitializer, props::Epsilon>
    rmsnorm_props; /**< rmsnorm layer properties */

  NNTR_EXPORT static std::vector<ClContext::SharedPtrClKernel> &
  getLayerKernelPtrs();

  enum Kernels {
    RMSNORM_CL,
    RMSNORM_CL_FP16,
  };
};
} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RMSNORM_LAYER_CL__ */
