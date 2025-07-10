// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024  Niket Agarwal <niket.a@samsung.com>
 *
 * @file   swiglu_cl.h
 * @date   6th June 2024
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __SWIGLU_LAYER_CL_H__
#define __SWIGLU_LAYER_CL_H__

#include <cl_context.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <layer_impl_cl.h>
#include <node_exporter.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <utility>

namespace nntrainer {

/**
 * @brief A SwiGLU layer
 *
 */
class SwiGLULayerCl final : public LayerImplCl {

public:
  /**
   * @brief Construct a new SwiGLU layer object
   *
   */
  NNTR_EXPORT SwiGLULayerCl() : LayerImplCl(), swiglu_props(props::Print()) {}

  /**
   * @brief Destroy the SwiGLU layer object
   *
   */
  NNTR_EXPORT ~SwiGLULayerCl() {}

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
   * @copydoc bool supportBackwarding() const
   */
  NNTR_EXPORT bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  NNTR_EXPORT void
  exportTo(Exporter &exporter,
           const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  NNTR_EXPORT const std::string getType() const override {
    return SwiGLULayerCl::type;
  };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  NNTR_EXPORT void setProperty(const std::vector<std::string> &values) override;

  static constexpr const char *type = "swiglu";

  /**
   * @brief Process data and dimensions for swiglu operation
   * @param[in] input1 Tensor
   * @param[in] input2 Tensor
   * @param[in] result Tensor
   */
  NNTR_EXPORT void swigluProcess(Tensor const &in1, Tensor const &in2,
                              Tensor &result);

  /**
   * @brief     swiglu computation
   * @param[in] matAdata float * for Input Vector A
   * @param[in] vecXdata float * for Input Vector X
   * @param[in] vecYdata float * for Output Vector Y
   * @param[in] dim1 number of elements in input vector A
   * @param[in] dim1 number of elements in input vector X
   */
  NNTR_EXPORT void swiglu_cl(const float *matAdata, const float *vecXdata,
                          float *vecYdata, unsigned int dim1,
                          unsigned int dim2);

#ifdef ENABLE_FP16
  /**
   * @brief     fp16 swiglu computation
   * @param[in] matAdata fp16 * for Input Vector A
   * @param[in] vecXdata fp16 * for Input Vector X
   * @param[in] vecYdata fp16 * for Output Vector Y
   * @param[in] dim1 number of elements in input vector A
   * @param[in] dim1 number of elements in input vector X
   */
  NNTR_EXPORT void swiglu_cl_fp16(const _FP16 *matAdata, const _FP16 *vecXdata,
                               _FP16 *vecYdata, unsigned int dim1,
                               unsigned int dim2);
#endif

  /**
   * @brief     Register OpenCL kernels for SwiGLU layer. This should be called
   */
  NNTR_EXPORT static bool registerClKernels(ClContext *global_cl_context);

private:
  std::tuple<props::Print> swiglu_props; /**< swiglu layer properties : unit -
                                            number of output neurons */

  NNTR_EXPORT static std::vector<ClContext::SharedPtrClKernel> &
  getLayerKernelPtrs();

  enum Kernels { SWIGLU_CL, SWIGLU_CL_FP16 }; /** kernels enum */
};

} // namespace nntrainer

#endif /* __SWIGLU_LAYER_CL_H__ */
