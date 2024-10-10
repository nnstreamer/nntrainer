// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024  Niket Agarwal <niket.a@samsung.com>
 *
 * @file   swiglu_cl.h
 * @date   6th June 2024
 * @brief  Implementation of SwiGLU activation function
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __SWIGLU_LAYER_CL_H__
#define __SWIGLU_LAYER_CL_H__

#include <cl_context.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <layer_impl.h>
#include <node_exporter.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <utility>

namespace nntrainer {

/**
 * @brief A SwiGLU layer
 *
 */
class SwiGLULayerCl final : public Layer {

private:
  inline static ClContext cl_context_ref;

public:
  /**
   * @brief Construct a new SwiGLU layer object
   *
   */
  SwiGLULayerCl() : Layer(), swiglu_props(props::Print()) {}

  /**
   * @brief Destroy the SwiGLU layer object
   *
   */
  ~SwiGLULayerCl() {}

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
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return SwiGLULayerCl::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "swiglu";

  static opencl::Kernel kernel_swiglu;
  static opencl::Kernel kernel_swiglu_fp16;

  std::tuple<props::Print> swiglu_props; /**< swiglu layer properties : unit -
                                            number of output neurons */

  /**
   * @brief Process data and dimensions for swiglu operation
   * @param[in] input1 Tensor
   * @param[in] input2 Tensor
   * @param[in] result Tensor
   */
  void swigluProcess(Tensor const &in1, Tensor const &in2, Tensor &result);

  /**
   * @brief     swiglu computation
   * @param[in] matAdata float * for Input Vector A
   * @param[in] vecXdata float * for Input Vector X
   * @param[in] vecYdata float * for Output Vector Y
   * @param[in] dim1 number of elements in input vector A
   * @param[in] dim1 number of elements in input vector X
   */
  void swiglu_cl(const float *matAdata, const float *vecXdata, float *vecYdata,
                 unsigned int dim1, unsigned int dim2);

#ifdef ENABLE_FP16
  /**
   * @brief     fp16 swiglu computation
   * @param[in] matAdata fp16 * for Input Vector A
   * @param[in] vecXdata fp16 * for Input Vector X
   * @param[in] vecYdata fp16 * for Output Vector Y
   * @param[in] dim1 number of elements in input vector A
   * @param[in] dim1 number of elements in input vector X
   */
  void swiglu_cl_fp16(const __fp16 *matAdata, const __fp16 *vecXdata,
                      __fp16 *vecYdata, unsigned int dim1, unsigned int dim2);
#endif
};

} // namespace nntrainer

#endif /* __SWIGLU_LAYER_CL_H__ */
