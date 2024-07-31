// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   transpose_cl.h
 * @date   31 July 2024
 * @brief  Implementation of transpose layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __TRANSPOSE_LAYER_CL_H__
#define __TRANSPOSE_LAYER_CL_H__

#include <common_properties.h>
#include <layer_devel.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {

/**
 * @brief A tranpose layer.
 *
 */
class TransposeLayerCl final : public Layer {
public:
  /**
   * @brief Construct a new transpose layer object
   *
   */
  TransposeLayerCl() : Layer(), transpose_props(props::Print()) {}

  /**
   * @brief Destroy the transpose layer object
   *
   */
  ~TransposeLayerCl() {}

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
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TransposeLayerCl::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "transpose";

  static opencl::Kernel kernel_transpose;
  static opencl::Kernel kernel_transpose_fp16;

  std::tuple<props::Print> transpose_props; /**< transpose layer properties :
                                            unit - number of output neurons */

  /**
   * @brief Process data and dimensions for transpose operation
   * @param[in] input Tensor
   * @param[in] result Tensor
   * @param[in] RunLayerContext reference
   */
  void TransposeProcess(Tensor const &in, Tensor &result,
                        RunLayerContext &context);

  /**
   * @brief     transpose computation
   * @param[in] input float * for Input Tensor
   * @param[in] res float * for Output Tensor
   * @param[in] input_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input_channels   represents the channels of the input tensor
   * @param[in] input_height   represents the height of the input tensor
   * @param[in] input_width   represents the width of the input tensor
   * @param[in] context RunLayerContext reference
   */
  void transpose_cl(const float *in, float *res, unsigned int input_batch_size,
                    unsigned int input_channels, unsigned int input_height,
                    unsigned int input_width, RunLayerContext &context);

  /**
   * @brief     transpose computation
   * @param[in] input fp16 * for Input Tensor
   * @param[in] res fp16 * for Output Tensor
   * @param[in] input_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input_channels   represents the channels of the input tensor
   * @param[in] input_height   represents the height of the input tensor
   * @param[in] input_width   represents the width of the input tensor
   * @param[in] context RunLayerContext reference
   */
  void transpose_cl_fp16(const __fp16 *in, __fp16 *res,
                         unsigned int input_batch_size,
                         unsigned int input_channels, unsigned int input_height,
                         unsigned int input_width, RunLayerContext &context);
};
} // namespace nntrainer

#endif /* __TRANSPOSE_LAYER_CL_H__ */
