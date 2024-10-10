// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024  Niket Agarwal <niket.a@samsung.com>
 *
 * @file   reshape_cl.h
 * @date   18 June 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Reshape GPU Layer Implementation
 *
 */

#ifndef __RESHAPE_LAYER_CL_H__
#define __RESHAPE_LAYER_CL_H__
#ifdef __cplusplus

#include <cl_context.h>
#include <common_properties.h>
#include <layer_devel.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {
/**
 * @class   Reshape Layer
 * @brief   Reshape Layer
 */
class ReshapeLayerCl : public Layer {

private:
  inline static ClContext cl_context_ref;

public:
  /**
   * @brief     Constructor of Reshape Layer
   */
  ReshapeLayerCl() : Layer() {}

  /**
   * @brief     Destructor of Reshape Layer
   */
  ~ReshapeLayerCl() = default;

  /**
   *  @brief  Move constructor of ReshapeLayer.
   *  @param[in] ReshapeLayer &&
   */
  ReshapeLayerCl(ReshapeLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ReshapeLayer to be moved.
   */
  ReshapeLayerCl &operator=(ReshapeLayerCl &&rhs) = default;

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
   int from, unsigned int to, bool training)
   */
  void incremental_forwarding(RunLayerContext &context, unsigned int from,
                              unsigned int to, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(RunLayerContext &context) override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::supportInPlace()
   */
  bool supportInPlace() const override { return true; }

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ReshapeLayerCl::type; };

  inline static const std::string type = "reshape";

  static opencl::Kernel kernel_copy;
  static opencl::Kernel kernel_copy_fp16;

  /**
   * @brief Process data and dimensions for reshape operation
   * @param[in] input Tensor
   * @param[in] result Tensor
   */
  void ReshapeProcess(Tensor const &input, Tensor &result);

  /**
   * @brief     copy computation
   * @param[in] input float * for Input Tensor
   * @param[in] res float * for Output Tensor
   * @param[in] input_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input_channels   represents the channels of the input tensor
   * @param[in] input_height   represents the height of the input tensor
   * @param[in] input_width   represents the width of the input tensor
   */
  void copy_cl(const float *input, float *res, unsigned int input_batch_size,
               unsigned int input_channels, unsigned int input_height,
               unsigned int input_width);

#ifdef ENABLE_FP16
  /**
   * @brief     copy computation
   * @param[in] input fp16 * for Input Tensor
   * @param[in] res fp16 * for Output Tensor
   * @param[in] input_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input_channels   represents the channels of the input tensor
   * @param[in] input_height   represents the height of the input tensor
   * @param[in] input_width   represents the width of the input tensor
   */
  void copy_cl_fp16(const __fp16 *input, __fp16 *res,
                    unsigned int input_batch_size, unsigned int input_channels,
                    unsigned int input_height, unsigned int input_width);
#endif

protected:
  std::tuple<props::TargetShape>
    reshape_props; /**< reshape properties : target_shape after reshape */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __RESHAPE_LAYER_CL_H__ */
