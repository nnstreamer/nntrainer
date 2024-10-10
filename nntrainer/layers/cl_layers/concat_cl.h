// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024  Niket Agarwal <niket.a@samsung.com>
 *
 * @file   concat_cl.h
 * @date   2 July 2024
 * @brief  Implementation of Concat Layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __CONCAT_LAYER_CL_H__
#define __CONCAT_LAYER_CL_H__
#ifdef __cplusplus

#include <cl_context.h>
#include <common_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <layer_impl.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>
#include <tensor_dim.h>
#include <utility>

namespace nntrainer {

/**
 * @class   Concat Layer
 * @brief   Concat Layer
 */
class ConcatLayerCl : public Layer {

private:
  inline static ClContext cl_context_ref;

public:
  /**
   * @brief     Constructor of Concat Layer
   */
  ConcatLayerCl();

  /**
   * @brief     Destructor of Concat Layer
   */
  ~ConcatLayerCl() = default;

  /**
   *  @brief  Move constructor of ConcatLayer.
   *  @param[in] ConcatLayer &&
   */
  ConcatLayerCl(ConcatLayerCl &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs ConcatLayer to be moved.
   */
  ConcatLayerCl &operator=(ConcatLayerCl &&rhs) = default;

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
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return ConcatLayerCl::type; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::supportBackwarding()
   */
  bool supportBackwarding() const override { return false; }

  /**
   * @copydoc Layer::setProperty(const PropertyType type, const std::string
   * &value)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "concat";

  static opencl::Kernel kernel_concat_axis3;
  static opencl::Kernel kernel_concat_axis3_fp16;
  static opencl::Kernel kernel_concat_axis2;
  static opencl::Kernel kernel_concat_axis2_fp16;
  static opencl::Kernel kernel_concat_axis1;
  static opencl::Kernel kernel_concat_axis1_fp16;

  /**
   * @brief Process data and dimensions for concat
   * @param[in] input1 Tensor
   * @param[in] input2 Tensor
   * @param[in] result Tensor
   */
  void ConcatProcess(Tensor const &in1, Tensor const &in2, Tensor &result);

  /**
   * @brief     concat computation for axis 3
   * @param[in] matAdata float * for Input Tensor A
   * @param[in] vecXdata float * for Input Tensor X
   * @param[in] vecYdata float * for Output Tensor Y
   * @param[in] input1_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input1_channels   represents the channels of the input tensor
   * @param[in] input1_height   represents the height of the input tensor
   * @param[in] input1_width   represents the width of the input tensor A
   * @param[in] input2_width   represents the width of the input tensor X
   */
  void concat_cl_axis3(const float *matAdata, const float *vecXdata,
                       float *vecYdata, unsigned int input1_batch_size,
                       unsigned int input1_channels, unsigned int input1_height,
                       unsigned int input1_width, unsigned int input2_width);

  /**
   * @brief     concat computation for axis 2
   * @param[in] matAdata float * for Input Tensor A
   * @param[in] vecXdata float * for Input Tensor X
   * @param[in] vecYdata float * for Output Tensor Y
   * @param[in] input1_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input1_channels   represents the channels of the input tensor
   * @param[in] input1_width   represents the width of the input tensor
   * @param[in] input1_height   represents the height of the input tensor A
   * @param[in] input2_height   represents the height of the input tensor X
   */
  void concat_cl_axis2(const float *matAdata, const float *vecXdata,
                       float *vecYdata, unsigned int input1_batch_size,
                       unsigned int input1_channels, unsigned int input1_width,
                       unsigned int input1_height, unsigned int input2_height);

  /**
   * @brief     concat computation for axis 1
   * @param[in] matAdata float * for Input Tensor A
   * @param[in] vecXdata float * for Input Tensor X
   * @param[in] vecYdata float * for Output Tensor Y
   * @param[in] input1_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input1_height   represents the height of the input tensor
   * @param[in] input1_width   represents the width of the input tensor
   * @param[in] input1_channels   represents the channels of the input tensor A
   * @param[in] input2_channels   represents the channels of the input tensor X
   */
  void concat_cl_axis1(const float *matAdata, const float *vecXdata,
                       float *vecYdata, unsigned int input1_batch_size,
                       unsigned int input1_height, unsigned int input1_width,
                       unsigned int input1_channels,
                       unsigned int input2_channels);

#ifdef ENABLE_FP16
  /**
   * @brief     concat computation for axis 3 fp16
   * @param[in] matAdata fp16 * for Input Tensor A
   * @param[in] vecXdata fp16 * for Input Tensor X
   * @param[in] vecYdata fp16 * for Output Tensor Y
   * @param[in] input1_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input1_channels   represents the channels of the input tensor
   * @param[in] input1_height   represents the height of the input tensor
   * @param[in] input1_width   represents the width of the input tensor A
   * @param[in] input2_width   represents the width of the input tensor X
   */
  void concat_cl_axis3_fp16(const __fp16 *matAdata, const __fp16 *vecXdata,
                            __fp16 *vecYdata, unsigned int input1_batch_size,
                            unsigned int input1_channels,
                            unsigned int input1_height,
                            unsigned int input1_width,
                            unsigned int input2_width);

  /**
   * @brief     concat computation for axis 2 fp16
   * @param[in] matAdata fp16 * for Input Tensor A
   * @param[in] vecXdata fp16 * for Input Tensor X
   * @param[in] vecYdata fp16 * for Output Tensor Y
   * @param[in] input1_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input1_channels   represents the channels of the input tensor
   * @param[in] input1_width   represents the width of the input tensor
   * @param[in] input1_height   represents the height of the input tensor A
   * @param[in] input2_height   represents the height of the input tensor X
   */
  void concat_cl_axis2_fp16(const __fp16 *matAdata, const __fp16 *vecXdata,
                            __fp16 *vecYdata, unsigned int input1_batch_size,
                            unsigned int input1_channels,
                            unsigned int input1_width,
                            unsigned int input1_height,
                            unsigned int input2_height);

  /**
   * @brief     concat computation for axis 1 fp16
   * @param[in] matAdata fp16 * for Input Tensor A
   * @param[in] vecXdata fp16 * for Input Tensor X
   * @param[in] vecYdata fp16 * for Output Tensor Y
   * @param[in] input1_batch_size  represents the number of samples in the input
   * tensor
   * @param[in] input1_height   represents the height of the input tensor
   * @param[in] input1_width   represents the width of the input tensor
   * @param[in] input1_channels   represents the channels of the input tensor A
   * @param[in] input2_channels   represents the channels of the input tensor X
   */
  void concat_cl_axis1_fp16(const __fp16 *matAdata, const __fp16 *vecXdata,
                            __fp16 *vecYdata, unsigned int input1_batch_size,
                            unsigned int input1_height,
                            unsigned int input1_width,
                            unsigned int input1_channels,
                            unsigned int input2_channelst);
#endif
private:
  std::tuple<props::ConcatDimension> concat_props;
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __CONCAT_LAYER_CL_H__ */
