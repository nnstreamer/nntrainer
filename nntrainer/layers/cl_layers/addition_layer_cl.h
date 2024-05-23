// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   addition_layer_cl.h
 * @date   17 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief	 This is Addition Layer Class Class for Neural Network with OpenCl
 * implementation
 */

#ifndef __ADDITION_LAYER_CL_H__
#define __ADDITION_LAYER_CL_H__
#ifdef __cplusplus

#include <common_properties.h>
#include <layer_devel.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

#define CREATE_IF_EMPTY_DIMS(tensor, ...) \
  do {                                    \
    if (tensor.empty())                   \
      tensor = Tensor(__VA_ARGS__);       \
  } while (0);

namespace nntrainer {

/**
 * @class   AdditionLayerCL
 * @brief   Addition Layer
 */
class AdditionLayerCL : public Layer {
public:
  /**
   * @brief     Constructor of Addition Layer
   */
  AdditionLayerCL() : Layer(), add_props(props::Print()) {}

  /**
   * @brief     Destructor of Addition Layer
   */
  ~AdditionLayerCL(){};

  /**
   *  @brief  Move constructor of AdditionLayer.
   *  @param[in] AdditionLayer &&
   */
  AdditionLayerCL(AdditionLayerCL &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @parma[in] rhs AdditionLayer to be moved.
   */
  AdditionLayerCL &operator=(AdditionLayerCL &&rhs) = default;

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
   * @brief declaring static kernel objects
   */
  static opencl::Kernel kernel_addition;

  /**
   * @brief Process data and dimensions for add operation used in addition layer
   * @param[in] input Tensor
   * @param[in] result Tensor
   * @param[in] RunLayerContext reference
   */
  void AddProcess(Tensor const &input, Tensor &result,
                  RunLayerContext &context);

  /**
   * @brief     addition : sum of all input vectors
   * @param[in] input float * for input
   * @param[in] res float * for result/output
   * @param[in] size number of elements in input vector
   * @param[in] context RunLayerContext reference
   */
  void addition_cl(const float *input, float *res, unsigned int size,
                   RunLayerContext &context);

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return AdditionLayerCL::type; };

  std::tuple<props::Print>
    add_props; /**< fc layer properties : unit - number of output neurons */

  inline static const std::string type = "addition";
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ADDITION_LAYER_H__ */
