// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   addition_layer_cl.h
 * @date   28 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief	 This is Addition Layer Class Class for Neural Network with OpenCl
 * implementation
 */

#ifndef __ADDITION_LAYER_CL_H__
#define __ADDITION_LAYER_CL_H__
#ifdef __cplusplus

#include <cl_context.h>
#include <common_properties.h>
#include <layer_devel.h>
#include <layer_impl_cl.h>

namespace nntrainer {

/**
 * @class   AdditionLayerCL
 * @brief   Addition Layer
 */
class AdditionLayerCL : public LayerImplCl {
public:
  /**
   * @brief     Constructor of Addition Layer
   */
  AdditionLayerCL() : LayerImplCl(), add_props(props::Print()) {}

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

  /**
   * @brief     Register Cl Kernels for `AdditionLayerCl`, bypassing the
   * registration process since it does not require any specific kernels. This
   * function simply returns `true` because `AdditionLayerCl` does not rely on
   * the specific kernels for the layer.
   */
  static bool registerClKernels() { return true; };

  inline static const std::string type = "addition";

private:
  std::tuple<props::Print>
    add_props; /**< fc layer properties : unit - number of output neurons */
};

} // namespace nntrainer

#endif /* __cplusplus */
#endif /* __ADDITION_LAYER_H__ */
