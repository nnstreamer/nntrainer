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
#include <layer_impl_cl.h>
#include <opencl_buffer.h>
#include <opencl_kernel.h>

namespace nntrainer {

/**
 * @brief A tranpose layer.
 *
 */
class TransposeLayerCl final : public LayerImplCl {
public:
  /**
   * @brief Construct a new transpose layer object
   *
   */
  TransposeLayerCl() : LayerImplCl(), transpose_props(props::Print()) {}

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
  bool supportBackwarding() const override { return false; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override{};

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return TransposeLayerCl::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @brief     Register Cl Kernels for `TransposeLayerCl`, bypassing the
   * registration process since it does not require any specific kernels. This
   * function simply returns `true` because `TransposeLayerCl` does not rely on
   * the specific kernels for the layer.
   */
  static bool registerClKernels() { return true; };

  static constexpr const char *type = "transpose";

private:
  std::tuple<props::Print> transpose_props; /**< transpose layer properties :
                                            unit - number of output neurons */
};
} // namespace nntrainer

#endif /* __TRANSPOSE_LAYER_CL_H__ */
