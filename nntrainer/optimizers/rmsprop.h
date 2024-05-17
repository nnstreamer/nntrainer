// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   rmsprop.h
 * @date   17 May 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the RMSProp optimizer.
 */
#ifndef __RMSPROP_H__
#define __RMSPROP_H__
#ifdef __cplusplus

#include <tuple>

#include <optimizer_common_properties.h>
#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @class   RMSProp optimizer class
 * @brief   RMSProp optimizer
 */
class RMSProp : public Optimizer {
public:
  /**
   * @brief Construct a new Adam object
   *
   */
  RMSProp();

  /**
   * @brief Destroy the Adam object
   *
   */
  ~RMSProp();

  /**
   * @copydoc Optimizer::getDefaultLearningRate()
   *
   */
  double getDefaultLearningRate() const override { return 0.001; }

  /**
   * @copydoc applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return RMSProp::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim> getOptimizerVariableDim(const TensorDim &dim) override;

  /**
   * @copydoc Optimizer::exportTo(Exporter &exporter, const
   * ml::train::ExportMethods& method)
   */
  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  inline static const std::string type = "rmsprop";

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

private:
  std::tuple<props::Rho, props::PropsEpsilon, props::TorchRef> rmsprop_props;

  /**
   * @brief Get updated learning rate
   *
   * @param ll learning rate
   *
   * @return updated learning rate
   */
  double getUpdatedLearningRate(unsigned int iteration, double ll) const;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __RMSPROP_H__ */
