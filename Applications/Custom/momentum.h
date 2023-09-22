// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   momentum.h
 * @date   31 March 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Momentum optimizer.
 */
#ifndef __MOMENTUM_H__
#define __MOMENTUM_H__
#ifdef __cplusplus
#include <tuple>

#include <base_properties.h>
#include <node_exporter.h>
#include <optimizer_context.h>
#include <optimizer_devel.h>

namespace custom {

/**
 * @brief   momentum property
 */
class PropsM : public nntrainer::Property<double> {
public:
  static constexpr const char *key = "momentum"; /**< unique key to access */
  using prop_tag = nntrainer::double_prop_tag;   /**< property type */
};

/**
 * @class   Momentum optimizer class
 * @brief   Momentum optimizer
 */
class Momentum final : public nntrainer::Optimizer {
public:
  /**
   * @brief     Constructor of Optimizer Class
   */
  Momentum();

  /**
   * @brief     Destructor of Optimizer Class
   */
  ~Momentum() = default;

  double getDefaultLearningRate() const override { return 0.001; }

  /**
   * @copydoc applyGradient(Weight &weight, int tensor_idx, double updated_lr,
   * int iteration)
   */
  void applyGradient(nntrainer::RunOptimizerContext &context);

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const { return Momentum::type; }

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<ml::train::TensorDim>
  getOptimizerVariableDim(const ml::train::TensorDim &dim) override;

  /**
   * @copydoc Optimizer::exportTo(Exporter &exporter, const
   * ml::train::ExportMethods& method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  inline static const std::string type = "momentum";

private:
  std::tuple<PropsM> momentum_props; /** momentum for grad */
};
} // namespace custom

#endif /* __cplusplus */
#endif /* __MOMENTUM_H__ */
