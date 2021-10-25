// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   adam.h
 * @date   6 October 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is the Adam optimizer.
 */
#ifndef __ADAM_H__
#define __ADAM_H__
#ifdef __cplusplus

#include <tuple>

#include <base_properties.h>
#include <optimizer_impl.h>

namespace nntrainer {

/**
 * @brief Beta 1 props
 *
 */
class PropsB1 : public Property<double> {
public:
  static constexpr const char *key = "beta1"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};

/**
 * @brief Beta 2 props
 *
 */
class PropsB2 : public Property<double> {
public:
  static constexpr const char *key = "beta2"; /**< unique key to access */
  using prop_tag = double_prop_tag;           /**< property type */
};

/**
 * @brief epsilon props
 * @todo move this to common props
 *
 */
class PropsEpsilon : public Property<double> {
public:
  static constexpr const char *key = "epsilon"; /**< unique key to access */
  using prop_tag = double_prop_tag;             /**< property type */
};

/**
 * @class   Adam optimizer class
 * @brief   Adam optimizer
 */
class Adam : public OptimizerImpl {
public:
  /**
   * @brief Construct a new Adam object
   *
   */
  Adam();

  /**
   * @brief Destroy the Adam object
   *
   */
  ~Adam();

  /**
   * @copydoc applyGradient(RunOptimizerContext &context)
   */
  void applyGradient(RunOptimizerContext &context) override;

  /**
   * @copydoc Optimizer::getType()
   */
  const std::string getType() const override { return Adam::type; }

  /**
   * @copydoc   getLearningRate(int iteration)
   */
  double getLearningRate(size_t iteration) const override;

  /**
   * @copydoc Optimizer::getOptimizerVariableDim(const TensorDim &dim)
   */
  std::vector<TensorDim> getOptimizerVariableDim(const TensorDim &dim) override;

  /**
   * @copydoc Optimizer::exportTo(Exporter &exporter, const ExportMethods&
   * method)
   */
  void exportTo(Exporter &exporter, const ExportMethods &method) const override;

  inline static const std::string type = "adam";

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

private:
  std::tuple<PropsB1, PropsB2, PropsEpsilon> adam_props;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __ADAM_H__ */
