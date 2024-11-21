// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Daniel Jang <minhyukjang@snu.ac.kr>
 *
 * @file   adamw.h
 * @date   3 November 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Daniel Jang <minhyukjang@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is the AdamW Optimizer.
 */
#ifndef __ADAMW_H__
#define __ADAMW_H__
#ifdef __cplusplus

#include <tuple>

#include <adam.h>

#include <base_properties.h>
#include <optimizer_devel.h>

namespace nntrainer {

/**
 * @class   AdamW Optimizer class
 * @brief   AdamW Optimizer
 */
class AdamW : public Optimizer {
public:
  /**
   * @brief Construct a new AdamW object
   *
   */
  AdamW();

  /**
   * @brief Destroy the AdamW object
   *
   */
  ~AdamW();

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
  const std::string getType() const override { return AdamW::type; }

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

  inline static const std::string type = "adamw";

  /**
   * @copydoc Optimizer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

private:
  std::tuple<PropsB1, PropsB2, PropsEpsilon, TorchRef> adam_props;
};
} /* namespace nntrainer */

#endif /* __cplusplus */
#endif /* __ADAMW_H__ */
