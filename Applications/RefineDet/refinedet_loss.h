// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   refinedet_loss.h
 * @date   16 November 2020
 * @brief  This file contains refinedet loss layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#ifndef __REFINEDET_LOSS_H__
#define __REFINEDET_LOSS_H__

#include <layer_context.h>
#include <loss_layer.h>
#include <node_exporter.h>

namespace custom {

/**
 * @brief loss layer for RefineDet
 *
 */
class RefineDetLoss final : public nntrainer::LossLayer {
public:
  /**
   * @brief Construct RefineDet loss object
   *
   * @param exponent_ exponent
   */
  RefineDetLoss() : LossLayer() {}

  /**
   * @brief Destroy the RefineDet loss object
   *
   */
  ~RefineDetLoss() {}

  /**
   * @copydoc Layer::finalize(InitLayerContext &context)
   */
  void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @copydoc Layer::forwarding(RunLayerContext &context, bool training)
   */
  void forwarding(nntrainer::RunLayerContext &context, bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override {}

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return RefineDetLoss::type; };

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const override { return true; }

  inline static const std::string type = "refinedet_loss";

private:
  std::array<unsigned int, 8> wt_idx; /**< indices of the weights */
  std::vector<unsigned int> positive_mask;
  std::vector<unsigned int> pos_neg_mask;
  std::vector<std::vector<float>> anchor_gt_label_yx;
  std::vector<std::vector<float>> anchor_gt_label_hw;
  std::vector<unsigned int> gt_class_labels;
  unsigned int num_positive_anchors;
};

} // namespace custom

#endif /* __REFINDET_LOSS_H__ */
