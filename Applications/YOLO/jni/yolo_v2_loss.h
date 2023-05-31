// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   yolo_v2_loss.h
 * @date   07 March 2023
 * @brief  This file contains the yolo v2 loss layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#ifndef __YOLO_V2_LOSS_LAYER_H__
#define __YOLO_V2_LOSS_LAYER_H__

#include <string>

#include <acti_func.h>
#include <base_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <node_exporter.h>

namespace custom {

namespace props {

/**
 * @brief maximum object number in 1 image for given dataset
 *
 */
class MaxObjectNumber final : public nntrainer::PositiveIntegerProperty {
public:
  MaxObjectNumber(const unsigned &value = 1);
  static constexpr const char *key = "max_object_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief class number for given dataset
 *
 */
class ClassNumber final : public nntrainer::PositiveIntegerProperty {
public:
  ClassNumber(const unsigned &value = 1);
  static constexpr const char *key = "class_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief grid height number
 *
 */
class GridHeightNumber final : public nntrainer::PositiveIntegerProperty {
public:
  GridHeightNumber(const unsigned &value = 1);
  static constexpr const char *key = "grid_height_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

/**
 * @brief grid width number
 *
 */
class GridWidthNumber final : public nntrainer::PositiveIntegerProperty {
public:
  GridWidthNumber(const unsigned &value = 1);
  static constexpr const char *key = "grid_width_number";
  using prop_tag = nntrainer::uint_prop_tag;
};

} // namespace props

/**
 * @brief Yolo V2 loss layer
 *
 */
class YoloV2LossLayer final : public nntrainer::Layer {
public:
  /**
   * @brief Construct a new YoloV2Loss Layer object
   *
   */
  YoloV2LossLayer();

  /**
   * @brief Destroy the YoloV2Loss Layer object
   *
   */
  ~YoloV2LossLayer() {}

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
   * @copydoc Layer::exportTo(Exporter &exporter, ExportMethods method)
   */
  void exportTo(nntrainer::Exporter &exporter,
                const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  void setBatch(nntrainer::RunLayerContext &context,
                unsigned int batch) override;

  /**
   * @copydoc bool supportBackwarding() const
   */
  bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::requireLabel()
   */
  bool requireLabel() const { return true; }

  /**
   * @copydoc Layer::getType()
   */
  const std::string getType() const override { return YoloV2LossLayer::type; };

  inline static const std::string type = "yolo_v2_loss";

private:
  static constexpr unsigned int NUM_ANCHOR = 5;
  const float anchors_w_buf[NUM_ANCHOR] = {1.3221, 3.19275, 5.05587, 9.47112,
                                           11.2364};
  const float anchors_h_buf[NUM_ANCHOR] = {1.73145, 4.00944, 8.09892, 4.84053,
                                           10.0071};
  const nntrainer::Tensor anchors_w;
  const nntrainer::Tensor anchors_h;
  nntrainer::Tensor anchors_ratio;

  nntrainer::ActiFunc sigmoid; /** sigmoid activation operation */
  nntrainer::ActiFunc softmax; /** softmax activation operation */

  std::tuple<props::MaxObjectNumber, props::ClassNumber,
             props::GridHeightNumber, props::GridWidthNumber>
    yolo_v2_loss_props;
  std::array<unsigned int, 22> wt_idx; /**< indices of the weights */

  /**
   * @brief find responsible anchors per object
   */
  unsigned int find_responsible_anchors(float bbox_ratio);

  /**
   * @brief generate ground truth, mask from labels
   */
  void generate_ground_truth(nntrainer::RunLayerContext &context);
};

} // namespace custom

#endif /* __YOLO_V2_LOSS_LAYER_H__ */
