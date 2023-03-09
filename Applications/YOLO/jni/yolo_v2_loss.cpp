// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Hyeonseok Lee <hs89.lee@samsung.com>
 *
 * @file   yolo_v2_loss.cpp
 * @date   07 March 2023
 * @brief  This file contains the yolo v2 loss layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include "yolo_v2_loss.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum YoloV2LossParams {
  bbox_x_pred,
  bbox_y_pred,
  bbox_w_pred,
  bbox_h_pred,
  confidence_pred,
  class_pred,
  bbox_w_pred_anchor,
  bbox_h_pred_anchor,
  bbox_x_gt,
  bbox_y_gt,
  bbox_w_gt,
  bbox_h_gt,
  confidence_gt,
  class_gt,
  bbox_class_mask,
  iou_mask
};

namespace props {
MaxObjectNumber::MaxObjectNumber(const unsigned &value) { set(value); }
ClassNumber::ClassNumber(const unsigned &value) { set(value); }
GridHeightNumber::GridHeightNumber(const unsigned &value) { set(value); }
GridWidthNumber::GridWidthNumber(const unsigned &value) { set(value); }
ImageHeightSize::ImageHeightSize(const unsigned &value) { set(value); }
ImageWidthSize::ImageWidthSize(const unsigned &value) { set(value); }
} // namespace props

YoloV2LossLayer::YoloV2LossLayer() :
  anchors_w({1, 1, NUM_ANCHOR, 1}, anchors_w_buf),
  anchors_h({1, 1, NUM_ANCHOR, 1}, anchors_h_buf),
  sigmoid(nntrainer::ActivationType::ACT_SIGMOID, false),
  softmax(nntrainer::ActivationType::ACT_SOFTMAX, false),
  yolo_v2_loss_props(props::MaxObjectNumber(), props::ClassNumber(),
                     props::GridHeightNumber(), props::GridWidthNumber(),
                     props::ImageHeightSize(), props::ImageWidthSize()) {
  anchors_ratio = anchors_w.divide(anchors_h);
  wt_idx.fill(std::numeric_limits<unsigned>::max());
}

void YoloV2LossLayer::finalize(nntrainer::InitLayerContext &context) {
  /** NYI */
}

void YoloV2LossLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  /** NYI */
}

void YoloV2LossLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  /** NYI */
}

void YoloV2LossLayer::exportTo(nntrainer::Exporter &exporter,
                               const ml::train::ExportMethods &method) const {
  /** NYI */
}

void YoloV2LossLayer::setProperty(const std::vector<std::string> &values) {
  /** NYI */
}

unsigned int YoloV2LossLayer::find_responsible_anchors(float bbox_ratio) {
  /** NYI */
  return 0;
}

void YoloV2LossLayer::generate_ground_truth(
  nntrainer::Tensor &bbox_x_pred, nntrainer::Tensor &bbox_y_pred,
  nntrainer::Tensor &bbox_w_pred, nntrainer::Tensor &bbox_h_pred,
  nntrainer::Tensor &labels, nntrainer::Tensor &bbox_x_gt,
  nntrainer::Tensor &bbox_y_gt, nntrainer::Tensor &bbox_w_gt,
  nntrainer::Tensor &bbox_h_gt, nntrainer::Tensor &confidence_gt,
  nntrainer::Tensor &class_gt, nntrainer::Tensor &bbox_class_mask,
  nntrainer::Tensor &iou_mask) {
  /** NYI */
}

#ifdef PLUGGABLE

nntrainer::Layer *create_yolo_v2_loss_layer() {
  auto layer = new YoloV2LossLayer();
  return layer;
}

void destory_yolo_v2_loss_layer(nntrainer::Layer *layer) { delete layer; }

/**
 * @note ml_train_layer_pluggable defines the entry point for nntrainer to
 * register a plugin layer
 */
extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_yolo_v2_loss_layer,
                                                   destory_yolo_v2_loss_layer};
}

#endif
} // namespace custom
