// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   refinedet_loss.cpp
 * @date   16 November 2020
 * @brief  This file contains refinedet loss layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <regex>
#include <nntrainer_error.h>
#include <math.h>
#include <acti_func.h>
#include <vector>

#include "concat_layer.h"
#include "layer_context.h"
#include "refinedet_loss.h"
#include "tensor.h"
#include "tensor_wrap_specs.h"

namespace custom {

static constexpr size_t SINGLE_INOUT_IDX = 0;

const unsigned int feature_map_size1 = 28;
const unsigned int feature_map_size2 = 14;
const unsigned int feature_map_size3 = 4;
const unsigned int feature_map_size4 = 2;
const unsigned int num_ratios = 3;
const unsigned int num_anchors = num_ratios * (
  feature_map_size1 * feature_map_size1 + 
  feature_map_size2 * feature_map_size2 + 
  feature_map_size3 * feature_map_size3 + 
  feature_map_size4 * feature_map_size4);
const unsigned int num_classes = 20;
const unsigned int max_gt_boxes = 5;
const float positive_anchor_threshold = 0.5;

using nntrainer::Tensor;
using nntrainer::TensorDim;

void RefineDetLoss::setProperty(const std::vector<std::string> &values) {
  if (!values.empty()) {
    std::string msg = "[RefineDetLoss] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw std::invalid_argument(msg);
  }
}

void RefineDetLoss::finalize(nntrainer::InitLayerContext &context) {
  const TensorDim &in_dim = context.getInputDimensions()[SINGLE_INOUT_IDX];
  TensorDim out_dim = in_dim;
  out_dim.height(max_gt_boxes);
  out_dim.width(5 + num_classes);
  context.setOutputDimensions({out_dim});
  unsigned int batch_size = in_dim.batch();
  input_tensor_idx = context.requestTensor(in_dim, "input", nntrainer::Tensor::Initializer::NONE, true,
    nntrainer::TensorLifespan::FORWARD_DERIV_LIFESPAN);
  // positive_mask = std::vector<std::vector<unsigned int>>(batch_size);
  // pos_neg_mask = std::vector<std::vector<unsigned int>>(batch_size);
  // anchor_gt_label_yx = std::vector<std::vector<std::vector<float>>>(batch_size);
  // anchor_gt_label_hw = std::vector<std::vector<std::vector<float>>>(batch_size);
  // gt_class_labels = std::vector<std::vector<unsigned int>>(batch_size);
  // num_positive_anchors = std::vector<unsigned int>(batch_size);
}

std::vector<Tensor> create_anchors_(
  const unsigned int& anchor_size,
  const unsigned int& stride,
  const unsigned int& feature_map_size
  ) {
  const std::vector<float> anchor_ratios = {0.5, 1, 2};
  
  // Coordinates of the center of anchor box
  Tensor anchor_yx = Tensor(feature_map_size, feature_map_size, anchor_ratios.size(), 2);
  for (unsigned int b = 0; b < anchor_yx.batch(); b++ ) {
    for (unsigned int c = 0; c < anchor_yx.channel(); c++) {
        for (unsigned int h = 0; h < anchor_yx.height(); h++) {
            for(unsigned int w = 0; w < anchor_yx.width(); w++) {
                anchor_yx.setValue(b, c, h, w, 
                ((b + 0.5) * (1 - w) + (c + 0.5) * w) * stride);
            }
        }
    }
  }

  std::vector<std::vector<float>> priors;
  for (float ratio: anchor_ratios) {
    priors.push_back({anchor_size * ((float)pow(ratio, 0.5)), anchor_size / ((float)pow(ratio, 0.5))});
  }

  // Sizes of anchor box
  Tensor anchor_hw = Tensor(feature_map_size, feature_map_size, anchor_ratios.size(), 2);
  for (unsigned int b = 0; b < anchor_hw.batch(); b++ ) {
    for (unsigned int c = 0; c < anchor_hw.channel(); c++) {
        for (unsigned int h = 0; h < anchor_hw.height(); h++) {
            for(unsigned int w = 0; w < anchor_hw.width(); w++) {
                anchor_hw.setValue(b, c, h, w, priors[h][w]);
            }
        }
    }
  }

  unsigned int total_len = anchor_yx.size();
  anchor_yx.reshape({1, 1, total_len / 2, 2});
  anchor_hw.reshape({1, 1, total_len / 2, 2});

  return {anchor_yx, anchor_hw};
}

std::array<Tensor, 2> create_anchors() {
  std::vector<Tensor> anchors1 = create_anchors_(8*4, 8, feature_map_size1);
  std::vector<Tensor> anchors2 = create_anchors_(16*4, 16, feature_map_size2);
  std::vector<Tensor> anchors3 = create_anchors_(32*4, 32, feature_map_size3);
  std::vector<Tensor> anchors4 = create_anchors_(64*4,  64,feature_map_size4);
  Tensor anchor_yx = Tensor::cat({anchors1[0], anchors2[0], anchors3[0], anchors4[0]}, 2);
  Tensor anchor_hw = Tensor::cat({anchors1[1], anchors2[1], anchors3[1], anchors4[1]}, 2);
  return {anchor_yx, anchor_hw}; 
}

float cross_entropy(Tensor& x, std::vector<unsigned int> l) {
  float loss = 0;
  Tensor output = Tensor(x.getDim());
  nntrainer::ActiFunc::softmax(x, output);
  for (unsigned int a = 0; a < output.height(); a++) {
    loss += log(output.getValue(0, 0, a, l[a]) + 1e-10);
  }
  // return -loss / output.height();
  return -loss;
}

float cross_entropy_with_mask(Tensor& x, std::vector<unsigned int> mask, std::vector<unsigned int> l) {
  float loss = 0;
  Tensor output = Tensor(x.getDim());
  nntrainer::ActiFunc::softmax(x, output);
  for (unsigned int a = 0; a < output.height(); a++) {
    if (!mask[a]) {continue;}
    loss += log(output.getValue(0, 0, a, l[a]) + 1e-10);
  }
  return -loss;
}

std::vector<std::pair<unsigned int, float>> cross_entropy_per_anchor(Tensor& x, std::vector<unsigned int> l) {
  std::vector<std::pair<unsigned int, float>> idx_loss(l.size());
  Tensor output = Tensor(x.getDim());
  nntrainer::ActiFunc::softmax(x, output);
  for (unsigned int a = 0; a < output.height(); a++) {
    idx_loss.push_back(std::pair<unsigned int, float>(a, log(output.getValue(0, 0, a, l[a]) + 1e-10)));
  }
  return idx_loss;
}

float smooth_l1(Tensor& x, Tensor& y, const std::vector<unsigned int> l) {
  x.subtract_i(y);
  x.apply_i([&](float val) {
      if (val < 0) {
          val = -val;
      }
      if (val < 1) {
          return 0.5 * val * val;
      } else {
          return val - 0.5;
      }
  });
  x = x.sum(3);
  for (unsigned int i = 0; i < x.height(); i++) {
    if (!l[i]) {
      for(unsigned int j = 0; j < x.width(); j++) {
        x.setValueInt(i * x.width() + j, 0);
      }
    }
  }
  return x.sum(2).getValue(0);
}

// box2 dim=1
std::vector<float> calc_iou(Tensor box1_yx, Tensor box1_hw, Tensor box2_yx, Tensor box2_hw) {
  // Get upper left and lower right corner, from centor and size
  Tensor box1_yx1 = box1_yx.add(box1_hw.divide(2));
  Tensor box1_yx2 = box1_yx.add(box1_hw.divide(2));

  std::vector<Tensor> box1_yx1_split = box1_yx1.split(2, 3);
  Tensor box1_y1 = box1_yx1_split[0];
  Tensor box1_x1 = box1_yx1_split[1];
  std::vector<Tensor> box1_yx2_split = box1_yx2.split(2, 3);
  Tensor box1_y2 = box1_yx2_split[0];
  Tensor box1_x2 = box1_yx2_split[1];

  // Get upper left and lower right corner, from centor and size
  Tensor box2_yx1 = box2_yx.add(box2_hw.divide(2));
  Tensor box2_yx2 = box2_yx.add(box2_hw.divide(2));

  std::vector<Tensor> box2_yx1_split = box2_yx1.split(2, 3);
  Tensor box2_y1 = box2_yx1_split[0];
  Tensor box2_x1 = box2_yx1_split[1];
  std::vector<Tensor> box2_yx2_split = box2_yx2.split(2, 3);
  Tensor box2_y2 = box2_yx2_split[0];
  Tensor box2_x2 = box2_yx2_split[1];

  auto min_func = [&](Tensor &bbox1_xy, Tensor &bbox2_xy, Tensor &intersection_xy) {
    std::transform(bbox1_xy.getData(), bbox1_xy.getData() + bbox1_xy.size(), intersection_xy.getData(),
                   [&bbox2_xy](float x1) { return std::min(x1, bbox2_xy.getValue(0)); });
  };
  auto max_func = [&](Tensor &bbox1_xy, Tensor &bbox2_xy, Tensor &intersection_xy) {
    std::transform(bbox1_xy.getData(), bbox1_xy.getData() + bbox1_xy.size(), intersection_xy.getData(),
                   [&bbox2_xy](float x1) { return std::max(x1, bbox2_xy.getValue(0)); });
  };

  unsigned int num_anc = box1_x1.getDim()[2];
  Tensor inter_x1(num_anc);
  Tensor inter_y1(num_anc);
  Tensor inter_x2(num_anc);
  Tensor inter_y2(num_anc);
  max_func(box1_x1, box2_x1, inter_x2);
  min_func(box1_x2, box2_x2, inter_x1);
  max_func(box1_y1, box2_y1, inter_y2);
  min_func(box1_y2, box2_y2, inter_y1);


  std::vector<Tensor> box1_hw_split = box1_hw.split(2, 3);
  Tensor box1_h = box1_hw_split[0];
  Tensor box1_w = box1_hw_split[1];

  // Assuming box2 = [h, w]
  unsigned int box2_h = box2_hw.getValue(0);
  unsigned int box2_w = box2_hw.getValue(1);
  Tensor inter_area = inter_x2.subtract(inter_x1).apply(nntrainer::ActiFunc::relu).multiply(
  inter_y2.subtract(inter_y1).apply(nntrainer::ActiFunc::relu));
  inter_area.reshape({1, 1, inter_area.size(), 1});
  Tensor iou = inter_area.divide(box1_h.multiply(box1_w).add(box2_h * box2_w).subtract(inter_area));
  std::vector<float> iou_vec = std::vector<float>(iou.getData(), iou.getData() + num_anc);
  return iou_vec;
}

void RefineDetLoss::forwarding(nntrainer::RunLayerContext &context, bool training) { 
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &input = context.getTensor(input_tensor_idx);
  input.copyData(input_);
  Tensor output = Tensor(1,1,1,1);
  output.setZero();
  Tensor &gt = context.getLabel(SINGLE_INOUT_IDX);
  
  std::vector<Tensor> input_split = input.split({2, 2, 2, 2, 2, num_classes}, 3);
  Tensor& arm_yx = input_split[0];
  Tensor& arm_hw = input_split[1];
  Tensor& arm_conf = input_split[2];
  Tensor& odm_yx = input_split[3];
  Tensor& odm_hw = input_split[4];
  Tensor& odm_conf = input_split[5];
  std::vector<Tensor> gt_split = gt.split({1, 2, 2, num_classes}, 3);
  Tensor& gt_is_label = gt_split[0];
  Tensor& gt_yx = gt_split[1];
  Tensor& gt_hw = gt_split[2];
  gt_yx.add_i(gt_hw);
  gt_yx.divide_i(2);
  gt_hw.subtract_i(gt_yx);
  gt_hw.multiply_i(2);
  Tensor& gt_class = gt_split[3];
  std::array<Tensor, 2> anchors = create_anchors();

  unsigned int num_gt_boxes = 0;
  for (unsigned int i = 0; i < max_gt_boxes; i++) {
    if (gt_is_label.getValue(i) == 0) {
       break;
    }
    else {
      num_gt_boxes++;
    }
  }

  unsigned int anchors_num = anchors[0].height();

  positive_mask = {};
  pos_neg_mask = {};
  anchor_gt_label_yx = {};
  anchor_gt_label_hw = {};
  gt_class_labels = {};
  num_positive_anchors = {};

  for (unsigned int b = 0; b < arm_conf.batch(); b++) {
    Tensor arm_conf_ = arm_conf.getBatchSlice(b, 1);
    Tensor arm_yx_ = arm_yx.getBatchSlice(b, 1);
    Tensor arm_hw_ = arm_hw.getBatchSlice(b, 1);
    Tensor odm_conf_ = odm_conf.getBatchSlice(b, 1);
    Tensor odm_yx_ = odm_yx.getBatchSlice(b, 1);
    Tensor odm_hw_ = odm_hw.getBatchSlice(b, 1);
    Tensor gt_class_ = gt_class.getBatchSlice(b, 1);
    Tensor gt_yx_ = gt_yx.getBatchSlice(b, 1);
    Tensor gt_hw_ = gt_hw.getBatchSlice(b, 1);

    // Search anchor with best iou or higher iou than 0.5
    std::set<unsigned int> positive_idx_set = {};
    std::vector<int> anchor_gt_label_idx(anchors_num, -1);
    std::vector<float> anchor_gt_label_iou(anchors_num);
    // Reset and create again
    anchor_gt_label_yx.push_back(std::vector<std::vector<float>>(anchors_num, {0,0}));
    anchor_gt_label_hw.push_back(std::vector<std::vector<float>>(anchors_num, {0,0}));
    std::vector<Tensor> gt_yx_boxes = gt_yx_.split(gt_class_.height(), 2);
    std::vector<Tensor> gt_hw_boxes = gt_hw_.split(gt_class_.height(), 2);
    for (unsigned int gt = 0; gt < num_gt_boxes; gt++) {
      std::vector<float> anc_gt_iou = calc_iou(anchors[0], anchors[1], gt_yx_boxes[gt], gt_hw_boxes[gt]);
      unsigned int max_idx = std::max_element(anc_gt_iou.begin(), anc_gt_iou.end()) - anc_gt_iou.begin();
      for (unsigned int i = 0; i < num_anchors; i++) {
        if (anchor_gt_label_iou[i] < anc_gt_iou[i]) {
          anchor_gt_label_idx[i] = gt;
          anchor_gt_label_iou[i] = anc_gt_iou[i];
          anchor_gt_label_yx[b][i] = {gt_yx_boxes[gt].getValue(0, 0, 0, 0), gt_yx_boxes[gt].getValue(0, 0, 0, 1)};
          anchor_gt_label_hw[b][i] = {gt_hw_boxes[gt].getValue(0, 0, 0, 0), gt_hw_boxes[gt].getValue(0, 0, 0, 1)};
        }
      }
      positive_idx_set.insert(max_idx);
      for (unsigned int i = 0; i < anchors_num; i++) {
        if (anc_gt_iou[i] > 0.5) {
          positive_idx_set.insert(i);
        }
      }
    }

    num_positive_anchors.push_back(positive_idx_set.size());
    // Reset positive mask and create again
    positive_mask.push_back(std::vector<unsigned int>(anchors_num));
    for (auto& i : positive_idx_set) {
      positive_mask[b][i] = 1;
    }

    // ARM loss
    output.add_i(cross_entropy(arm_conf_, positive_mask[b]) / num_positive_anchors[b]);
    auto log_ = [&](float val) {return (float)log(1e-10 + val);};
    Tensor gt_yx_ratio = Tensor(anchor_gt_label_yx[b]).subtract(anchors[0]).divide(anchors[1]);
    Tensor gt_hw_log = Tensor(anchor_gt_label_hw[b]).divide(anchors[1]).apply(log_);
    Tensor gt_yxhw = Tensor::cat({gt_yx_ratio, gt_hw_log}, 3);
    Tensor arm_yxhw = Tensor::cat({arm_yx_, arm_hw_}, 3);
    output.add_i(smooth_l1(arm_yxhw, gt_yxhw, positive_mask[b]) / num_positive_anchors[b]);

    // ODM loss
    // Negative anchor filtering
    unsigned int num_negative_anchors = anchors_num - num_positive_anchors[b];
    std::vector<unsigned int> negative_mask(anchors_num);
    std::transform(positive_mask[b].begin(), positive_mask[b].end(), negative_mask.begin(), [&](unsigned int val){return 1 - val;});
    for (unsigned int i = 0; i < anchors_num; i++) {
      if (arm_conf_.getValue(2 * i + 1) > 0.99 && negative_mask[i]) {
        negative_mask[i] = 0;
        num_negative_anchors--;
      }
    } 

    // Hard negative mining
    std::vector<std::pair<unsigned int, float>> arm_loss_per_anchor = cross_entropy_per_anchor(arm_conf_, positive_mask[b]);
    sort(arm_loss_per_anchor.begin(), arm_loss_per_anchor.end(), 
      [&](std::pair<unsigned int, float> v1, std::pair<unsigned int, float>v2) {return v1.second < v2.second;});
    if (num_negative_anchors > 3 * num_positive_anchors[b]) {
      num_negative_anchors = 3 * num_positive_anchors[b];
      arm_loss_per_anchor.resize(num_negative_anchors);
    }

    // Should NMS be done here??
    std::vector<Tensor> anchors_split0 = anchors[0].split(anchors[0].height(), 2);
    std::vector<Tensor> anchors_split1 = anchors[1].split(anchors[1].height(), 2);
    for (unsigned int i = 0; i < num_negative_anchors; i++) {
      unsigned int ith = arm_loss_per_anchor[i].first;
      Tensor& ith_anchor_yx = anchors_split0[ith];
      Tensor& ith_anchor_hw = anchors_split1[ith];
      std::vector<float> ith_anchor_iou = calc_iou(anchors[0], anchors[1], ith_anchor_yx, ith_anchor_hw);
      for(unsigned int j = 0; j < ith_anchor_iou.size(); j++) {
        if (i == j) {continue;}
        if (ith_anchor_iou[j] > 0.7 && negative_mask[j]) {
          negative_mask[j] = 0;
          num_negative_anchors--;
        }
      }
    }

    Tensor odm_yxhw = Tensor::cat({odm_yx_, odm_hw_}, 3);
    // Reset ODM mask and create again
    pos_neg_mask.push_back(std::vector<unsigned int>(anchors_num));
    gt_class_labels.push_back(std::vector<unsigned int>(anchors_num));
    std::vector<Tensor> gt_class_split = gt_class_.split(gt_class_.height(), 2);
    std::vector<Tensor> odm_conf_split = odm_conf_.split(odm_conf_.height(), 2);
    for (unsigned int i = 0; i < anchors_num; i++) {
      unsigned int odm_argmax = std::max_element(odm_conf_split[i].getData(), 
        odm_conf_split[i].getData() + odm_conf_.width()) - odm_conf_split[i].getData();
      pos_neg_mask[b][i] = (positive_mask[b][i] + negative_mask[i]) * odm_argmax;
      if (anchor_gt_label_idx[i] == -1) {
        gt_class_labels[b][i] = 0;
      }
      else {
        unsigned int gt_class_argmax = std::max_element(gt_class_split[anchor_gt_label_idx[i]].getData(), 
          gt_class_split[anchor_gt_label_idx[i]].getData() + gt_class_split[anchor_gt_label_idx[i]].width())
           - gt_class_split[anchor_gt_label_idx[i]].getData();
        gt_class_labels[b][i] = gt_class_argmax;
      }
    }
    output.add_i(cross_entropy_with_mask(odm_conf_, pos_neg_mask[b], gt_class_labels[b]) / num_positive_anchors[b]);
    output.add_i(smooth_l1(odm_yxhw, gt_yxhw, pos_neg_mask[b]) / num_positive_anchors[b]);
  }
  output.divide_i(arm_conf.batch());
  LossLayer::updateLoss(context, output);
}

// Assume 2d
void cross_entropy_derivative(Tensor& x, std::vector<unsigned int> l, 
  Tensor& x_deriv, unsigned int num_positive_anchors) {
  nntrainer::ActiFunc::softmax(x, x_deriv);
  Tensor label = Tensor(1, 1, l.size(), x.width());
  for (unsigned int i = 0; i < l.size(); i++) {
    label.setValue(0, 0, i, l[i], 1);
  }
  x_deriv.subtract_i(label);
  x_deriv.divide_i(num_positive_anchors);
}

void cross_entropy_with_mask_derivative(Tensor& x, std::vector<unsigned int> mask, std::vector<unsigned int> l, 
  Tensor& x_deriv, unsigned int num_positive_anchors) {
  nntrainer::ActiFunc::softmax(x, x_deriv);
  Tensor label = Tensor(1, 1, l.size(), x.width());
  for (unsigned int i = 0; i < l.size(); i++) {
    label.setValue(0, 0, i, l[i], 1);
  }
  x_deriv.subtract_i(label);
  for (unsigned int i = 0; i < mask.size(); i++) {
    if(!mask[i]) {
      for (unsigned int j = 0; j < x.width(); j++) {
        x_deriv.setValue(0, 0, i, j, 0);
      }
    }
  }
  x_deriv.divide_i(num_positive_anchors);
}

// splitted gradient for yx/hw
void smooth_l1_derivative(Tensor& x, Tensor& y, const std::vector<unsigned int> l, 
  Tensor& x_deriv1, Tensor& x_deriv2) {
  x.subtract_i(y);
  x.apply_i([&](float val) {
      if (-1 < val && val < 1) {
          return val;
      } else if (val < 0) {
          return -1.0f;
      } else if (val > 0) {
        return 1.0f;
      } else {
        return 0.0f;
      }
  });
  for (unsigned int i = 0; i < x.height(); i++) {
    if (!l[i]) {
      for(unsigned int j = 0; j < x.width(); j++) {
        x.setValueInt(i * x.width() + j, 0);
      }
    }
  }
  
  std::vector<Tensor> x_split = x.split(2, 3);
  x_deriv1.add_i(x_split[0]);
  x_deriv2.add_i(x_split[1]);
}

void RefineDetLoss::calcDerivative(nntrainer::RunLayerContext &context) {
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  std::vector<Tensor> outgoing_derivative_split = outgoing_derivative.split({2, 2, 2, 2, 2, num_classes}, 3);
  Tensor& arm_yx_deriv = outgoing_derivative_split[0];
  Tensor& arm_hw_deriv = outgoing_derivative_split[1];
  Tensor& arm_conf_deriv = outgoing_derivative_split[2];
  Tensor& odm_yx_deriv = outgoing_derivative_split[3];
  Tensor& odm_hw_deriv = outgoing_derivative_split[4];
  Tensor& odm_conf_deriv = outgoing_derivative_split[5];

  Tensor &input = context.getTensor(input_tensor_idx);

  Tensor &gt = context.getLabel(SINGLE_INOUT_IDX);
  std::vector<Tensor> input_split = input.split({2, 2, 2, 2, 2, num_classes}, 3);
  Tensor& arm_yx = input_split[0];
  Tensor& arm_hw = input_split[1];
  Tensor& arm_conf = input_split[2];
  Tensor& odm_yx = input_split[3];
  Tensor& odm_hw = input_split[4];
  Tensor& odm_conf = input_split[5];
  std::vector<Tensor> gt_split = gt.split({1, 2, 2, num_classes}, 3);
  Tensor& gt_yx = gt_split[1];
  Tensor& gt_hw = gt_split[2];
  gt_yx.add_i(gt_hw);
  gt_yx.divide_i(2);
  gt_hw.subtract_i(gt_yx);
  gt_hw.multiply_i(2);
  Tensor& gt_class = gt_split[3];
  std::array<Tensor, 2> anchors = create_anchors();
  for (unsigned int b = 0; b < input.batch(); b++) {
    Tensor arm_yx_deriv_ = arm_yx_deriv.getBatchSlice(b, 1);
    Tensor arm_hw_deriv_ = arm_hw_deriv.getBatchSlice(b, 1);
    Tensor arm_conf_deriv_ = arm_conf_deriv.getBatchSlice(b, 1);
    Tensor odm_yx_deriv_ = odm_yx_deriv.getBatchSlice(b, 1);
    Tensor odm_hw_deriv_ = odm_hw_deriv.getBatchSlice(b, 1);
    Tensor odm_conf_deriv_ = odm_conf_deriv.getBatchSlice(b, 1);

    Tensor arm_yx_ = arm_yx.getBatchSlice(b, 1);
    Tensor arm_hw_ = arm_hw.getBatchSlice(b, 1);
    Tensor arm_conf_ = arm_conf.getBatchSlice(b, 1);
    Tensor odm_yx_ = odm_yx.getBatchSlice(b, 1);
    Tensor odm_hw_ = odm_hw.getBatchSlice(b, 1);
    Tensor odm_conf_ = odm_conf.getBatchSlice(b, 1);

    cross_entropy_derivative(arm_conf_, positive_mask[b], arm_conf_deriv_, num_positive_anchors[b]);
    auto log_ = [&](float val) {return (float)log(val + 1e-10);};
    Tensor gt_yx_ratio = Tensor(anchor_gt_label_yx[b]).subtract(anchors[0]).divide(anchors[1]);
    Tensor gt_hw_log = Tensor(anchor_gt_label_hw[b]).divide(anchors[1]).apply(log_);
    Tensor gt_yxhw = Tensor::cat({gt_yx_ratio, gt_hw_log}, 3);
    Tensor arm_yxhw = Tensor::cat({arm_yx_, arm_hw_}, 3);
    smooth_l1_derivative(arm_yxhw, gt_yxhw, pos_neg_mask[b], arm_yx_deriv_, arm_hw_deriv_);
    arm_yx_deriv_.divide_i(num_positive_anchors[b]);
    arm_hw_deriv_.divide_i(num_positive_anchors[b]);
    cross_entropy_with_mask_derivative(odm_conf_, pos_neg_mask[b], gt_class_labels[b], odm_conf_deriv_, num_positive_anchors[b]);
    Tensor odm_yxhw = Tensor::cat({odm_yx_, odm_hw_}, 1);
    smooth_l1_derivative(odm_yxhw, gt_yxhw, pos_neg_mask[b], odm_yx_deriv_, odm_hw_deriv_);
    odm_yx_deriv_.divide_i(num_positive_anchors[b]);
    odm_hw_deriv_.divide_i(num_positive_anchors[b]);
  }
}

} // namespace custom
