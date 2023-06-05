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
#include <iostream>
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

const unsigned int feature_map_size1 = 40;
const unsigned int feature_map_size2 = 20;
const unsigned int feature_map_size3 = 5;
const unsigned int feature_map_size4 = 3;
const unsigned int num_ratios = 3;
const unsigned int num_anchors = num_ratios * (
  feature_map_size1 * feature_map_size1 + 
  feature_map_size2 * feature_map_size2 + 
  feature_map_size3 * feature_map_size3 + 
  feature_map_size4 * feature_map_size4);
const unsigned int num_classes = 20;
const unsigned int num_gt_boxes = 50;
const float positive_anchor_threshold = 0.5;

using nntrainer::Tensor;
using nntrainer::TensorDim;
// using nntrainer::TensorLifespan;


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
  out_dim.width(4 + num_classes);
  context.setOutputDimensions({out_dim});
  unsigned int batch_size = in_dim.batch();
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
  // std::cout << "cross_entropy_with_mask start" << std::endl;
  float loss = 0;
  Tensor output = Tensor(x.getDim());
  nntrainer::ActiFunc::softmax(x, output);
  for (unsigned int a = 0; a < output.height(); a++) {
    if (!mask[a]) {continue;}
    loss += log(output.getValue(0, 0, a, l[a]) + 1e-10);
  }
  // std::cout << "cross_entropy_with_mask end" << std::endl;
  // return -loss / output.height();
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
  // std::cout << "smooth_l1 start" << std::endl;
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
  x.sum(1);
  for (unsigned int a = 0; a < x.height(); a++) {
    if (!l[a]) {
      x.setValueInt(a, 0);
    }
  }
  // std::cout << "smooth_l1 end" << std::endl;
  return x.sum(0).getValue(0);
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
  // inter_area.print(std::cout);
  Tensor iou = inter_area.divide(box1_h.multiply(box1_w).add(box2_h * box2_w).subtract(inter_area));
  // iou.print(std::cout);
  std::vector<float> iou_vec = std::vector<float>(iou.getData(), iou.getData() + num_anc);
  return iou_vec;
}

void RefineDetLoss::forwarding(nntrainer::RunLayerContext &context, bool training) { 
  // std::cout << "refinedet forwarding start" << std::endl;
  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  // Tensor &output = context.getOutput(SINGLE_INOUT_IDX);
  Tensor output = Tensor(1,1,1,1);
  output.setZero();
  Tensor &gt = context.getLabel(SINGLE_INOUT_IDX);
  input.setRandUniform(2.0, 8.0);
  gt.setRandUniform(12.0, 18.0);
  
  std::vector<Tensor> input_split = input.split({2, 2, 2, 2, 2, num_classes}, 3);
  Tensor& arm_yx = input_split[0];
  Tensor& arm_hw = input_split[1];
  Tensor& arm_conf = input_split[2];
  Tensor& odm_yx = input_split[3];
  Tensor& odm_hw = input_split[4];
  Tensor& odm_conf = input_split[5];
  std::vector<Tensor> gt_split = gt.split({2, 2, num_classes}, 3);
  Tensor& gt_yx = gt_split[0];
  Tensor& gt_hw = gt_split[1];
  Tensor& gt_class = gt_split[2];
  std::array<Tensor, 2> anchors = create_anchors();
  for (auto& anc : anchors) {anc.setRandUniform(100.0, 1000.0);}
  gt_yx.setRandUniform(100.0, 1000.0);
  gt_hw.setRandUniform(100.0, 1000.0);

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
    // std::vector<std::vector<float>> anchor_gt_label_yx(anchors_num, {0,0});
    // std::vector<std::vector<float>> anchor_gt_label_hw(anchors_num, {0,0});
    // Reset and create again
    anchor_gt_label_yx.push_back(std::vector<std::vector<float>>(anchors_num, {0,0}));
    anchor_gt_label_hw.push_back(std::vector<std::vector<float>>(anchors_num, {0,0}));
    std::vector<Tensor> gt_yx_boxes = gt_yx_.split(gt_class_.height(), 2);
    std::vector<Tensor> gt_hw_boxes = gt_hw_.split(gt_class_.height(), 2);
    for (unsigned int gt = 0; gt < gt_class_.height(); gt++) {
      std::vector<float> anc_gt_iou = calc_iou(anchors[0], anchors[1], gt_yx_boxes[gt], gt_hw_boxes[gt]);
      unsigned int max_idx = std::max_element(anc_gt_iou.begin(), anc_gt_iou.end()) - anc_gt_iou.begin();
      // std::cout << "max_idx " << max_idx << std::endl;
      if (anchor_gt_label_iou[max_idx] < anc_gt_iou[max_idx]) {
        // std::cout << max_idx << ", " << gt << std::endl;
        anchor_gt_label_idx[max_idx] = gt;
        anchor_gt_label_iou[max_idx] = anc_gt_iou[max_idx];
        anchor_gt_label_yx[b][max_idx] = {gt_yx_boxes[gt].getValue(0, 0, 0, 0), gt_yx_boxes[gt].getValue(0, 0, 0, 1)};
        anchor_gt_label_hw[b][max_idx] = {gt_hw_boxes[gt].getValue(0, 0, 0, 0), gt_hw_boxes[gt].getValue(0, 0, 0, 1)};
      }
      positive_idx_set.insert(max_idx);
      for (unsigned int i = 0; i < anchors_num; i++) {
        if (anc_gt_iou[i] > 0.5) {
          positive_idx_set.insert(i);
        }
      }
    }
    num_positive_anchors.push_back(positive_idx_set.size());
    // std::vector<unsigned int> positive_mask(anchors_num);
    // Reset positive mask and create again
    positive_mask.push_back(std::vector<unsigned int>(anchors_num));
    for (auto& i : positive_idx_set) {
      positive_mask[b][i] = 1;
    }

    // ARM loss
    output.add_i(cross_entropy(arm_conf_, positive_mask[b]) / num_positive_anchors[b]);
    auto log_ = [&](float val) {return (float)log(val);};
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

    Tensor odm_yxhw = Tensor::cat({odm_yx_, odm_hw_}, 1);
    // std::vector<unsigned int> pos_neg_mask(anchors_num);
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
        unsigned int gt_class_argmax = std::max_element(gt_class_split[i].getData(), 
          gt_class_split[i].getData() + gt_class_.width()) - gt_class_split[i].getData();
        gt_class_labels[b][i] = gt_class_argmax;
      }
    }
    output.add_i(cross_entropy_with_mask(odm_conf_, pos_neg_mask[b], gt_class_labels[b]) / num_positive_anchors[b]);
    output.add_i(smooth_l1(odm_yxhw, gt_yxhw, pos_neg_mask[b]) / num_positive_anchors[b]);
  }
  output.divide_i(arm_conf.batch());
  LossLayer::updateLoss(context, output);
  // std::cout << "refinedet forwarding end" << std::endl;
}

// Assume 2d
void cross_entropy_derivative(Tensor& x, std::vector<unsigned int> l, 
  Tensor& x_deriv, unsigned int num_positive_anchors) {
  // std::cout << "cross_entropy_derivative 1" << std::endl;
  nntrainer::ActiFunc::softmax(x, x_deriv);
  Tensor label = Tensor(1, 1, l.size(), x.width());
  for (unsigned int i = 0; i < l.size(); i++) {
    label.setValue(0, 0, i, l[i], 1);
  }
  x_deriv.subtract_i(label);
  x_deriv.divide_i(num_positive_anchors);
  // std::cout << "cross_entropy_derivative 2" << std::endl;
}

void cross_entropy_with_mask_derivative(Tensor& x, std::vector<unsigned int> mask, std::vector<unsigned int> l, 
  Tensor& x_deriv, unsigned int num_positive_anchors) {
  // std::cout << "cross_entropy_with_mask_derivative 1" << std::endl;
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
  // std::cout << "cross_entropy_with_mask_derivative 2" << std::endl;
}

// splitted gradient for yx/hw
void smooth_l1_derivative(Tensor& x, Tensor& y, const std::vector<unsigned int> l, 
  Tensor& x_deriv1, Tensor& x_deriv2, unsigned int num_positive_anchors) {
  // std::cout << "smooth_l1_derivative 1" << std::endl;
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
  std::vector<Tensor> x_split = x.split(2, 3);
  x_deriv1.add_i(x_split[0]);
  x_deriv1.divide_i(num_positive_anchors);
  x_deriv2.add_i(x_split[1]);
  x_deriv2.divide_i(num_positive_anchors);
  // std::cout << "smooth_l1_derivative 2" << std::endl;
}

void RefineDetLoss::calcDerivative(nntrainer::RunLayerContext &context) {
  // std::cout << "refinedet derivative start" << std::endl;
  Tensor &outgoing_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  std::vector<Tensor> outgoing_derivative_split = outgoing_derivative.split({2, 2, 2, 2, 2, num_classes}, 3);
  Tensor& arm_yx_deriv = outgoing_derivative_split[0];
  Tensor& arm_hw_deriv = outgoing_derivative_split[1];
  Tensor& arm_conf_deriv = outgoing_derivative_split[2];
  Tensor& odm_yx_deriv = outgoing_derivative_split[3];
  Tensor& odm_hw_deriv = outgoing_derivative_split[4];
  Tensor& odm_conf_deriv = outgoing_derivative_split[5];

  Tensor &input = context.getInput(SINGLE_INOUT_IDX);
  Tensor &gt = context.getLabel(SINGLE_INOUT_IDX);
  std::vector<Tensor> input_split = input.split({2, 2, 2, 2, 2, num_classes}, 3);
  Tensor& arm_yx = input_split[0];
  Tensor& arm_hw = input_split[1];
  Tensor& arm_conf = input_split[2];
  Tensor& odm_yx = input_split[3];
  Tensor& odm_hw = input_split[4];
  Tensor& odm_conf = input_split[5];
  std::vector<Tensor> gt_split = gt.split({2, 2, num_classes}, 3);
  Tensor& gt_yx1 = gt_split[0];
  Tensor& gt_yx2 = gt_split[1];
  Tensor gt_yx = gt_yx1.add(gt_yx2).divide(2);
  Tensor gt_hw = gt_yx2.subtract(gt_yx1);
  Tensor& gt_class = gt_split[2];
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
    auto log_ = [&](float val) {return (float)log(val);};
    Tensor gt_yx_ratio = Tensor(anchor_gt_label_yx[b]).subtract(anchors[0]).divide(anchors[1]);
    Tensor gt_hw_log = Tensor(anchor_gt_label_hw[b]).divide(anchors[1]).apply(log_);
    Tensor gt_yxhw = Tensor::cat({gt_yx_ratio, gt_hw_log}, 3);
    Tensor arm_yxhw = Tensor::cat({arm_yx_, arm_hw_}, 3);
    smooth_l1_derivative(arm_yxhw, gt_yxhw, pos_neg_mask[b], arm_yx_deriv_, arm_hw_deriv_, num_positive_anchors[b]);
    cross_entropy_with_mask_derivative(odm_conf_, pos_neg_mask[b], gt_class_labels[b], odm_conf_deriv_, num_positive_anchors[b]);
    Tensor odm_yxhw = Tensor::cat({odm_yx_, odm_hw_}, 1);
    smooth_l1_derivative(arm_yxhw, gt_yxhw, pos_neg_mask[b], odm_yx_deriv_, odm_hw_deriv_, num_positive_anchors[b]);
  }
  // std::cout << "refinedet derivative end" << std::endl;
}

// #ifdef PLUGGABLE

// nntrainer::Layer *create_pow_layer() {
//   auto layer = new PowLayer();
//   std::cout << "power created\n";
//   return layer;
// }

// void destory_pow_layer(nntrainer::Layer *layer) {
//   std::cout << "power deleted\n";
//   delete layer;
// }

// extern "C" {
// nntrainer::LayerPluggable ml_train_layer_pluggable{create_pow_layer,
//                                                    destory_pow_layer};
// }

// #endif

} // namespace custom
