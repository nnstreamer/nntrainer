// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   centroid_knn.cpp
 * @date   09 Jan 2021
 * @brief  This file contains the simple nearest neighbor layer
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * @details This layer takes centroid and calculate l2 distance
 */

#include <iostream>
#include <limits>
#include <regex>
#include <sstream>

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tensor.h>
#include <weight.h>

#include <centroid_knn.h>
#include <simpleshot_utils.h>

namespace simpleshot {
namespace layers {

int CentroidKNN::setProperty(std::vector<std::string> values) {
  util::Entry e;

  std::vector<std::string> unhandled_values;

  for (auto &val : values) {
    try {
      e = util::getKeyValue(val);
    } catch (std::invalid_argument &e) {
      std::cerr << e.what() << std::endl;
      return ML_ERROR_INVALID_PARAMETER;
    }

    if (e.key == "num_class") {
      try {
        num_class = std::stoul(e.value);
        if (num_class == 0) {
          std::cerr << "[CentroidKNN] num_class cannot be zero" << std::endl;
          return ML_ERROR_INVALID_PARAMETER;
        }
      } catch (std::invalid_argument &e) {
        std::cerr << e.what() << std::endl;
        return ML_ERROR_INVALID_PARAMETER;
      } catch (std::out_of_range &e) {
        std::cerr << e.what() << std::endl;
        return ML_ERROR_INVALID_PARAMETER;
      }
    } else {
      unhandled_values.push_back(val);
    }
  }

  return nntrainer::LayerV1::setProperty(unhandled_values);
}

int CentroidKNN::initialize(nntrainer::Manager &manager) {
  if (input_dim[0].channel() != 1 || input_dim[0].height() != 1) {
    ml_logw("centroid nearest layer is designed for flattend feature for now, "
            "please check");
  }

  if (num_class == 0) {
    ml_loge("Error: num_class must be a positive non-zero integer");
    return ML_ERROR_INVALID_PARAMETER;
  }

  output_dim[0] = nntrainer::TensorDim({num_class});

  /// weight is a distance map that contains centroid of features of each class
  auto map_dim =
    nntrainer::TensorDim({num_class, input_dim[0].getFeatureLen()});

  /// samples seen for the current run to calculate the centroid
  auto samples_seen = nntrainer::TensorDim({num_class});

  if (weights.empty()) {
    weights.reserve(2);
    weights.emplace_back(map_dim, nntrainer::WeightInitializer::WEIGHT_ZEROS,
                         nntrainer::WeightRegularizer::NONE, 1.0f, false,
                         "centroidNN:map");
    weights.emplace_back(samples_seen,
                         nntrainer::WeightInitializer::WEIGHT_ZEROS,
                         nntrainer::WeightRegularizer::NONE, 1.0f, false,
                         "centroidNN:num_samples");
    manager.trackWeights(weights);
  } else {
    weights[0].reset(map_dim, nntrainer::WeightInitializer::WEIGHT_ZEROS,
                     nntrainer::WeightRegularizer::NONE, 1.0f, false);
    weights[1].reset(samples_seen, nntrainer::WeightInitializer::WEIGHT_ZEROS,
                     nntrainer::WeightRegularizer::NONE, 1.0f, false);
  }

  return ML_ERROR_NONE;
}

void CentroidKNN::forwarding(bool training) {
  auto &hidden_ = net_hidden[0]->getVariableRef();
  auto &input_ = net_input[0]->getVariableRef();
  auto &label = net_hidden[0]->getGradientRef();

  if (training && label.uninitialized()) {
    throw std::invalid_argument(
      "[CentroidKNN] forwarding requires label feeded");
  }

  auto &map = weightAt(0).getVariableRef();
  auto &num_samples = weightAt(1).getVariableRef();
  auto feature_len = input_dim[0].getFeatureLen();

  auto get_distance = [](const nntrainer::Tensor &a,
                         const nntrainer::Tensor &b) {
    return -a.subtract(b).l2norm();
  };

  if (training) {
    auto ans = label.argmax();

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      auto saved_feature =
        map.getSharedDataTensor({feature_len}, ans[b] * feature_len);

      //  nntrainer::Tensor::Map(map.getData(), {feature_len},
      // ans[b] * feature_len);
      auto num_sample = num_samples.getValue(0, 0, 0, ans[b]);
      auto current_feature = input_.getBatchSlice(b, 1);
      saved_feature.multiply_i(num_sample);
      saved_feature.add_i(current_feature);
      saved_feature.divide_i(num_sample + 1);
      num_samples.setValue(0, 0, 0, ans[b], num_sample + 1);
    }
  }

  for (unsigned int i = 0; i < num_class; ++i) {
    auto saved_feature =
      map.getSharedDataTensor({feature_len}, i * feature_len);
    // nntrainer::Tensor::Map(map.getData(), {feature_len}, i * feature_len);

    auto num_sample = num_samples.getValue(0, 0, 0, i);

    for (unsigned int b = 0; b < input_.batch(); ++b) {
      auto current_feature = input_.getBatchSlice(b, 1);

      if (num_sample == 0) {
        hidden_.setValue(b, 0, 0, i, std::numeric_limits<float>::min());
      } else {
        hidden_.setValue(b, 0, 0, i,
                         get_distance(current_feature, saved_feature));
      }
    }
  }
}

void CentroidKNN::calcDerivative() {
  throw std::invalid_argument("[CentroidKNN::calcDerivative] This Layer "
                              "does not support backward propagation");
}

} // namespace layers
} // namespace simpleshot
