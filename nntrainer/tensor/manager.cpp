// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	manager.cpp
 * @date	2 Dec 2020
 * @brief	This is NNtrainer manager for all weights, i/o and intermediate
 * tensors
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <functional>
#include <vector>

#include <manager.h>

namespace nntrainer {

/**
 * @brief     Add weight to be tracked and updated with nntrainer
 */
void Manager::trackWeight(std::reference_wrapper<Weight> w) {
  std::vector<std::reference_wrapper<Weight>> temp = {w};
  weights.emplace_back(temp);
}

/**
 * @brief     Add weights to be tracked and updated with nntrainer
 */
void Manager::trackWeights(std::vector<Weight> &ws) {
  std::vector<std::reference_wrapper<Weight>> layer_weights;
  layer_weights.reserve(ws.size());

  size_t weight_size = 0;

  for (auto &w : ws) {
    layer_weights.emplace_back(std::ref(w));
    if (w.getTrainable())
      weight_size += w.getDim().getDataLen();
  }

  weights.push_back(layer_weights);

  max_weight_size = std::max(max_weight_size, weight_size);
}

/**
 * @brief Allocate and initialize the weight variable
 */
void Manager::initialize() {
  Tensor shared_grad;
  if (max_weight_size > 0 && enable_gradient_memory_opt)
    shared_grad = Tensor(max_weight_size);

  for (auto &l_w : weights) {
    size_t offset = 0;
    for (auto &w : l_w) {
      Weight &weight = w.get();
      if (weight.getTrainable() && enable_gradient_memory_opt) {
        weight.initialize(
          shared_grad.getSharedDataTensor(weight.getDim(), offset));
        offset += weight.getDim().getDataLen();
      } else {
        weight.initialize();
      }
    }
  }
}

/**
 * @brief Track the inputs/ouputs of the layer
 */
void Manager::TrackLayerInOuts(const std::string layer_name,
                               const std::vector<TensorDim> &input_dim) {
  int cnt = 0;
  auto base_name = layer_name + ":Input";

  std::vector<std::shared_ptr<Var_Grad>> in_out;
  in_out.reserve(input_dim.size());

  for (auto const &dim : input_dim) {
    in_out.emplace_back(std::make_shared<Var_Grad>(
      dim, false, base_name + std::to_string(cnt++)));
  }

  in_outs.push_back(in_out);
}

} // namespace nntrainer
