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

#include <activation_layer.h>
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
 * still derivative memory needs to be allocated
 */
std::vector<std::shared_ptr<Var_Grad>> &
Manager::TrackLayerInOuts(const std::string &layer_type,
                          const std::string &layer_name,
                          const std::vector<TensorDim> &input_dim) {
  int cnt = 0;
  auto base_name = layer_name + ":InOut";
  bool is_act_layer = layer_type == ActivationLayer::type;

  size_t inout_derivative_size = 0;

  std::vector<std::shared_ptr<Var_Grad>> in_out;
  in_out.reserve(input_dim.size());

  for (auto const &dim : input_dim) {
    in_out.emplace_back(
      std::make_shared<Var_Grad>(dim, true, base_name + std::to_string(cnt++)));
    if (is_act_layer)
      inout_derivative_size += dim.getDataLen();
  }

  in_outs.push_back(in_out);
  is_act_type.push_back(is_act_layer);

  max_derivative_size = std::max(max_derivative_size, inout_derivative_size);
  return in_outs.back();
}

void Manager::untrackLayerInOuts(const std::string layer_name) {
  auto var_name = layer_name + ":InOut" + std::to_string(0);

  for (unsigned int cnt = 0; cnt < in_outs.size(); cnt++) {
    if (!in_outs[cnt].empty() && in_outs[cnt][0]->getName() == var_name) {
      in_outs.erase(in_outs.begin() + cnt);
      is_act_type.erase(is_act_type.begin() + cnt);
      break;
    }
  }
}

/**
 * @brief Initialize the inputs/outputs for the layer
 */
void Manager::initializeInOuts(bool trainable) {
  Tensor shared_deriv;
  if (max_derivative_size > 0 && enable_activation_memory_opt)
    shared_deriv = Tensor(max_derivative_size);

  size_t count = 0;
  for (unsigned int idx = 0; idx < in_outs.size(); idx++) {
    auto &l_io = in_outs[idx];
    size_t offset = 0;
    bool is_last_layer = idx == in_outs.size() - 1;
    for (auto &io : l_io) {
      if (enable_derivative_memory_opt && !is_last_layer) {
        if (is_act_type[count] && enable_activation_memory_opt) {
          io->initialize(
            shared_deriv.getSharedDataTensor(io->getDim(), offset));
          offset += io->getDim().getDataLen();
        } else {
          io->initializeShared();
        }
      } else {
        if (is_last_layer)
          io->initialize(Tensor(), true);
        else
          io->initialize(Tensor(), trainable);
      }
    }
    count += 1;
  }
}

} // namespace nntrainer
