// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghoon Kang <dhkang01@snu.ac.kr>
 *
 * @file   channel_shuffle.cpp
 * @date   23 April 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghoon Kang <dhkang01@snu.ac.kr>
 * @bug    No known bugs except for NYI items
 * @brief  This is Channel Shuffle Layer Class for Neural Network
 *
 */

#include <algorithm>
#include <cstring>
#include <limits>
#include <string>

#include <channel_shuffle.h>
#include <cpu_backend.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <profiler.h>
#include <tensor_dim.h>
#include <thread>
#include <util_func.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

ChannelShuffle::ChannelShuffle() : channel_shuffle_props(props::SplitNumber()) {}

void ChannelShuffle::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Channel Shuffle layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  unsigned int num_groups = std::get<props::SplitNumber>(channel_shuffle_props);

  NNTR_THROW_IF(in_dim.channel() % num_groups != 0, std::invalid_argument)
    << "Number of channels must be divisible by number of groups";

  // Output dimensions are same as input dimensions
  context.setOutputDimensions({in_dim});
}

void ChannelShuffle::forwarding(RunLayerContext &context, bool training) {
  /**
 * Channel Shuffle Operation:
 * 
 * Input Tensor: [N, C, H, W] where:
 * - N: batch size
 * - C: number of channels
 * - H: height
 * - W: width
 * 
 * Let's say we have:
 * - C = 12 channels
 * - G = 3 groups (specified by SplitNumber)
 * 
 * Step 1: Reshape into groups
 * [N, C, H, W] -> [N, G, C/G, H, W]
 * Example
 * Before: [N, 12, H, W]
 * After:  [N, 3, 4, H, W]
 * 
 * Step 2: Transpose groups
 * [N, G, C/G, H, W] -> [N, C/G, G, H, W]
 * Example
 * Before: [N, 3, 4, H, W]
 * After:  [N, 4, 3, H, W]
 * 
 * Step 3: Reshape back
 * [N, C/G, G, H, W] -> [N, C, H, W]
 * Example
 * Before: [N, 4, 3, H, W]
 * After:  [N, 12, H, W]
 * 
 * Visualization:
 * Original:    [1,2,3,4,5,6,7,8,9,10,11,12]
 * After Step1: [[1, 2, 3, 4], 
 *               [5, 6, 7, 8], 
 *               [9,10,11,12]]
 * After Step2: [[1, 5, 9], 
 *               [2, 6,10], 
 *               [3, 7,11], 
 *               [4, 8,12]]
 * After Step3: [1,5,9,2,6,10,3,7,11,4,8,12]
   */
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                           void *user_data) {
    for (unsigned int b = s; b < e; ++b) {
      Tensor out = hidden_.getBatchSlice(b, 1);
      Tensor in_sub = input_.getBatchSlice(b, 1);
      //TODO: Implement channel shuffle operation
    }
  };

  auto workers = ParallelBatch(forwarding_job, input_.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    forwarding_job(0, input_.batch(), 0, nullptr);
  }
}

void ChannelShuffle::calcDerivative(RunLayerContext &context) {
  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  auto compute_derivative = [&](unsigned int s, unsigned int e, unsigned int pid,
                               void *user_data) {
    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);
      //TODO: Implement channel shuffle operation
    }
  };

  auto workers = ParallelBatch(compute_derivative, derivative.batch(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    compute_derivative(0, derivative.batch(), 0, nullptr);
  }
}

void ChannelShuffle::calcGradient(RunLayerContext &context) {
  // Channel Shuffle layer has no weights to update
  // No gradient calculation needed
}

void ChannelShuffle::exportTo(Exporter &exporter,
                            const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(channel_shuffle_props, method, this);
}

void ChannelShuffle::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, channel_shuffle_props);
  LayerImpl::setProperty(remain_props);
}

} // namespace nntrainer 