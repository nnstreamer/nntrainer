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

/**
 * @brief Helper function to perform channel shuffle transpose operation
 * @param input Input tensor to transpose
 * @param output Output tensor to store transposed result
 * @param num_groups Number of groups for channel shuffle
 * @param channels_per_group Number of channels per group
 */
static void channel_shuffle_transpose(const Tensor &input, Tensor &output,
                                    unsigned int num_groups,
                                    unsigned int channels_per_group) {
  const TensorDim &dim = input.getDim();
  
  // Transpose operation: [N, G, C/G, H*W] -> [N, C/G, G, H*W]
  for (unsigned int n = 0; n < dim.batch(); ++n) {
    for (unsigned int g = 0; g < num_groups; ++g) {
      for (unsigned int c = 0; c < channels_per_group; ++c) {
        for (unsigned int hw = 0; hw < dim.width(); ++hw) {
          float val = input.getValue<float>(n, g, c, hw);
          output.setValue(n, c, g, hw, val);
        }
      }
    }
  }
}

ChannelShuffle::ChannelShuffle() : channel_shuffle_props(props::SplitNumber()) {}

void ChannelShuffle::finalize(InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Channel Shuffle layer takes only one input";

  const TensorDim &in_dim = context.getInputDimensions()[0];

  unsigned int num_groups = std::get<props::SplitNumber>(channel_shuffle_props).get();
  
  // If split_number is 0, find the smallest divisor of channel count that is greater than 1 and less than channel
  if (num_groups == 0) {
    unsigned int channel_count = in_dim.channel();
    for (unsigned int i = 2; i < channel_count; i++) {
      if (channel_count % i == 0) {
        num_groups = i;
        break;
      }
    }

    NNTR_THROW_IF(num_groups == 0, std::invalid_argument)
      << "Input split_number is 0, and channel count is prime number";

    std::get<props::SplitNumber>(channel_shuffle_props).set(num_groups);
  }

  // Validate split_number
  NNTR_THROW_IF(num_groups <= 1, std::invalid_argument)
    << "Number of groups must be greater than 1";
    
  NNTR_THROW_IF(num_groups >= in_dim.channel(), std::invalid_argument)
    << "Number of groups must be less than number of channels";

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

  unsigned int num_groups = std::get<props::SplitNumber>(channel_shuffle_props);
  unsigned int channels_per_group = input_.channel() / num_groups;

  // Calculate dimensions once before parallel section
  TensorDim group_dim = input_.getDim(); // [N, C, H, W]
  group_dim.width(group_dim.width() * group_dim.height());
  group_dim.height(channels_per_group);
  group_dim.channel(num_groups); // [N, G, C/G, H*W]
  group_dim.batch(1); // For batch slice

  TensorDim transposed_dim = group_dim; // [1, G, C/G, H*W]
  transposed_dim.channel(channels_per_group);
  transposed_dim.height(num_groups); // [1, C/G, G, H*W]

  TensorDim original_dim = hidden_.getDim(); // [N, C, H, W]
  original_dim.batch(1); // For batch slice

  auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                           void *user_data) {
    for (unsigned int b = s; b < e; ++b) {
      Tensor out = hidden_.getBatchSlice(b, 1);
      Tensor in_sub = input_.getBatchSlice(b, 1);

      // Step 1: Reshape into groups
      // [1, C, H, W] -> [1, G, C/G, H*W]
      in_sub.reshape(group_dim);

      // Step 2: Transpose groups
      // [1, G, C/G, H*W] -> [1, C/G, G, H*W]
      out.reshape(transposed_dim);
      channel_shuffle_transpose(in_sub, out, num_groups, channels_per_group);

      // Step 3: Reshape back to original dimensions
      // [1, C/G, G, H*W] -> [1, C, H, W]
      out.reshape(original_dim);
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
  /**
   * Channel Shuffle Derivative Operation:
   * 
   * Input Derivative Tensor: [N, C, H, W] where:
   * - N: batch size
   * - C: number of channels
   * - H: height
   * - W: width
   * 
   * Let's say we have:
   * - C = 12 channels
   * - G = 3 groups (specified by SplitNumber)
   * 
   * Forward operation:
   * Original:    [1,2,3,4,5,6,7,8,9,10,11,12]
   * After Step1: [[1, 2, 3, 4], 
   *               [5, 6, 7, 8], 
   *               [9,10,11,12]]
   * After Step2: [[1, 5, 9], 
   *               [2, 6,10], 
   *               [3, 7,11], 
   *               [4, 8,12]]
   * After Step3: [1,5,9,2,6,10,3,7,11,4,8,12]
   * 
   * Inverse operation (derivative):
   * Original:    [1,5,9,2,6,10,3,7,11,4,8,12]  (output from forward)
   * After Step1: [[1, 5, 9], 
   *               [2, 6,10], 
   *               [3, 7,11], 
   *               [4, 8,12]]
   * After Step2: [[1, 2, 3, 4], 
   *               [5, 6, 7, 8], 
   *               [9,10,11,12]]
   * After Step3: [1,2,3,4,5,6,7,8,9,10,11,12]  (back to original)
   * 
   * Step 1: Reshape into groups
   * [N, C, H, W] -> [N, C/G, G, H*W]
   * Example
   * Before: [N, 12, H, W]
   * After:  [N, 4, 3, H*W]
   * 
   * Step 2: Transpose groups (inverse of forward operation)
   * [N, C/G, G, H*W] -> [N, G, C/G, H*W]
   * Example
   * Before: [N, 4, 3, H*W]
   * After:  [N, 3, 4, H*W]
   * 
   * Step 3: Reshape back
   * [N, G, C/G, H*W] -> [N, C, H, W]
   * Example
   * Before: [N, 3, 4, H*W]
   * After:  [N, 12, H, W]
   */
  const Tensor &derivative = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &input_derivative = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  unsigned int num_groups = std::get<props::SplitNumber>(channel_shuffle_props);
  unsigned int channels_per_group = derivative.channel() / num_groups;

  // Calculate dimensions once before parallel section
  TensorDim group_dim = derivative.getDim();
  group_dim.width(group_dim.width() * group_dim.height());
  group_dim.height(num_groups);
  group_dim.channel(channels_per_group);  // First reshape to [N, C/G, G, H*W]
  group_dim.batch(1); // For batch slice
  
  TensorDim transposed_dim = group_dim; // [1, C/G, G, H*W]
  transposed_dim.height(channels_per_group);
  transposed_dim.channel(num_groups); // [1, G, C/G, H*W]
  
  TensorDim original_dim = input_derivative.getDim(); // [N, C, H, W]
  original_dim.batch(1); // For batch slice

  auto compute_derivative = [&](unsigned int s, unsigned int e, unsigned int pid,
                               void *user_data) {
    for (unsigned int b = s; b < e; ++b) {
      Tensor deriv_sub = derivative.getBatchSlice(b, 1);
      Tensor in_deriv_sub = input_derivative.getBatchSlice(b, 1);

      // Step 1: Reshape into groups
      // [1, C, H, W] -> [1, C/G, G, H*W]
      deriv_sub.reshape(group_dim);

      // Step 2: Transpose groups (inverse of forward operation)
      // [1, C/G, G, H*W] -> [1, G, C/G, H*W]
      in_deriv_sub.reshape(transposed_dim);
      channel_shuffle_transpose(deriv_sub, in_deriv_sub, num_groups, channels_per_group);

      // Step 3: Reshape back to original dimensions
      // [1, G, C/G, H*W] -> [1, C, H, W]
      in_deriv_sub.reshape(original_dim);
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