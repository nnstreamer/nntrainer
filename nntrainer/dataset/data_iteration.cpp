// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   data_iteration.cpp
 * @date   11 Aug 2021
 * @brief  This file contains iteration and sample class
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */
#include <data_iteration.h>

#include <algorithm>

#include <nntrainer_error.h>
#include <tensor.h>
#include <tensor_dim.h>

namespace nntrainer {

namespace {

/**
 * @brief check if all the dimension has the same batch, this is required
 * assumption for the creation of Iteration
 *
 */
bool isBatchSame(const std::vector<ml::train::TensorDim> &input_dims,
                 const std::vector<ml::train::TensorDim> &label_dims) {
  if (input_dims.empty()) {
    /// requires at least one input
    return false;
  }

  unsigned int reference_batch = input_dims.front().batch();
  auto pred = [reference_batch](const TensorDim &dim) {
    return dim.batch() == reference_batch;
  };

  return std::all_of(input_dims.begin(), input_dims.end(), pred) &&
         std::all_of(label_dims.begin(), label_dims.end(), pred);
}

/**
 * @brief slice vectors of tensors in to batch direction
 *
 * @param batched_tensors batched tensor
 * @param b  batch
 * @return std::vector<Tensor> sliced tensor
 */
std::vector<Tensor> sliceTensor(const std::vector<Tensor> &batched_tensors,
                                unsigned int b) {
  std::vector<Tensor> sliced_tensor;
  sliced_tensor.reserve(batched_tensors.size());
  std::transform(batched_tensors.begin(), batched_tensors.end(),
                 std::back_inserter(sliced_tensor),
                 [b](const Tensor &t) { return t.getBatchSlice(b, 1); });
  return sliced_tensor;
};

std::vector<Sample> unpackIteration(Iteration &iter) {
  auto b = iter.batch();

  std::vector<Sample> samples;
  samples.reserve(b);

  for (decltype(b) i = 0; i < b; ++i) {
    samples.emplace_back(iter, i);
  }

  return samples;
}

} // namespace

Iteration::Iteration(const std::vector<ml::train::TensorDim> &input_dims,
                     const std::vector<ml::train::TensorDim> &label_dims) :
  inputs(input_dims.begin(), input_dims.end()),
  labels(label_dims.begin(), label_dims.end()) {

  NNTR_THROW_IF(!isBatchSame(input_dims, label_dims), std::invalid_argument)
    << "check batch size is all the same for all the input and label";

  samples = unpackIteration(*this);
}

Sample::Sample(const Iteration &iter, unsigned int batch) :
  inputs(sliceTensor(iter.getInputsRef(), batch)),
  labels(sliceTensor(iter.getLabelsRef(), batch)) {}

} // namespace nntrainer
