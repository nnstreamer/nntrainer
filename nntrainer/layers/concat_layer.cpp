// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   concat_layer.cpp
 * @date   27 Oct 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Concat Layer Class for Neural Network
 *
 * @todo merge concat and split layer to a common implementation
 */

#include <cstring>
#include <vector>

#include <concat_layer.h>
#include <layer_context.h>
#include <nntr_threads.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tensor_dim.h>
#include <util_func.h>

namespace nntrainer {
ConcatLayer::ConcatLayer() : Layer(), leading_helper_dim(1) {}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ConcatLayer::finalize(InitLayerContext &context) {
  auto &concat_dimension_prop = std::get<props::ConcatDimension>(concat_props);
  /** for backward compatibility, default concat dimension will be channel */
  /// @todo this is hacky way to force concat dimension to width if channel
  /// dimension is taken, this is because recurrent realizer, return sequence
  /// exploits concat layer but have no control over where to stack/axis
  unsigned int concat_dimension =
    context.getInputDimensions().front().channel() > 1 ? 3 : 1;
  if (!concat_dimension_prop.empty())
    concat_dimension = concat_dimension_prop.get();

  /**
   * The concat is only done along the axis dimension.
   * For example, consider 2 inputs a, b with dimensions [b,c,h,w] each
   * 1. concat_dimension = 1, output_dim = [b,c_a+c_b,h,w]
   * 2. concat_dimension = 2, output_dim = [b,c,h_a+h_b,w]
   * 3. concat_dimension = 3, output_dim = [b,c,h,w_a+w_b]
   */
  auto const &input_dims = context.getInputDimensions();
  const TensorDim &input_dim_0 = input_dims[SINGLE_INOUT_IDX];
  unsigned int concat_dim_val = input_dim_0.getTensorDim(concat_dimension);

  for (unsigned int idx = 1; idx < input_dims.size(); ++idx) {
    const TensorDim &dim = input_dims[idx];

    for (unsigned int i = 0; i < ml::train::TensorDim::getNumDim(); ++i) {
      if (i == concat_dimension)
        continue;
      NNTR_THROW_IF(input_dim_0[i] != dim[i], std::runtime_error)
        << "Error: concat layer requires same shape from all input layers "
           "along non-concat dimension";
    }
    concat_dim_val += dim[concat_dimension];
  }

  TensorDim output_dim = input_dim_0;
  output_dim.setTensorDim(concat_dimension, concat_dim_val);

  context.setOutputDimensions({output_dim});

  /**
   * Setup output_reshape_helper to which output will be reshaped in forwarding
   * to facilitate easier processing.
   *
   * The helper shape consolidates all the dimensions before the axis
   * together and all the dimensions after the axis to facilitate
   * easier splitting of the data.
   */
  leading_helper_dim = 1;
  output_reshape_helper.channel(1);
  output_reshape_helper.height(1);
  output_reshape_helper.width(1);
  for (unsigned int idx = 1; idx < concat_dimension; ++idx) {
    leading_helper_dim *= output_dim.getTensorDim(idx);
  }

  output_reshape_helper.height(output_dim.getTensorDim(concat_dimension));

  for (unsigned int idx = concat_dimension + 1;
       idx < ml::train::TensorDim::getNumDim(); ++idx) {
    output_reshape_helper.width(output_reshape_helper.width() *
                                output_dim.getTensorDim(idx));
  }

  /**
   * Setup input_reshape_helper to which inputs will be reshaped in forwarding
   * to facilitate easier processing.
   */
  input_reshape_helper.resize(input_dims.size());
  for (unsigned int idx = 0; idx < input_reshape_helper.size(); idx++) {
    input_reshape_helper[idx] = output_reshape_helper;
    input_reshape_helper[idx].height(
      input_dims[idx].getTensorDim(concat_dimension));
  }

  setBatch(input_dims[SINGLE_INOUT_IDX].batch());
}

void ConcatLayer::forwarding(RunLayerContext &context, bool training) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();
  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size = output_reshape_helper.width();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);

    /** loop over the dimensions before the concat dimension */
    for (unsigned int batch = 0; batch < output.batch(); batch++) {
      /** loop over the concat dimension itself */
      for (unsigned int count = 0; count < irh.height(); count++) {
        Tensor dest_tensor = Tensor::Map<float>(
          output.getAddress<float>(batch, 0, output_height_offset + count, 0),
          data_copy_size * sizeof(float), {1, 1, 1, data_copy_size});
        const Tensor source_tensor = Tensor::Map<float>(
          input.getAddress(batch, 0, count, 0), data_copy_size * sizeof(float),
          {1, 1, 1, data_copy_size});
        dest_tensor.copy(source_tensor);
      }
    }

    input.reshape(in_dim);
    output_height_offset += irh.height();
  }

  output.reshape(out_dim);
}

void ConcatLayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor &output = context.getOutput(SINGLE_INOUT_IDX);

  const TensorDim out_dim = output.getDim();
  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size = output_reshape_helper.width();

  // @todo: this implementation is only works when axis is 3(width). Consider
  // for other axes
  unsigned int batch_channel = out_dim.batch() * out_dim.channel();

  // std::vector<unsigned int > vec;
  unsigned int offset[100];

  unsigned int num_workers =
    NNTR_NUM_THREADS > context.getNumInputs() ? 1 : NNTR_NUM_THREADS;
  unsigned int chunk =
    (context.getNumInputs() + (num_workers - 1)) / num_workers;

  unsigned int sum = 0;
  offset[0] = 0;
  for (unsigned int i = 1; i < num_workers; ++i) {
    unsigned int from = (i - 1) * chunk;
    unsigned int to = i * chunk;

    for (unsigned int j = from; j < to; ++j) {
      sum += input_reshape_helper[j].height();
    }
    offset[i] = sum;
  }

  /**
   * Below sets the pad area values to zero
   * it is faster to do this way than seting selective area to zero
   */
  auto forwarding_job = [&](unsigned int s, unsigned int e, unsigned int pid,
                            void *user_data) {
    unsigned int output_height_offset = offset[pid];
    for (unsigned int idx = s; idx < e; idx++) {
      Tensor &input = context.getInput(idx);
      const TensorDim in_dim = input.getDim();
      auto const &irh = input_reshape_helper[idx];
      input.reshape(irh);

      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = batch_channel * from;
           batch < batch_channel * to; batch++) {
        /** loop over the concat dimension itself */
        for (unsigned int count = 0; count < irh.height(); count++) {
          Tensor dest_tensor = Tensor::Map(
            output.getAddress(batch, 0, output_height_offset + count, 0),
            data_copy_size * sizeof(float), {1, 1, 1, data_copy_size});
          const Tensor source_tensor = Tensor::Map(
            input.getAddress(batch, 0, count, 0),
            data_copy_size * sizeof(float), {1, 1, 1, data_copy_size});
          dest_tensor.copy(source_tensor);
        }
      }

      input.reshape(in_dim);
      output_height_offset += irh.height();
    }
  };

  auto workers = ParallelBatch(forwarding_job, context.getNumInputs(), nullptr);

  if (workers.getNumWorkers() > 1) {
    workers.run();
  } else {
    forwarding_job(0, context.getNumInputs(), 0, nullptr);
  }

  output.reshape(out_dim);
}

void ConcatLayer::calcDerivative(RunLayerContext &context) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor output = context.getIncomingDerivative(SINGLE_INOUT_IDX);

  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size = output_reshape_helper.width();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getOutgoingDerivative(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);

    /** loop over the dimensions before the concat dimension */
    for (unsigned int batch = 0; batch < output.batch(); batch++) {
      /** loop over the concat dimension itself */
      for (unsigned int count = 0; count < irh.height(); count++) {
        const Tensor source_tensor = Tensor::Map<float>(
          output.getAddress(batch, 0, output_height_offset + count, 0),
          data_copy_size * sizeof(float), {1, 1, 1, data_copy_size});
        Tensor dest_tensor = Tensor::Map<float>(input.getAddress(batch, 0, count, 0),
                                         data_copy_size * sizeof(float),
                                         {1, 1, 1, data_copy_size});
        dest_tensor.copy(source_tensor);
      }
    }

    input.reshape(in_dim);
    output_height_offset += irh.height();
  }
}

void ConcatLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, concat_props);
  NNTR_THROW_IF(!remain_props.empty(), std::invalid_argument)
    << "[ConcatLayer] Unknown Layer Properties count " +
         std::to_string(values.size());
}

void ConcatLayer::exportTo(Exporter &exporter,
                           const ml::train::ExportMethods &method) const {
  Layer::exportTo(exporter, method);
  exporter.saveResult(concat_props, method, this);
}

} /* namespace nntrainer */
