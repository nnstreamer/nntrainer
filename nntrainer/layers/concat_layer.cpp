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

  bool is_nchw = (context.getFormat() == Tformat::NCHW);

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
  if (!is_nchw)
    output_reshape_helper.setFormat(Tformat::NHWC);

  output_reshape_helper.channel(1);
  output_reshape_helper.height(1);
  output_reshape_helper.width(1);

  std::vector<unsigned int> dims_order;
  if (is_nchw)
    dims_order = {0, 1, 2, 3};
  else
    dims_order = {0, 2, 3, 1};

  for (unsigned int idx = 1; idx < 4; ++idx) {
    if (dims_order[idx] == concat_dimension)
      break;
    leading_helper_dim *= output_dim.getTensorDim(dims_order[idx]);
  }

  is_nchw
    ? output_reshape_helper.height(output_dim.getTensorDim(concat_dimension))
    : output_reshape_helper.width(output_dim.getTensorDim(concat_dimension));

  auto dim_pos =
    std::find(dims_order.begin(), dims_order.end(), concat_dimension);

  for (dim_pos++; dim_pos < dims_order.end(); dim_pos++) {
    unsigned int idx = *dim_pos;
    is_nchw ? output_reshape_helper.width(output_reshape_helper.width() *
                                          output_dim.getTensorDim(idx))
            : output_reshape_helper.channel(output_reshape_helper.channel() *
                                            output_dim.getTensorDim(idx));
  }

  /**
   * Setup input_reshape_helper to which inputs will be reshaped in forwarding
   * to facilitate easier processing.
   */
  input_reshape_helper.resize(input_dims.size());
  for (unsigned int idx = 0; idx < input_reshape_helper.size(); idx++) {
    input_reshape_helper[idx] = output_reshape_helper;
    is_nchw ? input_reshape_helper[idx].height(
                input_dims[idx].getTensorDim(concat_dimension))
            : input_reshape_helper[idx].width(
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

  bool is_nchw = (output.getFormat() == Tformat::NCHW);

  const TensorDim out_dim = output.getDim();
  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size =
    is_nchw ? output_reshape_helper.width() : output_reshape_helper.channel();
  TensorDim::TensorType tensor_type = out_dim.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);

    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** loop over the concat dimension itself */
        for (unsigned int count = 0;
             count < (is_nchw ? irh.height() : irh.width()); count++) {
          Tensor dest_tensor = Tensor::Map<float>(
            output.getAddress<float>(
              batch, 0, is_nchw ? (output_height_offset + count) : 0,
              is_nchw ? 0 : output_height_offset + count),
            data_copy_size * sizeof(float),
            {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
             tensor_type});
          const Tensor source_tensor = Tensor::Map<float>(
            input.getAddress<float>(batch, 0, is_nchw ? count : 0,
                                    is_nchw ? 0 : count),
            data_copy_size * sizeof(float),
            {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
             tensor_type});
          dest_tensor.copy(source_tensor);
        }
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** loop over the concat dimension itself */
        for (unsigned int count = 0;
             count < (is_nchw ? irh.height() : irh.width()); count++) {
          Tensor dest_tensor = Tensor::Map<_FP16>(
            output.getAddress<_FP16>(
              batch, 0, is_nchw ? (output_height_offset + count) : 0,
              is_nchw ? 0 : output_height_offset + count),
            data_copy_size * sizeof(_FP16),
            {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
             tensor_type});
          const Tensor source_tensor = Tensor::Map<_FP16>(
            input.getAddress<_FP16>(batch, 0, is_nchw ? count : 0,
                                    is_nchw ? 0 : count),
            data_copy_size * sizeof(_FP16),
            {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
             tensor_type});
          dest_tensor.copy(source_tensor);
        }
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    input.reshape(in_dim);
    output_height_offset += is_nchw ? irh.height() : irh.width();
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

  bool is_nchw = (output.getFormat() == Tformat::NCHW);

  const TensorDim out_dim = output.getDim();
  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size =
    is_nchw ? output_reshape_helper.width() : output_reshape_helper.channel();
  TensorDim::TensorType tensor_type = out_dim.getTensorType();

  // @todo: this implementation is only works when axis is 3(width). Consider
  // for other axes
  unsigned int batch_channel = out_dim.batch() * out_dim.channel();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getInput(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);

    /** loop over the dimensions before the concat dimension */
    for (unsigned int batch = batch_channel * from; batch < batch_channel * to;
         batch++) {
      /** loop over the concat dimension itself */
      for (unsigned int count = 0;
           count < (is_nchw ? irh.height() : irh.width()); count++) {
        Tensor dest_tensor =
          Tensor::Map(output.getAddress(
                        batch, 0, is_nchw ? (output_height_offset + count) : 0,
                        is_nchw ? 0 : output_height_offset + count),
                      data_copy_size * sizeof(float),
                      {1, is_nchw ? 1 : data_copy_size, 1,
                       is_nchw ? data_copy_size : 1, tensor_type});
        const Tensor source_tensor = Tensor::Map(
          input.getAddress(batch, 0, is_nchw ? count : 0, is_nchw ? 0 : count),
          data_copy_size * sizeof(float),
          {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
           tensor_type});
        dest_tensor.copy(source_tensor);
      }
    }

    input.reshape(in_dim);
    output_height_offset += is_nchw ? irh.height() : irh.width();
  }

  output.reshape(out_dim);
}

void ConcatLayer::calcDerivative(RunLayerContext &context) {
  /**
   * @todo avoid copy by creating input here as a shared_tensor of the output
   * here and then this layer can be in_place as well
   */
  Tensor output = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  bool is_nchw = (output.getFormat() == Tformat::NCHW);

  output.reshape(output_reshape_helper);
  unsigned int output_height_offset = 0;
  unsigned int data_copy_size =
    is_nchw ? output_reshape_helper.width() : output_reshape_helper.channel();
  TensorDim::TensorType tensor_type = output.getTensorType();

  for (unsigned int idx = 0; idx < context.getNumInputs(); idx++) {
    Tensor &input = context.getOutgoingDerivative(idx);
    const TensorDim in_dim = input.getDim();
    auto const &irh = input_reshape_helper[idx];
    input.reshape(irh);

    if (in_dim.getDataType() == TensorDim::DataType::FP32) {
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** loop over the concat dimension itself */
        for (unsigned int count = 0;
             count < (is_nchw ? irh.height() : irh.width()); count++) {
          const Tensor source_tensor = Tensor::Map<float>(
            output.getAddress<float>(
              batch, 0, is_nchw ? (output_height_offset + count) : 0,
              is_nchw ? 0 : output_height_offset + count),
            data_copy_size * sizeof(float),
            {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
             tensor_type});
          Tensor dest_tensor = Tensor::Map<float>(
            input.getAddress<float>(batch, 0, is_nchw ? count : 0,
                                    is_nchw ? 0 : count),
            data_copy_size * sizeof(float),
            {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
             tensor_type});
          dest_tensor.copy(source_tensor);
        }
      }
    } else if (in_dim.getDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      /** loop over the dimensions before the concat dimension */
      for (unsigned int batch = 0; batch < output.batch(); batch++) {
        /** loop over the concat dimension itself */
        for (unsigned int count = 0;
             count < (is_nchw ? irh.height() : irh.width()); count++) {
          const Tensor source_tensor = Tensor::Map<_FP16>(
            output.getAddress<_FP16>(
              batch, 0, is_nchw ? (output_height_offset + count) : 0,
              is_nchw ? 0 : output_height_offset + count),
            data_copy_size * sizeof(_FP16),
            {1, is_nchw ? 1 : data_copy_size, 1, is_nchw ? data_copy_size : 1,
             tensor_type});
          Tensor dest_tensor =
            Tensor::Map<_FP16>(input.getAddress<_FP16>(batch, 0, count, 0),
                               data_copy_size * sizeof(_FP16),
                               {1, is_nchw ? 1 : data_copy_size, 1,
                                is_nchw ? data_copy_size : 1, tensor_type});
          dest_tensor.copy(source_tensor);
        }
      }
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    }

    input.reshape(in_dim);
    output_height_offset += is_nchw ? irh.height() : irh.width();
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
