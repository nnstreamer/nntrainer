// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   tflite_layer.cpp
 * @date   26 October 2020
 * @brief  This is class to encapsulate tflite as a layer of Neural Network
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <tflite_layer.h>

namespace nntrainer {

void TfLiteLayer::setDimensions(const std::vector<int> &tensor_idx_list,
                                std::vector<TensorDim> &dim, bool is_output) {
  unsigned int num_tensors = tensor_idx_list.size();
  dim.resize(num_tensors);

  for (unsigned int i = 0; i < num_tensors; i++) {
    unsigned int tensor_idx = tensor_idx_list[i];
    if (is_output && interpreter->tensor(tensor_idx)->type != kTfLiteFloat32)
      throw exception::not_supported(
        "Data type other than float32 not supported");

    unsigned int num_dims = interpreter->tensor(tensor_idx)->dims->size;
    if (num_dims > MAXDIM)
      throw exception::not_supported("Number of dimensions exceed the support");

    /** This puts the unused dimensions to the outer dimensions */
    for (size_t dim_idx = 0; dim_idx < num_dims; dim_idx++)
      dim[i].setTensorDim(
        MAXDIM - dim_idx - 1,
        interpreter->tensor(tensor_idx)->dims->data[num_dims - dim_idx - 1]);
  }
}

void TfLiteLayer::finalize(InitLayerContext &context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;

  model = tflite::FlatBufferModel::BuildFromFile(modelfile.c_str());
  if (!model)
    throw std::invalid_argument("Failed to build tflite model");

  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
  if (!interpreter)
    throw std::invalid_argument("Failed to build tflite interpreter");

  if (interpreter->AllocateTensors() != kTfLiteOk)
    throw std::runtime_error("Failed to allocate tensors!");

  std::vector<TensorDim> dims;
  setDimensions(interpreter->inputs(), dims, false);
  const std::vector<TensorDim> &input_dims = context.getInputDimensions();

  if (dims.size() != input_dims.size())
    throw std::invalid_argument("Provided number of input dimensions mismatch");

  for (size_t idx = 0; idx < dims.size(); idx++) {
    if (dims[idx] != input_dims[idx])
      throw std::invalid_argument("Input dimensions mismatch");
  }

  std::vector<TensorDim> output_dims;
  setDimensions(interpreter->outputs(), output_dims, true);
  context.setOutputDimensions(output_dims);
}

void TfLiteLayer::setProperty(const std::vector<std::string> &values) {
  /// @todo: deprecate this in favor of loadProperties
  for (unsigned int i = 0; i < values.size(); ++i) {
    std::string key;
    std::string value;
    std::stringstream ss;

    if (getKeyValue(values[i], key, value) != ML_ERROR_NONE) {
      throw std::invalid_argument("Error parsing the property: " + values[i]);
    }

    if (value.empty()) {
      ss << "value is empty: key: " << key << ", value: " << value;
      throw std::invalid_argument(ss.str());
    }

    /// @note this calls derived setProperty if available
    setProperty(key, value);
  }
}

void TfLiteLayer::setProperty(const std::string &type_str,
                              const std::string &value) {
  using PropertyType = nntrainer::Layer::PropertyType;
  nntrainer::Layer::PropertyType type =
    static_cast<nntrainer::Layer::PropertyType>(parseLayerProperty(type_str));

  switch (type) {
  case PropertyType::modelfile: {
    modelfile = value;
  } break;
  default:
    std::string msg = "[TfLiteLayer] Unknown Layer Property Key for value " +
                      std::string(value);
    throw exception::not_supported(msg);
  }
}

void TfLiteLayer::forwarding(RunLayerContext &context, bool training) {
  auto in_indices = interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++)
    interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(context.getInput(idx).getData());

  auto out_indices = interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(context.getOutput(idx).getData());
  }

  int status = interpreter->Invoke();
  if (status != kTfLiteOk)
    throw std::runtime_error("Invoke failed");

#ifdef DEBUG
  std::vector<TensorDim> out_tf_dim;
  setDimensions(interpreter->outputs(), out_tf_dim, true);
  if (out_tf_dim.size() != output_dim.size()) {
    throw std::invalid_argument(
      "[TfliteLayer::forward] number of output dimension does not match");
  }

  for (unsigned int i = 0; i < out_tf_dim.size(); ++i) {
    if (output_dim[i] != out_tf_dim[i]) {
      throw std::invalid_argumetns(
        "[TfliteLayer::forward] output dimension does not match");
    }
  }
#endif
}

void TfLiteLayer::calcDerivative(RunLayerContext &context) {
  throw exception::not_supported(
    "calcDerivative is not supported for tflite layer");
}
} /* namespace nntrainer */
