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

int TfLiteLayer::initialize(Manager &manager) {
  tflite::ops::builtin::BuiltinOpResolver resolver;

  model = tflite::FlatBufferModel::BuildFromFile(modelfile.c_str());
  if (!model)
    return ML_ERROR_INVALID_PARAMETER;

  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);
  if (!interpreter)
    return ML_ERROR_INVALID_PARAMETER;

  if (interpreter->AllocateTensors() != kTfLiteOk)
    throw std::runtime_error("Failed to allocate tensors!");

  std::vector<TensorDim> dims;
  setDimensions(interpreter->inputs(), dims, false);
  setDimensions(interpreter->outputs(), output_dim, true);

  if (input_dim.size() && input_dim[0].getTensorDim(0) != 0) {
    if (dims.size() != input_dim.size())
      throw std::invalid_argument(
        "Provided number of input dimensions mismatch");

    for (size_t idx = 0; idx < dims.size(); idx++) {
      if (dims[idx] != input_dim[idx])
        throw std::invalid_argument("Input dimensions mismatch");
    }
  } else {
    input_dim.resize(dims.size());
    std::copy(dims.begin(), dims.end(), input_dim.begin());
  }

  return ML_ERROR_NONE;
}

void TfLiteLayer::setProperty(const PropertyType type,
                              const std::string &value) {
  switch (type) {
  case PropertyType::modelfile: {
    if (!value.empty())
      modelfile = value;
  } break;
  default:
    LayerV1::setProperty(type, value);
    break;
  }
}

void TfLiteLayer::forwarding(bool training) {
#ifdef DEBUG
  if (net_input.size() != input_dim.size())
    throw std::invalid_argument("Provided number of input dimensions mismatch");

  for (unsigned int idx = 0; idx < input_dim.size(); idx++) {
    if (net_input[idx]->getDim() != input_dim[idx])
      throw std::invalid_argument("Input dimensions mismatch");
  }
#endif
  auto in_indices = interpreter->inputs();
  for (size_t idx = 0; idx < net_input.size(); idx++)
    interpreter->tensor(in_indices[idx])->data.raw =
      (char *)net_input[idx]->getVariableRef().getData();

  auto out_indices = interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(net_hidden[idx]->getVariableRef().getData());
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

void TfLiteLayer::copy(std::shared_ptr<LayerV1> l) {
  LayerV1::copy(l);

  std::shared_ptr<TfLiteLayer> from = std::static_pointer_cast<TfLiteLayer>(l);
  this->modelfile = from->modelfile;
}

void TfLiteLayer::calcDerivative() {
  throw exception::not_supported(
    "calcDerivative is not supported for tflite layer");
}
} /* namespace nntrainer */
