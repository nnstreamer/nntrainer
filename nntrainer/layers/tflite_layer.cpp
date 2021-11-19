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

#include <base_properties.h>
#include <layer_context.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <tflite_layer.h>
#include <util_func.h>

namespace nntrainer {

/**
 * @brief TflModelPath property
 *
 */
class PropsTflModelPath : public Property<std::string> {
public:
  static constexpr const char *key = "model_path"; /**< unique key to access */
  using prop_tag = str_prop_tag;                   /**< property type */

  static constexpr const char ending[] = ".tflite";
  static constexpr unsigned int ending_len = 7;
  /**
   * @brief check is valid
   *
   * @param v value to check
   * @return bool true if valid
   */
  bool isValid(const std::string &v) const override;
};

bool PropsTflModelPath::isValid(const std::string &v) const {
  if (v.size() < ending_len) {
    return false;
  }
  std::string ext(v.end() - ending_len, v.end());
  std::for_each(ext.end() - ending_len, ext.end(),
                [](char &c) { c = ::tolower(c); });

  /// check if path ends with .tflite
  if (!endswith(ext, ending)) {
    return false;
  }
  std::ifstream file(v, std::ios::binary | std::ios::ate);
  return file.good();
}

TfLiteLayer::TfLiteLayer() :
  Layer(),
  tfl_layer_props(new PropsType(PropsTflModelPath())),
  interpreter(nullptr),
  model(nullptr) {}

TfLiteLayer::~TfLiteLayer() {}

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
    if (num_dims > ml::train::TensorDim::MAXDIM)
      throw exception::not_supported("Number of dimensions exceed the support");

    /** This puts the unused dimensions to the outer dimensions */
    for (size_t dim_idx = 0; dim_idx < num_dims; dim_idx++)
      dim[i].setTensorDim(
        ml::train::TensorDim::MAXDIM - dim_idx - 1,
        interpreter->tensor(tensor_idx)->dims->data[num_dims - dim_idx - 1]);
  }
}

void TfLiteLayer::finalize(InitLayerContext &context) {
  tflite::ops::builtin::BuiltinOpResolver resolver;

  model = tflite::FlatBufferModel::BuildFromFile(
    std::get<PropsTflModelPath>(*tfl_layer_props).get().c_str());
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
  auto left_values = loadProperties(values, *tfl_layer_props);
  NNTR_THROW_IF(!left_values.empty(), std::invalid_argument)
    << "There are unparsed properties, " << left_values.front();
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
  if (out_tf_dim.size() != context.getNumOutputs()) {
    throw std::invalid_argument(
      "[TfliteLayer::forward] number of output dimension does not match");
  }

  for (unsigned int i = 0; i < out_tf_dim.size(); ++i) {
    if (context.getOutput(i).getDim() != out_tf_dim[i]) {
      throw std::invalid_argument(
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
