// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file	nnstreamer_layer.cpp
 * @date	26 October 2020
 * @brief	This is class to encapsulate nnstreamer as a layer of Neural Network
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * @todo: provide input/output dimensions to nnstreamer for certain frameworks
 * @todo: support transposing the data to support NCHW nntrainer data to NHWC
 * nnstreamer data
 */

#include <lazy_tensor.h>
#include <nnstreamer_layer.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <parse_util.h>
#include <util_func.h>

namespace nntrainer {

const std::string NNStreamerLayer::type = "backbone_nnstreamer";

int NNStreamerLayer::nnst_info_to_tensor_dim(ml_tensors_info_h &out_res,
                                             TensorDim &dim) {
  int status = ML_ERROR_NONE;
  unsigned int count;
  ml_tensor_type_e type;
  ml_tensor_dimension dim_;

  status = ml_tensors_info_get_count(out_res, &count);
  if (status != ML_ERROR_NONE)
    return status;

  if (count > 1)
    return ML_ERROR_NOT_SUPPORTED;

  status = ml_tensors_info_get_tensor_type(out_res, 0, &type);
  if (status != ML_ERROR_NONE)
    return status;

  if (type != ML_TENSOR_TYPE_FLOAT32)
    return ML_ERROR_NOT_SUPPORTED;

  if (ML_TENSOR_RANK_LIMIT > MAXDIM)
    return ML_ERROR_NOT_SUPPORTED;

  status = ml_tensors_info_get_tensor_dimension(out_res, 0, dim_);
  if (status != ML_ERROR_NONE)
    return status;

  for (size_t i = 0; i < MAXDIM; i++)
    dim.setTensorDim(i, dim_[i]);

  /* reverse the dimension as nnstreamer stores dimension in reverse way */
  dim.reverse();

  return status;
}

NNStreamerLayer::~NNStreamerLayer() { finalizeError(ML_ERROR_NONE); }

int NNStreamerLayer::finalizeError(int status) {
  if (in_res) {
    ml_tensors_info_destroy(in_res);
    in_res = nullptr;
  }

  if (out_res) {
    ml_tensors_info_destroy(out_res);
    out_res = nullptr;
  }

  if (in_data_cont) {
    ml_tensors_data_destroy(in_data_cont);
    in_data_cont = nullptr;
  }

  if (out_data_cont) {
    ml_tensors_data_destroy(out_data_cont);
    out_data_cont = nullptr;
  }

  if (single) {
    ml_single_close(single);
    single = nullptr;
  }

  return status;
}

int NNStreamerLayer::initialize(Manager &manager) {
  int status = ML_ERROR_NONE;
  TensorDim in_dim;

  status = ml_single_open(&single, modelfile.c_str(), NULL, NULL,
                          ML_NNFW_TYPE_ANY, ML_NNFW_HW_AUTO);
  if (status != ML_ERROR_NONE)
    return finalizeError(status);

  /* input tensor in filter */
  status = ml_single_get_input_info(single, &in_res);
  if (status != ML_ERROR_NONE)
    return finalizeError(status);

  status = nnst_info_to_tensor_dim(in_res, in_dim);
  if (status != ML_ERROR_NONE)
    return finalizeError(status);

  if (input_dim[0].getTensorDim(0) != 0 && input_dim[0] != in_dim) {
    ml_loge("Set tensor info does not match info from the framework");
    return finalizeError(ML_ERROR_INVALID_PARAMETER);
  } else {
    input_dim[0] = in_dim;
  }

  /* input tensor in filter */
  status = ml_single_get_output_info(single, &out_res);
  if (status != ML_ERROR_NONE)
    return finalizeError(status);

  status = nnst_info_to_tensor_dim(out_res, output_dim[0]);
  if (status != ML_ERROR_NONE)
    return finalizeError(status);

  /* generate input data container */
  status = ml_tensors_data_create(in_res, &in_data_cont);
  if (status != ML_ERROR_NONE)
    return finalizeError(status);

  size_t in_data_size;
  status =
    ml_tensors_data_get_tensor_data(in_data_cont, 0, &in_data, &in_data_size);
  if (status != ML_ERROR_NONE)
    return finalizeError(status);

  if (in_data_size != input_dim[0].getDataLen() * sizeof(float))
    return finalizeError(ML_ERROR_INVALID_PARAMETER);

  return status;
}

void NNStreamerLayer::setTrainable(bool train) {
  if (train)
    throw exception::not_supported(
      "NNStreamer layer does not support training");

  Layer::setTrainable(false);
}

void NNStreamerLayer::setProperty(const PropertyType type,
                                  const std::string &value) {
  switch (type) {
  case PropertyType::modelfile: {
    if (!value.empty())
      modelfile = value;
  } break;
  default:
    Layer::setProperty(type, value);
    break;
  }
}

void NNStreamerLayer::forwarding(sharedConstTensors in) {
  size_t data_size;
  Tensor input = *in[0];
  Tensor &hidden_ = net_hidden[0]->var;

  std::copy(input.getData(), input.getData() + input.length(),
            (float *)in_data);

  int status = ml_single_invoke(single, in_data_cont, &out_data_cont);
  if (status != ML_ERROR_NONE)
    throw std::runtime_error("Failed to forward nnstreamer backbone");

  status =
    ml_tensors_data_get_tensor_data(out_data_cont, 0, &out_data, &data_size);
  if (status != ML_ERROR_NONE) {
    ml_tensors_data_destroy(out_data_cont);
    out_data_cont = nullptr;
    throw std::runtime_error("Failed to forward nnstreamer backbone");
  }

  if (data_size != hidden_.getSize()) {
    ml_tensors_data_destroy(out_data_cont);
    out_data_cont = nullptr;
    throw std::runtime_error("Output size mismatch from nnstreamer backbone.");
  }

  std::copy((float *)out_data, (float *)((char *)out_data + data_size),
            hidden_.getData());
}

void NNStreamerLayer::copy(std::shared_ptr<Layer> l) {
  Layer::copy(l);

  std::shared_ptr<NNStreamerLayer> from =
    std::static_pointer_cast<NNStreamerLayer>(l);
  this->modelfile = from->modelfile;
}

void NNStreamerLayer::calcDerivative(sharedConstTensors derivative) {
  throw exception::not_supported(
    "calcDerivative is not supported for nnstreamer layer");
}
} /* namespace nntrainer */
