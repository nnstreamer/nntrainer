// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 DongHak Park <donghak.park@samsung.com>
 *
 * @file flatbuffer_interpreter.cpp
 * @date 09 February 2023
 * @brief NNTrainer *.flatbuffer Interpreter
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <flatbuffer_interpreter.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <utility>

#include <app_context.h>
#include <bn_realizer.h>
#include <execution_mode.h>
#include <fc_layer.h>
#include <flatbuffer_opnode.h>
#include <layer.h>
#include <layer_node.h>
#include <loss_realizer.h>
#include <model.h>
#include <network_graph.h>
#include <nntrainer_error.h>
#include <nntrainer_schema_generated.h>
#include <node_exporter.h>
#include <optimizer.h>
#include <tensor.h>

#include "flatbuffers/util.h"

static constexpr const char *FUNC_TAG = "[FLATBUFFER INTERPRETER]";
namespace nntrainer {

namespace {
/**
 * @brief After finishing building, call this to save to a file
 *
 * @param builder flatbuffer builder
 * @param out out
 */
void builder2file(const flatbuffers::FlatBufferBuilder &builder,
                  const std::string &out) {
  uint8_t *buf = builder.GetBufferPointer();
  size_t size = builder.GetSize();
  flatbuffers::Verifier v(buf, size);

  NNTR_THROW_IF(!nntr::VerifyModelBuffer(v), std::invalid_argument)
    << FUNC_TAG << "Verifying serialized model failed";

  std::ofstream os(out, std::ios_base::binary);
  const size_t error_buflen = 100;
  char error_buf[error_buflen];
  NNTR_THROW_IF(!os.good(), std::invalid_argument)
    << FUNC_TAG << "failed to open, reason : "
    << strerror_r(errno, error_buf, error_buflen);

  std::streamsize sz = static_cast<std::streamsize>(builder.GetSize());
  NNTR_THROW_IF(sz < 0, std::invalid_argument)
    << FUNC_TAG << "builder size :" << builder.GetSize()
    << " is too big. It cannot be represented by std::streamsize";

  os.write((char *)builder.GetBufferPointer(), sz);
  os.close();
}

} // namespace

/**
 * @brief Export NNTrainer GraphRepresentation to flatbuffer (circle_plus)
 * format
 *
 * @param representation GraphRepresentation to export
 * @param out circle_plus file (.nntr)
 */
void FlatBufferInterpreter::serialize(const GraphRepresentation &representation,
                                      const std::string &out) {

  GraphRepresentation graph = representation;
  flatbuffers::FlatBufferBuilder fbb;

  flatbuffers::Offset<flatbuffers::Vector<float>> tensor_data;
  flatbuffers::Offset<flatbuffers::Vector<float>> tensor_data2;
  flatbuffers::Offset<flatbuffers::Vector<int32_t>> tensor_dim;
  flatbuffers::Offset<flatbuffers::Vector<int32_t>> tensor_dim2;

  for (auto &node : representation) {
    auto variables = node->getWeights();
    auto name = node->getName();

    if (variables.size() == 0)
      continue;
    else {
      auto weights = node->getWeight(0);
      tensor_data = fbb.CreateVector(weights.getData(), weights.size());
      tensor_dim =
        fbb.CreateVector<int>({static_cast<int32_t>(weights.batch()),
                               static_cast<int32_t>(weights.channel()),
                               static_cast<int32_t>(weights.width()),
                               static_cast<int32_t>(weights.height())});

      auto weights2 = node->getWeight(1);
      tensor_data2 = fbb.CreateVector(weights2.getData(), weights2.size());
      tensor_dim2 =
        fbb.CreateVector<int>({static_cast<int32_t>(weights2.batch()),
                               static_cast<int32_t>(weights2.channel()),
                               static_cast<int32_t>(weights2.width()),
                               static_cast<int32_t>(weights2.height())});
    }
  }

  // build Tensor

  auto tensor = nntr::CreateTensor(fbb, nntr::TensorType_FLOAT32, tensor_dim,
                                   fbb.CreateString("FC_weight"), tensor_data);
  auto tensor2 = nntr::CreateTensor(fbb, nntr::TensorType_FLOAT32, tensor_dim2,
                                    fbb.CreateString("FC_bias"), tensor_data2);

  auto tensors =
    fbb.CreateVector<flatbuffers::Offset<nntr::Tensor>>({tensor, tensor2});

  // build Layer
  auto layer_name = fbb.CreateString("FC1");
  auto layer = nntr::CreateLayers(
    fbb, nntr::LayerTypes_FULLY_CONNECTED, layer_name, nntr::LayerOptions_NONE,
    0, 0, 0, nntr::ActivationType_NONE, tensors, tensors, tensors);

  auto layers = fbb.CreateVector<flatbuffers::Offset<nntr::Layers>>({layer});

  // Create NetworkGraph
  auto network_name = fbb.CreateString("circle_fc_NetworkGraph");
  auto network_graph =
    nntr::CreateNetworkGraph(fbb, network_name, 0, 0, layers);
  auto network_graphs =
    fbb.CreateVector<flatbuffers::Offset<nntr::NetworkGraph>>({network_graph});

  // Create Model

  auto name = fbb.CreateString("FC Test Model");

  fbb.Finish(nntr::CreateModel(fbb, name, 10, 128, nntr::OptimizerType_SGD,
                               nntr::LRSchedulerType_CONSTANT, nntr::LossType_MSE, network_graphs));

  builder2file(fbb, out);
}

template <typename T>
static std::string withKey(const std::string &key, const T &value) {
  std::stringstream ss;
  ss << key << "=" << value;
  return ss.str();
}

template <typename T>
static std::string withKey(const std::string &key,
                           std::initializer_list<T> value) {
  if (std::empty(value)) {
    throw std::invalid_argument("empty data cannot be converted");
  }

  std::stringstream ss;
  ss << key << "=";

  auto iter = value.begin();
  for (; iter != value.end() - 1; ++iter) {
    ss << *iter << ',';
  }
  ss << *iter;

  return ss.str();
}

GraphRepresentation FlatBufferInterpreter::deserialize(const std::string &in) {
  // using LayerHandle = std::shared_ptr<ml::train::Layer>;
  // using ModelHandle = std::unique_ptr<ml::train::Model>;
  using ml::train::createLayer;

  flatbuffers::FlatBufferBuilder fbb;

  auto file_name = in;
  std::string binary_buffer;
  bool file_load_status =
    flatbuffers::LoadFile(file_name.c_str(), true, &binary_buffer);

  fbb.PushBytes(reinterpret_cast<unsigned char *>(
                  const_cast<char *>(binary_buffer.c_str())),
                binary_buffer.size());

  auto model = nntr::GetModel(fbb.GetCurrentBufferPointer());

  auto network_graph = model->network_graph();
  auto layers = network_graph->begin()->layers();
  std::string layer_type;
  if (layers->begin()->type() == 0)
    layer_type = "fully_connected";

  // std::vector<LayerHandle> layer_handles;
  // layer_handles.push_back(
  //   createLayer(layer_type, {withKey("name", "FC1"), withKey("unit", 4)}));

  // std::string loss_type;
  // if (model->loss()->type() == 0)
  //   loss_type = "mse";

  // ModelHandle model_handle = ml::train::createModel(
  //   ml::train::ModelType::NEURAL_NET, {withKey("loss", loss_type)});

  // for (auto layer : layer_handles)
  //   model_handle->addLayer(layer);

  // model_handle->setProperty(
  //   {withKey("batch_size", 128), withKey("epochs", 10)});
  // std::string optimizer_type;
  // if (model->optimizer()->type() == 0)
  //   optimizer_type = "sgd";
  // auto optimizer = ml::train::createOptimizer(
  //   optimizer_type, {withKey("learning_rate", 0.001)});
  // model_handle->setOptimizer(std::move(optimizer));
  return {};
}

} // namespace nntrainer