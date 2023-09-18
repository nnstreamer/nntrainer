/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	nntrainer_test_util.cpp
 * @date	28 April 2020
 * @brief	This is util functions for test
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "nntrainer_test_util.h"
#include <app_context.h>
#include <climits>
#include <iostream>
#include <layer_node.h>
#include <multiout_realizer.h>
#include <nntrainer_error.h>
#include <random>
#include <regex>
#include <tensor.h>

#define num_class 10
#define batch_size 16
#define feature_size 62720

static std::mt19937 rng(0);

/**
 * @brief replace string and save it in file
 * @param[in] from string to be replaced
 * @param[in] to string to be replaced to
 * @param[in] file file to perform the action on
 * @param[in] init_config file contents to be initialized with if file not found
 */
void replaceString(const std::string &from, const std::string &to,
                   const std::string file, std::string init_config) {
  size_t start_pos = 0;
  std::string s;
  std::ifstream file_stream(file.c_str(), std::ifstream::in);
  if (file_stream.good()) {
    s.assign((std::istreambuf_iterator<char>(file_stream)),
             std::istreambuf_iterator<char>());
    file_stream.close();
  } else {
    s = init_config;
  }
  while ((start_pos = s.find(from, start_pos)) != std::string::npos) {
    s.replace(start_pos, from.size(), to);
    start_pos += to.size();
  }

  std::ofstream data_file(file.c_str());
  data_file << s;
  data_file.close();
}

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] outVec
 * @param[out] outLabel
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
static bool getData(std::ifstream &F, float *outVec, float *outLabel,
                    unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();

  uint64_t position =
    (uint64_t)((feature_size + num_class) * (uint64_t)id * sizeof(float));

  if (position > file_length) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  F.read((char *)outVec, sizeof(float) * feature_size);
  F.read((char *)outLabel, sizeof(float) * num_class);

  return true;
}

DataInformation::DataInformation(unsigned int num_samples,
                                 const std::string &filename) :
  count(0),
  num_samples(num_samples),
  file(filename, std::ios::in | std::ios::binary),
  idxes(num_samples) {
  std::iota(idxes.begin(), idxes.end(), 0);
  std::shuffle(idxes.begin(), idxes.end(), rng);
  rng.seed(0);
  if (!file.good()) {
    throw std::invalid_argument("given file is not good, filename: " +
                                filename);
  }
}

static auto getDataSize = [](const std::string &file_name) {
  std::ifstream f(file_name, std::ios::in | std::ios::binary);
  NNTR_THROW_IF(!f.good(), std::invalid_argument)
    << "cannot find " << file_name;
  f.seekg(0, std::ios::end);
  long file_size = f.tellg();
  return static_cast<unsigned int>(
    file_size / ((num_class + feature_size) * sizeof(float)));
};

std::string train_filename = getResPath("trainingSet.dat", {"test"});
std::string valid_filename = getResPath("trainingSet.dat", {"test"});

DataInformation createTrainData() {
  return DataInformation(getDataSize(train_filename), train_filename);
}

DataInformation createValidData() {
  return DataInformation(getDataSize(valid_filename), valid_filename);
}

/**
 * @brief      get data which size is batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto data = reinterpret_cast<DataInformation *>(user_data);

  getData(data->file, *outVec, *outLabel, data->idxes.at(data->count));
  data->count++;
  if (data->count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
    std::shuffle(data->idxes.begin(), data->idxes.end(), data->rng);
  }

  return ML_ERROR_NONE;
}

/**
 * @brief return a tensor filled with contant value with dimension
 */
nntrainer::Tensor constant(float value, unsigned int batch,
                           unsigned int channel, unsigned int height,
                           unsigned int width) {
  nntrainer::Tensor t(batch, channel, height, width);
  t.setValue(value);

  return t;
}

nntrainer::Tensor ranged(unsigned int batch, unsigned int channel,
                         unsigned int height, unsigned int width,
                         nntrainer::Tformat fm) {
  nntrainer::Tensor t(batch, channel, height, width, fm);
  unsigned int i = 0;
  return t.apply([&](float in) { return i++; });
}

nntrainer::Tensor randUniform(unsigned int batch, unsigned int channel,
                              unsigned int height, unsigned int width,
                              float min, float max) {
  nntrainer::Tensor t(batch, channel, height, width);
  t.setRandUniform(min, max);
  return t;
}

const std::string
getResPath(const std::string &filename,
           const std::initializer_list<const char *> fallback_base) {
  static const char *prefix = std::getenv("NNTRAINER_RESOURCE_PATH");
  static const char *fallback_prefix = "./res";

  std::stringstream ss;
  if (prefix != nullptr) {
    ss << prefix << '/' << filename;
    return ss.str();
  }

  ss << fallback_prefix;
  for (auto &folder : fallback_base) {
    ss << '/' << folder;
  }

  ss << '/' << filename;

  return ss.str();
}

nntrainer::GraphRepresentation
makeGraph(const std::vector<LayerRepresentation> &layer_reps) {
  static auto &ac = nntrainer::AppContext::Global();
  nntrainer::GraphRepresentation graph_rep;

  for (const auto &layer_representation : layer_reps) {
    /// @todo Use unique_ptr here
    std::shared_ptr<nntrainer::LayerNode> layer = nntrainer::createLayerNode(
      ac.createObject<nntrainer::Layer>(layer_representation.first),
      layer_representation.second);
    graph_rep.push_back(layer);
  }

  return graph_rep;
}

nntrainer::GraphRepresentation makeCompiledGraph(
  const std::vector<LayerRepresentation> &layer_reps,
  std::vector<std::unique_ptr<nntrainer::GraphRealizer>> &realizers,
  const std::string &loss_layer) {
  static auto &ac = nntrainer::AppContext::Global();

  nntrainer::GraphRepresentation graph_rep;
  auto model_graph = nntrainer::NetworkGraph();

  for (auto &layer_representation : layer_reps) {
    std::shared_ptr<nntrainer::LayerNode> layer = nntrainer::createLayerNode(
      ac.createObject<nntrainer::Layer>(layer_representation.first),
      layer_representation.second);
    graph_rep.push_back(layer);
  }

  for (auto &realizer : realizers) {
    graph_rep = realizer->realize(graph_rep);
  }

  for (auto &layer : graph_rep) {
    model_graph.addLayer(layer);
  }

  model_graph.compile(loss_layer);

  graph_rep.clear();
  for (auto &node : model_graph.getLayerNodes()) {
    graph_rep.push_back(node);
  }

  return graph_rep;
}

void sizeCheckedReadTensor(nntrainer::Tensor &t, std::ifstream &file,
                           const std::string &error_msg) {
  unsigned int sz = 0;
  nntrainer::checkedRead(file, (char *)&sz, sizeof(unsigned));
  NNTR_THROW_IF(t.getDim().getDataLen() != sz, std::invalid_argument)
    << "[ReadFail] dimension does not match at " << error_msg << " sz: " << sz
    << " dimsize: " << t.getDim().getDataLen() << '\n';
  t.read(file);
}
