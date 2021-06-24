// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   main.cpp
 * @date   24 Jun 2021
 * @todo   move resnet model creating to separate sourcefile
 * @brief  task runner for the resnet
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

/**
 * @brief make "key=value" from key and value
 *
 * @tparam T type of a value
 * @param key key
 * @param value value
 * @return std::string with "key=value"
 */
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

/**
 * @brief resnet block
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param filters number of filters
 * @param kernel_size number of kernel_size
 * @param downsample downsample to make output size 0
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> resnetBlock(const std::string &block_name,
                                     const std::string &input_name, int filters,
                                     int kernel_size, bool downsample) {
  using ml::train::createLayer;

  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto create_conv =
    [&with_name, filters](const std::string &name, int kernel_size, int stride,
                          int padding, const std::string &input_layer) {
      std::vector<std::string> props{
        with_name(name),
        withKey("stride", {stride, stride}),
        withKey("filters", filters),
        withKey("kernel_size", {kernel_size, kernel_size}),
        withKey("padding", {padding, padding}),
        withKey("input_layers", input_layer)};

      return createLayer("conv2d", props);
    };

  auto create_batch_relu = [&with_name](const std::string &name) {
    return createLayer("batch_normalization",
                       {with_name(name), "activation=relu"});
  };

  /** residual path */
  LayerHandle a1 = create_conv("a1", 3, downsample ? 2 : 1, 1, input_name);
  LayerHandle a2 = create_batch_relu("a2");
  LayerHandle a3 = create_conv("a3", 3, 1, 1, scoped_name("a2"));

  /** skip path */
  LayerHandle b1 = nullptr;
  if (downsample) {
    b1 = create_conv("b1", 1, 2, 0, input_name);
  }

  const std::string skip_name = b1 ? scoped_name("b1") : input_name;

  LayerHandle c1 = createLayer(
    "Addition",
    {with_name("c1"), withKey("input_layers", {scoped_name("a3"), skip_name})});

  LayerHandle c2 = create_batch_relu(""); /// use block_name itself.

  if (downsample) {
    return {a1, a2, a3, b1, c1, c2};
  } else {
    return {a1, a2, a3, c1, c2};
  }
}

/**
 * @brief Create resnet 18
 *
 * @return vector of layers that contain full graph of resnet18
 */
std::vector<LayerHandle> createResnet18Graph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(
    createLayer("conv2d", {
                            withKey("name", "conv0"),
                            withKey("kernel_size", {3, 3}),
                            withKey("stride", {1, 1}),
                            withKey("padding", {1, 1}),
                            withKey("bias_initializer", "zeros"),
                            withKey("weight_initializer", "xavier_uniform"),
                          }));

  layers.push_back(
    createLayer("batch_normalization", {withKey("name", "first_bn_relu"),
                                        withKey("activation", "relu")}));

  std::vector<std::vector<LayerHandle>> blocks;

  blocks.push_back(resnetBlock("conv1_0", "first_bn_relu", 64, 3, false));
  blocks.push_back(resnetBlock("conv1_1", "conv1_0", 64, 3, false));
  blocks.push_back(resnetBlock("conv2_0", "conv1_1", 128, 3, true));
  blocks.push_back(resnetBlock("conv2_1", "conv2_0", 128, 3, false));
  blocks.push_back(resnetBlock("conv3_0", "conv2_1", 256, 3, true));
  blocks.push_back(resnetBlock("conv3_1", "conv3_0", 256, 3, false));
  blocks.push_back(resnetBlock("conv4_0", "conv3_1", 256, 3, true));
  blocks.push_back(resnetBlock("conv4_1", "conv4_0", 256, 3, false));

  for (auto &block : blocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  layers.push_back(createLayer("pooling2d", {withKey("name", "last_p1"),
                                             withKey("pooling", "average"),
                                             withKey("pool_size", {4, 4})}));

  layers.push_back(createLayer("flatten", {withKey("name", "last_f1")}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("unit", 100), withKey("activation", "softmax")}));

  return layers;
}

/// @todo update createResnet18 to be more generic
ModelHandle createResnet18() {
  ModelHandle model =
    ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                           {withKey("loss", "cross"),
                            withKey("batch_size", 128), withKey("epochs", 60)});

  for (auto layers : createResnet18Graph()) {
    model->addLayer(layers);
  }

  return model;
}

ml_train_datagen_cb train_cb, valid_cb;

void create_and_run() {
  ModelHandle model = createResnet18();

  auto optimizer = ml::train::createOptimizer("adam");
  model->setOptimizer(std::move(optimizer));
}

int main() {
  try {
    create_and_run();
  } catch (std::exception &e) {
    std::cerr << "uncaught error! error: " << e.what();
    return 1;
  }

  return 0;
}
