// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   03 March 2023
 * @todo   replace backbone to original darknet of yolo v2
 * @brief  application example for YOLO v2
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <app_context.h>
#include <det_dataloader.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include "yolo_v2_loss.h"

#include <reorg_layer.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using UserDataType = std::unique_ptr<nntrainer::util::DirDataLoader>;

const unsigned int ANCHOR_NUMBER = 5;

const unsigned int MAX_OBJECT_NUMBER = 4;
const unsigned int CLASS_NUMBER = 4;
const unsigned int GRID_HEIGHT_NUMBER = 13;
const unsigned int GRID_WIDTH_NUMBER = 13;
const unsigned int IMAGE_HEIGHT_SIZE = 416;
const unsigned int IMAGE_WIDTH_SIZE = 416;
const unsigned int BATCH_SIZE = 4;
const unsigned int EPOCHS = 3;
const char *TRAIN_DIR_PATH = "/TRAIN_DIR/";
const char *VALIDATION_DIR_PATH = "/VALID_DIR/";
// const std::string MODEL_INIT_BIN_PATH = "/home/user/MODEL_INIT_BIN_PATH.bin";

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DirDataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

int validData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DirDataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

std::array<UserDataType, 2> createDetDataGenerator(const char *train_dir,
                                                   const char *valid_dir,
                                                   int max_num_label, int c,
                                                   int h, int w) {
  UserDataType train_data(new nntrainer::util::DirDataLoader(
    train_dir, max_num_label, c, h, w, true));
  UserDataType valid_data(new nntrainer::util::DirDataLoader(
    valid_dir, max_num_label, c, h, w, false));

  return {std::move(train_data), std::move(valid_data)};
}

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
 * @brief yolo block
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param filters number of filters
 * @param kernel_size number of kernel_size
 * @param downsample downsample to make output size 0
 * @return std::vector<LayerHandle> vectors of layers
 */
std::vector<LayerHandle> yoloBlock(const std::string &block_name,
                                   const std::string &input_name, int filters,
                                   int kernel_size, bool downsample) {
  using ml::train::createLayer;

  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto createConv = [&with_name, filters](const std::string &name,
                                          int kernel_size, int stride,
                                          const std::string &padding,
                                          const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", padding),
      withKey("input_layers", input_layer)};

    return createLayer("conv2d", props);
  };

  /** construct basic layer **/
  LayerHandle a1 = createConv("a1", kernel_size, 1, "same", input_name);

  if (downsample) {
    LayerHandle a2 = createLayer("batch_normalization",
                                 {with_name("a2"), withKey("momentum", "0.9"),
                                  withKey("activation", "leaky_relu")});

    LayerHandle a3 = createLayer(
      "pooling2d", {withKey("name", block_name), withKey("stride", {2, 2}),
                    withKey("pooling", "max"), withKey("pool_size", {2, 2})});

    return {a1, a2, a3};
  } else {
    LayerHandle a2 =
      createLayer("batch_normalization",
                  {withKey("name", block_name), withKey("momentum", "0.9"),
                   withKey("activation", "leaky_relu")});

    return {a1, a2};
  }
}

/**
 * @brief Create yolo v2 light
 *
 * @return vector of layers that contain full graph of yolo v2 light
 */
ModelHandle YOLO() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input",
    {withKey("name", "input0"),
     withKey("input_shape", "3:" + std::to_string(IMAGE_HEIGHT_SIZE) + ":" +
                              std::to_string(IMAGE_WIDTH_SIZE))}));

  std::vector<std::vector<LayerHandle>> blocks;

  blocks.push_back(yoloBlock("conv1", "input0", 32, 3, true));
  blocks.push_back(yoloBlock("conv2", "conv1", 64, 3, true));
  blocks.push_back(yoloBlock("conv3", "conv2", 128, 3, false));
  blocks.push_back(yoloBlock("conv4", "conv3", 64, 1, false));
  blocks.push_back(yoloBlock("conv5", "conv4", 128, 3, true));
  blocks.push_back(yoloBlock("conv6", "conv5", 256, 3, false));
  blocks.push_back(yoloBlock("conv7", "conv6", 128, 1, false));
  blocks.push_back(yoloBlock("conv8", "conv7", 256, 3, true));
  blocks.push_back(yoloBlock("conv9", "conv8", 512, 3, false));
  blocks.push_back(yoloBlock("conv10", "conv9", 256, 1, false));
  blocks.push_back(yoloBlock("conv11", "conv10", 512, 3, false));
  blocks.push_back(yoloBlock("conv12", "conv11", 256, 1, false));
  blocks.push_back(yoloBlock("conv13", "conv12", 512, 3, false));

  blocks.push_back({createLayer(
    "pooling2d", {withKey("name", "conv_a_pool"), withKey("stride", {2, 2}),
                  withKey("pooling", "max"), withKey("pool_size", {2, 2}),
                  withKey("input_layers", "conv13")})});
  blocks.push_back(yoloBlock("conv_a1", "conv_a_pool", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a2", "conv_a1", 512, 1, false));
  blocks.push_back(yoloBlock("conv_a3", "conv_a2", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a4", "conv_a3", 512, 1, false));
  blocks.push_back(yoloBlock("conv_a5", "conv_a4", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a6", "conv_a5", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a7", "conv_a6", 1024, 3, false));

  blocks.push_back(yoloBlock("conv_b", "conv13", 64, 1, false));

  blocks.push_back(
    {createLayer("reorg_layer", {withKey("name", "re_organization"),
                                 withKey("input_layers", "conv_b")})});

  blocks.push_back(
    {createLayer("concat", {withKey("name", "concat"),
                            withKey("input_layers", "conv_a7, re_organization"),
                            withKey("axis", 1)})});

  blocks.push_back(yoloBlock("conv_out1", "concat", 1024, 3, false));

  blocks.push_back(
    {createLayer("conv2d", {
                             withKey("name", "conv_out2"),
                             withKey("filters", 5 * (5 + CLASS_NUMBER)),
                             withKey("kernel_size", {1, 1}),
                             withKey("stride", {1, 1}),
                             withKey("padding", "same"),
                             withKey("input_layers", "conv_out1"),
                           })});

  for (auto &block : blocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  layers.push_back(createLayer("permute", {
                                            withKey("name", "permute"),
                                            withKey("direction", {2, 3, 1}),
                                          }));

  layers.push_back(createLayer(
    "reshape",
    {
      withKey("name", "reshape"),
      withKey("target_shape",
              std::to_string(GRID_HEIGHT_NUMBER * GRID_WIDTH_NUMBER) + ":" +
                std::to_string(ANCHOR_NUMBER) + ":" +
                std::to_string(5 + CLASS_NUMBER)),
    }));

  layers.push_back(createLayer(
    "yolo_v2_loss", {
                      withKey("name", "yolo_v2_loss"),
                      withKey("max_object_number", MAX_OBJECT_NUMBER),
                      withKey("class_number", CLASS_NUMBER),
                      withKey("grid_height_number", GRID_HEIGHT_NUMBER),
                      withKey("grid_width_number", GRID_WIDTH_NUMBER),
                    }));

  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

int main(int argc, char *argv[]) {
  // print start time
  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::cout << "started computation at " << std::ctime(&start_time)
            << std::endl;

  // set training config and print it
  std::cout << "batch_size: " << BATCH_SIZE << " epochs: " << EPOCHS
            << std::endl;

  try {
    // create YOLO v2 model
    ModelHandle model = YOLO();
    model->setProperty({withKey("batch_size", BATCH_SIZE),
                        withKey("epochs", EPOCHS),
                        withKey("save_path", "yolov2.bin")});

    // create optimizer
    auto optimizer = ml::train::createOptimizer(
      "adam", {"learning_rate=0.001", "epsilon=1e-8", "torch_ref=true"});
    model->setOptimizer(std::move(optimizer));

    // compile and initialize model
    model->compile();
    model->initialize();
    model->save("./yolov2.ini", ml::train::ModelFormat::MODEL_FORMAT_INI);
    // model->load(MODEL_INIT_BIN_PATH);

    // create train and validation data
    std::array<UserDataType, 2> user_datas;
    user_datas = createDetDataGenerator(TRAIN_DIR_PATH, VALIDATION_DIR_PATH,
                                        MAX_OBJECT_NUMBER, 3, IMAGE_HEIGHT_SIZE,
                                        IMAGE_WIDTH_SIZE);
    auto &[train_user_data, valid_user_data] = user_datas;

    auto dataset_train = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
    auto dataset_valid = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                      std::move(dataset_train));
    model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                      std::move(dataset_valid));

    model->train();
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }

  // print end time and duration
  auto end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);
  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";
}
