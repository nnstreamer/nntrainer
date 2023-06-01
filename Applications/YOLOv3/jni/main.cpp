// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   main.cpp
 * @date   01 June 2023
 * @brief  application example for YOLO v3
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
const char *TRAIN_DIR_PATH = "/home/user/TRAIN_DIR/";
const char *VALIDATION_DIR_PATH = "/home/user/VALID_DIR/";
const std::string MODEL_INIT_BIN_PATH = "./init_model.bin";

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
 * @brief Convolution block
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param kernel_size kernel size
 * @param num_filters number of filters
 * @param stride stride
 * @param padding padding
 * @return std::vector<LayerHandle> vector of layers
 */
std::vector<LayerHandle> convBlock(const std::string &block_name,
                                   const std::string &input_layer,
                                   int kernel_size, int num_filters, int stride,
                                   int padding) {
  using ml::train::createLayer;

  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };

  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto createConv = [&with_name, &kernel_size, &num_filters, &stride, &padding](
                      const std::string &name, const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("filters", num_filters),
      withKey("stride", {stride, stride}),
      withKey("padding", padding),
      withKey("disable_bias", "true"),
      withKey("input_layers", input_layer)};

    return createLayer("conv2d", props);
  };

  LayerHandle conv = createConv("conv", input_layer);
  LayerHandle bn_act =
    createLayer("batch_normalization",
                {withKey("name", block_name), withKey("momentum", "0.9"),
                 withKey("activation", "leaky_relu")});
  return {conv, bn_act};
}

/**
 * @brief Darknet block
 *
 * @param block_name name of the block
 * @param input_name name of the input
 * @param kernel_size kernel size
 * @param num_filters number of filters
 * @param stride stride
 * @param padding padding
 * @return std::vector<LayerHandle> vector of layers
 */
std::vector<LayerHandle> darknetBlock(const std::string &block_name,
                                      std::string input_layer, int num_filters,
                                      int repeat) {
  auto scoped_name = [&block_name](const std::string &layer_name, int uid) {
    return block_name + "/" + layer_name + "_" + std::to_string(uid);
  };
  using ml::train::createLayer;
  std::string output_layer_name;
  std::vector<std::vector<LayerHandle>> blocks;
  for (int i = 0; i < repeat; i++) {
    blocks.push_back(
      convBlock(scoped_name("c1", i), input_layer, 1, num_filters / 2, 1, 0));
    blocks.push_back(convBlock(scoped_name("c2", i), scoped_name("c1", i), 3,
                               num_filters, 1, 1));
    output_layer_name = (repeat - 1 != i) ? scoped_name("res", i) : block_name;
    blocks.push_back({createLayer(
      "addition",
      {withKey("name", output_layer_name),
       withKey("input_layers", input_layer + ", " + scoped_name("c2", i))})});
    input_layer = scoped_name("res", i);
  }

  std::vector<LayerHandle> layers;
  for (auto &block : blocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  return layers;
}

/**
 * @brief Create DarkNet53 backbone
 *
 * @return vector of layers that contain full graph of darknet53 backbone
 */
ModelHandle Darknet53() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input",
    {withKey("name", "input0"),
     withKey("input_shape", "3:" + std::to_string(IMAGE_HEIGHT_SIZE) + ":" +
                              std::to_string(IMAGE_WIDTH_SIZE))}));

  std::vector<std::vector<LayerHandle>> blocks;
  blocks.push_back(convBlock("conv1", "input0", 3, 32, 1, 1));
  blocks.push_back(convBlock("conv2", "conv1", 3, 64, 2, 1));
  blocks.push_back(darknetBlock("block1", "conv2", 64, 1));
  blocks.push_back(convBlock("conv3", "block1", 3, 128, 2, 1));
  blocks.push_back(darknetBlock("block2", "conv3", 128, 2));
  blocks.push_back(convBlock("conv4", "block2", 3, 256, 2, 1));
  blocks.push_back(darknetBlock("block3", "conv4", 256, 8));
  blocks.push_back(convBlock("conv5", "block3", 3, 512, 2, 1));
  blocks.push_back(darknetBlock("block4", "conv5", 512, 8));
  blocks.push_back(convBlock("conv6", "block4", 3, 1024, 2, 1));
  blocks.push_back(darknetBlock("block5", "conv6", 1024, 4));

  for (auto &block : blocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  for (auto layer : layers) {
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
    // create Darknet53 model
    ModelHandle model = Darknet53();
    model->setProperty({withKey("batch_size", BATCH_SIZE),
                        withKey("epochs", EPOCHS),
                        withKey("save_path", "darknet53.bin")});

    // create optimizer
    auto optimizer = ml::train::createOptimizer(
      "adam", {"learning_rate=0.001", "epsilon=1e-8", "torch_ref=true"});
    model->setOptimizer(std::move(optimizer));

    // compile and initialize model
    model->compile();
    model->initialize();

    model->summarize(std::cout,
                     ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);
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
