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
#include <engine.h>
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <util_func.h>

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
    return nntrainer::withKey("name", scoped_name(layer_name));
  };

  auto createConv = [&with_name, filters](const std::string &name,
                                          int kernel_size, int stride,
                                          const std::string &padding,
                                          const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      nntrainer::withKey("stride", {stride, stride}),
      nntrainer::withKey("filters", filters),
      nntrainer::withKey("kernel_size", {kernel_size, kernel_size}),
      nntrainer::withKey("padding", padding),
      nntrainer::withKey("disable_bias", "true"),
      nntrainer::withKey("input_layers", input_layer)};

    return createLayer("conv2d", props);
  };

  /** construct basic layer **/
  LayerHandle a1 = createConv("a1", kernel_size, 1, "same", input_name);

  if (downsample) {
    LayerHandle a2 =
      createLayer("batch_normalization",
                  {with_name("a2"), nntrainer::withKey("momentum", "0.9"),
                   nntrainer::withKey("epsilon", 0.00001),
                   nntrainer::withKey("activation", "leaky_relu")});

    LayerHandle a3 =
      createLayer("pooling2d", {nntrainer::withKey("name", block_name),
                                nntrainer::withKey("stride", {2, 2}),
                                nntrainer::withKey("pooling", "max"),
                                nntrainer::withKey("pool_size", {2, 2})});

    return {a1, a2, a3};
  } else {
    LayerHandle a2 = createLayer(
      "batch_normalization", {nntrainer::withKey("name", block_name),
                              nntrainer::withKey("momentum", "0.9"),
                              nntrainer::withKey("epsilon", 0.00001),
                              nntrainer::withKey("activation", "leaky_relu")});

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
    "input", {nntrainer::withKey("name", "input0"),
              nntrainer::withKey("input_shape",
                                 "3:" + std::to_string(IMAGE_HEIGHT_SIZE) +
                                   ":" + std::to_string(IMAGE_WIDTH_SIZE))}));

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

  blocks.push_back(
    {createLayer("pooling2d", {nntrainer::withKey("name", "conv_a_pool"),
                               nntrainer::withKey("stride", {2, 2}),
                               nntrainer::withKey("pooling", "max"),
                               nntrainer::withKey("pool_size", {2, 2}),
                               nntrainer::withKey("input_layers", "conv13")})});
  blocks.push_back(yoloBlock("conv_a1", "conv_a_pool", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a2", "conv_a1", 512, 1, false));
  blocks.push_back(yoloBlock("conv_a3", "conv_a2", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a4", "conv_a3", 512, 1, false));
  blocks.push_back(yoloBlock("conv_a5", "conv_a4", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a6", "conv_a5", 1024, 3, false));
  blocks.push_back(yoloBlock("conv_a7", "conv_a6", 1024, 3, false));

  blocks.push_back(yoloBlock("conv_b", "conv13", 64, 1, false));

  blocks.push_back({createLayer(
    "reorg_layer", {nntrainer::withKey("name", "re_organization"),
                    nntrainer::withKey("input_layers", "conv_b")})});

  blocks.push_back({createLayer(
    "concat", {nntrainer::withKey("name", "concat"),
               nntrainer::withKey("input_layers", "conv_a7, re_organization"),
               nntrainer::withKey("axis", 1)})});

  blocks.push_back(yoloBlock("conv_out1", "concat", 1024, 3, false));

  blocks.push_back({createLayer(
    "conv2d", {
                nntrainer::withKey("name", "conv_out2"),
                nntrainer::withKey("filters", 5 * (5 + CLASS_NUMBER)),
                nntrainer::withKey("kernel_size", {1, 1}),
                nntrainer::withKey("stride", {1, 1}),
                nntrainer::withKey("padding", "same"),
                nntrainer::withKey("input_layers", "conv_out1"),
              })});

  for (auto &block : blocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  layers.push_back(
    createLayer("permute", {
                             nntrainer::withKey("name", "permute"),
                             nntrainer::withKey("direction", {2, 3, 1}),
                           }));

  layers.push_back(createLayer(
    "reshape",
    {
      nntrainer::withKey("name", "reshape"),
      nntrainer::withKey(
        "target_shape", std::to_string(GRID_HEIGHT_NUMBER * GRID_WIDTH_NUMBER) +
                          ":" + std::to_string(ANCHOR_NUMBER) + ":" +
                          std::to_string(5 + CLASS_NUMBER)),
    }));

  layers.push_back(
    createLayer("yolo_v2_loss",
                {
                  nntrainer::withKey("name", "yolo_v2_loss"),
                  nntrainer::withKey("max_object_number", MAX_OBJECT_NUMBER),
                  nntrainer::withKey("class_number", CLASS_NUMBER),
                  nntrainer::withKey("grid_height_number", GRID_HEIGHT_NUMBER),
                  nntrainer::withKey("grid_width_number", GRID_WIDTH_NUMBER),
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
    auto &ct_engine = nntrainer::Engine::Global();

    auto app_context = static_cast<nntrainer::AppContext *>(
      ct_engine.getRegisteredContext("cpu"));

    app_context->registerFactory(nntrainer::createLayer<custom::ReorgLayer>);
    app_context->registerFactory(
      nntrainer::createLayer<custom::YoloV2LossLayer>);
  } catch (std::invalid_argument &e) {
    std::cerr << "failed to register factory, reason: " << e.what()
              << std::endl;
    return 1;
  }

  try {
    // create YOLO v2 model
    ModelHandle model = YOLO();
    model->setProperty({nntrainer::withKey("batch_size", BATCH_SIZE),
                        nntrainer::withKey("epochs", EPOCHS),
                        nntrainer::withKey("save_path", "yolov2.bin")});

    // create optimizer
    auto optimizer = ml::train::createOptimizer(
      "adam", {"learning_rate=0.001", "epsilon=1e-8", "torch_ref=true"});
    int status = model->setOptimizer(std::move(optimizer));
    if (status) {
      throw std::invalid_argument("failed to set optimizer");
    }

    // compile and initialize model
    status = model->compile();
    if (status) {
      throw std::invalid_argument("model compilation failed!");
    }
    status = model->initialize();
    if (status) {
      throw std::invalid_argument("model initialization failed!");
    }
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
