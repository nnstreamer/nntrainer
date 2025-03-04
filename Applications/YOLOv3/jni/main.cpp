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

#include <upsample_layer.h>
#include <yolo_v3_loss.h>

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
const unsigned int EPOCHS = 2;
const char *TRAIN_DIR_PATH = "/home/user/train_dir/";

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

std::array<UserDataType, 1> createDetDataGenerator(const char *train_dir,
                                                   int max_num_label, int c,
                                                   int h, int w) {
  UserDataType train_data(
    new nntrainer::util::DirDataLoader(train_dir, max_num_label, c, h, w, true,
                                       {
                                         {BATCH_SIZE, 1, 4, 5},
                                         {BATCH_SIZE, 1, 4, 5},
                                         {BATCH_SIZE, 1, 4, 5},
                                       }));

  return {std::move(train_data)};
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
 * @return Vector of layers that construct darknet block
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
 * @return Vector of layers that contain full graph of darknet53 backbone
 */
std::vector<LayerHandle> Darknet53() {
  using ml::train::createLayer;

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

  return layers;
}

/**
 * @brief Create YOLOv3 backbone
 *
 * @return Model handle of YOLOv3
 */
ModelHandle YOLOv3() {
  using ml::train::createLayer;

  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  std::vector<LayerHandle> layers = Darknet53();
  std::vector<std::vector<LayerHandle>> fp3, fp2, fp1, neck3_2, neck2_1;
  std::vector<std::vector<LayerHandle>> head3, head2, head1;

  // feature pyramid for large object
  fp3.push_back(convBlock("fp3_1", "block5", 1, 512, 1, 0));
  fp3.push_back(convBlock("fp3_2", "fp3_1", 3, 1024, 1, 1));
  fp3.push_back(convBlock("fp3_3", "fp3_2", 1, 512, 1, 0));
  fp3.push_back(convBlock("fp3_4", "fp3_3", 3, 1024, 1, 1));
  fp3.push_back(convBlock("fp3", "fp3_4", 1, 512, 1, 0));

  for (auto &block : fp3) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // connection for medium object
  neck3_2.push_back(convBlock("neck3_2_1", "fp3", 1, 256, 1, 0));
  neck3_2.push_back(
    {createLayer("upsample", {withKey("name", "neck3_2"),
                              withKey("input_layers", "neck3_2_1")})});

  for (auto &block : neck3_2) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // feature pyramid for medium object
  fp2.push_back(
    {createLayer("concat", {withKey("name", "fp2_1"),
                            withKey("input_layers", "neck3_2, block4"),
                            withKey("axis", "1")})});
  fp2.push_back(convBlock("fp2_2", "fp2_1", 1, 256, 1, 0));
  fp2.push_back(convBlock("fp2_3", "fp2_2", 3, 512, 1, 1));
  fp2.push_back(convBlock("fp2_4", "fp2_3", 1, 256, 1, 0));
  fp2.push_back(convBlock("fp2_5", "fp2_4", 3, 512, 1, 1));
  fp2.push_back(convBlock("fp2", "fp2_5", 1, 256, 1, 0));

  for (auto &block : fp2) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // connection for small object
  neck2_1.push_back(convBlock("neck2_1_1", "fp2", 1, 128, 1, 0));
  neck2_1.push_back(
    {createLayer("upsample", {withKey("name", "neck2_1"),
                              withKey("input_layers", "neck2_1_1")})});

  for (auto &block : neck2_1) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // feature pyramid for small object
  fp1.push_back(
    {createLayer("concat", {withKey("name", "fp1_1"),
                            withKey("input_layers", "neck2_1, block3"),
                            withKey("axis", "1")})});
  fp1.push_back(convBlock("fp1_2", "fp1_1", 1, 128, 1, 0));
  fp1.push_back(convBlock("fp1_3", "fp1_2", 3, 256, 1, 1));
  fp1.push_back(convBlock("fp1_4", "fp1_3", 1, 128, 1, 0));
  fp1.push_back(convBlock("fp1_5", "fp1_4", 3, 256, 1, 1));
  fp1.push_back(convBlock("fp1", "fp1_5", 1, 128, 1, 0));

  for (auto &block : fp1) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // head for large object
  head3.push_back(convBlock("head3_1", "fp3", 3, 1024, 1, 1));
  head3.push_back(
    convBlock("head3", "head3_1", 1, 3 * (5 + CLASS_NUMBER), 1, 0));
  head3.push_back({createLayer("permute", {
                                            withKey("name", "h3_permute"),
                                            withKey("direction", {2, 3, 1}),
                                          })});
  head3.push_back({createLayer(
    "reshape", {
                 withKey("name", "h3_reshape"),
                 withKey("target_shape", // grid : anchor : 5 + num_classes
                         std::to_string(13 * 13) + ":" + std::to_string(3) +
                           ":" + std::to_string(5 + CLASS_NUMBER)),
               })});
  head3.push_back({createLayer(
    "yolo_v3_loss",
    {withKey("name", "loss_for_large"),
     withKey("max_object_number", MAX_OBJECT_NUMBER),
     withKey("class_number", CLASS_NUMBER), withKey("grid_height_number", 13),
     withKey("grid_width_number", 13), withKey("scale", 1)})});

  for (auto &block : head3) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // head for medium object
  head2.push_back(convBlock("head2_1", "fp2", 3, 512, 1, 1));
  head2.push_back(
    convBlock("head2", "head2_1", 1, 3 * (5 + CLASS_NUMBER), 1, 0));
  head2.push_back({createLayer("permute", {
                                            withKey("name", "h2_permute"),
                                            withKey("direction", {2, 3, 1}),
                                          })});
  head2.push_back({createLayer(
    "reshape", {
                 withKey("name", "h2_reshape"),
                 withKey("target_shape", // grid : anchor : 5 + num_classes
                         std::to_string(26 * 26) + ":" + std::to_string(3) +
                           ":" + std::to_string(5 + CLASS_NUMBER)),
               })});
  head2.push_back({createLayer(
    "yolo_v3_loss",
    {withKey("name", "loss_for_medium"),
     withKey("max_object_number", MAX_OBJECT_NUMBER),
     withKey("class_number", CLASS_NUMBER), withKey("grid_height_number", 26),
     withKey("grid_width_number", 26), withKey("scale", 2)})});

  for (auto &block : head2) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // head for small object
  head1.push_back(convBlock("head1_1", "fp1", 3, 256, 1, 1));
  head1.push_back(
    convBlock("head1", "head1_1", 1, 3 * (5 + CLASS_NUMBER), 1, 0));
  head1.push_back({createLayer("permute", {
                                            withKey("name", "h1_permute"),
                                            withKey("direction", {2, 3, 1}),
                                          })});
  head1.push_back({createLayer(
    "reshape", {
                 withKey("name", "h1_reshape"),
                 withKey("target_shape", // grid : anchor : 5 + num_classes
                         std::to_string(52 * 52) + ":" + std::to_string(3) +
                           ":" + std::to_string(5 + CLASS_NUMBER)),
               })});
  head1.push_back({createLayer(
    "yolo_v3_loss",
    {withKey("name", "loss_for_small"),
     withKey("max_object_number", MAX_OBJECT_NUMBER),
     withKey("class_number", CLASS_NUMBER), withKey("grid_height_number", 52),
     withKey("grid_width_number", 52), withKey("scale", 3)})});

  for (auto &block : head1) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  // Regist layers to model
  for (auto &layer : layers) {
    model->addLayer(layer);
  }

  return model;
}

int main(int argc, char *argv[]) {
  try {
    // print start time
    auto start = std::chrono::system_clock::now();
    std::time_t start_time = std::chrono::system_clock::to_time_t(start);
    std::cout << "started computation at " << std::ctime(&start_time)
              << std::endl;

    // set training config and print it
    std::cout << "batch_size: " << BATCH_SIZE << " epochs: " << EPOCHS
              << std::endl;

    try {
      auto &app_context = nntrainer::AppContext::Global();
      app_context.registerFactory(
        nntrainer::createLayer<custom::UpsampleLayer>);
    } catch (std::exception &e) {
      std::cerr << "failed to register factory, reason: " << e.what()
                << std::endl;
      return 1;
    }

    try {
      auto &app_context = nntrainer::AppContext::Global();
      app_context.registerFactory(
        nntrainer::createLayer<custom::YoloV3LossLayer>);
    } catch (std::exception &e) {
      std::cerr << "failed to register yolov3 loss, reason: " << e.what()
                << std::endl;
      return 1;
    }

    try {
      // create YOLOv3 model
      ModelHandle model = YOLOv3();
      model->setProperty({withKey("batch_size", BATCH_SIZE),
                          withKey("epochs", EPOCHS),
                          withKey("save_path", "darknet53.bin")});

      // create optimizer
      auto optimizer = ml::train::createOptimizer(
        "adam", {"learning_rate=0.000001", "epsilon=1e-8", "torch_ref=true"});
      model->setOptimizer(std::move(optimizer));

      // compile and initialize model
      model->compile();
      model->initialize();

      model->summarize(std::cout,
                       ml_train_summary_type_e::ML_TRAIN_SUMMARY_MODEL);

      // create train and validation data
      std::array<UserDataType, 1> user_datas;
      user_datas = createDetDataGenerator(TRAIN_DIR_PATH, MAX_OBJECT_NUMBER, 3,
                                          IMAGE_HEIGHT_SIZE, IMAGE_WIDTH_SIZE);
      auto &[train_user_data] = user_datas;

      auto dataset_train = ml::train::createDataset(
        ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

      model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                        std::move(dataset_train));

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
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what() << "\n";
    return 1;
  }
}
