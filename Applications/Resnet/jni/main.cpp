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
#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#if defined(ENABLE_TEST)
#include <gtest/gtest.h>
#endif

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <cifar_dataloader.h>

#ifdef PROFILE
#include <profiler.h>
#endif

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

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
                                     int kernel_size, bool downsample,
                                     bool pre_trained) {
  using ml::train::createLayer;

  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto create_conv = [&with_name, filters,
                      pre_trained](const std::string &name, int kernel_size,
                                   int stride, const std::string &padding,
                                   const std::string &input_layer) {
    std::vector<std::string> props{
      with_name(name),
      withKey("stride", {stride, stride}),
      withKey("filters", filters),
      withKey("kernel_size", {kernel_size, kernel_size}),
      withKey("padding", padding),
      withKey("input_layers", input_layer),
      withKey("trainable", pre_trained ? "true" : "false")};

    return createLayer("conv2d", props);
  };

/** residual path */
#if defined(ENABLE_TFLITE_INTERPRETER)
  LayerHandle a1 = create_conv("a1", 3, downsample ? 2 : 1, "same", input_name);
#else
  LayerHandle a1 = create_conv("a1", 3, downsample ? 2 : 1,
                               downsample ? "1,1" : "same", input_name);
#endif
  LayerHandle a2 =
    createLayer("batch_normalization",
                {with_name("a2"), withKey("activation", "relu"),
                 withKey("momentum", "0.9"), withKey("epsilon", "0.00001"),
                 withKey("trainable", pre_trained ? "true" : "false")});
  LayerHandle a3 = create_conv("a3", 3, 1, "same", scoped_name("a2"));

  /** skip path */
  LayerHandle b1 = nullptr;
  if (downsample) {
#if defined(ENABLE_TFLITE_INTERPRETER)
    b1 = create_conv("b1", 1, 2, "same", input_name);
#else
    b1 = create_conv("b1", 1, 2, "0,0", input_name);
#endif
  }

  const std::string skip_name = b1 ? scoped_name("b1") : input_name;

  LayerHandle c1 = createLayer(
    "Addition",
    {with_name("c1"), withKey("input_layers", {scoped_name("a3"), skip_name})});

  LayerHandle c2 =
    createLayer("batch_normalization",
                {withKey("name", block_name), withKey("activation", "relu"),
                 withKey("momentum", "0.9"), withKey("epsilon", "0.00001"),
                 withKey("trainable", "false")});

  if (downsample) {
    return {b1, a1, a2, a3, c1, c2};
  } else {
    return {a1, a2, a3, c1, c2};
  }
}

/**
 * @brief Create resnet 18
 *
 * @return vector of layers that contain full graph of resnet18
 */
std::vector<LayerHandle> createResnet18Graph(bool pre_trained) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "3:32:32")}));

  layers.push_back(createLayer(
    "conv2d", {withKey("name", "conv0"), withKey("filters", 64),
               withKey("kernel_size", {3, 3}), withKey("stride", {1, 1}),
               withKey("padding", "same"), withKey("bias_initializer", "zeros"),
               withKey("weight_initializer", "xavier_uniform"),
               withKey("trainable", pre_trained ? "true" : "false")}));

  layers.push_back(createLayer(
    "batch_normalization",
    {withKey("name", "first_bn_relu"), withKey("activation", "relu"),
     withKey("momentum", "0.9"), withKey("epsilon", "0.00001"),
     withKey("trainable", pre_trained ? "true" : "false")}));

  std::vector<std::vector<LayerHandle>> blocks;

  blocks.push_back(
    resnetBlock("conv1_0", "first_bn_relu", 64, 3, false, pre_trained));
  blocks.push_back(
    resnetBlock("conv1_1", "conv1_0", 64, 3, false, pre_trained));
  blocks.push_back(
    resnetBlock("conv2_0", "conv1_1", 128, 3, true, pre_trained));
  blocks.push_back(
    resnetBlock("conv2_1", "conv2_0", 128, 3, false, pre_trained));
  blocks.push_back(
    resnetBlock("conv3_0", "conv2_1", 256, 3, true, pre_trained));
  blocks.push_back(
    resnetBlock("conv3_1", "conv3_0", 256, 3, false, pre_trained));
  blocks.push_back(
    resnetBlock("conv4_0", "conv3_1", 512, 3, true, pre_trained));
  blocks.push_back(
    resnetBlock("conv4_1", "conv4_0", 512, 3, false, pre_trained));

  for (auto &block : blocks) {
    layers.insert(layers.end(), block.begin(), block.end());
  }

  layers.push_back(createLayer(
    "pooling2d", {withKey("name", "last_p1"), withKey("pooling", "average"),
                  withKey("pool_size", {4, 4}), withKey("stride", "4,4")}));

  layers.push_back(createLayer("flatten", {withKey("name", "last_f1")}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("unit", 100), withKey("activation", "softmax")}));

  return layers;
}

/// @todo update createResnet18 to be more generic
ModelHandle createResnet18(bool pre_trained = false) {
/// @todo support "LOSS : cross" for TF_Lite Exporter
#if (defined(ENABLE_TFLITE_INTERPRETER) && !defined(ENABLE_TEST))
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});
#else
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "cross")});
#endif

  for (auto &layer : createResnet18Graph(pre_trained)) {
    model->addLayer(layer);
  }

  return model;
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

int validData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::util::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

#if defined(ENABLE_TEST)
TEST(Resnet_Training, verify_accuracy) {
  EXPECT_FLOAT_EQ(training_loss, 4.389328);
  EXPECT_FLOAT_EQ(validation_loss, 11.611803);
}
#endif

/// @todo maybe make num_class also a parameter
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data,
                  UserDataType &valid_user_data) {
  // set option for transfer learning
  const bool transfer_learning = false;
  std::string pretrained_bin_path = "./pretrained_resnet18.bin";

  // setup model
  ModelHandle model = createResnet18(transfer_learning);
  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("save_path", "resnet_full.bin")});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  status = model->initialize();
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  auto dataset_valid = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));
  model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                    std::move(dataset_valid));

  if (transfer_learning)
    model->load(pretrained_bin_path);
  model->train();

#if defined(ENABLE_TEST)
  training_loss = model->getTrainingLoss();
  validation_loss = model->getValidationLoss();
#elif defined(ENABLE_TFLITE_INTERPRETER)
  model->exports(ml::train::ExportMethods::METHOD_TFLITE, "resnet_test.tflite");
#endif
}

std::array<UserDataType, 2>
createFakeDataGenerator(unsigned int batch_size,
                        unsigned int simulated_data_size,
                        unsigned int data_split) {
  UserDataType train_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 3, 32, 32}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));
  UserDataType valid_data(new nntrainer::util::RandomDataLoader(
    {{batch_size, 3, 32, 32}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));

  return {std::move(train_data), std::move(valid_data)};
}

std::array<UserDataType, 2>
createRealDataGenerator(const std::string &directory, unsigned int batch_size,
                        unsigned int data_split) {

  UserDataType train_data(new nntrainer::util::Cifar100DataLoader(
    directory + "/train.bin", batch_size, data_split));
  UserDataType valid_data(new nntrainer::util::Cifar100DataLoader(
    directory + "/test.bin", batch_size, data_split));

  return {std::move(train_data), std::move(valid_data)};
}

int main(int argc, char *argv[]) {
  if (argc < 5) {
    std::cerr
      << "usage: ./main [{data_directory}|\"fake\"] [batchsize] [data_split] "
         "[epoch] \n"
      << "when \"fake\" is given, original data size is assumed 512 for both "
         "train and validation\n";
    return EXIT_FAILURE;
  }

  auto start = std::chrono::system_clock::now();
  std::time_t start_time = std::chrono::system_clock::to_time_t(start);
  std::cout << "started computation at " << std::ctime(&start_time)
            << std::endl;

#ifdef PROFILE
  auto listener =
    std::make_shared<nntrainer::profile::GenericProfileListener>();
  nntrainer::profile::Profiler::Global().subscribe(listener);
#endif

  std::string data_dir = argv[1];
  unsigned int batch_size = std::stoul(argv[2]);
  unsigned int data_split = std::stoul(argv[3]);
  unsigned int epoch = std::stoul(argv[4]);

  std::cout << "data_dir: " << data_dir << ' ' << "batch_size: " << batch_size
            << " data_split: " << data_split << " epoch: " << epoch
            << std::endl;

  /// warning: the data loader will be destroyed at the end of this function,
  /// and passed as a pointer to the databuffer
  std::array<UserDataType, 2> user_datas;

  try {
    if (data_dir == "fake") {
      user_datas = createFakeDataGenerator(batch_size, 512, data_split);
    } else {
      user_datas = createRealDataGenerator(data_dir, batch_size, data_split);
    }
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  auto &[train_user_data, valid_user_data] = user_datas;

  try {
    createAndRun(epoch, batch_size, train_user_data, valid_user_data);
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what()
              << std::endl;
    return EXIT_FAILURE;
  }
  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  std::time_t end_time = std::chrono::system_clock::to_time_t(end);

  std::cout << "finished computation at " << std::ctime(&end_time)
            << "elapsed time: " << elapsed_seconds.count() << "s\n";

#ifdef PROFILE
  std::cout << *listener;
#endif

  int status = EXIT_SUCCESS;
#if defined(ENABLE_TEST)
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error during InitGoogleTest" << std::endl;
    return EXIT_FAILURE;
  }

  try {
    status = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error during RUN_ALL_TESTS()" << std::endl;
  }
#endif

  return status;
}
