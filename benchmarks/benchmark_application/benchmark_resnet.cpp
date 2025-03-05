/**
 * Copyright (C) 2024 Donghak Park <donghak.park@samsung.com>
 *
 * @file   benchmark_resnet.cpp
 * @date   15 Aug 2024
 * @brief  benchmark test for resnet application
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <array>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include "benchmark/benchmark.h"
#include <fake_data_gen.h>

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

uint64_t get_cpu_freq() {
#if not defined(_WIN32)
  unsigned int freq = 0;
  char cur_cpu_name[512];
  int cpu = sched_getcpu();
  snprintf(cur_cpu_name, sizeof(cur_cpu_name),
           "/sys/devices/system/cpu/cpu%d/cpufreq/scaling_cur_freq", cpu);

  FILE *f = fopen(cur_cpu_name, "r");
  if (f != nullptr) {
    if (fscanf(f, "%d", &freq) != 0) {
      fclose(f);
      return uint64_t(freq) * 1000;
    }
    fclose(f);
  }
#endif
  return 0;
}

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
                                     int kernel_size, bool downsample) {
  using ml::train::createLayer;

  auto scoped_name = [&block_name](const std::string &layer_name) {
    return block_name + "/" + layer_name;
  };
  auto with_name = [&scoped_name](const std::string &layer_name) {
    return withKey("name", scoped_name(layer_name));
  };

  auto create_conv = [&with_name, filters](const std::string &name,
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

  LayerHandle a1 = create_conv("a1", 3, downsample ? 2 : 1,
                               downsample ? "1,1" : "same", input_name);
  LayerHandle a2 =
    createLayer("batch_normalization",
                {with_name("a2"), withKey("activation", "relu"),
                 withKey("momentum", "0.9"), withKey("epsilon", "0.00001")});
  LayerHandle a3 = create_conv("a3", 3, 1, "same", scoped_name("a2"));

  /** skip path */
  LayerHandle b1 = nullptr;
  if (downsample) {
    b1 = create_conv("b1", 1, 2, "0,0", input_name);
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
std::vector<LayerHandle> createResnet18Graph() {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", "3:32:32")}));

  layers.push_back(createLayer(
    "conv2d", {withKey("name", "conv0"), withKey("filters", 64),
               withKey("kernel_size", {3, 3}), withKey("stride", {1, 1}),
               withKey("padding", "same"), withKey("bias_initializer", "zeros"),
               withKey("weight_initializer", "xavier_uniform")}));

  layers.push_back(createLayer(
    "batch_normalization",
    {withKey("name", "first_bn_relu"), withKey("activation", "relu"),
     withKey("momentum", "0.9"), withKey("epsilon", "0.00001")}));

  std::vector<std::vector<LayerHandle>> blocks;

  blocks.push_back(resnetBlock("conv1_0", "first_bn_relu", 64, 3, false));
  blocks.push_back(resnetBlock("conv1_1", "conv1_0", 64, 3, false));
  blocks.push_back(resnetBlock("conv2_0", "conv1_1", 128, 3, true));
  blocks.push_back(resnetBlock("conv2_1", "conv2_0", 128, 3, false));
  blocks.push_back(resnetBlock("conv3_0", "conv2_1", 256, 3, true));
  blocks.push_back(resnetBlock("conv3_1", "conv3_0", 256, 3, false));
  blocks.push_back(resnetBlock("conv4_0", "conv3_1", 512, 3, true));
  blocks.push_back(resnetBlock("conv4_1", "conv4_0", 512, 3, false));

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

ModelHandle createResnet18(bool pre_trained = false) {
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "cross")});

  for (auto &layer : createResnet18Graph()) {
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

void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data,
                  UserDataType &valid_user_data) {

  // setup model
  ModelHandle model = createResnet18();
  model->setProperty(
    {withKey("batch_size", batch_size), withKey("epochs", epochs)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});

  model->setOptimizer(std::move(optimizer));
  model->compile();
  model->initialize();

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  auto dataset_valid = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));
  model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                    std::move(dataset_valid));

  model->train();
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

static void Test_ResnetFull(benchmark::State &state) {

  unsigned int batch_size = 1;
  unsigned int data_split = 128;
  unsigned int epoch = 10;

  std::cout << "batch_size: " << batch_size << " data_split: " << data_split
            << " epoch: " << epoch << std::endl;

  std::array<UserDataType, 2> user_datas;
  user_datas = createFakeDataGenerator(batch_size, 512, data_split);
  auto &[train_user_data, valid_user_data] = user_datas;
  auto check_freq = get_cpu_freq();
  state.counters["check_freq"] = check_freq;
  for (auto _ : state) {
    createAndRun(epoch, batch_size, train_user_data, valid_user_data);
  }
}

BENCHMARK(Test_ResnetFull);
BENCHMARK_MAIN();
