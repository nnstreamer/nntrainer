// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    NNTrianer.cpp
 * @date    08 Sept 2021
 * @see     https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is simple nntrainer implementaiton with JNI
 *
 */

#include <android/log.h>
#include <jni.h>

#include <array>
#include <chrono>
#include <ctime>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

#include <layer.h>
#include <model.h>
#include <optimizer.h>

#include <cifar_dataloader.h>

#define LOG_TAG "nntrainer"

#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#define WIDTH 32
#define HEIGHT 32

using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;

using UserDataType = std::unique_ptr<nntrainer::util::DataLoader>;

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

  LayerHandle a1 = create_conv("a1", 3, downsample ? 2 : 1, "same", input_name);
  LayerHandle a2 = createLayer(
    "batch_normalization", {with_name("a2"), withKey("activation", "relu")});
  LayerHandle a3 = create_conv("a3", 3, 1, "same", scoped_name("a2"));

  /** skip path */
  LayerHandle b1 = nullptr;
  if (downsample) {
    b1 = create_conv("b1", 1, 2, "same", input_name);
  }

  const std::string skip_name = b1 ? scoped_name("b1") : input_name;

  LayerHandle c1 = createLayer(
    "Addition",
    {with_name("c1"), withKey("input_layers", {scoped_name("a3"), skip_name})});

  LayerHandle c2 =
    createLayer("batch_normalization",
                {withKey("name", block_name), withKey("activation", "relu")});

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

  layers.push_back(createLayer("input", {
                                          withKey("name", "inputlayer"),
                                          withKey("input_shape", "3:32:32"),
                                        }));

  layers.push_back(
    createLayer("conv2d", {
                            withKey("name", "conv0"),
                            withKey("filters", 64),
                            withKey("kernel_size", {3, 3}),
                            withKey("stride", {1, 1}),
                            withKey("padding", "same"),
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
  blocks.push_back(resnetBlock("conv4_0", "conv3_1", 512, 3, true));
  blocks.push_back(resnetBlock("conv4_1", "conv4_0", 512, 3, false));

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

/**
 * @brief create Neural Network Graph
 *
 * @param input_dim : Input Dimension, 1:1:17
 * @param label_size : label size (number of rooms)
 * @return Layers
 */
std::vector<LayerHandle> createGraph(std::string input_dim,
                                     const int label_size) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  layers = createResnet18Graph();

  return layers;
}

/**
 * @brief create Neural Network Model
 *
 * @param input_dim : Input Dimension, 1:1:17
 * @param label_size : label size (number of rooms)
 * @return Model
 */
ModelHandle createNetwork(std::string input_dim, unsigned int label_size) {
  ModelHandle model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                             {withKey("loss", "mse")});

  for (auto layers : createGraph(input_dim, label_size)) {
    model->addLayer(layers);
  }

  return model;
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

/**
 * @brief run train
 *
 * @param epochs
 * @param batch_size
 * @param input_dim : 1:1:17
 * @param label_len : label size - number of rooms
 * @param train_user_data : user data set for train
 * @param valid_user_data : user data set for validation
 * @param bin_file_path : binary weight savging data path
 * @param best_bin_file_path : binary weight savging data path for best accuracy
 * epoch
 */
void createAndRun(float *data, unsigned int epochs, unsigned int batch_size,
                  unsigned int data_size, unsigned int data_len,
                  unsigned int label_len, const char *image_dir,
                  const char *bin_file_path, const char *best_bin_file_path) {

  std::string input_dim;

  input_dim = "3:32:32";

  unsigned int data_split = 1;

  ModelHandle model = createNetwork(input_dim, label_len);

  model->setProperty({withKey("batch_size", batch_size),
                      withKey("epochs", epochs),
                      withKey("save_path", bin_file_path),
                      withKey("save_best_path", best_bin_file_path)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=1e-4"});

  model->setOptimizer(std::move(optimizer));
  LOGW("done opt\n");

  int status = model->compile();

  LOGW("done compile\n");

  status = model->initialize();
  LOGW("done initialize\n");

  unsigned int vd_size = data_size * 0.1;

  if (vd_size < batch_size)
    vd_size = batch_size;

  unsigned int td_size = data_size - vd_size;

  if (td_size < batch_size) {
    LOGW("done data\n");
    throw std::invalid_argument(
      "training size and validation size is greater than batch_size");
  }

  LOGW("done initialize %d = %d + %d\n", data_size, td_size, vd_size);
  if (td_size < 1.0)
    throw std::invalid_argument("training data size needs to be at least 1");
  LOGW("done data\n");

  std::array<UserDataType, 2> user_datas;

  user_datas = createFakeDataGenerator(batch_size, 512, data_split);

  auto &[train_user_data, valid_user_data] = user_datas;

  // LOGW("loading %s\n",
  //      (std::string(image_dir) + "/pretrained_resnet18.bin").c_str());
  // model->load(std::string(image_dir) + "/pretrained_resnet18.bin");
  // LOGW("load weight\n");

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());

  auto dataset_valid = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                    std::move(dataset_train));
  model->setDataset(ml::train::DatasetModeType::MODE_VALID,
                    std::move(dataset_valid));

  LOGW("ready dataset\n");

  LOGW("start train\n");
  model->train();
}

/**
 * @brief test the network
 *
 * @param data vector which has data + label
 * @param[out] out : result of neural network
 * @param data_size : number of data
 * @param data_len : length of one data
 * @param label_len : length of label
 * @param bin_file_path : file location of binary weight file to be loaded
 * @return UserDataType
 */
void testing(float *data, float *out, unsigned data_size, unsigned int data_len,
             unsigned int label_len, const char *dir_path,
             const char *bin_file_path, unsigned int &correct) {
  unsigned int d_size = data_size;
  unsigned int d_len = data_len;
  unsigned int l_len = label_len;
  UserDataType user_data(new nntrainer::util::RandomDataLoader(
    {{d_size, 3, 32, 32}}, {{d_size, 1, 1, l_len}}, d_size));

  LOGW("%s %d %d %d\n", bin_file_path, data_size, d_len, label_len);

  float *input = new float[d_len];
  float *label = new float[l_len];
  bool last = false;
  std::string input_dim;

  input_dim = "3:32:32";

  ModelHandle model = createNetwork(input_dim, l_len);
  model->setProperty({withKey("save_path", "weight.bin")});
  auto optimizer = ml::train::createOptimizer(
    "adam", {"learning_rate=1e-3", "beta1=0.9", "beta2=0.999", "epsilon=1e-7"});
  model->setOptimizer(std::move(optimizer));

  int status = model->compile();

  status = model->initialize();

  model->load(bin_file_path);

  LOGW("careteDataGen");

  int right = 0;
  int count = 0;

  while (!last) {
    std::vector<float *> result;
    std::vector<float *> in;
    std::vector<float *> l;
    user_data->next(&input, &label, &last);

    in.push_back(input);
    l.push_back(label);

    result = model->inference(1, in, l);
    int index = count * (l_len);

    int result_ans = 0;
    float results_val = 0.0;
    int label_ans = 0;
    float label_val = 0.0;

    for (unsigned int i = 0; i < l_len; ++i) {
      out[index + i] = result[0][i];
      if (result[0][i] > results_val) {
        results_val = result[0][i];
        result_ans = i;
      }
      if (label[i] > label_val) {
        label_val = label[i];
        label_ans = i;
      }
    }

    if (result_ans == label_ans)
      right++;

    LOGI("%d need to be %d\n", result_ans, label_ans);
    in.clear();
    l.clear();

    count++;
  }

  correct = right;

  delete[] input;
  delete[] label;
}

/**
 * @brief train the network
 *
 * @param traindata vector which has data + label
 * @param b_size : batch size
 * @param ep : epochs
 * @param d_size : number of data
 * @param d_len : length of one data
 * @param l_len : length of label
 * @param bin_file_path : file location of binary weight file
 * @param best_bin_file_path : file location of binary weight file with best
 * accuracy
 * @return int
 */
int train(float *traindata, int b_size, int ep, unsigned int d_size,
          unsigned int d_len, unsigned int l_len, const char *image_dir,
          const char *bin_file_path, const char *best_bin_file_path) {
  unsigned int batch_size = b_size;
  unsigned int epoch = ep;

  try {
    createAndRun(traindata, epoch, batch_size, d_size, d_len, l_len, image_dir,
                 bin_file_path, best_bin_file_path);
  } catch (std::exception &e) {
    std::cerr << "uncaught error while running! details: " << e.what() << '\n';
    return 1;
  }

  return 0;
}

void shuffle(float *data, int d_size, int d_len) {
  std::mt19937 rng;
  std::vector<float> temp;
  std::vector idxes = std::vector<unsigned int>(d_size);
  std::iota(idxes.begin(), idxes.end(), 0);
  std::shuffle(idxes.begin(), idxes.end(), rng);

  std::string o = "";
  for (unsigned int i = 0; i < d_size; ++i) {
    int idx = idxes[i] * (d_len + 1);
    for (unsigned int j = 0; j < d_len + 1; ++j) {
      temp.push_back(data[idx + j]);
      o = o + " " + std::to_string(data[idx + j]);
    }
    o = o + "\n";
  }

  LOGW("%s", o.c_str());
  for (unsigned int i = 0; i < d_size * (d_len + 1); ++i) {
    data[i] = temp[i];
  }
}

/**
 * @brief jni interface for train
 *
 * @param train_values vector which has data + label
 * @param batch_size : batch size
 * @param epoch : epochs
 * @param data_size : number of data
 * @param data_len : length of one data
 * @param label_len : length of label
 * @param bin_file_path : file location of binary weight file
 * @param best_bin_file_path : file location of binary weight file with best
 * accuracy
 * @return int
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_samsung_sr_nntr_nntrainer_NNTrainerNative_nntrainerTrain(
  JNIEnv *env, jobject thiz, jfloatArray train_values, jint batch_size,
  jint data_size, jint data_len, jint label_len, jstring directory_path,
  jstring bin_file_path, jstring best_bin_file_path, int epoch) {
  jfloat *taluesjf = env->GetFloatArrayElements(train_values, 0);
  float *train_data = taluesjf;
  const char *file_path = env->GetStringUTFChars(bin_file_path, 0);
  const char *best_file_path = env->GetStringUTFChars(best_bin_file_path, 0);
  const char *dir_path = env->GetStringUTFChars(directory_path, 0);
  int ep = epoch;
  int b_size = batch_size;
  int d_size = data_size;
  int d_len = data_len;
  int l_len = label_len;

  try {
    train(train_data, b_size, ep, d_size, d_len, l_len, dir_path, file_path,
          best_file_path);
  } catch (std::exception &e) {
    LOGW("error: %s", e.what());
  }

  LOGW("Trainer Done");

  return 0;
}

/**
 * @brief jni interface for testing
 *
 * @param test_data vector which has data + label
 * @param[out] output : result
 * @param data_size : number of data
 * @param data_len : length of one data
 * @param label_len : length of label
 * @param bin_file_path : file location of binary weight file to be loadded.
 * @return int
 */
extern "C" JNIEXPORT jint JNICALL
Java_com_samsung_sr_nntr_nntrainer_NNTrainerNative_nntrainerTest(
  JNIEnv *env, jobject thiz, jfloatArray test_data, jfloatArray output,
  jint data_size, jint data_len, jint label_len, jstring directory_path,
  jstring bin_file_path, jint correct) {
  float *data = env->GetFloatArrayElements(test_data, nullptr);
  float *out = env->GetFloatArrayElements(output, nullptr);
  const char *dir_path = env->GetStringUTFChars(directory_path, 0);
  const char *file_path = env->GetStringUTFChars(bin_file_path, 0);

  int d_size = data_size;
  int d_len = data_len;
  int l_len = label_len;
  unsigned int right = 0;

  LOGW("Start Testing %d %d %d", d_size, d_len, l_len);
  testing(data, out, d_size, d_len, l_len, dir_path, file_path, right);

  for (unsigned int i = 0; i < d_size; ++i) {
    std::string s = "";
    int index = i * (l_len);
    for (unsigned int j = 0; j < l_len; ++j) {
      s = s + std::to_string(out[index + j]) + " ";
    }
    LOGW("%s\n", s.c_str());
  }

  env->ReleaseFloatArrayElements(output, out, 0);
  correct = right;

  jint ret = right;
  LOGW("Testing Done: %d", right);

  return ret;
}
