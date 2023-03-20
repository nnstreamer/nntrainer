// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijong.moon@samsung.com>
 *
 * @file   resnet.cpp
 * @date   20 March 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @author Hyeonseok Lee <hs89.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Resnet Application for android
 *
 */

#include "resnet.h"
#include "tensor_dim.h"
#include <ctime>
#include <sstream>

/** cache loss values post training for test */
float training_loss = 0.0;
float validation_loss = 0.0;

ml::train::RunStats training;
ml::train::RunStats validation;
ModelHandle model;
bool stop = false;
std::string test_result = "";
std::string infer_result = "";
bool model_destroyed = true;
bool last = false;

std::vector<std::string> split(std::string input, char delimiter) {
  std::vector<std::string> answer;
  std::stringstream ss(input);
  std::string temp;
  while (getline(ss, temp, delimiter)) {
    answer.push_back(temp);
  }
  return answer;
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

void setStop() {
  stop = true;
  last = true;
}

bool stop_cb(void *userdata) {
  bool *ret = reinterpret_cast<bool *>(userdata);
  return *ret;
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

  /** residual path */
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
std::vector<LayerHandle> createResnet18Graph(std::string input_shape,
                                             unsigned int unit) {
  using ml::train::createLayer;

  std::vector<LayerHandle> layers;

  ml::train::TensorDim dim(input_shape);

  layers.push_back(createLayer(
    "input", {withKey("name", "input0"), withKey("input_shape", input_shape)}));

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

  layers.push_back(createLayer(
    "pooling2d", {withKey("name", "last_p1"), withKey("pooling", "average"),
                  withKey("pool_size", {dim.height() / 8, dim.width() / 8})}));

  layers.push_back(createLayer("flatten", {withKey("name", "last_f1")}));
  layers.push_back(
    createLayer("fully_connected",
                {withKey("unit", unit), withKey("activation", "softmax")}));

  return layers;
}

/// @todo update createResnet18 to be more generic
ml::train::Model *createResnet18(std::string input_shape, unsigned int unit) {
  /// @todo support "LOSS : cross" for TF_Lite Exporter
  ANDROID_LOG_D("create Model in JNI");
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET,
                                 {withKey("loss", "cross")});

  for (auto layer : createResnet18Graph(input_shape, unit)) {
    model->addLayer(layer);
  }
  ANDROID_LOG_D("create Model in JNI DONE");
  ANDROID_LOG_D("--------------------------------------");
  model_destroyed = false;
  return model.get();
}

int trainData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::resnet::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

int validData_cb(float **input, float **label, bool *last, void *user_data) {
  auto data = reinterpret_cast<nntrainer::resnet::DataLoader *>(user_data);

  data->next(input, label, last);
  return 0;
}

/// @todo maybe make num_class also a parameter
void createAndRun(unsigned int epochs, unsigned int batch_size,
                  UserDataType &train_user_data, UserDataType &valid_user_data,
                  std::string bin_path, std::string best_path,
                  ml::train::Model *model_) {

  stop = false;

  model_->setProperty(
    {withKey("batch_size", batch_size), withKey("epochs", epochs),
     withKey("save_path", bin_path), withKey("save_best_path", best_path)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model_->setOptimizer(std::move(optimizer));

  ANDROID_LOG_D("start compile");
  int status = model_->compile();
  ANDROID_LOG_D("compile finished");
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  ANDROID_LOG_D("start initialize");
  status = model_->initialize();
  ANDROID_LOG_D("initialize finished");
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  ANDROID_LOG_D("start load");
  // model_->load ("/data/local/tmp/resnet/pretrained_resnet18.bin",
  //     ml::train::ModelFormat::MODEL_FORMAT_BIN);
  ANDROID_LOG_D("load finished");

  auto dataset_train = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, trainData_cb, train_user_data.get());
  auto dataset_valid = ml::train::createDataset(
    ml::train::DatasetType::GENERATOR, validData_cb, valid_user_data.get());

  model_->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                     std::move(dataset_train));
  model_->setDataset(ml::train::DatasetModeType::MODE_VALID,
                     std::move(dataset_valid));

  ANDROID_LOG_D("start train");
  model_->train({}, stop_cb, &stop);
  ANDROID_LOG_D("train finished");
}

std::array<UserDataType, 2>
createFakeDataGenerator(unsigned int batch_size,
                        unsigned int simulated_data_size,
                        unsigned int data_split) {
  UserDataType train_data(new nntrainer::resnet::RandomDataLoader(
    {{batch_size, 3, 256, 256}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));
  UserDataType valid_data(new nntrainer::resnet::RandomDataLoader(
    {{batch_size, 3, 256, 256}}, {{batch_size, 1, 1, 100}},
    simulated_data_size / data_split));

  return {std::move(train_data), std::move(valid_data)};
}

std::array<UserDataType, 2>
createDirDataGenerator(const std::string dir, float split_ratio,
                       unsigned int label_len, unsigned int channel,
                       unsigned int width, unsigned int height, bool is_train) {

  UserDataType train_data(new nntrainer::resnet::DirDataLoader(
    (dir).c_str(), split_ratio, label_len, channel, width, height, true));
  UserDataType valid_data(new nntrainer::resnet::DirDataLoader(
    (dir).c_str(), split_ratio, label_len, channel, width, height, false));
  return {std::move(train_data), std::move(valid_data)};
}

int init(int argc, char *argv[], ml::train::Model *model_) {
  if (argc < 11) {
    std::cerr
      << "usage: ./main [{data_directory}|\"fake\"] [batchsize] [data_split] "
         "[epoch] [channel] [height] [width] [ bin path ] [ bin best path ] "
         "[num class]\n"
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
  unsigned int channel = std::stoul(argv[5]);
  unsigned int height = std::stoul(argv[6]);
  unsigned int width = std::stoul(argv[7]);
  std::string bin_path = argv[8];
  std::string bin_best_path = argv[9];
  unsigned int num_class = std::stoul(argv[10]);
  float split_ratio = 0.1;

  ANDROID_LOG_D("---data_dir: %s", data_dir.c_str());
  ANDROID_LOG_D("---batch_size: %d", batch_size);
  ANDROID_LOG_D("---data_split: %d", data_split);
  ANDROID_LOG_D("--- epoch: %d", epoch);
  ANDROID_LOG_D("---input shape: %d:%d:%d", channel, height, width);
  ANDROID_LOG_D("num_class: %d", num_class);
  ANDROID_LOG_D("bin path: %s", bin_path.c_str());
  ANDROID_LOG_D("bin best path: %s", bin_best_path.c_str());

  std::array<UserDataType, 2> user_datas;

  try {
    if (data_dir == "fake") {
      user_datas = createFakeDataGenerator(batch_size, 512, data_split);
    } else {
      user_datas = createDirDataGenerator(
        (std::string(data_dir) + "/train/").c_str(), split_ratio, num_class,
        channel, width, height, true);
    }
  } catch (const std::exception &e) {
    std::cerr << "uncaught error while creating data generator! details: "
              << e.what() << std::endl;
    return EXIT_FAILURE;
  }

  auto &[train_user_data, valid_user_data] = user_datas;

  try {
    createAndRun(epoch, batch_size, train_user_data, valid_user_data, bin_path,
                 bin_best_path, model_);
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

  model_destroyed = true;
  return 0;
}

std::string displayProgress(const int count, float loss, int batch_size) {
  int barWidth = 20;
  std::stringstream ssInt;
  ssInt << count * batch_size;
  std::string str = ssInt.str();
  int len = str.size();
  std::string ret;

  int pad_left = (barWidth - len) / 2;
  int pad_right = barWidth - pad_left - len;
  std::string out_str =
    std::string(pad_left, ' ') + str + std::string(pad_right, ' ');

  ret = " [ " + out_str + " ] " + " ( Training Loss: " + std::to_string(loss) +
        " ) ";

  return ret;
}

std::string getTestingStatus() { return test_result; }

bool modelDestroyed() { return model_destroyed; }

std::string testModel(int argc, char *argv[], ml::train::Model *model_) {
  if (argc < 11) {
    std::cerr
      << "usage: ./main [{data_directory}|\"fake\"] [batchsize] [data_split] "
         "[epoch] [channel] [height] [width] [ bin path ] [ bin best path ] "
         "[num class]\n"
      << "when \"fake\" is given, original data size is assumed 512 for both "
         "train and validation\n";
    throw std::invalid_argument("wrong argument");
  }

  stop = false;
  std::string data_dir = argv[1];
  unsigned int batch_size = std::stoul(argv[2]);
  unsigned int data_split = std::stoul(argv[3]);
  unsigned int epoch = std::stoul(argv[4]);
  unsigned int channel = std::stoul(argv[5]);
  unsigned int height = std::stoul(argv[6]);
  unsigned int width = std::stoul(argv[7]);
  std::string bin_path = argv[8];
  std::string bin_best_path = argv[9];
  unsigned int num_class = std::stoul(argv[10]);

  test_result = "";

  ANDROID_LOG_D("---data_dir: %s", data_dir.c_str());
  ANDROID_LOG_D("---batch_size: %d", batch_size);
  ANDROID_LOG_D("--- epoch: %d", epoch);
  ANDROID_LOG_D("---input shape: %d:%d:%d", channel, height, width);
  ANDROID_LOG_D("num_class: %d", num_class);
  ANDROID_LOG_D("bin path: %s", bin_path.c_str());
  ANDROID_LOG_D("bin best path: %s", bin_best_path.c_str());

  float *input = new float[channel * height * width];

  float *label = new float[num_class];

  std::array<UserDataType, 2> user_datas;

  last = false;
  std::string input_dim = std::to_string(channel) + ":" +
                          std::to_string(height) + ":" + std::to_string(width);

  model_->setProperty({withKey("batch_size", batch_size),
                       withKey("epochs", epoch), withKey("save_path", bin_path),
                       withKey("save_best_path", bin_best_path)});

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model_->setOptimizer(std::move(optimizer));

  ANDROID_LOG_D("start compile");
  int status = model_->compile();
  ANDROID_LOG_D("compile finished");
  if (status) {
    delete[] input;
    delete[] label;
    throw std::invalid_argument("model compilation failed!");
  }

  ANDROID_LOG_D("start initialize");
  status = model_->initialize();
  ANDROID_LOG_D("initialize finished");
  if (status) {
    delete[] input;
    delete[] label;
    throw std::invalid_argument("model initialization failed!");
  }

  model_->load(bin_best_path);
  user_datas =
    createDirDataGenerator((std::string(data_dir) + "/test").c_str(), 0.0,
                           num_class, channel, width, height, false);

  auto &[test_user_data, dummy] = user_datas;

  int right = 0;
  int count = 0;
  float accuracy;

  while (!last && !stop) {
    std::vector<float *> result;
    std::vector<float *> in;
    std::vector<float *> l;
    test_user_data->next(&input, &label, &last);

    std::string file_name =
      static_cast<nntrainer::resnet::DirDataLoader *>(test_user_data.get())
        ->getCurFileName();

    in.push_back(input);
    l.push_back(label);
    result = model_->inference(1, in, l);
    int index = count * (num_class);
    bool answer = false;

    int result_ans = 0;
    float result_val = 0.0;
    int label_ans = 0;
    float label_val = 0.0;

    for (unsigned int i = 0; i < num_class; ++i) {
      if (result[0][i] > result_val) {
        result_val = result[0][i];
        result_ans = i;
      }
      if (label[i] > label_val) {
        label_val = label[i];
        label_ans = i;
      }
    }

    if (result_ans == label_ans) {
      answer = true;
      right++;
    }

    std::vector<std::string> splited = split(file_name, '/');
    test_result += " " + std::to_string(count) + " ] " + (splited.rbegin()[1]) +
                   "/" + (splited.rbegin()[0]) + " : correct? " +
                   (answer ? "RIGHT" : "WRONG") + "\n";

    in.clear();
    l.clear();

    count++;
  }

  accuracy = ((float)right / (float)count) * 100.0;

  test_result += "\n\nAccuarcy ( " + std::to_string(right) + " / " +
                 std::to_string(count) + " )  : " + std::to_string(accuracy) +
                 " %\n";
  model_destroyed = true;

  delete[] input;
  delete[] label;

  return test_result;
}

static int argmax(float *vec, unsigned int num_class) {
  int ret = 0;
  float val = 0.0;
  for (unsigned int i = 0; i < num_class; i++) {
    if (val < vec[i]) {
      val = vec[i];
      ret = i;
    }
  }
  return ret;
}

std::string inferModel(int argc, char *argv[], uint8_t *pBmp,
                       ml::train::Model *model_) {
  ANDROID_LOG_D("%d", argc);
  if (argc < 6) {
    std::cerr
      << "usage: ./main [{data_directory}|\"fake\"] [batchsize] [data_split] "
         "[epoch] [channel] [height] [width] [ bin path ] [ bin best path ] "
         "[num class]\n"
      << "when \"fake\" is given, original data size is assumed 512 for both "
         "train and validation\n";
    throw std::invalid_argument("wrong argument");
  }

  stop = false;
  std::string file_path = argv[0];
  unsigned int channel = std::stoul(argv[1]);
  unsigned int height = std::stoul(argv[2]);
  unsigned int width = std::stoul(argv[3]);
  unsigned int num_class = std::stoul(argv[4]);
  std::string bin_best_path = argv[5];

  infer_result = "";

  float *input = new float[channel * height * width];
  int fs = width * height;
  int fs2 = fs * 2;

  auto start = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < height; i++) {
    int hh = i * width;
    int h4 = hh * 4;
    for (unsigned j = 0; j < width; j++) {
      int cc = j * 4;
      input[hh + j] = pBmp[h4 + cc];
      input[fs + hh + j] = pBmp[h4 + cc + 1];
      input[fs2 + hh + j] = pBmp[h4 + cc + 2];
    }
  }

  auto end = std::chrono::system_clock::now();

  std::chrono::duration<double> elapsed_seconds = end - start;
  ANDROID_LOG_D("prepare input time");

  ANDROID_LOG_D("prepare input : %f", elapsed_seconds.count());

  std::string input_dim = std::to_string(channel) + ":" +
                          std::to_string(height) + ":" + std::to_string(width);

  auto optimizer = ml::train::createOptimizer("adam", {"learning_rate=0.001"});
  model_->setOptimizer(std::move(optimizer));

  ANDROID_LOG_D("start compile");
  int status = model_->compile();
  ANDROID_LOG_D("compile finished");
  if (status) {
    throw std::invalid_argument("model compilation failed!");
  }

  ANDROID_LOG_D("start initialize");
  status = model_->initialize();
  ANDROID_LOG_D("initialize finished");
  if (status) {
    throw std::invalid_argument("model initialization failed!");
  }

  model_->load(bin_best_path);

  auto end1 = std::chrono::system_clock::now();

  elapsed_seconds = end1 - end;

  ANDROID_LOG_D("prepare model : %f", elapsed_seconds.count());

  std::vector<float *> in;
  std::vector<float *> result;
  std::vector<float *> label;
  in.push_back(input);

  result = model_->inference(1, in, label);

  int result_ans = argmax(result[0], num_class);

  auto end2 = std::chrono::system_clock::now();

  elapsed_seconds = end2 - end1;

  ANDROID_LOG_D("infer input : %f", elapsed_seconds.count());

  infer_result += "class is : " + std::to_string(result_ans) + "\n";
  infer_result += "-----------------------------------\n";

  for (unsigned int i = 0; i < num_class; ++i)
    infer_result += std::to_string(result[0][i]) + " ";

  infer_result += "\n";

  in.clear();

  model_destroyed = true;

  delete[] input;

  return infer_result;
}
