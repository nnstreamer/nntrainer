// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   05 Oct 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug	   No known bugs except for NYI items
 * @brief  This is VGG Example with
 *
 */

#include "bitmap_helpers.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdlib.h>

#include "databuffer.h"
#include "databuffer_func.h"
#include "neuralnet.h"
#include "nntrainer_error.h"
#include "tensor.h"
#include <vector>

/**
 * @brief     Data size for each category
 */
const unsigned int num_class = 100;

const unsigned int num_train = 100;

const unsigned int num_val = 20;

const unsigned int batch_size = 128;

const unsigned int feature_size = 3072;

unsigned int train_count = 0;
unsigned int val_count = 0;

int width = 32;

int height = 32;

int channel = 3;

unsigned int seed;

std::string resource;

float training_loss = 0.0;
float validation_loss = 0.0;
float last_batch_loss = 0.0;

typedef struct {
  unsigned int remain;
  std::vector<unsigned int> duplication;
} counting_info_s;

counting_info_s count_train, count_val;

/**
 * @brief Extract Feature from images
 * @param[out] input_img feature of image
 * @param[out] input_label label of image
 * @param[in] index image index to get feature
 * @param[in] type validation or training
 */
void ExtractFeatures(std::string p, std::vector<float> &input_img,
                     std::vector<float> &input_label, unsigned int index,
                     std::string type) {

  std::string total_label[100] = {
    "apple",      "bridge",    "cockroach",  "hamster",      "motorcycle",
    "plain",      "seal",      "table",      "willow_tree",  "aquarium_fish",
    "bus",        "couch",     "house",      "mountain",     "plate",
    "shark",      "tank",      "wolf",       "baby",         "butterfly",
    "crab",       "kangaroo",  "mouse",      "poppy",        "shrew",
    "telephone",  "woman",     "bear",       "camel",        "crocodile",
    "keyboard",   "mushroom",  "porcupine",  "skunk",        "television",
    "worm",       "beaver",    "can",        "cup",          "lamp",
    "oak_tree",   "possum",    "skyscraper", "tiger",        "bed",
    "castle",     "dinosaur",  "lawn_mower", "orange",       "rabbit",
    "snail",      "tractor",   "bee",        "caterpillar",  "dolphin",
    "leopard",    "orchid",    "raccoon",    "snake",        "train",
    "beetle",     "cattle",    "elephant",   "lion",         "otter",
    "ray",        "spider",    "trout",      "bicycle",      "chair",
    "flatfish",   "lizard",    "palm_tree",  "road",         "squirrel",
    "tulip",      "bottle",    "chimpanzee", "forest",       "lobster",
    "pear",       "rocket",    "streetcar",  "turtle",       "bowl",
    "clock",      "fox",       "man",        "pickup_truck", "rose",
    "sunflower",  "wardrobe",  "boy",        "cloud",        "girl",
    "maple_tree", "pine_tree", "sea",        "sweet_pepper", "whale"};

  std::string path = p + "/train/";

  std::stringstream ss;

  unsigned int class_id = 0;
  bool val = false;

  if (!type.compare("val")) {
    class_id = index / num_val;
    path += total_label[class_id];
    val = true;
  } else {
    class_id = index / num_train;
    path += total_label[class_id];
  }

  if (val) {
    ss << std::setw(4) << std::setfill('0') << (500 - (index % num_val));
  } else {
    ss << std::setw(4) << std::setfill('0') << ((index % num_train) + 1);
  }

  std::string img = path + "/" + ss.str() + ".bmp";

  uint8_t *in = tflite::label_image::read_bmp(img, &width, &height, &channel);

  input_img.resize(channel * width * height);

  for (int c = 0; c < channel; ++c) {
    unsigned int I = c * height * width;
    for (int h = 0; h < height; ++h) {
      unsigned int J = h * width;
      for (int w = 0; w < width; ++w) {
        input_img[I + J + w] = in[I + J + w];
      }
    }
  }

  input_label.resize(num_class);
  input_label[class_id] = 1.0;

  delete[] in;
}

/**
 * @brief  get random number
 * @param[in] min minimum value
 * @param[in] max maximum value
 */
static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand_r(&seed);
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

/**
 * @brief      get data which size is batch for train Directly from image
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_train(float **outVec, float **outLabel, bool *last,
                   void *user_data) {
  std::vector<int> memI;

  unsigned int count = 0;

  if (count_train.remain < batch_size) {
    count_train.remain = num_class * num_train;
    count_train.duplication.clear();
    count_train.duplication.resize(count_train.remain);
    for (unsigned int i = 0; i < count_train.remain; ++i)
      count_train.duplication[i] = i;
    *last = true;
    return 0;
  }

  while (count < batch_size) {
    int nomI = rangeRandom(0, count_train.remain - 1);
    memI.push_back(count_train.duplication[nomI]);
    count_train.remain--;
    count++;
    count_train.duplication.erase(count_train.duplication.begin() + nomI);
  }

  for (unsigned int i = 0; i < count; ++i) {
    std::vector<float> input, label;
    unsigned int index = memI[i];
    ExtractFeatures(resource, input, label, index, "train");
    for (unsigned int j = 0; j < input.size(); ++j)
      outVec[0][i * input.size() + j] = input[j];
    for (unsigned int j = 0; j < label.size(); ++j)
      outLabel[0][i * label.size() + j] = label[j];
  }
  *last = false;
  return 0;
}

/**
 * @brief      get data which size is batch for validation Directly from image
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_val(float **outVec, float **outLabel, bool *last,
                 void *user_data) {
  std::vector<int> memI;
  if (count_val.remain < batch_size) {
    count_val.remain = num_class * num_val;
    count_val.duplication.clear();
    count_val.duplication.resize(count_val.remain);
    for (unsigned int i = 0; i < count_val.remain; ++i)
      count_val.duplication[i] = i;
    *last = true;
    return 0;
  }

  unsigned int count = 0;

  while (count < batch_size) {
    int nomI = rangeRandom(0, count_val.remain - 1);
    memI.push_back(count_val.duplication[nomI]);
    count_val.remain--;
    count++;
    count_val.duplication.erase(count_val.duplication.begin() + nomI);
  }

  for (unsigned int i = 0; i < count; ++i) {
    std::vector<float> input, label;
    unsigned int index = memI[i];
    ExtractFeatures(resource, input, label, index, "val");
    for (unsigned int j = 0; j < input.size(); ++j)
      outVec[0][i * input.size() + j] = input[j];
    for (unsigned int j = 0; j < label.size(); ++j)
      outLabel[0][i * label.size() + j] = label[j];
  }
  *last = false;
  return 0;
}

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] outVec
 * @param[out] outLabel
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
bool getData(std::ifstream &F, std::vector<float> &outVec,
             std::vector<float> &outLabel, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position =
    (uint64_t)((feature_size + num_class) * (uint64_t)id * sizeof(float));

  if (position > file_length) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  for (unsigned int i = 0; i < feature_size; i++)
    F.read((char *)&outVec[i], sizeof(float));
  for (unsigned int i = 0; i < num_class; i++)
    F.read((char *)&outLabel[i], sizeof(float));

  return true;
}

/**
 * @brief      get data which size is batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_train_file(float **outVec, float **outLabel, bool *last,
                        void *user_data) {
  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = num_train;

  // std::string filename = "vgg_trainingSet.dat";
  // std::ifstream F(filename, std::ios::in | std::ios::binary);

  if (data_size * num_class - train_count < batch_size) {
    *last = true;
    train_count = 0;
    return ML_ERROR_NONE;
  }

  count = 0;
  for (unsigned int i = train_count; i < train_count + batch_size; i++) {
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size, 0);
    l.resize(num_class, 0);

    // getData(F, o, l, i);

    for (unsigned int j = 0; j < feature_size; ++j)
      outVec[0][count * feature_size + j] = o[j];
    for (unsigned int j = 0; j < num_class; ++j)
      outLabel[0][count * num_class + j] = l[j];
    count++;
  }

  // F.close();
  *last = false;
  train_count += batch_size;
  return ML_ERROR_NONE;
}

/**
 * @brief      get data which size is batch for validation
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_val_file(float **outVec, float **outLabel, bool *last,
                      void *user_data) {

  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = num_val;

  // std::string filename = "vgg_valSet.dat";
  // std::ifstream F(filename, std::ios::in | std::ios::binary);

  if (data_size * num_class - val_count < batch_size) {
    *last = true;
    val_count = 0;
    return ML_ERROR_NONE;
  }

  count = 0;
  for (unsigned int i = val_count; i < val_count + batch_size; i++) {
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(num_class);

    // getData(F, o, l, i);

    for (unsigned int j = 0; j < feature_size; ++j)
      outVec[0][count * feature_size + j] = o[j];
    for (unsigned int j = 0; j < num_class; ++j)
      outLabel[0][count * num_class + j] = l[j];
    count++;
  }

  // F.close();
  *last = false;
  val_count += batch_size;
  return ML_ERROR_NONE;
}

int main(int argc, char *argv[]) {
  int status = 0;

  if (argc < 3) {
    std::cout << "./nntrainer_vgg vgg.ini resource\n";
    exit(-1);
  }

  seed = time(NULL);
  srand(seed);

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];
  resource = args[1];

  srand(time(NULL));
  std::vector<std::vector<float>> inputVector, outputVector;
  std::vector<std::vector<float>> inputValVector, outputValVector;

  std::vector<float> input_, label_;

  count_train.remain = num_class * num_train;
  count_train.duplication.resize(count_train.remain);

  count_val.remain = num_class * num_val;
  count_val.duplication.resize(count_val.remain);

  for (unsigned int i = 0; i < count_train.remain; ++i)
    count_train.duplication[i] = i;

  for (unsigned int i = 0; i < count_val.remain; ++i)
    count_val.duplication[i] = i;

  std::shared_ptr<nntrainer::DataBufferFromCallback> DB =
    std::make_shared<nntrainer::DataBufferFromCallback>();
  DB->setGeneratorFunc(nntrainer::DatasetDataUsageType::DATA_TRAIN,
                       getBatch_train_file);
  DB->setGeneratorFunc(nntrainer::DatasetDataUsageType::DATA_VAL,
                       getBatch_val_file);

  /**
   * @brief     Neural Network Create & Initialization
   */
  nntrainer::NeuralNetwork NN;
  try {
    NN.loadFromConfig(config);
  } catch (...) {
    std::cerr << "Error during loadFromConfig" << std::endl;
    return 0;
  }

  try {
    NN.compile();
    NN.initialize();
  } catch (...) {
    std::cerr << "Error during init" << std::endl;
    return 0;
  }

  try {
    NN.readModel();
    NN.setDataBuffer((DB));
    NN.train();
    training_loss = NN.getTrainingLoss();
    validation_loss = NN.getValidationLoss();
    last_batch_loss = NN.getLoss();
  } catch (...) {
    std::cerr << "Error during train" << std::endl;
    return 0;
  }

  return status;
}
