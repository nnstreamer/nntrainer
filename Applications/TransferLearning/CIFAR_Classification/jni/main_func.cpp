/**
 * Copyright (C) 2019 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	main_func.cpp
 * @date	04 December 2019
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Classification Example with one FC Layer
 *              The base model for feature extractor is mobilenet v2 with
 * 1280*7*7 feature size. It read the Classification.ini in res directory and
 * run according to the configuration.
 *
 */
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include <dataset.h>
#include <model.h>

#include <bitmap_helpers.h>

#include <engine.h>
#define TRAINING true

/**
 * @brief     Data size for each category
 */
const unsigned int total_train_data_size = 100;

const unsigned int total_val_data_size = 10;

const unsigned int total_test_data_size = 100;

const unsigned int buffer_size = 100;

/**
 * @brief     Number of category : Three
 */
const unsigned int total_label_size = 10;

const unsigned int batch_size = 32;

const unsigned int image_size = 224 * 224 * 3;

unsigned int SEED = 1234;

std::string data_path;

using namespace std;

bool duplicate[total_label_size * total_train_data_size];
bool valduplicate[total_label_size * total_val_data_size];

string total_label[10] = {"airplane", "automobile", "bird",  "cat",  "deer",
                          "dog",      "frog",       "horse", "ship", "truck"};
/**
 * @brief     step function
 * @param[in] x value to be distinguished
 * @retval 0.0 or 1.0
 */
float stepFunction(float x) {
  if (x > 0.9) {
    return 1.0;
  }

  if (x < 0.1) {
    return 0.0;
  }

  return x;
}

/**
 * @brief     Generate Random integer value between min to max
 * @param[in] min : minimum value
 * @param[in] max : maximum value
 * @retval    min < random value < max
 */
static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand_r(&SEED);
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

/**
 * @brief     Load Image data to the given container
 * @param[in] filename name of the file
 * @param[out] image container to load the image data
 */
void getImage(const string filename, float *image) {
  int width, height, channels;
  uint8_t *in =
    tflite::label_image::read_bmp(filename, &width, &height, &channels);

  if (width * height * channels != image_size)
    throw std::runtime_error("Dataset file image size error");

  for (size_t i = 0; i < image_size; i++) {
    image[i] = ((float)in[i]) / 255.0;
  }

  delete[] in;
}

/**
 * @brief     Get the image with the given index and its label
 * @param[in] data_path path containing the data
 * @param[out] input container to hold the input
 * @param[out] output container to hold the output
 * @param[in] n index of the data to be loaded
 * @param[in] is_val if this image is for training or validation
 */
void getNthImage(std::string data_path, float *input, float *output, int n,
                 bool is_val = false) {
  std::string path = data_path;
  unsigned int data_size = total_train_data_size;
  if (is_val)
    data_size = total_val_data_size;

  int label_idx = n / data_size;
  int image_idx = n % data_size;

  path += "train/" + total_label[label_idx] + "/";

  std::stringstream ss;
  if (is_val)
    ss << std::setw(4) << std::setfill('0') << (5000 - image_idx);
  else
    ss << std::setw(4) << std::setfill('0') << (image_idx + 1);

  path += ss.str() + ".bmp";

  getImage(path, input);

  memset(output, 0, total_label_size * sizeof(float));
  output[label_idx] = 1;
}

/**
 * @brief     Fills a batch of data into the containers
 * @param[out] outVev input data
 * @param[out] outLabel label data
 * @param[out] last if this is last data
 * @param[in] is_val if this data is for validation or training
 */
int getBatch(float **outVec, float **outLabel, bool *last, bool is_val) {
  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = total_train_data_size;
  bool *dupl = duplicate;

  if (is_val) {
    data_size = total_val_data_size;
    dupl = valduplicate;
  }

  for (unsigned int i = 0; i < total_label_size * data_size; i++) {
    if (!dupl[i])
      count++;
  }

  if (count < batch_size) {
    for (unsigned int i = 0; i < total_label_size * data_size; ++i) {
      dupl[i] = false;
    }
    *last = true;
    return 0;
  }

  count = 0;
  while (count < batch_size) {
    int nomI = rangeRandom(0, total_label_size * data_size - 1);
    if (!dupl[nomI]) {
      memI.push_back(nomI);
      dupl[nomI] = true;
      count++;
    }
  }

  for (unsigned int i = 0; i < count; i++) {
    getNthImage(data_path, &outVec[0][i * image_size],
                &outLabel[0][i * total_label_size], memI[i], is_val);
  }

  *last = false;
  return 0;
}

/**
 * @brief      get data which size is batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_train(float **outVec, float **outLabel, bool *last,
                   void *user_data) {
  return getBatch(outVec, outLabel, last, false);
}

/**
 * @brief      get data which size is batch for validation
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_val(float **outVec, float **outLabel, bool *last,
                 void *user_data) {
  return getBatch(outVec, outLabel, last, true);
}

/**
 * @brief     create NN
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "./nntrainer_classification Config.ini resources\n";
    exit(0);
  }
  const vector<string> args(argv + 1, argv + argc);
  std::string config = args[0];
  data_path = args[1] + "/";

  /// @todo add api version of this
  try {
    nntrainer::Engine::Global().setWorkingDirectory(data_path);
  } catch (std::invalid_argument &e) {
    std::cerr << "setting data_path failed, pwd is used instead";
  }

  srand(SEED);
  std::vector<std::vector<float>> inputVector, outputVector;
  std::vector<std::vector<float>> inputValVector, outputValVector;
  std::vector<std::vector<float>> inputTestVector, outputTestVector;

  /* This is to check duplication of data */
  memset(duplicate, 0, sizeof(bool) * total_label_size * total_train_data_size);
  memset(valduplicate, 0,
         sizeof(bool) * total_label_size * total_val_data_size);

  /**
   * @brief     Data buffer Create & Initialization
   */
  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_val;
  try {
    dataset_train =
      createDataset(ml::train::DatasetType::GENERATOR, getBatch_train);
    dataset_val =
      createDataset(ml::train::DatasetType::GENERATOR, getBatch_val);
  } catch (...) {
    std::cerr << "Error creating dataset" << std::endl;
    return 1;
  }

  std::unique_ptr<ml::train::Model> model;
  /**
   * @brief     Neural Network Create & Initialization
   */
  try {
    model = createModel(ml::train::ModelType::NEURAL_NET);
    model->load(config, ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN);
  } catch (...) {
    std::cerr << "Error during loadFromConfig" << std::endl;
    return 1;
  }
  try {
    model->compile();
    model->initialize();
  } catch (...) {
    std::cerr << "Error during init" << std::endl;
    return 1;
  }
  model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
  model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);

  /**
   * @brief     Neural Network Train & validation
   */
  try {
    model->train();
  } catch (...) {
    std::cerr << "Error during train" << std::endl;
    return 1;
  }

  /**
   * @brief     Finalize NN
   */
  return 0;
}
