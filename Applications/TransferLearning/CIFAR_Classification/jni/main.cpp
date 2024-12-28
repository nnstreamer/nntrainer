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
 * @file	main.cpp
 * @date	04 December 2019
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 * @brief	This is Classification Example with one FC Layer
 *              The base model for feature extractor is mobilenet v2 with
 * 1280*7*7 feature size. It read the Classification.ini in res directory and
 * run according to the configureation.
 *
 */

#include "bitmap_helpers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/gen_op_registration.h"
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include <engine.h>
#include <neuralnet.h>
#include <tensor.h>

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

/**
 * @brief     Max Epochs
 */
const unsigned int iteration = 3000;

const unsigned int batch_size = 32;

const unsigned int feature_size = 62720;

using namespace std;

/**
 * @brief     location of resources ( ../../res/ )
 */
string data_path;

bool duplicate[total_label_size * total_train_data_size];
bool valduplicate[total_label_size * total_val_data_size];

unsigned int seed;

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
    x = rand_r(&seed);
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

/**
 * @brief     Get Feature vector from tensorflow lite
 *            This creates interpreter & inference with ssd tflite
 * @param[in] filename input file path
 * @param[out] feature_input save output of tflite
 */
void getFeature(const string filename, vector<float> &feature_input) {
  int input_dim[4];
  int output_dim[4];
  std::string model_path = data_path + "mobilenetv2.tflite";
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  assert(model != NULL);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  const std::vector<int> &input_idx_list = interpreter->inputs();
  const std::vector<int> &output_idx_list = interpreter->outputs();

  for (int i = 0; i < 4; i++) {
    input_dim[i] = 1;
    output_dim[i] = 1;
  }

  int len = interpreter->tensor(input_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(input_idx_list[0])->dims->data,
                    interpreter->tensor(input_idx_list[0])->dims->data + len,
                    input_dim);
  len = interpreter->tensor(output_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(output_idx_list[0])->dims->data,
                    interpreter->tensor(output_idx_list[0])->dims->data + len,
                    output_dim);

  int output_number_of_pixels = 1;
  int wanted_channels = input_dim[0];
  int wanted_height = input_dim[1];
  int wanted_width = input_dim[2];

  for (int k = 0; k < 4; k++)
    output_number_of_pixels *= input_dim[k];

  int _input = interpreter->inputs()[0];

  float *output;
  uint8_t *in = tflite::label_image::read_bmp(filename, &wanted_width,
                                              &wanted_height, &wanted_channels);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors!" << std::endl;
    exit(0);
  }

  for (int l = 0; l < output_number_of_pixels; l++) {
    (interpreter->typed_tensor<float>(_input))[l] = ((float)in[l]) / 255.0;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke!" << std::endl;
    exit(0);
  }

  output = interpreter->typed_output_tensor<float>(0);

  std::cout << input_dim[0] << " " << input_dim[1] << " " << input_dim[2] << " "
            << input_dim[3] << std::endl;
  std::cout << output_dim[0] << " " << output_dim[1] << " " << output_dim[2]
            << " " << output_dim[3] << std::endl;

  for (unsigned int l = 0; l < feature_size; l++) {
    feature_input[l] = output[l];
  }

  delete[] in;
}

/**
 * @brief     Extract the features from all three categories
 * @param[in] p data path
 * @param[out] feature_input save output of tflite
 * @param[out] feature_output save label data
 */
void ExtractFeatures(std::string p, vector<vector<float>> &feature_input,
                     vector<vector<float>> &feature_output, std::string type,
                     std::ofstream &f) {
  string total_label[10] = {"airplane", "automobile", "bird",  "cat",  "deer",
                            "dog",      "frog",       "horse", "ship", "truck"};

  int data_size = total_train_data_size;
  bool val = false;

  if (!type.compare("val")) {
    data_size = total_val_data_size;
    val = true;
  } else if (!type.compare("test")) {
    data_size = total_test_data_size;
    val = false;
  }

  int trainingSize = total_label_size * data_size;

  feature_input.resize(trainingSize);
  feature_output.resize(trainingSize);

  for (unsigned int i = 0; i < total_label_size; i++) {
    std::string path = p;
    if (!type.compare("val") || !type.compare("training")) {
      path += "train/" + total_label[i];
    } else if (!type.compare("test")) {
      path += "test/" + total_label[i];
    }

    for (int j = 0; j < data_size; j++) {
      std::string img = path + "/";
      std::stringstream ss;

      if (val) {
        ss << std::setw(4) << std::setfill('0') << (5000 - j);
      } else {
        ss << std::setw(4) << std::setfill('0') << (j + 1);
      }

      img += ss.str() + ".bmp";
      printf("%s\n", img.c_str());

      std::vector<float> _input, _output;
      _input.resize(feature_size);
      _output.resize(total_label_size);

      getFeature(img, _input);
      _output[i] = 1;

      for (unsigned int k = 0; k < feature_size; ++k)
        f.write((char *)&_input[k], sizeof(float));

      for (unsigned int k = 0; k < total_label_size; ++k)
        f.write((char *)&_output[k], sizeof(float));
    }
  }
}

bool getData(std::ifstream &F, std::vector<float> &outVec,
             std::vector<float> &outLabel, int id) {
  long pos = F.tellg();
  F.seekg(pos + (feature_size + total_label_size) * id);
  for (unsigned int i = 0; i < feature_size; i++)
    F.read((char *)&outVec[i], sizeof(float));
  for (unsigned int i = 0; i < total_label_size; i++)
    F.read((char *)&outLabel[i], sizeof(float));

  return true;
}

bool getBatchSize(std::string type,
                  std::vector<std::vector<std::vector<float>>> &outVec,
                  std::vector<std::vector<std::vector<float>>> &outLabel) {
  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  int data_size = total_train_data_size;

  std::string filename = type + "Set.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  for (unsigned int i = 0; i < total_label_size * data_size; i++) {
    if (!duplicate[i])
      count++;
  }

  if (count < batch_size)
    return false;

  count = 0;
  while (count < batch_size) {
    int nomI = rangeRandom(0, total_label_size * data_size - 1);
    if (!duplicate[nomI]) {
      memI.push_back(nomI);
      duplicate[nomI] = true;
      count++;
    }
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<std::vector<float>> out;
    std::vector<std::vector<float>> outL;
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(total_label_size);

    getData(F, o, l, memI[i]);

    out.push_back(o);
    outL.push_back(l);

    outVec.push_back(out);
    outLabel.push_back(outL);
  }

  F.close();
  return true;
}

void save(std::vector<std::vector<float>> inVec,
          std::vector<std::vector<float>> inLabel, std::string type) {
  std::string file = type + "Set.dat";

  unsigned int data_size = total_train_data_size;

  if (!type.compare("val")) {
    data_size = total_val_data_size;
  } else if (!type.compare("test")) {
    data_size = total_test_data_size;
  }

  std::ofstream TrainigSet(file, std::ios::out | std::ios::binary);
  for (unsigned int i = 0; i < total_label_size * data_size; ++i) {
    for (unsigned int j = 0; j < feature_size; ++j) {
      TrainigSet.write((char *)&inVec[i][j], sizeof(float));
    }
    for (unsigned int j = 0; j < total_label_size; ++j)
      TrainigSet.write((char *)&inLabel[i][j], sizeof(float));
  }
}

bool read(std::vector<std::vector<float>> &inVec,
          std::vector<std::vector<float>> &inLabel, std::string type) {
  std::string file = type + "Set.dat";

  file = data_path + file;
  std::ifstream TrainingSet(file, std::ios::in | std::ios::binary);

  if (!TrainingSet.good()) {
    std::cerr << "try reading file path but failed, path: " << file
              << std::endl;
    return false;
  }

  return true;
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
  data_path = args[1] + '/';

  /// @todo add api version of this
  try {
    nntrainer::Engine::Global().setWorkingDirectory(data_path);
  } catch (std::invalid_argument &e) {
    std::cerr << "setting data_path failed, pwd is used instead";
  }

  seed = time(NULL);
  srand(seed);

  std::vector<std::vector<float>> inputVector, outputVector;
  std::vector<std::vector<float>> inputValVector, outputValVector;
  std::vector<std::vector<float>> inputTestVector, outputTestVector;

  try {
    if (!read(inputVector, outputVector, "training")) {
      /**
       * @brief     Extract Feature
       */
      std::string filename = data_path + "trainingSet.dat";
      std::ofstream f(filename, std::ios::out | std::ios::binary);

      ExtractFeatures(data_path, inputVector, outputVector, "training", f);

      f.close();
    }
  } catch (...) {
    std::cerr << "Error during open file: " << std::endl;
    return 1;
  }

  try {
    if (!read(inputValVector, outputValVector, "val")) {
      /**
       * @brief     Extract Feature
       */
      std::string filename = data_path + "valSet.dat";
      std::ofstream f(filename, std::ios::out | std::ios::binary);

      ExtractFeatures(data_path, inputValVector, outputValVector, "val", f);

      f.close();
    }
  } catch (...) {
    std::cerr << "Error during open file: " << std::endl;
    return 1;
  }

  try {
    if (!read(inputTestVector, outputTestVector, "test")) {
      /**
       * @brief     Extract Feature
       */
      std::string filename = data_path + "testSet.dat";
      std::ofstream f(filename, std::ios::out | std::ios::binary);

      ExtractFeatures(data_path, inputTestVector, outputTestVector, "test", f);

      f.close();
    }
  } catch (...) {
    std::cerr << "Error during open file: " << std::endl;
    return 1;
  }

  /**
   * @brief     Neural Network Create & Initialization
   */
  nntrainer::NeuralNetwork NN;
  int status = ML_ERROR_NONE;
  try {
    NN.load(config, ml::train::ModelFormat::MODEL_FORMAT_INI);
    // NN.load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);

    status = NN.compile();
    if (status != ML_ERROR_NONE)
      return status;

    status = NN.initialize();
    if (status != ML_ERROR_NONE)
      return status;
  } catch (...) {
    std::cerr << "Error during init" << std::endl;
    return 1;
  }

  try {
    NN.train();
  } catch (...) {
    std::cerr << "Error during train" << std::endl;
    return 1;
  }

  try {
    if (!TRAINING) {
      std::string img = data_path;
      std::vector<float> featureVector, resultVector;
      featureVector.resize(feature_size);
      getFeature(img, featureVector);

      nntrainer::Tensor X;

      X = nntrainer::Tensor({featureVector}, {nntrainer::Tformat::NCHW,
                                              nntrainer::Tdatatype::FP32});
      NN.forwarding({MAKE_SHARED_TENSOR(X)})[0]->apply<float>(stepFunction);
    }
  } catch (...) {
    std::cerr << "Error while forwarding the model" << std::endl;
    return 1;
  }

  return 0;
}
