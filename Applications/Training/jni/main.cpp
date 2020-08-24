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
 * @brief	This is Transfer Learning Example with one FC Layer
 *
 *              Inputs : Three Categories ( Happy, Sad, Soso ) with
 *                       5 pictures for each category
 *              Feature Extractor : ssd_mobilenet_v2_coco_feature.tflite
 *                                  ( modified to use feature extracter )
 *              Classifier : One Fully Connected Layer
 *
 */

#include "bitmap_helpers.h"
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <time.h>

#include "neuralnet.h"
#include "tensor.h"

/**
 * @brief     Data size for each category
 */
#define TOTAL_DATA_SIZE 5

/**
 * @brief     Number of category : Three
 */
#define TOTAL_LABEL_SIZE 3

/**
 * @brief     Number of Test Set
 */
#define TOTAL_TEST_SIZE 8

/**
 * @brief     Max Epochs
 */
#define ITERATION 1000

using namespace std;

/**
 * @brief     location of resources ( ../../res/ )
 */
string data_path;

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
 * @brief     Get Feature vector from tensorflow lite
 *            This creates interpreter & inference with ssd tflite
 * @param[in] filename input file path
 * @param[out] feature_input save output of tflite
 */
void getFeature(const string filename, vector<float> &feature_input) {
  int input_size;
  int output_size;
  int *output_idx_list;
  int *input_idx_list;
  int inputDim[4];
  int outputDim[4];
  int input_idx_list_len = 0;
  int output_idx_list_len = 0;
  std::string model_path = data_path + "ssd_mobilenet_v2_coco_feature.tflite";
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  assert(model != NULL);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  input_size = interpreter->inputs().size();
  output_size = interpreter->outputs().size();

  input_idx_list = new int[input_size];
  output_idx_list = new int[output_size];

  int t_size = interpreter->tensors_size();
  for (int i = 0; i < t_size; i++) {
    for (int j = 0; j < input_size; j++) {
      if (strcmp(interpreter->tensor(i)->name, interpreter->GetInputName(j)) ==
          0)
        input_idx_list[input_idx_list_len++] = i;
    }
    for (int j = 0; j < output_size; j++) {
      if (strcmp(interpreter->tensor(i)->name, interpreter->GetOutputName(j)) ==
          0)
        output_idx_list[output_idx_list_len++] = i;
    }
  }
  for (int i = 0; i < 4; i++) {
    inputDim[i] = 1;
    outputDim[i] = 1;
  }

  int len = interpreter->tensor(input_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(input_idx_list[0])->dims->data,
                    interpreter->tensor(input_idx_list[0])->dims->data + len,
                    inputDim);
  len = interpreter->tensor(output_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(output_idx_list[0])->dims->data,
                    interpreter->tensor(output_idx_list[0])->dims->data + len,
                    outputDim);

  int output_number_of_pixels = 1;
  int wanted_channels = inputDim[0];
  int wanted_height = inputDim[1];
  int wanted_width = inputDim[2];

  for (int k = 0; k < 4; k++)
    output_number_of_pixels *= inputDim[k];

  int _input = interpreter->inputs()[0];

  uint8_t *in;
  float *output;
  in = tflite::label_image::read_bmp(filename, &wanted_width, &wanted_height,
                                     &wanted_channels);
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors!" << std::endl;
    exit(0);
  }

  for (int l = 0; l < output_number_of_pixels; l++) {
    (interpreter->typed_tensor<float>(_input))[l] =
      ((float)in[l] - 127.5f) / 127.5f;
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke!" << std::endl;
    exit(0);
  }

  output = interpreter->typed_output_tensor<float>(0);

  for (int l = 0; l < 128; l++) {
    feature_input[l] = output[l];
  }

  delete[] input_idx_list;
  delete[] output_idx_list;
  delete[] in;
}

/**
 * @brief     Extract the features from all three categories
 * @param[in] p data path
 * @param[out] feature_input save output of tflite
 * @param[out] feature_output save label data
 */
void ExtractFeatures(std::string p, vector<vector<float>> &feature_input,
                     vector<vector<float>> &feature_output) {
  string total_label[TOTAL_LABEL_SIZE] = {"happy", "sad", "soso"};

  int trainingSize = TOTAL_LABEL_SIZE * TOTAL_DATA_SIZE;

  feature_input.resize(trainingSize);
  feature_output.resize(trainingSize);

  int count = 0;

  for (int i = 0; i < TOTAL_LABEL_SIZE; i++) {
    std::string path = p;
    path += total_label[i];

    for (int j = 0; j < TOTAL_DATA_SIZE; j++) {
      std::string img = path + "/";
      img += total_label[i] + std::to_string(j + 1) + ".bmp";
      printf("%s\n", img.c_str());

      feature_input[count].resize(128);

      getFeature(img, feature_input[count]);
      feature_output[count].resize(TOTAL_LABEL_SIZE);
      feature_output[count][i] = 1;
      count++;
    }
  }
}

/**
 * @brief     create NN
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "./TransferLearning Config.ini resources\n";
    exit(0);
  }
  const vector<string> args(argv + 1, argv + argc);
  std::string config = args[0];
  data_path = args[1];

  srand(time(NULL));
  std::string ini_file = data_path + "ini.bin";
  std::vector<std::vector<float>> inputVector, outputVector;
  /**
   * @brief     Extract Feature
   */
  ExtractFeatures(data_path, inputVector, outputVector);

  /**
   * @brief     Neural Network Create & Initialization
   */
  nntrainer::NeuralNetwork NN;

  try {
    NN.loadFromConfig(config);
    NN.init();
  } catch (...) {
    std::cerr << "Error during initiation" << std::endl;
    NN.finalize();
    return -1;
  }

  /**
   * @brief     back propagation
   */
  for (int i = 0; i < ITERATION; i++) {
    for (unsigned int j = 0; j < inputVector.size(); j++) {
      nntrainer::Tensor in, out;
      try {
        in = nntrainer::Tensor({inputVector[j]});
      } catch (...) {
        std::cerr << "Error during tensor initialization" << std::endl;
        NN.finalize();
        return -1;
      }
      try {
        out = nntrainer::Tensor({outputVector[j]});
      } catch (...) {
        std::cerr << "Error during tensor initialization" << std::endl;
        NN.finalize();
        return -1;
      }

      try {
        NN.backwarding(MAKE_SHARED_TENSOR(in), MAKE_SHARED_TENSOR(out), i);
      } catch (...) {
        std::cerr << "Error during backwarding the model" << std::endl;
        NN.finalize();
        return -1;
      }
    }
    cout << "#" << i + 1 << "/" << ITERATION << " - Loss : " << NN.getLoss()
         << endl;
    NN.setLoss(0.0);
  }

  /**
   * @brief     test
   */
  for (int i = 0; i < TOTAL_TEST_SIZE; i++) {
    std::string path = data_path;
    path += "testset";
    printf("\n[%s]\n", path.c_str());
    std::string img = path + "/";
    img += "test" + std::to_string(i + 1) + ".bmp";
    printf("%s\n", img.c_str());

    std::vector<float> featureVector, resultVector;
    featureVector.resize(128);
    getFeature(img, featureVector);
    nntrainer::Tensor X;
    try {
      X = nntrainer::Tensor({featureVector});
      NN.forwarding(MAKE_SHARED_TENSOR(X))->apply(stepFunction);
    } catch (...) {
      std::cerr << "Error during forwaring the model" << std::endl;
      NN.finalize();
      return -1;
    }
  }

  /**
   * @brief     Finalize NN
   */
  NN.finalize();
}
