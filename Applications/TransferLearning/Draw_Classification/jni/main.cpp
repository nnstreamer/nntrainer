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

#include <nntrainer.h>

/** Number of dimensions for the input data */
#define MAX_DIM 4

/** Data size for each category */
#define NUM_DATA_PER_LABEL 5

/** Size of each label (number of label categories) */
#define LABEL_SIZE 3

/** Size of each input */
#define INPUT_SIZE 128

/** Number of test data points */
#define TOTAL_TEST_SIZE 8

/** Total number of data points in an epoch */
#define EPOCH_SIZE LABEL_SIZE *NUM_DATA_PER_LABEL

/** Max Epochs */
#define EPOCHS 1000

/** labels values */
const std::string label_names[LABEL_SIZE] = {"happy", "sad", "soso"};

/** Vectors containing the training data */
std::vector<std::vector<float>> inputVector, labelVector;
unsigned int iteration = 0;

/**
 * @brief     step function
 * @param[in] x value to be distinguished
 * @retval 0.0 or 1.0
 */
// float stepFunction(float x) {
//   if (x > 0.9) {
//     return 1.0;
//   }
//
//   if (x < 0.1) {
//     return 0.0;
//   }
//
//   return x;
// }

struct TFLiteData {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::string data_path;

  int output_number_of_pixels;
  int inputDimReq[MAX_DIM];
};

void setupTensorflowLiteModel(const std::string &data_path,
                              TFLiteData &tflite_data) {
  int input_size;
  int output_size;
  int len;
  int outputDim[MAX_DIM];

  tflite_data.data_path = data_path;
  std::string model_path = data_path + "ssd_mobilenet_v2_coco_feature.tflite";
  tflite_data.model =
    tflite::FlatBufferModel::BuildFromFile(model_path.c_str());
  if (tflite_data.model == NULL)
    throw std::runtime_error("Unable to build model from file");

  tflite::InterpreterBuilder(*tflite_data.model.get(),
                             tflite_data.resolver)(&tflite_data.interpreter);

  if (tflite_data.interpreter->AllocateTensors() != kTfLiteOk)
    throw std::runtime_error("Failed to allocate tensors!");

  input_size = tflite_data.interpreter->inputs().size();
  output_size = tflite_data.interpreter->outputs().size();

  if (input_size > 1 || output_size > 1)
    throw std::runtime_error("Model is expected with single input and output");

  for (int i = 0; i < MAX_DIM; i++) {
    tflite_data.inputDimReq[i] = 1;
    outputDim[i] = 1;
  }

  int input_idx = tflite_data.interpreter->inputs()[0];
  len = tflite_data.interpreter->tensor(input_idx)->dims->size;
  std::reverse_copy(tflite_data.interpreter->tensor(input_idx)->dims->data,
                    tflite_data.interpreter->tensor(input_idx)->dims->data +
                      len,
                    tflite_data.inputDimReq);

  int output_idx = tflite_data.interpreter->outputs()[0];
  len = tflite_data.interpreter->tensor(output_idx)->dims->size;
  std::reverse_copy(
    tflite_data.interpreter->tensor(output_idx)->dims->data,
    tflite_data.interpreter->tensor(output_idx)->dims->data + len, outputDim);

  tflite_data.output_number_of_pixels = 1;
  for (int k = 0; k < MAX_DIM; k++)
    tflite_data.output_number_of_pixels *= tflite_data.inputDimReq[k];
}

/**
 * @brief     Get Feature vector from tensorflow lite
 *            This creates interpreter & inference with ssd tflite
 * @param[in] filename input file path
 * @param[out] feature_input save output of tflite
 */
void getInputFeature(const TFLiteData &tflite_data, const std::string filename,
                     std::vector<float> &feature_input) {
  uint8_t *in;
  int inputDim[MAX_DIM] = {1, 1, 1, 1};
  in = tflite::label_image::read_bmp(filename, inputDim, inputDim + 1,
                                     inputDim + 2);

  int input_img_size = 1;
  for (int idx = 0; idx < MAX_DIM; idx++) {
    input_img_size *= inputDim[idx];
  }

  if (tflite_data.output_number_of_pixels != input_img_size) {
    delete in;
    throw std::runtime_error("Input size does not match the required size");
  }

  int input_idx = tflite_data.interpreter->inputs()[0];
  for (int l = 0; l < tflite_data.output_number_of_pixels; l++) {
    (tflite_data.interpreter->typed_tensor<float>(input_idx))[l] =
      ((float)in[l] - 127.5f) / 127.5f;
  }

  if (tflite_data.interpreter->Invoke() != kTfLiteOk)
    std::runtime_error("Failed to invoke.");

  float *output = tflite_data.interpreter->typed_output_tensor<float>(0);
  for (int l = 0; l < INPUT_SIZE; l++) {
    feature_input[l] = output[l];
  }

  delete[] in;
}

/**
 * @brief     Extract the features from pretrained model
 * @param[in] data_path data path
 * @param[out] input_data output of tflite model (input for the nntrainer model)
 * @param[out] label_data one hot label data
 */
void extractFeatures(const TFLiteData &tflite_data,
                     std::vector<std::vector<float>> &input_data,
                     std::vector<std::vector<float>> &label_data) {
  int trainingSize = LABEL_SIZE * NUM_DATA_PER_LABEL;

  input_data.resize(trainingSize, std::vector<float>(INPUT_SIZE));
  /** resize label data to size and initialize to 0 */
  label_data.resize(trainingSize, std::vector<float>(LABEL_SIZE, 0));

  for (int i = 0; i < LABEL_SIZE; i++) {
    for (int j = 0; j < NUM_DATA_PER_LABEL; j++) {
      std::string label_file = label_names[i] + std::to_string(j + 1) + ".bmp";
      std::string img =
        tflite_data.data_path + "/" + label_names[i] + "/" + label_file;

      int count = i * NUM_DATA_PER_LABEL + j;
      getInputFeature(tflite_data, img, input_data[count]);
      label_data[count][i] = 1;
    }
  }
}

/**
 * Data generator callback
 */
int getBatch_train(float **input, float **label, bool *last, void *user_data) {
  if (iteration >= EPOCH_SIZE) {
    *last = true;
    iteration = 0;
    return ML_ERROR_NONE;
  }

  for (int idx = 0; idx < INPUT_SIZE; idx++) {
    input[0][idx] = inputVector[iteration][idx];
  }

  for (int idx = 0; idx < LABEL_SIZE; idx++) {
    label[0][idx] = labelVector[iteration][idx];
  }

  *last = false;
  iteration += 1;
  return ML_ERROR_NONE;
}

/**
 * @brief     create NN
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  int status = ML_ERROR_NONE;
  if (argc < 3) {
    std::cout << "./TransferLearning Config.ini resources\n";
    exit(0);
  }

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];

  /** location of resources ( ../../res/ ) */
  std::string data_path = args[1];

  srand(time(NULL));

  TFLiteData tflite_data;
  setupTensorflowLiteModel(data_path, tflite_data);

  /** Extract features from the already trained model */
  extractFeatures(tflite_data, inputVector, labelVector);

  /** Neural Network Create & Initialization */
  ml_train_model_h handle = NULL;
  ml_train_dataset_h dataset = NULL;

  status = ml_train_model_construct_with_conf(config.c_str(), &handle);
  if (status != ML_ERROR_NONE) {
    std::cerr << "Failed to construct the model" << std::endl;
    return status;
  }

  status = ml_train_model_compile(handle, NULL);
  if (status != ML_ERROR_NONE) {
    std::cerr << "Failed to compile the model" << std::endl;
    ml_train_model_destroy(handle);
    return status;
  }

  /** Set the dataset from generator */
  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  NULL, NULL);
  if (status != ML_ERROR_NONE) {
    std::cerr << "Failed to create the dataset" << std::endl;
    ml_train_model_destroy(handle);
    return status;
  }

  status = ml_train_dataset_set_property(dataset, "buffer_size=100", NULL);
  if (status != ML_ERROR_NONE) {
    std::cerr << "Failed to set property for the dataset" << std::endl;
    ml_train_dataset_destroy(dataset);
    ml_train_model_destroy(handle);
    return status;
  }

  status = ml_train_model_set_dataset(handle, dataset);
  if (status != ML_ERROR_NONE) {
    std::cerr << "Failed to set dataset to the dataset" << std::endl;
    ml_train_dataset_destroy(dataset);
    ml_train_model_destroy(handle);
    return status;
  }

  /**
   * @brief     back propagation
   */
  std::stringstream epoch_string;
  epoch_string << "epochs=" << EPOCHS << std::endl;
  status = ml_train_model_run(handle, epoch_string.str().c_str(), NULL);
  if (status != ML_ERROR_NONE) {
    std::cerr << "Failed to train the model" << std::endl;
    ml_train_model_destroy(handle);
    return status;
  }

  /** destroy the model */
  ml_train_model_destroy(handle);

  /**
   * @brief     test
   */
  // for (int i = 0; i < TOTAL_TEST_SIZE; i++) {
  //   std::string path = data_path;
  //   path += "testset";
  //   printf("\n[%s]\n", path.c_str());
  //   std::string img = path + "/";
  //   img += "test" + std::to_string(i + 1) + ".bmp";
  //   printf("%s\n", img.c_str());

  //   std::vector<float> featureVector, resultVector;
  //   featureVector.resize(128);
  //   getFeature(img, featureVector);
  //   nntrainer::Tensor X;
  //   try {
  //     X = nntrainer::Tensor({featureVector});
  //     NN.forwarding(MAKE_SHARED_TENSOR(X))->apply(stepFunction);
  //   } catch (...) {
  //     std::cerr << "Error during forwaring the model" << std::endl;
  //     NN.finalize();
  //     return -1;
  //   }
  // }
}
