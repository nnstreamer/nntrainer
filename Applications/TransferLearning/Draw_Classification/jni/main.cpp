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

#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#if defined(__TIZEN__)
#include <gtest/gtest.h>

#include <nnstreamer-single.h>
#include <nnstreamer.h>
#include <nntrainer_internal.h>
#else
#include "tensorflow/contrib/lite/interpreter.h"
#include "tensorflow/contrib/lite/kernels/register.h"
#include "tensorflow/contrib/lite/model.h"
#include "tensorflow/contrib/lite/string_util.h"
#include "tensorflow/contrib/lite/tools/gen_op_registration.h"
#include <cmath>
#include <fstream>
#include <iostream>
#endif

#include "bitmap_helpers.h"
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

/** Minimum softmax value threshold to make a confident threshold */
#define PREDICTION_THRESHOLD 0.9

/** labels values */
const char *label_names[LABEL_SIZE] = {"happy", "sad", "soso"};

/** Vectors containing the training data */
float inputVector[EPOCH_SIZE][INPUT_SIZE];
float labelVector[EPOCH_SIZE][LABEL_SIZE];

/** Benchmark output values */
const float test_output_benchmark[TOTAL_TEST_SIZE] = {
  0.99669778, 0.96033746, 0.99192446, 0.98053128,
  0.95911789, 0.99331927, 0.55696899, 0.46636438};

/** Container to hold the output values when running */
float test_output[TOTAL_TEST_SIZE];

/** set float array to 0 */
void array_set_zero(float *data, size_t num_elem) {
  for (size_t idx = 0; idx < num_elem; idx++) {
    data[idx] = 0.0;
  }
}

#if defined(__TIZEN__)
int getInputFeature(ml_single_h single, const char *test_file_path,
                    float *feature_input) {
  int status = ML_ERROR_NONE;
  ml_tensors_info_h in_res;
  ml_tensors_data_h input = NULL, output;
  void *raw_data;
  size_t data_size;
  int inputDim[MAX_DIM] = {1, 1, 1, 1};
  uint8_t *in = NULL;

  /* input tensor in filter */
  status = ml_single_get_input_info(single, &in_res);
  if (status != ML_ERROR_NONE)
    return status;

  /* generate dummy data */
  status = ml_tensors_data_create(in_res, &input);
  if (status != ML_ERROR_NONE)
    goto fail_in_info;

  status = ml_tensors_data_get_tensor_data(input, 0, &raw_data, &data_size);
  if (status != ML_ERROR_NONE)
    goto fail_in_data;

  in = tflite::label_image::read_bmp(test_file_path, inputDim, inputDim + 1,
                                     inputDim + 2);

  for (size_t l = 0; l < data_size / sizeof(float); l++) {
    ((float *)raw_data)[l] = ((float)(in[l]) - 127.5f) / 127.5f;
  }
  delete[] in;

  status = ml_single_invoke(single, input, &output);
  if (status != ML_ERROR_NONE)
    goto fail_in_data;

  status = ml_tensors_data_get_tensor_data(output, 0, &raw_data, &data_size);
  if (status != ML_ERROR_NONE)
    goto fail_out_data;

  memcpy(feature_input, raw_data, INPUT_SIZE * sizeof(float));

fail_out_data:
  ml_tensors_data_destroy(output);

fail_in_data:
  ml_tensors_data_destroy(input);

fail_in_info:
  ml_tensors_info_destroy(in_res);

  return status;
}

int setupSingleModel(const char *data_path, ml_single_h *single) {
  int status;

  char *test_model = NULL;
  status = asprintf(&test_model, "%s/%s", data_path,
                    "ssd_mobilenet_v2_coco_feature.tflite");
  if (status < 0 || test_model == NULL)
    return -errno;

  status = ml_single_open(single, test_model, NULL, NULL,
                          ML_NNFW_TYPE_TENSORFLOW_LITE, ML_NNFW_HW_ANY);
  free(test_model);

  return status;
}

int extractFeatures(const char *data_path, float input_data[][INPUT_SIZE],
                    float label_data[][LABEL_SIZE]) {
  ml_single_h single;
  unsigned int count = 0;
  int status;

  status = setupSingleModel(data_path, &single);
  if (status != ML_ERROR_NONE)
    return status;

  for (int i = 0; i < LABEL_SIZE; i++) {
    for (int j = 0; j < NUM_DATA_PER_LABEL; j++) {

      char *test_file_path = NULL;
      status = asprintf(&test_file_path, "%s/%s/%s%d.bmp", data_path,
                        label_names[i], label_names[i], j + 1);
      if (status < 0 || test_file_path == NULL) {
        status = -errno;
        goto fail_exit;
      }

      count = i * NUM_DATA_PER_LABEL + j;
      array_set_zero(label_data[count], LABEL_SIZE);
      label_data[count][i] = 1;
      status = getInputFeature(single, test_file_path, input_data[count]);
      free(test_file_path);

      if (status != ML_ERROR_NONE)
        goto fail_exit;
    }
  }

fail_exit:
  status = ml_single_close(single);

  return status;
}
#else
/**
 * @brief Private data for Tensorflow lite object
 */
struct TFLiteData {
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model;
  std::string data_path;

  int output_number_of_pixels;
  int inputDimReq[MAX_DIM];
};

/**
 * @brief Load the tensorflow lite model and its metadata
 */
void setupTensorflowLiteModel(const std::string &data_path,
                              TFLiteData &tflite_data) {
  int input_size;
  int output_size;
  int len;
  int outputDim[MAX_DIM];

  tflite_data.data_path = data_path;
  std::string model_path =
    data_path + "/" + "ssd_mobilenet_v2_coco_feature.tflite";
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
                     float *feature_input) {
  uint8_t *in = NULL;
  int inputDim[MAX_DIM] = {1, 1, 1, 1};
  in = tflite::label_image::read_bmp(filename.c_str(), inputDim, inputDim + 1,
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
 * @param[in] tflite_data private data for tflite model
 * @param[out] input_data output of tflite model (input for the nntrainer model)
 * @param[out] label_data one hot label data
 */
void extractFeatures(const TFLiteData &tflite_data,
                     float input_data[][INPUT_SIZE],
                     float label_data[][LABEL_SIZE]) {
  for (int i = 0; i < LABEL_SIZE; i++) {
    for (int j = 0; j < NUM_DATA_PER_LABEL; j++) {
      std::string label_file = label_names[i] + std::to_string(j + 1) + ".bmp";
      std::string img =
        tflite_data.data_path + "/" + label_names[i] + "/" + label_file;

      int count = i * NUM_DATA_PER_LABEL + j;
      getInputFeature(tflite_data, img, input_data[count]);

      array_set_zero(label_data[count], LABEL_SIZE);
      label_data[count][i] = 1;
    }
  }
}
#endif

/**
 * Data generator callback
 */
int getBatch_train(float **input, float **label, bool *last, void *user_data) {
  static unsigned int iteration = 0;
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
 * @brief Train the model with the given config file path
 * @param[in] config Model config file path
 */
int trainModel(const char *config) {
  int status = ML_ERROR_NONE;

  /** Neural Network Create & Initialization */
  ml_train_model_h handle = NULL;
  ml_train_dataset_h dataset = NULL;

  status = ml_train_model_construct_with_conf(config, &handle);
  if (status != ML_ERROR_NONE) {
    return status;
  }

  status = ml_train_model_compile(handle, NULL);
  if (status != ML_ERROR_NONE) {
    ml_train_model_destroy(handle);
    return status;
  }

  /** Set the dataset from generator */
  status = ml_train_dataset_create_with_generator(&dataset, getBatch_train,
                                                  NULL, NULL);
  if (status != ML_ERROR_NONE) {
    ml_train_model_destroy(handle);
    return status;
  }

  status = ml_train_dataset_set_property(dataset, "buffer_size=100", NULL);
  if (status != ML_ERROR_NONE) {
    ml_train_dataset_destroy(dataset);
    ml_train_model_destroy(handle);
    return status;
  }

  status = ml_train_model_set_dataset(handle, dataset);
  if (status != ML_ERROR_NONE) {
    ml_train_dataset_destroy(dataset);
    ml_train_model_destroy(handle);
    return status;
  }

  /** Do the training */
  status = ml_train_model_run(handle, NULL);
  if (status != ML_ERROR_NONE) {
    ml_train_model_destroy(handle);
    return status;
  }

  /** destroy the model */
  status = ml_train_model_destroy(handle);
  return status;
}

#if defined(__TIZEN__)
void sink_cb(const ml_tensors_data_h data, const ml_tensors_info_h info,
             void *user_data) {
  static int test_file_idx = 1;
  int status = ML_ERROR_NONE;
  ml_tensor_dimension dim;
  float *raw_data;
  size_t data_size;
  int max_idx = -1;
  float max_val = 0; // last layer is softmax, so all values will be positive

  ml_tensors_info_get_tensor_dimension(info, 0, dim);

  status =
    ml_tensors_data_get_tensor_data(data, 0, (void **)&raw_data, &data_size);
  if (status != ML_ERROR_NONE)
    return;

  for (int i = 0; i < LABEL_SIZE; i++) {
    if (raw_data[i] > max_val && raw_data[i] > PREDICTION_THRESHOLD) {
      max_val = raw_data[i];
      max_idx = i;
    }
  }

  std::cout << "Label for test file test" << test_file_idx << ".bmp = ";
  if (max_idx >= 0)
    std::cout << label_names[max_idx] << " with softmax value = " << max_val
              << std::endl;
  else
    std::cout << "could not be predicted with enough confidence." << std::endl;

  if (max_val > 0)
    test_output[test_file_idx - 1] = max_val;
  else
    test_output[test_file_idx - 1] = raw_data[0];

  test_file_idx += 1;
}
#endif

/**
 * @brief Test the model with the given config file path
 * @param[in] data_path Path of the test data
 * @param[in] config Model config file path
 */
int testModel(const char *data_path, const char *model) {
#if defined(__TIZEN__)
  ml_single_h single;
  int status = ML_ERROR_NONE;
  ml_pipeline_h pipe;
  ml_pipeline_src_h src;
  ml_pipeline_sink_h sink;
  ml_tensors_info_h in_info;
  ml_tensors_data_h in_data;
  void *raw_data;
  size_t data_size;
  ml_tensor_dimension in_dim = {1, INPUT_SIZE, 1, 1};

  char pipeline[2048];
  snprintf(pipeline, sizeof(pipeline),
           "appsrc name=srcx ! "
           "other/"
           "tensor,dimension=(string)1:%d:1:1,type=(string)float32,framerate=("
           "fraction)0/1 ! "
           "tensor_filter framework=nntrainer model=\"%s\" input=1:%d:1:1 "
           "inputtype=float32 output=1:%d:1:1 outputtype=float32 ! tensor_sink "
           "name=sinkx",
           INPUT_SIZE, model, INPUT_SIZE, LABEL_SIZE);

  status = setupSingleModel(data_path, &single);
  if (status != ML_ERROR_NONE)
    goto fail_exit;

  status = ml_pipeline_construct(pipeline, NULL, NULL, &pipe);
  if (status != ML_ERROR_NONE)
    goto fail_single_close;

  status = ml_pipeline_src_get_handle(pipe, "srcx", &src);
  if (status != ML_ERROR_NONE)
    goto fail_pipe_destroy;

  status = ml_pipeline_sink_register(pipe, "sinkx", sink_cb, NULL, &sink);
  if (status != ML_ERROR_NONE)
    goto fail_src_release;

  status = ml_pipeline_start(pipe);
  if (status != ML_ERROR_NONE)
    goto fail_sink_release;

  ml_tensors_info_create(&in_info);
  ml_tensors_info_set_count(in_info, 1);
  ml_tensors_info_set_tensor_type(in_info, 0, ML_TENSOR_TYPE_FLOAT32);
  ml_tensors_info_set_tensor_dimension(in_info, 0, in_dim);

  for (int i = 0; i < TOTAL_TEST_SIZE; i++) {
    char *test_file_path;
    status =
      asprintf(&test_file_path, "%s/testset/test%d.bmp", data_path, i + 1);
    if (status < 0) {
      status = -errno;
      goto fail_info_release;
    }

    float featureVector[INPUT_SIZE];
    status = getInputFeature(single, test_file_path, featureVector);
    if (status != ML_ERROR_NONE)
      goto fail_info_release;

    status = ml_tensors_data_create(in_info, &in_data);
    if (status != ML_ERROR_NONE)
      goto fail_info_release;

    status = ml_tensors_data_get_tensor_data(in_data, 0, &raw_data, &data_size);
    if (status != ML_ERROR_NONE) {
      ml_tensors_data_destroy(&in_data);
      goto fail_info_release;
    }

    for (size_t ds = 0; ds < data_size / sizeof(float); ds++)
      ((float *)raw_data)[ds] = featureVector[ds];

    status = ml_pipeline_src_input_data(src, in_data,
                                        ML_PIPELINE_BUF_POLICY_AUTO_FREE);
    if (status != ML_ERROR_NONE) {
      ml_tensors_data_destroy(&in_data);
      goto fail_info_release;
    }

    /** No need to destroy data here, pipeline freed buffer automatically */
  }

  /** Sleep for 1 second for all the data to be received by sink callback */
  sleep(1);

fail_info_release:
  ml_tensors_info_destroy(in_info);

  status = ml_pipeline_stop(pipe);

fail_sink_release:
  status = ml_pipeline_sink_unregister(sink);

fail_src_release:
  status = ml_pipeline_src_release_handle(src);

fail_pipe_destroy:
  status = ml_pipeline_destroy(pipe);

fail_single_close:
  ml_single_close(single);

fail_exit:
  return status;
#else
  std::cout << "Testing of model only with TIZEN." << std::endl;
  return ML_ERROR_NONE;
#endif
}

#if defined(__TIZEN__)
/**
 * @brief  Test to verify that the draw classification app is successful
 */
TEST(DrawClassification, matchTestResult) {
  for (int idx = 0; idx < TOTAL_TEST_SIZE; idx++) {
    EXPECT_FLOAT_EQ(test_output_benchmark[idx], test_output[idx]);
  }
}
#endif

/**
 * @brief     create NN
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  int status = ML_ERROR_NONE;
  if (argc < 3) {
#if defined(__TIZEN__)
    ml_loge("./TransferLearning Config.ini resources.");
#else
    std::cout << "./TransferLearning Config.ini resources." << std::endl;
#endif
    return 1;
  }

#if defined(__TIZEN__)
  set_feature_state(SUPPORTED);
#endif

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];

  /** location of resources ( ../../res/ ) */
  std::string data_path = args[1];

  srand(time(NULL));

  /** Extract features from the pre-trained model */
#if defined(__TIZEN__)
  status = extractFeatures(data_path.c_str(), inputVector, labelVector);
  if (status != ML_ERROR_NONE)
    return 1;
#else
  TFLiteData tflite_data;
  try {
    setupTensorflowLiteModel(data_path, tflite_data);
    extractFeatures(tflite_data, inputVector, labelVector);
  } catch (...) {
    std::cout << "Running tflite model failed." << std::endl;
    return 1;
  }
#endif

  /** Do the training */
  status = trainModel(config.c_str());
  if (status != ML_ERROR_NONE)
    return 1;

  /** Test the trained model */
  status = testModel(data_path.c_str(), config.c_str());
  if (status != ML_ERROR_NONE)
    return 1;

#if defined(__TIZEN__)
  set_feature_state(NOT_CHECKED_YET);
#endif

#if defined(__TIZEN__)
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    status = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }
#endif

  return status;
}
