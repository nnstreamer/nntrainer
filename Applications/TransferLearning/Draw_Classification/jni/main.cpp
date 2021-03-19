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
 *              Classifier : One Fully Connected Layer
 *
 */

#if defined(NNSTREAMER_AVAILABLE) && defined(ENABLE_TEST)
#define APP_VALIDATE
#endif

#include <limits.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#if defined(NNSTREAMER_AVAILABLE)
#include <nnstreamer.h>
#endif
#include <nntrainer_internal.h>

#if defined(APP_VALIDATE)
#include <gtest/gtest.h>
#endif

#include "bitmap_helpers.h"
#include <app_context.h>
#include <nntrainer.h>

/** Number of dimensions for the input data */
#define MAX_DIM 4

/** Data size for each category */
#define NUM_DATA_PER_LABEL 5

/** Size of each label (number of label categories) */
#define LABEL_SIZE 3

/** Size of each input */
#define IMAGE_SIDE 300
#define IMAGE_CHANNELS 3
#define INPUT_SIZE IMAGE_SIDE *IMAGE_SIDE *IMAGE_CHANNELS

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

#if defined(APP_VALIDATE)
/** Benchmark output values */
const float test_output_benchmark[TOTAL_TEST_SIZE] = {
  0.99669778, 0.96033746, 0.99192446, 0.98053128,
  0.95911789, 0.99331927, 0.55696899, 0.46636438};
#endif

/** Container to hold the output values when running */
float test_output[TOTAL_TEST_SIZE];

/** set float array to 0 */
void array_set_zero(float *data, size_t num_elem) {
  for (size_t idx = 0; idx < num_elem; idx++) {
    data[idx] = 0.0;
  }
}

/**
 * @brief     Load input image data from bmp files and normalize to float
 * @param[in] filename input file path
 * @param[out] feature_input normalized images
 */
void getInputFeature(const std::string &filename, float *feature_input) {
  uint8_t *in = NULL;
  int inputDim[MAX_DIM] = {1, 1, 1, 1};
  in = tflite::label_image::read_bmp(filename.c_str(), inputDim, inputDim + 1,
                                     inputDim + 2);

  int input_img_size = 1;
  for (int idx = 0; idx < MAX_DIM; idx++) {
    input_img_size *= inputDim[idx];
  }

  if (INPUT_SIZE != input_img_size) {
    delete[] in;
    throw std::runtime_error("Input size does not match the required size");
  }

  for (int l = 0; l < INPUT_SIZE; l++) {
    feature_input[l] = ((float)in[l] - 127.5f) / 127.5f;
  }

  delete[] in;
}

void loadAllData(const std::string &data_path, float input_data[][INPUT_SIZE],
                 float label_data[][LABEL_SIZE]) {
  for (int i = 0; i < LABEL_SIZE; i++) {
    for (int j = 0; j < NUM_DATA_PER_LABEL; j++) {
      std::string label_file = label_names[i] + std::to_string(j + 1) + ".bmp";
      std::string img = data_path + "/" + label_names[i] + "/" + label_file;

      int count = i * NUM_DATA_PER_LABEL + j;
      getInputFeature(img, input_data[count]);

      array_set_zero(label_data[count], LABEL_SIZE);
      label_data[count][i] = 1;
    }
  }
}

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

#if defined(NNSTREAMER_AVAILABLE)
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

int getInputFeature_c(const std::string filename, float *feature_input) {
  try {
    getInputFeature(filename, feature_input);
  } catch (...) {
    return ML_ERROR_UNKNOWN;
  }

  return ML_ERROR_NONE;
}
#endif

/**
 * @brief Test the model with the given config file path
 * @param[in] data_path Path of the test data
 * @param[in] config Model config file path
 */
int testModel(const char *data_path, const char *model) {
#if defined(NNSTREAMER_AVAILABLE)
  int status = ML_ERROR_NONE;
  int new_status;
  ml_pipeline_h pipe;
  ml_pipeline_src_h src;
  ml_pipeline_sink_h sink;
  ml_tensors_info_h in_info;
  ml_tensors_data_h in_data;
  void *raw_data;
  size_t data_size;
  ml_tensor_dimension in_dim = {1, IMAGE_SIDE, IMAGE_SIDE, IMAGE_CHANNELS};

  char pipeline[2048];
  snprintf(
    pipeline, sizeof(pipeline),
    "appsrc name=srcx ! "
    "other/"
    "tensor,dimension=(string)%d:%d:%d:1,type=(string)float32,framerate=("
    "fraction)0/1 ! "
    "tensor_filter framework=nntrainer model=\"%s\" input=%d:%d:%d:1 "
    "inputtype=float32 output=%d outputtype=float32 ! tensor_sink "
    "name=sinkx",
    IMAGE_CHANNELS, IMAGE_SIDE, IMAGE_SIDE, model, IMAGE_CHANNELS, IMAGE_SIDE,
    IMAGE_SIDE, LABEL_SIZE);

  status = ml_pipeline_construct(pipeline, NULL, NULL, &pipe);
  if (status != ML_ERROR_NONE)
    goto fail_exit;

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
    status = getInputFeature_c(test_file_path, featureVector);
    free(test_file_path);
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
  new_status = ml_pipeline_stop(pipe);
  if (status == ML_ERROR_NONE && new_status != ML_ERROR_NONE) {
    status = new_status;
  }

fail_sink_release:
  new_status = ml_pipeline_sink_unregister(sink);
  if (status == ML_ERROR_NONE && new_status != ML_ERROR_NONE) {
    status = new_status;
  }

fail_src_release:
  new_status = ml_pipeline_src_release_handle(src);
  if (status == ML_ERROR_NONE && new_status != ML_ERROR_NONE) {
    status = new_status;
  }

fail_pipe_destroy:
  new_status = ml_pipeline_destroy(pipe);
  if (status == ML_ERROR_NONE && new_status != ML_ERROR_NONE) {
    status = new_status;
  }

fail_exit:
  return status;
#else
  std::cout << "Testing of model is supported without NNStreamer." << std::endl;
  return ML_ERROR_NONE;
#endif
}

#if defined(APP_VALIDATE)
/**
 * @brief  Test to verify that the draw classification app is successful
 * @note Enable this once caching is enabled for backbones and epochs to 1000
 */
TEST(DrawClassification, matchTestResult) {
  for (int idx = 0; idx < TOTAL_TEST_SIZE; idx++) {
    // EXPECT_FLOAT_EQ(test_output_benchmark[idx], test_output[idx]);
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

  /// @todo add capi version of this
  try {
    nntrainer::AppContext::Global().setWorkingDirectory(data_path);
  } catch (std::invalid_argument &e) {
    std::cerr << "setting data_path failed, pwd is used instead";
  }

  srand(time(NULL));

  /** Load input images */
  try {
    loadAllData(data_path, inputVector, labelVector);
  } catch (...) {
    std::cout << "Failed loading input images." << std::endl;
    return 1;
  }

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

#if defined(APP_VALIDATE)
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

  // please comment below if you are going to train continuously.
  try {
    const std::string data_path =
      nntrainer::AppContext::Global().getWorkingPath("model_draw_cls.bin");
    remove(data_path.c_str());
  } catch (std::exception &e) {
    std::cerr << "failed to get working data_path, reason: " << e.what();
    return 1;
  }

  return status;
}
