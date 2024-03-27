// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Donghak Park <donghak.park@samsung.com>
 *
 * @file unittest_tflite_export.cpp
 * @date 23 November 2023
 * @brief export test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <functional>
#include <gtest/gtest.h>
#include <memory>
#include <sstream>
#include <vector>

#include <app_context.h>
#include <flatten_realizer.h>
#include <layer.h>
#include <model.h>
#include <node_exporter.h>
#include <optimizer.h>
#include <realizer.h>
#include <stdlib.h>

#include <nntrainer_test_util.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tflite_interpreter.h>
#endif

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;
using LayerHandle = std::shared_ptr<ml::train::Layer>;
using ModelHandle = std::unique_ptr<ml::train::Model>;
using ml::train::createLayer;

std::vector<float> out;
std::vector<float> ans;
std::vector<float *> in_f;
std::vector<float *> l_f;

unsigned int seed = 0;

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
 * @brief Run TF Lite model with given input_vector's data
 *
 * @param tf_file_name tflite file name
 * @param input_vector input data
 * @return std::vector<float> output of tflite
 */
std::vector<float> run_tflite(std::string tf_file_name,
                              std::vector<float> input_vector) {
  std::vector<float> ret_vector;

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(tf_file_name.c_str());

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  auto in_indices = tf_interpreter->inputs();
  float *tf_input = tf_interpreter->typed_input_tensor<float>(0);

  for (unsigned int i = 0; i < input_vector.size(); i++) {
    tf_input[i] = input_vector[i];
  }

  int status = tf_interpreter->Invoke();

  auto out_indices = tf_interpreter->outputs();
  auto num_outputs = out_indices.size();
  auto out_tensor = tf_interpreter->tensor(out_indices[0]);
  auto out_size = out_tensor->bytes / sizeof(float);

  float *tf_output = tf_interpreter->typed_output_tensor<float>(0);
  for (size_t idx = 0; idx < out_size; idx++) {
    ret_vector.push_back(tf_output[idx]);
  }
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  return ret_vector;
}

void data_clear() {
  out.clear();
  ans.clear();
  in_f.clear();
  l_f.clear();
}

/**
 * @brief Simple Fully Connected Layer export TEST
 */
TEST(nntrainerInterpreterTflite, simple_fc) {

  nntrainer::TfliteInterpreter interpreter;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "mse")});

  nn_model->addLayer(createLayer(
    "input", {withKey("name", "in0"), withKey("input_shape", "1:1:1:1")}));
  nn_model->addLayer(createLayer("fully_connected",
                                 {withKey("name", "fc0"), withKey("unit", 2)}));
  nn_model->addLayer(createLayer("fully_connected",
                                 {withKey("name", "fc1"), withKey("unit", 1)}));

  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  data_clear();
  unsigned int data_size = 1 * 1 * 1 * 1;
  std::vector<float> input_data;
  float *nntr_input = new float[data_size];

  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
  }

  in_f.push_back(nntr_input);
  auto answer_f = nn_model->inference(1, in_f, l_f);
  for (auto element : answer_f) {
    ans.push_back(*element);
  }
  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE,
                    "simple_fc.tflite");

  out = run_tflite("simple_fc.tflite", input_data);

  for (size_t i = 0; i < out.size(); i++)
    EXPECT_NEAR(out[i], ans[i], 0.000001f);

  if (remove("simple_fc.tflite")) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove tflite "
              << "simple_fc.tflite"
              << "failed, reason: "
              << strerror_r(errno, error_buf, error_buflen);
  }
}

/**
 * @brief Flatten Test for export (NCHW -> NHWC)
 *
 */
TEST(nntrainerInterpreterTflite, flatten_test) {

  nntrainer::TfliteInterpreter interpreter;
  nntrainer::FlattenRealizer fr;
  data_clear();

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=3:2:4"});

  auto flat = LayerRepresentation("flatten", {"name=flat", "input_layers=in0"});

  auto g = fr.realize(makeGraph({input0, flat}));

  nntrainer::NetworkGraph ng;

  /// @todo Disable Support Inplace  --> should be support Inplace()
  ng.setMemoryOptimizations(false);

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);
  interpreter.serialize(g, "flatten_test.tflite");
  ng.deallocateTensors();

  std::vector<float> input_data;

  int count = 0;
  for (int i = 0; i < 3 * 2 * 4; i++) {
    input_data.push_back(count);
    count++;
  }

  out = run_tflite("flatten_test.tflite", input_data);

  std::vector<float> ans = {0, 8,  16, 1, 9,  17, 2, 10, 18, 3, 11, 19,
                            4, 12, 20, 5, 13, 21, 6, 14, 22, 7, 15, 23};

  for (size_t i = 0; i < out.size(); i++)
    EXPECT_NEAR(out[i], ans[i], 0.000001f);

  if (remove("flatten_test.tflite")) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove tflite "
              << "flatten_test.tflite"
              << "failed, reason: "
              << strerror_r(errno, error_buf, error_buflen);
  }
}

/**
 * @brief Resnet Part export TEST
 */
TEST(nntrainerInterpreterTflite, part_of_resnet_0) {

  nntrainer::TfliteInterpreter interpreter;

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});

  auto averagepool0 = LayerRepresentation(
    "pooling2d", {"name=averagepool0", "input_layers=in0", "pooling=average",
                  "pool_size=1,1", "stride=1,1", "padding=valid"});

  auto reshape0 =
    LayerRepresentation("reshape", {"name=reshape0", "target_shape=1:1:1",
                                    "input_layers=averagepool0"});

  auto fc0 = LayerRepresentation(
    "fully_connected", {"name=fc0", "unit=1", "input_layers=reshape0",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto softmax0 = LayerRepresentation(
    "activation", {"name=softmax0", "activation=softmax", "input_layers=fc0"});

  auto g = makeGraph({input0, averagepool0, reshape0, fc0, softmax0});

  nntrainer::NetworkGraph ng;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "mse")});

  for (auto &node : g) {
    nn_model->addLayer(node);
  }

  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  data_clear();
  unsigned int data_size = 1 * 1 * 1 * 1;
  std::vector<float> input_data;
  float *nntr_input = new float[data_size];

  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
  }
  in_f.push_back(nntr_input);

  auto answer_f = nn_model->inference(1, in_f, l_f);
  for (auto element : answer_f) {
    ans.push_back(*element);
  }

  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE,
                    "part_of_resnet.tflite");

  out = run_tflite("part_of_resnet.tflite", input_data);

  for (size_t i = 0; i < out.size(); i++)
    EXPECT_NEAR(out[i], ans[i], 0.000001f);

  if (remove("part_of_resnet.tflite")) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove ini "
              << "part_of_resnet.tflite"
              << "failed, reason: "
              << strerror_r(errno, error_buf, error_buflen);
  }
}

/**
 * @brief MNIST Model export test
 */
TEST(nntrainerInterpreterTflite, MNIST_FULL_TEST) {

  nntrainer::TfliteInterpreter interpreter;

  ModelHandle nn_model = ml::train::createModel(
    ml::train::ModelType::NEURAL_NET, {withKey("loss", "mse")});
  nn_model->setProperty({withKey("memory_optimization", "false")});

  nn_model->addLayer(createLayer(
    "input", {withKey("name", "in0"), withKey("input_shape", "1:1:28:28")}));

  nn_model->addLayer(createLayer(
    "conv2d", {withKey("name", "conv0"), withKey("filters", 6),
               withKey("kernel_size", {5, 5}), withKey("stride", {1, 1}),
               withKey("padding", "same"), withKey("bias_initializer", "zeros"),
               withKey("weight_initializer", "xavier_uniform"),
               withKey("activation", "relu")}));

  nn_model->addLayer(createLayer(
    "pooling2d", {withKey("name", "pooling2d_p1"),
                  withKey("pooling", "average"), withKey("pool_size", {2, 2}),
                  withKey("stride", {2, 2}), withKey("padding", "same")}));

  nn_model->addLayer(createLayer(
    "conv2d", {withKey("name", "conv0"), withKey("filters", 12),
               withKey("kernel_size", {5, 5}), withKey("stride", {1, 1}),
               withKey("padding", "same"), withKey("bias_initializer", "zeros"),
               withKey("weight_initializer", "xavier_uniform"),
               withKey("activation", "relu")}));

  nn_model->addLayer(createLayer(
    "pooling2d", {withKey("name", "pooling2d_p1"),
                  withKey("pooling", "average"), withKey("pool_size", {2, 2}),
                  withKey("stride", {2, 2}), withKey("padding", "same")}));

  nn_model->addLayer(createLayer("flatten"));

  nn_model->addLayer(createLayer(
    "fully_connected", {withKey("name", "fc0"), withKey("unit", 10)}));

  EXPECT_EQ(nn_model->compile(), ML_ERROR_NONE);
  EXPECT_EQ(nn_model->initialize(), ML_ERROR_NONE);

  data_clear();
  unsigned int data_size = 1 * 1 * 28 * 28;
  std::vector<float> input_data;
  float nntr_input[28 * 28];

  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
  }

  in_f.push_back(nntr_input);
  auto answer_f = nn_model->inference(1, in_f, l_f);
  std::cout << "answer_length" << answer_f.size() << "\n";
  for (auto element : answer_f) {
    ans.push_back(*element);
  }
  nn_model->exports(ml::train::ExportMethods::METHOD_TFLITE,
                    "MNIST_FULL_TEST.tflite");

  out = run_tflite("MNIST_FULL_TEST.tflite", input_data);

  for (size_t i = 0; i < ans.size(); i++) {
    EXPECT_NEAR(out[i], ans[i], 0.000001f);
    std::cout << "out : " << out[i] << " ans : " << ans[i] << std::endl;
  }
  if (remove("MNIST_FULL_TEST.tflite")) {
    const size_t error_buflen = 100;
    char error_buf[error_buflen];
    std::cerr << "remove tflite "
              << "MNIST_FULL_TEST.tflite"
              << "failed, reason: "
              << strerror_r(errno, error_buf, error_buflen);
  }
}
