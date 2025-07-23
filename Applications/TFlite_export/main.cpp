// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Donghak Park <donghak.park@samsung.com>
 *
 * @file   TFLite_export.cpp
 * @date   18 July 2025
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  TF Lite Export Example
 *
 */

#include <iostream>
#include <model.h>
#include <optimizer.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#include <tflite_interpreter.h>
#endif

const std::string INI_FILE_NAME = "./res/fc_model.ini";
const std::string TF_LITE_MODEL_NAME = "fc_model.tflite";
// const std::string BIN_FILE_NAME = "./res/saved_model.bin";
const unsigned int data_size = 12;
const unsigned int output_size = 9;

std::vector<float *> input_data;
unsigned int seed = 0;

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

  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  tf_interpreter->AllocateTensors();

  auto &in_indices = tf_interpreter->inputs();
  float *tf_input = tf_interpreter->typed_input_tensor<float>(0);

  for (unsigned int i = 0; i < input_vector.size(); i++) {
    tf_input[i] = input_vector[i];
  }

  tf_interpreter->Invoke();

  auto out_indices = tf_interpreter->outputs();
  auto out_tensor = tf_interpreter->tensor(out_indices[0]);
  auto out_size = out_tensor->bytes / sizeof(float);

  float *tf_output = tf_interpreter->typed_output_tensor<float>(0);
  for (size_t idx = 0; idx < out_size; idx++) {
    ret_vector.push_back(tf_output[idx]);
  }

  return ret_vector;
}

int main(int argc, char *argv[]) {
  std::unique_ptr<ml::train::Model> model;

  model = createModel(ml::train::ModelType::NEURAL_NET);
  std::cout << "Model Create Done" << std::endl;

  model->load(INI_FILE_NAME, ml::train::ModelFormat::MODEL_FORMAT_INI);

  std::cout << "Model Load with ini file Done" << std::endl;

  auto optimizer = ml::train::createOptimizer("sgd");
  model->setOptimizer(std::move(optimizer));
  model->compile();
  std::cout << "Model Compile Done" << std::endl;

  model->initialize();
  std::cout << "Model Initialize Done" << std::endl;

  // model->load(BIN_FILE_NAME, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  std::cout << "Model Weight file Load Done" << std::endl;

  std::vector<float> tf_lite_input_data;
  float *nntr_input = new float[data_size];

  std::cout << "Input Data : { ";
  for (unsigned int i = 0; i < data_size; i++) {
    auto rand_float = static_cast<float>(rand_r(&seed) / (RAND_MAX + 1.0));
    tf_lite_input_data.push_back(rand_float);
    nntr_input[i] = rand_float;
    std::cout << rand_float << ", ";
  }
  std::cout << "} " << std::endl;

  input_data.push_back(nntr_input);
  auto answer_f = model->inference(1, input_data, {});

  std::cout << "NNTrainer Output Data : { ";
  for (unsigned int i = 0; i < output_size; i++) {
    std::cout << answer_f[0][i] << ", ";
  }
  std::cout << "} " << std::endl;

  model->exports(ml::train::ExportMethods::METHOD_TFLITE, TF_LITE_MODEL_NAME);
  std::cout << "Model Export Done" << std::endl;

  auto tf_lite_answer_f = run_tflite(TF_LITE_MODEL_NAME, tf_lite_input_data);

  std::cout << "TFLITE Output Data : { ";
  for (auto element : tf_lite_answer_f) {
    std::cout << element << ", ";
  }
  std::cout << "} " << std::endl;
  std::cout << "Export & Compare Done" << std::endl;

  delete[] nntr_input;
  return 0;
}
