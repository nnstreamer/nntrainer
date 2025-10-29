// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 * @file   main.cpp
 * @date   14 October 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * Updated on 16 Oct 2025 to add debug output for Android execution
 */

#include <fstream>
#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

void saveToRaw(const float *data, size_t size, const std::string &filename) {
  std::ofstream out(filename, std::ios::binary);
  if (!out) {
    std::cerr << "Error: Cannot open file " << filename << " for writing.\n";
    return;
  }

  out.write(reinterpret_cast<const char *>(data), size * sizeof(float));
  out.close();

  std::cout << std::endl << ".bin generated successfully !";
}

int main() {
  auto model = ml::train::createModel();

  std::cout << "--------------------------------------Create Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    // std::string path = "/storage_data/snap/sumon/sumon-98/nntrainer/Applications/ONNX/jni/qwen3_model_one_layer_no_cast.onnx";
    std::string path = "./qwen3_model_one_layer_no_cast.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Load Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    model->compile(ml::train::ExecutionMode::INFERENCE);
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Compile Model "
               "Done--------------------------------------"
            << std::endl;
  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Initialize Model Done--------------------------------------" << std::endl;
  std::cout << "Skipping model summary..." << std::endl;

  std::cout << "Starting model summary..." << std::endl;
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  std::cout << "Finished model summary." << std::endl;

  std::cout << "--------------------------------------Summarize Model "
               "Done--------------------------------------"
            << std::endl;

  std::cout << "Loading weights..." << std::endl;
  std::string weight_path = "./qwen_weights_one_layer_no_cast/";
  std::cout << "Loading weights from: " << weight_path << std::endl;
  std::cout << "Loading weights from: " << weight_path << std::endl;
  try {
        model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
        std::cout << "Weights loaded successfully" << std::endl;
      } catch (std::exception &e) {
        std::cerr << "Error during loading weights: " << e.what() << "\n";
        return 1;
      }
  std::cout<<"starting inferencing!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!";
  float *input = new float[1];
  float *sin = new float[128];
  float *cos = new float[128];
  float *epsilon = new float[1];

  input[0] = 52;

  for (int i = 0; i < 128; i++) {
    sin[i] = 0;
    cos[i] = 1;
  }
  epsilon[0] = 1e-6;

  std::vector<float *> in;

  in.push_back(epsilon);
  in.push_back(sin);
  in.push_back(cos);
  in.push_back(input);

  auto ans = model->inference(1, in);

  std::cout << "-------------------------------------------Inference "
               "Done--------------------------------------------"
            << std::endl;

  for (auto it : ans) {
    saveToRaw(it, 151936,
              "../../../../Applications/ONNX/jni/nntrainer_logits.bin");
  }

  return 0;
}
