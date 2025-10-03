// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   test_simple_model_no_bias_poc.cpp
 * @date   2 Oct 2025
 * @brief  POC for ONNX model with two FC layers (no bias) using single bin weight loading
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Niket
 * @bug    No known bugs except for NYI items
 */

#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

int main() {
  auto model = ml::train::createModel();

  std::cout << "--------------------------------------Create Model Done--------------------------------------" << std::endl;
  try {
    std::string path = "/home/code/niket/nntrainer_october/nntrainer/Applications/ONNX/jni/simple_model_no_bias.onnx";
    model->load(path, ml::train::ModelFormat::MODEL_FORMAT_ONNX);
  } catch (const std::exception &e) {
    std::cerr << "Error during load: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Load Model Done--------------------------------------" << std::endl;
  try {
    model->compile(ml::train::ExecutionMode::INFERENCE);
  } catch (const std::exception &e) {
    std::cerr << "Error during compile: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Compile Model Done--------------------------------------" << std::endl;
  try {
    model->initialize();
  } catch (const std::exception &e) {
    std::cerr << "Error during initialize: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Initialize Model Done--------------------------------------" << std::endl;
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);
  std::cout << "--------------------------------------Summarize Model Done--------------------------------------" << std::endl;

  // Load weights from single bin file
  std::cout << "Loading weights from single bin file..." << std::endl;
  try {
    model->load("/home/code/niket/nntrainer_october/nntrainer/Applications/ONNX/jni/simple_model_no_bias_weights.bin", ml::train::ModelFormat::MODEL_FORMAT_BIN);
    std::cout << "Successfully loaded weights from single bin file" << std::endl;
  } catch (const std::exception &e) {
    std::cerr << "Error during weight loading: " << e.what() << "\n";
    return 1;
  }

  // Prepare simple input data for the model
  // Input shape: [1, 4] -> FC1 (4->8) -> FC2 (8->2) -> Output: [1, 2]
  float *input = new float[4];
  
  // Initialize input data
  for(int i = 0; i < 4; i++){
    input[i] = static_cast<float>(i + 1);  // [1.0, 2.0, 3.0, 4.0]
  }

  std::vector<float *> in; 
  in.push_back(input);
  
  // Run inference
  auto ans = model->inference(1, in);

  std::cout << "-------------------------------------------Inference Done--------------------------------------------" << std::endl;
  
  // Print input
  std::cout << "Input: [";
  for(int i = 0; i < 4; i++){
    std::cout << input[i];
    if(i < 3) std::cout << ", ";
  }
  std::cout << "]" << std::endl;
  
  // Print output
  std::cout << "Output: [";
  for(size_t i = 0; i < ans.size(); i++) {
    for(int j = 0; j < 2; j++) {  // Output should be [1, 2]
      std::cout << ans[i][j];
      if(j < 1) std::cout << ", ";
    }
  }
  std::cout << "]" << std::endl;
  
  // Clean up
  delete[] input;
  
  return 0;
}
