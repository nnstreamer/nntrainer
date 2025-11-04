// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Sachin Singh <sachin.3@samsung.com>
 * @file   main.cpp
 * @date   14 October 2025
 * @brief  onnx example using nntrainer-onnx-api
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Sachin Singh <sachin.3@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <layer.h>
#include <model.h>
#include <nntrainer-api-common.h>
#include <optimizer.h>
#include <util_func.h>

void loadFromRaw(float *data, size_t size, const std::string &filename) {

  std::ifstream file(filename, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open file: " << filename << std::endl;
    return;
  }

  file.read(reinterpret_cast<char *>(data), size * sizeof(float));
  std::streamsize bytesRead = file.gcount();

  if (bytesRead != size * sizeof(float)) {
    std::cerr << "Warning: Expected " << size * sizeof(float)
              << " bytes, but read " << bytesRead << " bytes.\n";
  }

  file.close();
  return;
}

void saveToRaw(float *data, size_t size, const std::string &filename) {
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
    std::string path =
      "../../../../Applications/ONNX/python/qwen3/multi-token/qwen3_model.onnx";
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

  std::cout << "--------------------------------------Initialize Model "
               "Done--------------------------------------"
            << std::endl;
  model->summarize(std::cout, ML_TRAIN_SUMMARY_MODEL);

  std::cout << "--------------------------------------Summarize Model "
               "Done--------------------------------------"
            << std::endl;

  std::string weight_path =
    "../../../../Applications/ONNX/python/qwen3/multi-token/bins/";
  try {
    model->load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
  } catch (std::exception &e) {
    std::cerr << "Error during loading weights: " << e.what() << "\n";
    return 1;
  }

  std::cout << "--------------------------------------Loading weights "
               "Done--------------------------------------"
            << std::endl;

  const int max_embedding_length = 256;
  const int tokens_to_be_generated = 20;
  const int num_vocab = 151936;
  int curr_len = 0;

  float *input = new float[max_embedding_length];
  float *sin = new float[max_embedding_length * 128];
  float *cos = new float[max_embedding_length * 128];
  float *epsilon = new float[1];

  // Loading inputs
  loadFromRaw(
    input, max_embedding_length,
    "../../../../Applications/ONNX/python/qwen3/multi-token/input_tokens.bin");

  for (int i = 0; i < max_embedding_length; i++) {
    if (input[i] == 151643)
      break;
    ++curr_len;
  }

  // Loading rotary embeddings
  loadFromRaw(sin, max_embedding_length * 128,
              "../../../../Applications/ONNX/python/qwen3/multi-token/"
              "rotary_embeddings_sine.bin");
  loadFromRaw(cos, max_embedding_length * 128,
              "../../../../Applications/ONNX/python/qwen3/multi-token/"
              "rotary_embeddings_cosine.bin");

  epsilon[0] = 1e-6;

  for (int i = 0; i < tokens_to_be_generated; i++) {
    float *output = model->inference(1, {epsilon, sin, cos, input})[0];
    output = output + (int)(curr_len - 1) * (num_vocab);
    float token_id =
      std::distance(output, std::max_element(output, output + num_vocab));
    input[curr_len] = token_id;
    curr_len += 1;
  }

  saveToRaw(
    input, curr_len,
    "../../../../Applications/ONNX/python/qwen3/multi-token/output_tokens.bin");

  return 0;
}
