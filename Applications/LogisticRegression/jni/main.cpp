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
 * @brief	This is Binary Logistic Regression Example
 *
 *              Trainig set (dataset1.txt) : two colume data + result (1.0 or
 * 0.0) Configuration file : ../../res/LogisticRegression.ini Test set
 * (test.txt)
 */

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>

#include <databuffer.h>
#include <neuralnet.h>
#include <tensor.h>

std::string data_file;

const unsigned int total_train_data_size = 90;

unsigned int train_count = 0;

const unsigned int batch_size = 16;

const unsigned int feature_size = 2;

const unsigned int total_val_data_size = 10;

constexpr unsigned int SEED = 0;

bool training = false;

/**
 * @brief     step function
 * @param[in] x value to be distinguished
 * @retval 0.0 or 1.0
 */
float stepFunction(float x) {
  if (x > 0.5) {
    return 1.0;
  }

  if (x < 0.5) {
    return 0.0;
  }

  return x;
}

/**
 * @brief     get idth Data
 * @param[in] F file stream
 * @param[out] input feature data
 * @param[out] label label data
 * @param[in] id id th
 * @retval boolean true if there is no error
 */
bool getData(std::ifstream &F, float *input, float *label, unsigned int id) {
  std::string temp;
  F.clear();
  F.seekg(0, std::ios_base::beg);
  char c;
  unsigned int i = 0;
  while (F.get(c) && i < id)
    if (c == '\n')
      ++i;

  F.putback(c);

  if (!std::getline(F, temp))
    return false;

  std::istringstream buffer(temp);
  float x;
  for (unsigned int j = 0; j < feature_size; ++j) {
    buffer >> x;
    input[j] = x;
  }
  buffer >> x;
  label[0] = x;

  return true;
}

std::mt19937 rng;
std::vector<unsigned int> train_idxes;

/**
 * @brief     get a single data
 * @param[out] outVec feature data
 * @param[out] outLabel label data
 * @param[out] last end of data
 * @param[in] user_data user data
 * @retval int 0 if there is no error
 */
int getSample_train(float **outVec, float **outLabel, bool *last,
                    void *user_data) {
  std::ifstream dataFile(data_file);

  if (!getData(dataFile, *outVec, *outLabel, train_idxes.at(train_count))) {
    return -1;
  }
  train_count++;
  if (train_count < total_train_data_size) {
    *last = false;
  } else {
    *last = true;
    train_count = 0;
    std::shuffle(train_idxes.begin(), train_idxes.end(), rng);
  }

  return 0;
}

/**
 * @brief     create NN
 *            back propagation of NN
 * @param[in]  arg 1 : train / inference
 * @param[in]  arg 2 : configuration file path
 * @param[in]  arg 3 : resource path (dataset.txt or testset.txt)
 */
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout
      << "./LogisticRegression train (| inference) Config.ini data.txt\n";
    exit(1);
  }

  const std::string weight_path = "logistic_model.bin";
  train_idxes.resize(total_train_data_size);
  std::iota(train_idxes.begin(), train_idxes.end(), 0);
  rng.seed(SEED);

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[1];
  data_file = args[2];

  if (!args[0].compare("train"))
    training = true;

  srand(time(NULL));

  auto data_train = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                             getSample_train);

  /**
   * @brief     Create NN
   */
  std::vector<std::vector<float>> inputVector, outputVector;
  nntrainer::NeuralNetwork NN;

  /**
   * @brief     Initialize NN with configuration file path
   */

  try {
    NN.loadFromConfig(config);
    NN.compile();
    NN.initialize();
  } catch (...) {
    std::cerr << "Error during init" << std::endl;
    return 0;
  }

  if (training) {
    NN.setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                  std::move(data_train));

    try {
      NN.train({"save_path=" + weight_path});
    } catch (...) {
      std::cerr << "Error during train" << std::endl;
      return 0;
    }
  } else {
    try {
      NN.load(weight_path, ml::train::ModelFormat::MODEL_FORMAT_BIN);
    } catch (std::exception &e) {
      std::cerr << "Error during loading weights: " << e.what() << "\n";
      return 1;
    }
    std::ifstream dataFile(data_file);
    int cn = 0;
    for (unsigned int j = 0; j < total_val_data_size; ++j) {
      nntrainer::Tensor d;
      std::vector<float> o;
      std::vector<float> l;
      o.resize(feature_size);
      l.resize(1);

      getData(dataFile, o.data(), l.data(), j);

      try {
        float answer =
          NN.forwarding({MAKE_SHARED_TENSOR(nntrainer::Tensor({o}))},
                        {MAKE_SHARED_TENSOR(nntrainer::Tensor({l}))})[0]
            ->apply(stepFunction)
            .getValue(0, 0, 0, 0);
        std::cout << answer << " : " << l[0] << std::endl;
        cn += answer == l[0];
      } catch (...) {
        std::cerr << "Error during forwarding the model" << std::endl;
        return -1;
      }
    }
    std::cout << "[ Accuracy ] : "
              << ((float)(cn) / total_val_data_size) * 100.0 << "%"
              << std::endl;
  }

  /**
   * @brief     Finalize NN
   */
  return 0;
}
