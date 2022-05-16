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
#include <layer.h>
#include <model.h>
#include <optimizer.h>
#include <random>
#include <sstream>

std::string data_file;

const unsigned int total_train_data_size = 90;

unsigned int train_count = 0;

const unsigned int batch_size = 16;

const unsigned int feature_size = 2;

const unsigned int total_val_data_size = 10;

constexpr unsigned int SEED = 0;

bool training = false;

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

  /**
   * @brief     Create NN
   */
  std::unique_ptr<ml::train::Model> model;
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);

  /**
   * @brief     Initialize NN with configuration file path
   */

  try {
    model->load(config, ml::train::ModelFormat::MODEL_FORMAT_INI);
    model->compile();
    model->initialize();
  } catch (...) {
    std::cerr << "Error during init" << std::endl;
    return 0;
  }

  if (training) {
    auto data_train = ml::train::createDataset(
      ml::train::DatasetType::GENERATOR, getSample_train);

    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN,
                      std::move(data_train));

    try {
      model->train({"save_path=" + weight_path});
    } catch (...) {
      std::cerr << "Error during train" << std::endl;
      return 0;
    }
  } else {
    try {
      model->load(weight_path);
    } catch (std::exception &e) {
      std::cerr << "Error during loading weights: " << e.what() << "\n";
      return 1;
    }
    std::ifstream dataFile(data_file);
    int cn = 0;
    std::vector<float *> in;
    std::vector<float *> l;

    auto step = [](const float result) {
      if (result < 0.5) {
        return 0;
      } else
        return 1;
    };

    for (unsigned int j = 0; j < total_val_data_size; ++j) {

      float input[feature_size];
      float label[1];

      if (!getData(dataFile, input, label, j))
        std::cout << "error dring read file " << std::endl;

      try {

        std::vector<float *> answer;

        in.push_back(input);
        l.push_back(label);

        answer = model->inference(1, in, l);

        in.clear();
        l.clear();

        int c = step(answer[0][0]);

        if (c == int(label[0])) {
          cn++;
          std::cout << answer[0][0] << " - " << c << " : " << label[0]
                    << std::endl;
        } else {
          std::cout << " Something Wrong  " << answer[0][0] << " " << label[0]
                    << std::endl;
        }
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
