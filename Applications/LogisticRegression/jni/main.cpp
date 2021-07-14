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

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include <databuffer.h>
#include <neuralnet.h>
#include <tensor.h>

std::string data_file;

const unsigned int total_train_data_size = 90;

unsigned int train_count = 0;

const unsigned int batch_size = 16;

const unsigned int feature_size = 2;

const unsigned int total_val_data_size = 10;

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
 * @param[out] outVec feature data
 * @param[out] outLabel label data
 * @param[in] id id th
 * @retval boolean true if there is no error
 */
bool getData(std::ifstream &F, std::vector<float> &outVec,
             std::vector<float> &outLabel, unsigned int id) {
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
    outVec[j] = x;
  }
  buffer >> x;
  outLabel[0] = x;

  return true;
}

/**
 * @brief     get Data as much as batch size
 * @param[out] outVec feature data
 * @param[out] outLabel label data
 * @param[out] last end of data
 * @param[in] user_data user data
 * @retval int 0 if there is no error
 */
int getBatch_train(float **outVec, float **outLabel, bool *last,
                   void *user_data) {
  std::ifstream dataFile(data_file);
  unsigned int data_size = total_train_data_size;
  unsigned int count = 0;

  if (data_size - train_count < batch_size) {
    *last = true;
    train_count = 0;
    return 0;
  }

  for (unsigned int i = train_count; i < train_count + batch_size; ++i) {

    std::vector<float> o;
    std::vector<float> l;
    o.resize(feature_size);
    l.resize(1);

    if (!getData(dataFile, o, l, i)) {
      return -1;
    };

    for (unsigned int j = 0; j < feature_size; ++j)
      outVec[0][count * feature_size + j] = o[j];
    outLabel[0][count] = l[0];

    count++;
  }

  dataFile.close();
  *last = false;
  train_count += batch_size;
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

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[1];
  data_file = args[2];

  if (!args[0].compare("train"))
    training = true;

  srand(time(NULL));

  auto data_train =
    ml::train::createDataset(ml::train::DatasetType::GENERATOR, getBatch_train);

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
    NN.setDataset(ml::train::DatasetDataUsageType::DATA_TRAIN,
                  std::move(data_train));

    try {
      NN.train();
    } catch (...) {
      std::cerr << "Error during train" << std::endl;
      return 0;
    }
  } else {
    try {
      NN.readModel();
    } catch (std::exception &e) {
      std::cerr << "Error during readModel: " << e.what() << "\n";
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

      getData(dataFile, o, l, j);

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
