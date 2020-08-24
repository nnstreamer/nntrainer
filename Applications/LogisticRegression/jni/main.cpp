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

#include "neuralnet.h"
#include "tensor.h"
#define training true

std::string data_file;

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
 * @brief     create NN
 *            back propagation of NN
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path (dataset.txt or testset.txt)
 */
int main(int argc, char *argv[]) {
  if (argc < 3) {
    std::cout << "./LogisticRegression Config.ini data.txt\n";
    exit(0);
  }

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];
  data_file = args[1];

  srand(time(NULL));

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
    NN.init();
  } catch (...) {
    std::cerr << "Error during init" << std::endl;
    NN.finalize();
    return 0;
  }

  if (!training)
    NN.readModel();

  /**
   * @brief     Generate Trainig Set
   */
  std::ifstream dataFile(data_file);
  if (dataFile.is_open()) {
    std::string temp;
    int index = 0;
    while (std::getline(dataFile, temp)) {
      if (training && index % 10 == 1) {
        std::cout << temp << std::endl;
        index++;
        continue;
      }
      std::istringstream buffer(temp);
      std::vector<float> line;
      std::vector<float> out;
      float x;
      for (int i = 0; i < 2; i++) {
        buffer >> x;
        line.push_back(x);
      }
      inputVector.push_back(line);
      buffer >> x;
      out.push_back(x);
      outputVector.push_back(out);
      index++;
    }
  }

  /**
   * @brief     training NN ( back propagation )
   */
  if (training) {
    for (unsigned int i = 0; i < NN.getEpochs(); i++) {
      for (unsigned int j = 0; j < inputVector.size(); ++j) {
        std::vector<std::vector<float>> in, label;
        in.push_back(inputVector[j]);
        label.push_back(outputVector[j]);
        nntrainer::Tensor d, y;
        try {
          d = nntrainer::Tensor(in);
        } catch (...) {
          std::cerr << "Error during tensor construct" << std::endl;
          NN.finalize();
          return -1;
        }
        try {
          y = nntrainer::Tensor(label);
        } catch (...) {
          std::cerr << "Error during tensor construct" << std::endl;
          NN.finalize();
          return -1;
        }
        try {
          NN.backwarding(MAKE_SHARED_TENSOR(d), MAKE_SHARED_TENSOR(y), i);
        } catch (...) {
          std::cerr << "Error during backwarding the model" << std::endl;
          NN.finalize();
          return -1;
        }
      }
      std::cout << "#" << i + 1 << "/" << NN.getEpochs()
                << " - Loss : " << NN.getLoss() << std::endl;
      NN.setLoss(0.0);
    }
  } else {
    /**
     * @brief     forward propagation
     */
    int cn = 0;
    for (unsigned int j = 0; j < inputVector.size(); ++j) {
      std::vector<std::vector<float>> in, label;
      in.push_back(inputVector[j]);
      label.push_back(outputVector[j]);
      try {
        cn += NN.forwarding(MAKE_SHARED_TENSOR(in))
                ->apply(stepFunction)
                .getValue(0, 0, 0, 0) == label[0][0];
      } catch (...) {
        std::cerr << "Error during forwarding the model" << std::endl;
        NN.finalize();
        return -1;
      }
    }
    std::cout << "[ Accuracy ] : " << ((float)(cn) / inputVector.size()) * 100.0
              << "%" << std::endl;
  }

  /**
   * @brief     save Weight & Bias
   */
  NN.saveModel();

  /**
   * @brief     Finalize NN
   */
  NN.finalize();
}
