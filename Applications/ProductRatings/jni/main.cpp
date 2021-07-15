// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   10 March 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is a simple recommendation system Example
 *
 *              Trainig set (embedding_input.txt) : 4 colume data + result (1.0
 * or 0.0) Configuration file : ../../res/Embedding.ini
 *
 */

#include <cmath>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdlib.h>
#include <time.h>

#include <dataset.h>
#include <ml-api-common.h>
#include <neuralnet.h>
#include <tensor.h>

std::string data_file;

const unsigned int total_train_data_size = 25;

unsigned int train_count = 0;

const unsigned int batch_size = 20;

const unsigned int feature_size = 2;

const unsigned int total_val_data_size = 25;

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

  if (!std::getline(F, temp)) {
    return false;
  }

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

template <typename T> void loadFile(const char *filename, T &t) {
  std::ifstream file(filename);
  if (!file.good()) {
    throw std::runtime_error("could not read, check filename");
  }
  t.read(file);
  file.close();
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

  std::vector<float> o;
  std::vector<float> l;
  o.resize(feature_size);
  l.resize(1);

  for (unsigned int i = train_count; i < train_count + batch_size; ++i) {
    if (!getData(dataFile, o, l, i)) {
      return -1;
    }

    for (unsigned int j = 0; j < feature_size; ++j) {
      outVec[0][count * feature_size + j] = o[j];
    }
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
 * @param[in]  arg 3 : resource path (data) with below format
 * (int) (int) (float) #first data
 * ...
 * in each row represents user id, product id, rating (0 to 10)
 */
int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "./Embedding train (| inference) Config.ini data.txt\n";
    exit(1);
  }

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[1];
  data_file = args[2];

  if (!args[0].compare("train"))
    training = true;

  srand(time(NULL));

  std::shared_ptr<ml::train::Dataset> dataset;
  try {
    dataset = createDataset(ml::train::DatasetType::GENERATOR, getBatch_train,
                            getBatch_train);
  } catch (std::exception &e) {
    std::cerr << "Error creating dataset" << e.what() << std::endl;
    return 1;
  }

  /**
   * @brief     Create NN
   */
  std::vector<std::vector<float>> inputVector, outputVector;
  nntrainer::NeuralNetwork NN;
  /**
   * @brief     Initialize NN with configuration file path
   */

  try {
    auto status = NN.loadFromConfig(config);
    if (status != ML_ERROR_NONE) {
      std::cerr << "Error during loading" << std::endl;
      return 1;
    }

    status = NN.compile();
    if (status != ML_ERROR_NONE) {
      std::cerr << "Error during compile" << std::endl;
      return 1;
    }
    status = NN.initialize();
    if (status != ML_ERROR_NONE) {
      std::cerr << "Error during initialize" << std::endl;
      return 1;
    }
    NN.readModel();

    std::cout << "Input dimension: " << NN.getInputDimension()[0];

  } catch (std::exception &e) {
    std::cerr << "Unexpected Error during init " << e.what() << std::endl;
    return 1;
  }

  if (training) {
    NN.setDataset(dataset);
    try {
      NN.train({"batch_size=" + std::to_string(batch_size)});
    } catch (std::exception &e) {
      std::cerr << "Error during train " << e.what() << std::endl;
      return 1;
    }

    try {
      /****** testing with a golden data if any ********/
      nntrainer::Tensor golden(1, 1, 15, 8);

      loadFile("embedding_weight_golden.out", golden);
      golden.print(std::cout);

      nntrainer::Tensor weight_out_fc(1, 1, 32, 1);
      loadFile("fc_weight_golden.out", weight_out_fc);
      weight_out_fc.print(std::cout);
    } catch (...) {
      std::cerr << "Warning: during loading golden data\n";
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
          NN.inference({MAKE_SHARED_TENSOR(nntrainer::Tensor({o}))})[0]
            ->apply(stepFunction)
            .getValue(0, 0, 0, 0);

        std::cout << answer << " : " << l[0] << std::endl;
        cn += answer == l[0];
      } catch (...) {
        std::cerr << "Error during forwarding the model" << std::endl;
        return 1;
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
