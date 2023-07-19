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
#include <random>
#include <sstream>

#include <dataset.h>
#include <neuralnet.h>
#include <tensor.h>

std::string data_file;

constexpr unsigned int SEED = 0;

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
  uint *input_int = (uint *)input;
  uint x;
  for (unsigned int j = 0; j < feature_size; ++j) {
    buffer >> x;
    input_int[j] = x;
  }
  buffer >> x;
  label[0] = x;

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

  std::string weight_path = "product_ratings_model.bin";

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[1];
  data_file = args[2];

  if (!args[0].compare("train"))
    training = true;

  train_idxes.resize(total_train_data_size);
  std::iota(train_idxes.begin(), train_idxes.end(), 0);
  rng.seed(SEED);

  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_val;
  try {
    dataset_train =
      createDataset(ml::train::DatasetType::GENERATOR, getSample_train);
    dataset_val =
      createDataset(ml::train::DatasetType::GENERATOR, getSample_train);
  } catch (std::exception &e) {
    std::cerr << "Error creating dataset " << e.what() << std::endl;
    return 1;
  }

  /**
   * @brief     Create NN
   */
  nntrainer::NeuralNetwork NN;
  /**
   * @brief     Initialize NN with configuration file path
   */

  try {
    auto status = NN.loadFromConfig(config);
    if (status != 0) {
      std::cerr << "Error during loading" << std::endl;
      return 1;
    }

    status = NN.compile();
    if (status != 0) {
      std::cerr << "Error during compile" << std::endl;
      return 1;
    }
    status = NN.initialize();
    if (status != 0) {
      std::cerr << "Error during initialize" << std::endl;
      return 1;
    }

    std::cout << "Input dimension: " << NN.getInputDimension()[0];

  } catch (std::exception &e) {
    std::cerr << "Unexpected Error during init " << e.what() << std::endl;
    return 1;
  }

  if (training) {
    NN.setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    NN.setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);
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
          NN.inference({MAKE_SHARED_TENSOR(nntrainer::Tensor({o}, nntrainer::TensorDim::TensorType()))})[0]
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
