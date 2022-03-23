// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   main.cpp
 * @date   01 July 2020
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is MNIST Example with
 *
 * Input 28x28
 * Conv2D 5 x 5 : 6 filters, stride 1x1, padding=0,0
 * Pooling2D : 2 x 2, Average pooling, stride 1x1
 * Conv2D 6 x 5 x 5 : 12 filters, stride 1x1, padding=0,0
 * Pooling2D : 2 x 2, Average Pooling, stride 1x1
 * Flatten
 * Fully Connected Layer with 10 unit
 *
 */

#if defined(ENABLE_TEST)
#define APP_VALIDATE
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>
#include <vector>

#if defined(APP_VALIDATE)
#include <gtest/gtest.h>
#endif

#include <dataset.h>
#include <model.h>

#ifdef PROFILE
#include <profiler.h> // disable including this in android as profiler is not exposed yet to devel header
#endif

#define TRAINING true

#define VALIDATION false

constexpr unsigned int SEED = 0;

#if VALIDATION
/**
 * @brief     Data size for each category
 */
const unsigned int total_train_data_size = 32;

const unsigned int total_val_data_size = 32;

const unsigned int total_test_data_size = 32;

const unsigned int batch_size = 32;

#else

const unsigned int total_train_data_size = 100;

const unsigned int total_val_data_size = 100;

const unsigned int total_test_data_size = 100;

const unsigned int batch_size = 32;

#endif

/**
 * @brief     Number of category : Three
 */
const unsigned int total_label_size = 10;

unsigned int train_count = 0;
unsigned int val_count = 0;

const unsigned int feature_size = 784;

const float tolerance = 0.1;

std::string data_path;

float training_loss = 0.0;
float validation_loss = 0.0;

std::string filename = "mnist_trainingSet.dat";

/**
 * @brief     step function
 * @param[in] x value to be distinguished
 * @retval 0.0 or 1.0
 */
float stepFunction(float x) {
  if (x + tolerance > 1.0) {
    return 1.0;
  }

  if (x - tolerance < 0.0) {
    return 0.0;
  }

  return x;
}

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] input input
 * @param[out] label label
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
bool getData(std::ifstream &F, float *input, float *label, unsigned int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  uint64_t position = (uint64_t)((feature_size + total_label_size) *
                                 (uint64_t)id * sizeof(float));

  if (position > file_length) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  F.read((char *)input, sizeof(float) * feature_size);
  F.read((char *)label, sizeof(float) * total_label_size);

  return true;
}

/**
 * @brief UserData which stores information used to feed data from data callback
 *
 */
class DataInformation {
public:
  /**
   * @brief Construct a new Data Information object
   *
   * @param num_samples number of data
   * @param filename file name to read from
   */
  DataInformation(unsigned int num_samples, const std::string &filename);
  unsigned int count;
  unsigned int num_samples;
  std::ifstream file;
  std::vector<unsigned int> idxes;
  std::mt19937 rng;
};

DataInformation::DataInformation(unsigned int num_samples,
                                 const std::string &filename) :
  count(0),
  num_samples(num_samples),
  file(filename, std::ios::in | std::ios::binary),
  idxes(num_samples) {
  std::iota(idxes.begin(), idxes.end(), 0);
  rng.seed(SEED);
  std::shuffle(idxes.begin(), idxes.end(), rng);
  if (!file.good()) {
    throw std::invalid_argument("given file is not good, filename: " +
                                filename);
  }
}

/**
 * @brief      get data which size is batch for train
 * @param[out] outInput input vectors
 * @param[out] outLabel label vectors
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  auto data = reinterpret_cast<DataInformation *>(user_data);

  getData(data->file, *outVec, *outLabel, data->idxes.at(data->count));
  data->count++;
  if (data->count < data->num_samples) {
    *last = false;
  } else {
    *last = true;
    data->count = 0;
    std::shuffle(data->idxes.begin(), data->idxes.end(), data->rng);
  }

  return 0;
}

#if defined(APP_VALIDATE)
TEST(MNIST_training, verify_accuracy) {
  EXPECT_FLOAT_EQ(training_loss, 2.5698349);
  EXPECT_FLOAT_EQ(validation_loss, 2.5551746);
}
#endif

/**
 * @brief     create model
 *            Get Feature from tflite & run foword & back propatation
 * @param[in]  arg 1 : configuration file path
 * @param[in]  arg 2 : resource path
 */
int main(int argc, char *argv[]) {
  int status = 0;
#ifdef APP_VALIDATE
  status = remove("mnist_model.bin");
  if (status != 0) {
    std::cout << "Pre-existing model file doesn't exist.\n";
  }
#endif
  if (argc < 3) {
    std::cout << "./nntrainer_mnist mnist.ini dataset.dat\n";
    exit(0);
  }

#ifdef PROFILE
  nntrainer::profile::GenericProfileListener listener(
    &nntrainer::profile::Profiler::Global());
#endif

  const std::vector<std::string> args(argv + 1, argv + argc);
  std::string config = args[0];
  filename = args[1];

  std::unique_ptr<DataInformation> train_user_data;
  std::unique_ptr<DataInformation> valid_user_data;
  try {
    train_user_data =
      std::make_unique<DataInformation>(total_train_data_size, filename);
    valid_user_data =
      std::make_unique<DataInformation>(total_val_data_size, filename);
  } catch (std::invalid_argument &e) {
    std::cerr << "Error creating userdata for the data callback " << e.what()
              << std::endl;
    return 1;
  }

  /**
   * @brief     Data buffer Create & Initialization
   */
  std::shared_ptr<ml::train::Dataset> dataset_train, dataset_val;
  try {
    dataset_train = createDataset(ml::train::DatasetType::GENERATOR, getSample,
                                  train_user_data.get());
    dataset_val = createDataset(ml::train::DatasetType::GENERATOR, getSample,
                                valid_user_data.get());
  } catch (std::exception &e) {
    std::cerr << "Error creating dataset" << e.what() << std::endl;
    return 1;
  }

  std::unique_ptr<ml::train::Model> model;
  try {
    /**
     * @brief     Neural Network Create & Initialization
     */
    model = createModel(ml::train::ModelType::NEURAL_NET);
    model->load(config, ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN);
  } catch (std::exception &e) {
    std::cerr << "Error during loadFromConfig " << e.what() << std::endl;
    return 1;
  }

  try {
    model->compile();
    model->initialize();
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_val);
  } catch (std::exception &e) {
    std::cerr << "Error during init " << e.what() << std::endl;
    return 1;
  }

#if defined(APP_VALIDATE)
  try {
    model->setProperty({"epochs=5"});
  } catch (...) {
    std::cerr << "Error during setting epochs\n";
    return -1;
  }
#endif

  /**
   * @brief     Neural Network Train & validation
   */
  try {
    model->train();
    training_loss = model->getTrainingLoss();
    validation_loss = model->getValidationLoss();
  } catch (std::exception &e) {
    std::cerr << "Error during train " << e.what() << std::endl;
    return 0;
  }

#ifdef PROFILE
  std::cout << listener;
#endif

#if defined(APP_VALIDATE)
  try {
    testing::InitGoogleTest(&argc, argv);
  } catch (...) {
    std::cerr << "Error duing InitGoogleTest" << std::endl;
    return 0;
  }

  try {
    status = RUN_ALL_TESTS();
  } catch (...) {
    std::cerr << "Error duing RUN_ALL_TSETS()" << std::endl;
  }
#endif

  /**
   * @brief     Finalize model
   */
  return status;
}
