// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file   cifar_dataloader.h
 * @date   24 Jun 2021
 * @brief  dataloader for cifar 100
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug    No known bugs except for NYI items
 */
#include <tensor_dim.h>

#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace nntrainer::vgg {

using TensorDim = ml::train::TensorDim;

/**
 * @brief DataLoader interface used to load cifar data
 *
 */
class DataLoader {
public:
  /**
   * @brief Destroy the Data Loader object
   *
   */
  virtual ~DataLoader() {}

  /**
   * @brief create an iteration to fed to the generator callback
   *
   * @param[out] input list of inputs that is already allocated by nntrainer,
   * and this function is obliged to fill
   * @param[out] label list of label that is already allocated by nntrainer, and
   * this function is obliged to fill
   * @param[out] last  optional property to set when the epoch has finished
   */
  virtual void next(float **input, float **label, bool *last) = 0;
};

/**
 * @brief RandomData Generator
 *
 */
class RandomDataLoader final : public DataLoader {
public:
  /**
   * @brief Construct a new Random Data Loader object
   *
   * @param input_shapes input_shapes with appropriate batch
   * @param output_shapes label_shapes with appropriate batch
   * @param iteration     iteration per epoch
   */
  RandomDataLoader(const std::vector<TensorDim> &input_shapes,
                   const std::vector<TensorDim> &output_shapes, int iteration);

  /**
   * @brief Destroy the Random Data Loader object
   *
   */
  ~RandomDataLoader() {}

  /**
   * @copydoc void DataLoader::next(float **input, float**label, bool *last)
   */
  void next(float **input, float **label, bool *last);

private:
  unsigned int iteration;
  unsigned int iteration_for_one_epoch;

  std::vector<TensorDim> input_shapes;
  std::vector<TensorDim> output_shapes;

  std::mt19937 rng;
  std::uniform_int_distribution<int> input_dist;
  std::uniform_int_distribution<int> label_dist;
};

/**
 * @brief Cifar100DataLoader class
 *
 */
class Cifar100DataLoader final : public DataLoader {
public:
  /**
   * @brief Construct a new Cifar100 Data Loader object
   *
   * @param path path to read from
   * @param batch_size batch_size of current model
   * @param splits split divisor of the file 1 means using whole data, 2 means
   * half of the data, 10 means 10% of the data
   */
  Cifar100DataLoader(const std::string &path, int batch_size, int splits);

  /**
   * @brief Destroy the Cifar100 Data Loader object
   *
   */
  ~Cifar100DataLoader() {}

  /**
   * @copydoc void DataLoader::next(float **input, float**label, bool *last)
   */
  void next(float **input, float **label, bool *last);

private:
  inline static constexpr int ImageSize = 3 * 32 * 32;
  inline static constexpr int NumClass = 100;
  inline static constexpr int SampleSize =
    4 * (3 * 32 * 32 + 100); /**< 1 coarse label, 1 fine label, pixel size */

  unsigned int batch;
  unsigned int current_iteration;
  unsigned int iteration_per_epoch;

  std::mt19937 rng;
  std::ifstream file;
  std::vector<unsigned int> idxes; /**< index information for one epoch */
};

} // namespace nntrainer::vgg
