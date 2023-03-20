// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file    dataloader.h
 * @date    08 Sept 2021
 * @see     https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is simple dataloader for nntrainer
 *
 */

#include <tensor_dim.h>

#include <fstream>
#include <random>
#include <string>
#include <tuple>
#include <vector>

namespace nntrainer::resnet {

using TensorDim = ml::train::TensorDim;
typedef std::pair<int, int> ImageIdx;

/**
 * @brief read image
 *
 */
void read_image(const std::string path, float *input, uint &width,
                uint &height);

/**
 * @brief user data object
 *
 */
class DataLoader {
public:
  virtual ~DataLoader() {}

  /**
   * @brief generate a single data with label
   * @param[out] input : data
   * ex: -99 0.0 0.0 0.0 28.7776151452882 -67 -48 -86 -62 -74 -86
   *     0.0 0.0 0.0 0.0 0.0 0.0 ( non-normalized )
   * @param[out] label : label ( one hot )
   * ex: 0 0 0 1 0 0 0 0
   * @param[out] last : end of data
   *
   */
  virtual void next(float **input, float **label, bool *last) = 0;

protected:
  std::mt19937 rng;
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
   */
  ~RandomDataLoader() {}

  /**
   * @copydoc void DataLoader::next(float **input, float**label, bool *last)
   */
  void next(float **input, float **label, bool *last);

private:
  unsigned int iteration;
  unsigned int data_size;

  std::vector<TensorDim> input_shapes;
  std::vector<TensorDim> output_shapes;

  std::uniform_int_distribution<int> input_dist;
  std::uniform_int_distribution<int> label_dist;
};

/**
 * @brief user data object
 *
 */
class DirDataLoader final : public DataLoader {
public:
  /**
   * @brief Construct a new Directory Data Loader object
   *
   * @param directory directory path
   * @param split_ratio split ratio between training & validation
   * @param label_len size of label data
   * @param c size of channel
   * @param w size of width
   * @param h size of height
   * @param is_train true if training
   */
  DirDataLoader(const char *directory, float split_ratio, int label_len, int c,
                int w, int h, bool is_train);
  /**
   * @brief Destructor
   *
   */
  ~DirDataLoader() = default;

  /**
   * @copydoc void DataLoader::next(float **input, float**label, bool *last)
   */
  void next(float **input, float **label, bool *last);

  /**
   * @brief getter for current file name
   * @return current file name
   */
  std::string getCurFileName();

  /**
   * @brief setter for current file name
   */
  void setCurFileName(std::string s) { cur_file_name = s; };

private:
  unsigned int data_size;
  unsigned int label_len;
  unsigned int channel;
  unsigned int width;
  unsigned int height;
  bool is_train;
  unsigned int count;
  std::vector<unsigned int> num_images;
  std::mt19937 rng;
  std::vector<unsigned int> idxes;
  std::vector<std::pair<unsigned int, std::string>> data_list;
  std::string cur_file_name;
  std::string dir_path;
  unsigned int total_image_num;
};

} // namespace nntrainer::resnet
