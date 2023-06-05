// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   det_dataloader.h
 * @date   22 March 2023
 * @brief  dataloader for object detection dataset
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <random>
#include <string>
#include <tensor_dim.h>
#include <vector>

namespace nntrainer::util {

using TensorDim = ml::train::TensorDim;

/**
 * @brief user data object
 */
class DirDataLoader {
public:
  /**
   * @brief Construct a new Dir Data Loader object
   */
  DirDataLoader(const char *directory_, unsigned int max_num_label,
                unsigned int c, unsigned int w, unsigned int h, bool is_train_);
  /**
   * @brief Destroy the Dir Data Loader object
   */
  ~DirDataLoader(){};

  /**
   * @copydoc void DataLoader::next(float **input, float**label, bool *last)
   */
  void next(float **input, float **label, bool *last);

  /**
   * @brief getter for current file name
   * @return current file name
   */
  std::string getCurFileName() { return cur_file_name; };

  /**
   * @brief setter for current file name
   */
  void setCurFileName(std::string s) { cur_file_name = s; };

private:
  std::string dir_path;
  unsigned int data_size;
  unsigned int max_num_label;
  unsigned int channel;
  unsigned int height;
  unsigned int width;
  bool is_train;

  std::vector<std::pair<std::string, std::string>> data_list;
  std::vector<unsigned int> idxes;
  unsigned int count;
  std::string cur_file_name;

  // random number generator
  std::mt19937 rng;
};

} // namespace nntrainer::util
