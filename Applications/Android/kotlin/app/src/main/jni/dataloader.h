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
#include <vector>
#include <tuple>

namespace nntrainer::indoor
{

using TensorDim = ml::train::TensorDim;
typedef std::pair<int, int> ImageIdx;

/**
 * @brief user data object
 *
 */
class DataLoader
{
  public:
  DataLoader (float *data, int data_size, int data_len, int label_len);
  ~DataLoader () = default;

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
  void next (float **input, float **label, bool *last);

  private:
  float *data;
  unsigned int data_size;
  unsigned int label_len;
  unsigned int data_len;
  unsigned int SampleSize;
  unsigned int count;
  std::mt19937 rng;
  std::vector<unsigned int> idxes;
};

/**
 * @brief user data object
 *
 */
class ImageDataLoader
{
  public:
  ImageDataLoader (const char *directory, int data_size, int label_len, int w,
      int h, bool is_train);
  ~ImageDataLoader () = default;

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
  void next (float **input, float **label, bool *last);

  private:
  unsigned int data_size;
  unsigned int label_len;
  unsigned int width;
  unsigned int height;
  bool is_train;
  unsigned int count;
  std::vector<unsigned int> num_images;
  std::mt19937 rng;
  std::vector<unsigned int> idxes;
  std::string dir_path;
  std::vector<ImageIdx> datas;
  unsigned int total_image_num;

};
 
} // namespace nntrainer::indoor
