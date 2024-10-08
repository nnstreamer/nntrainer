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

#include "det_dataloader.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nntrainer_error.h>
#include <random>

namespace nntrainer::util {

namespace {
/**
 * @brief fill last to the given memory
 * @note this function increases iteration value, if last is set to true,
 * iteration resets to 0
 *
 * @param[in/out] iteration current iteration
 * @param data_size Data size
 * @return bool true if iteration has finished
 */
bool updateIteration(unsigned int &iteration, unsigned int data_size) {
  if (++iteration == data_size) {
    iteration = 0;
    return true;
  }
  return false;
};

} // namespace

// It supports bmp image file only now.
DirDataLoader::DirDataLoader(const char *directory_, unsigned int max_num_label,
                             unsigned int c, unsigned int w, unsigned int h,
                             bool is_train_,
                             const std::vector<TensorDim> &output_shapes) :
  iteration(0),
  max_num_label(max_num_label),
  channel(c),
  height(h),
  width(w),
  is_train(is_train_),
  output_shapes(output_shapes) {
  dir_path.assign(directory_);

  // set data list
  std::filesystem::directory_iterator itr(dir_path + "images");
  while (itr != std::filesystem::end(itr)) {
    // get image file name
    std::string img_file = itr->path().string();

    // check if it is bmp image file
    if (img_file.find(".bmp") == std::string::npos) {
      itr++;
      continue;
    }

    // set label file name
    std::string label_file = img_file;
    label_file.replace(label_file.find(".bmp"), 4, ".txt");
    label_file.replace(label_file.find("/images"), 7, "/annotations");

    // check if there is paired label file
    if (!std::filesystem::exists(label_file)) {
      itr++;
      continue;
    }

    // set data list
    data_list.push_back(make_pair(img_file, label_file));
    itr++;
  }

  // set index and shuffle data
  idxes = std::vector<unsigned int>(data_list.size());
  std::iota(idxes.begin(), idxes.end(), 0);
  if (is_train)
    std::shuffle(idxes.begin(), idxes.end(), rng);

  data_size = data_list.size();
  count = 0;
}

void read_image(const std::string path, float *input, uint &width,
                uint &height) {
  FILE *f = fopen(path.c_str(), "rb");

  if (f == nullptr)
    throw std::invalid_argument("Cannot open file: " + path);

  unsigned char info[54];
  size_t s = fread(info, sizeof(unsigned char), 54, f);

  unsigned int w = *(int *)&info[18];
  unsigned int h = *(int *)&info[22];

  if (w != width or h != height) {
    fclose(f);
    throw std::invalid_argument("the dimension of image file does not match" +
                                std::to_string(s));
  }

  int row_padded = (width * 3 + 3) & (~3);
  unsigned char *data = new unsigned char[row_padded];

  for (uint i = 0; i < height; i++) {
    s = fread(data, sizeof(unsigned char), row_padded, f);
    for (uint j = 0; j < width; j++) {
      input[height * (height - i - 1) + j] = (float)data[j * 3 + 2] / 255;
      input[(height * width) + height * (height - i - 1) + j] =
        (float)data[j * 3 + 1] / 255;
      input[(height * width) * 2 + height * (height - i - 1) + j] =
        (float)data[j * 3] / 255;
    }
  }

  delete[] data;
  fclose(f);
}

void DirDataLoader::next(float **input, float **label, bool *last) {
  auto fill_input = [this](float *input_, int index) {
    std::string img_file = data_list[index].first;
    read_image(img_file, input_, width, height);
  };

  auto fill_label = [this](float *label_, int index, int length) {
    // set label data
    std::string label_file = data_list[index].second;
    std::memset(label_, 0.0, sizeof(length));

    std::ifstream file(label_file);
    std::string cur_line;

    int line_idx = 0;
    while (getline(file, cur_line)) {
      std::stringstream ss(cur_line);
      std::string cur_value;

      int row_idx = 0;
      while (getline(ss, cur_value, ' ')) {
        if (row_idx == 0) {
          label_[line_idx * 5 + 4] = std::stof(cur_value);
        } else {
          label_[line_idx * 5 + row_idx - 1] = std::stof(cur_value) / 416;
        }
        row_idx++;
      }

      line_idx++;
      label_ += 5;
    }

    file.close();
  };

  fill_input(*input, idxes[count]);

  float **cur_label_tensor = label;
  for (unsigned int i = 0; i < output_shapes.size(); ++i) {
    fill_label(*cur_label_tensor, idxes[count],
               output_shapes.at(i).getFeatureLen());
    cur_label_tensor++;
  }
  if (updateIteration(iteration, data_size)) {
    std::shuffle(idxes.begin(), idxes.end(), rng);
    *last = true;
    count = 0;
  } else {
    *last = false;
    count++;
  }
}

} // namespace nntrainer::util
