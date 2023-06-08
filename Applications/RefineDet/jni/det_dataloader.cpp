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

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <nntrainer_error.h>
#include <random>

#include "bitmap_helpers.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/string_util.h"
#include "tensorflow/lite/tools/gen_op_registration.h"

namespace nntrainer::util {

// It supports bmp image file only now.
DirDataLoader::DirDataLoader(const char *directory_, unsigned int max_num_label,
                             unsigned int c, unsigned int w, unsigned int h,
                             bool is_train_) :
  max_num_label(max_num_label),
  channel(c),
  height(h),
  width(w),
  is_train(is_train_) {
  dir_path.assign(directory_);

  // set data list
  std::filesystem::directory_iterator itr(dir_path + "bmp_images");
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
    label_file.replace(label_file.find("/bmp_images"), 11, "/txt_annotations");

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

void read_image(const std::string path, std::vector<float> &image_data, uint &width,
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
      image_data[height * (height - i - 1) + j] = (float)data[j * 3 + 2] / 255;
      image_data[(height * width) + height * (height - i - 1) + j] =
        (float)data[j * 3 + 1] / 255;
      image_data[(height * width) * 2 + height * (height - i - 1) + j] =
        (float)data[j * 3] / 255;
    }
  }

  delete[] data;
  fclose(f);
}

void pass_backbone(float* in, float* out, std::array<int, 4> &in_dim, std::array<int, 4> &out_dim, const std::string model_path) {
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile(model_path.c_str());

  assert(model != NULL);
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  int input_dim[4];
  int output_dim[4];

  const std::vector<int> &input_idx_list = interpreter->inputs();
  const std::vector<int> &output_idx_list = interpreter->outputs();

  for (int i = 0; i < 4; i++) {
    input_dim[i] = 1;
    output_dim[i] = 1;
  }

  int len = interpreter->tensor(input_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(input_idx_list[0])->dims->data,
                    interpreter->tensor(input_idx_list[0])->dims->data + len,
                    input_dim);
  len = interpreter->tensor(output_idx_list[0])->dims->size;
  std::reverse_copy(interpreter->tensor(output_idx_list[0])->dims->data,
                    interpreter->tensor(output_idx_list[0])->dims->data + len,
                    output_dim);

  int input_number_of_pixels = 1;
  int output_number_of_pixels = 1;
  int wanted_height = input_dim[1];
  int wanted_width = input_dim[2];
  int wanted_channels = input_dim[3];

  for (int k = 0; k < 4; k++){
    input_number_of_pixels *= input_dim[k];
    output_number_of_pixels *= output_dim[k];
  }
    

  int _input = interpreter->inputs()[0];

  float *output;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors!" << std::endl;
    exit(0);
  }

  for (int l = 0; l < input_number_of_pixels; l++) {
    (interpreter->typed_tensor<float>(_input))[l] = (in[l]);
  }

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke!" << std::endl;
    exit(0);
  }

  output = interpreter->typed_output_tensor<float>(0);

  for (int l = 0; l < output_number_of_pixels; l++) {
    out[l] = output[l];
  }
}


void DirDataLoader::next(float **input, float **label, bool *last) {
  auto fill_one_sample = [this](float *input_, float *label_, int index) {
    // set input data
    std::string img_file = data_list[index].first;

    std::vector<float> image_data(width * height * 3);
    read_image(img_file, image_data, width, height);

    std::string data_path = "/home/bumkyu/nntrainer/Applications/RefineDet/";
    std::array<int, 4> input_dim = {1, 224, 224, 3};
    std::array<int, 4> output_dim = {1, 28, 28, 512};
    std::array<int, 4> input_dim2 = {1, 28, 28, 512};
    std::array<int, 4> output_dim2 = {1, 14, 14, 512};
    pass_backbone(image_data.data(), input_,  input_dim, output_dim, data_path + "vgg16_conv4_3.tflite");
    pass_backbone(input_, input_ + 28*28*512, input_dim2, output_dim2, data_path + "vgg16_conv5_3.tflite");

    // set label data
    std::string label_file = data_list[index].second;
    std::memset(label_, 0.0, 25 * sizeof(float) * max_num_label);

    std::ifstream file(label_file);
    std::string cur_line;

    int line_idx = 0;
    while (getline(file, cur_line)) {
      std::stringstream ss(cur_line);
      std::string cur_value;

      int row_idx = 0;
      while (getline(ss, cur_value, ' ')) {
        if (row_idx == 0) {
          label_[line_idx * 25] = 1;
        }
        label_[line_idx * 25 + 1 + row_idx] = std::stof(cur_value);
        row_idx++;
      }

      line_idx++;
    }

    file.close();
  };

  fill_one_sample(*input, *label, idxes[count]);

  count++;

  if (count < data_size) {
    *last = false;
  } else {
    *last = true;
    count = 0;
    std::shuffle(idxes.begin(), idxes.end(), rng);
  }
}

} // namespace nntrainer::util
