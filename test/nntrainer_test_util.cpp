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
 * @file	nntrainer_test_util.cpp
 * @date	28 April 2020
 * @brief	This is util functions for test
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include "nntrainer_test_util.h"
#include <climits>
#include <iostream>

#define num_class 10
#define mini_batch 16
#define feature_size 62720

static bool *duplicate;
static bool *valduplicate;
static bool alloc_train = false;
static bool alloc_val = false;

/**
 * @brief replace string and save it in file
 */
void replaceString(const std::string &from, const std::string &to,
                   const std::string n, std::string str) {
  size_t start_pos = 0;
  std::string s;
  std::ifstream file_stream(n.c_str(), std::ifstream::in);
  if (file_stream.good()) {
    s.assign((std::istreambuf_iterator<char>(file_stream)),
             std::istreambuf_iterator<char>());
    file_stream.close();
  } else {
    s = str;
  }
  while ((start_pos = s.find(from, start_pos)) != std::string::npos) {
    s.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }

  std::ofstream data_file(n.c_str());
  data_file << s;
  data_file.close();
}

/**
 * @brief     Generate Random integer value between min to max
 * @param[in] min : minimum value
 * @param[in] max : maximum value
 * @retval    min < random value < max
 */
static int rangeRandom(int min, int max) {
  int n = max - min + 1;
  int remainder = RAND_MAX % n;
  int x;
  do {
    x = rand();
  } while (x >= RAND_MAX - remainder);
  return min + x % n;
}

/**
 * @brief     load data at specific position of file
 * @param[in] F  ifstream (input file)
 * @param[out] outVec
 * @param[out] outLabel
 * @param[in] id th data to get
 * @retval true/false false : end of data
 */
static bool getData(std::ifstream &F, std::vector<float> &outVec,
                    std::vector<float> &outLabel, int id) {
  F.clear();
  F.seekg(0, std::ios_base::end);
  uint64_t file_length = F.tellg();
  if (id < 0)
    return false;
  uint64_t position = (feature_size + num_class) * id * sizeof(float);

  if (position > file_length || position > ULLONG_MAX) {
    return false;
  }
  F.seekg(position, std::ios::beg);
  for (unsigned int i = 0; i < feature_size; i++)
    F.read((char *)&outVec[i], sizeof(float));
  for (unsigned int i = 0; i < num_class; i++)
    F.read((char *)&outLabel[i], sizeof(float));

  return true;
}

/**
 * @brief      get data which size is mini batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] status for error handling
 * @retval true/false
 */
bool getMiniBatch_train(float *outVec, float *outLabel, int *status) {
  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  unsigned int data_size = 0;

  std::string filename = "trainingSet.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  if (F.good()) {
    F.seekg(0, std::ios::end);
    long file_size = F.tellg();
    data_size = static_cast<unsigned int>(
      file_size / ((num_class + feature_size) * sizeof(float)));
  }

  if (!alloc_train) {
    duplicate = (bool *)malloc(sizeof(bool) * data_size);
    for (unsigned int i = 0; i < data_size; ++i) {
      duplicate[i] = false;
    }
    alloc_train = true;
  }

  for (unsigned int i = 0; i < data_size; i++) {
    if (!duplicate[i])
      count++;
  }

  if (count < mini_batch) {
    free(duplicate);
    alloc_train = false;
    return false;
  }

  count = 0;
  while (count < mini_batch) {
    int nomI = rangeRandom(0, data_size - 1);
    if (!duplicate[nomI]) {
      memI.push_back(nomI);
      duplicate[nomI] = true;
      count++;
    }
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(num_class);

    getData(F, o, l, memI[i]);

    for (unsigned int j = 0; j < feature_size; ++j)
      outVec[i * feature_size + j] = o[j];
    for (unsigned int j = 0; j < num_class; ++j)
      outLabel[i * num_class + j] = l[j];
  }

  F.close();
  return true;
}

/**
 * @brief      get data which size is mini batch for validation
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] status for error handling
 * @retval true/false false : end of data
 */
bool getMiniBatch_val(float *outVec, float *outLabel, int *status) {

  std::vector<int> memI;
  std::vector<int> memJ;
  unsigned int count = 0;
  unsigned int data_size = 0;

  std::string filename = "trainingSet.dat";
  std::ifstream F(filename, std::ios::in | std::ios::binary);

  if (F.good()) {
    F.seekg(0, std::ios::end);
    long file_size = F.tellg();
    data_size = static_cast<unsigned int>(
      file_size / ((num_class + feature_size) * sizeof(float)));
  }

  if (!alloc_val) {
    valduplicate = (bool *)malloc(sizeof(bool) * data_size);
    for (unsigned int i = 0; i < data_size; ++i) {
      valduplicate[i] = false;
    }
    alloc_val = true;
  }

  for (unsigned int i = 0; i < data_size; i++) {
    if (!valduplicate[i])
      count++;
  }

  if (count < mini_batch) {
    free(valduplicate);
    alloc_val = false;
    return false;
  }

  count = 0;
  while (count < mini_batch) {
    int nomI = rangeRandom(0, data_size - 1);
    if (!valduplicate[nomI]) {
      memI.push_back(nomI);
      valduplicate[nomI] = true;
      count++;
    }
  }

  for (unsigned int i = 0; i < count; i++) {
    std::vector<float> o;
    std::vector<float> l;

    o.resize(feature_size);
    l.resize(num_class);

    getData(F, o, l, memI[i]);

    for (unsigned int j = 0; j < feature_size; ++j)
      outVec[i * feature_size + j] = o[j];
    for (unsigned int j = 0; j < num_class; ++j)
      outLabel[i * num_class + j] = l[j];
  }

  F.close();
  return true;
}
