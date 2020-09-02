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
 * @file	nntrainer_test_util.h
 * @date	28 April 2020
 * @brief	This is util functions for test
 * @see		https://github.com/nnstreamer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#ifndef __NNTRAINER_TEST_UTIL_H__
#define __NNTRAINER_TEST_UTIL_H__
#ifdef __cplusplus

#include "nntrainer_log.h"
#include <fstream>
#include <gtest/gtest.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <parse_util.h>
#include <tensor.h>
#include <unordered_map>

#define tolerance 10e-5

/** Enum values to get model accuracy and loss. Sync with internal CAPI header
 */
#define ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS 101
#define ML_TRAIN_SUMMARY_MODEL_VALID_LOSS 102
#define ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY 103

class IniSection {
public:
  IniSection(const std::string &section_name, const std::string &entry_str) :
    section_name(section_name) {
    setEntry(entry_str);
  }

  /**
   * @brief copy entry from @a from and overwrite entry and section_name
   */
  IniSection(IniSection &from, const std::string &section_name,
             const std::string &entry_str) :
    IniSection(from) {
    if (!section_name.empty()) {
      this->section_name = section_name;
    }
    if (!entry_str.empty()) {
      setEntry(entry_str);
    }
  }

  IniSection(IniSection &from, const std::string &entry_str) :
    IniSection(from, "", entry_str) {}

  IniSection() = default;
  ~IniSection() = default;

  void print(std::ostream &out) {
    out << '[' << section_name << ']' << std::endl;
    for (auto &it : entry)
      out << it.first << " = " << it.second << std::endl;
  }

  IniSection &operator+=(const IniSection &rhs) {
    setEntry(rhs.entry);
    return *this;
  }

  IniSection operator+(const IniSection &rhs) const {
    return IniSection(*this) += rhs;
  }

  IniSection &operator+=(const std::string &s) {
    setEntry(s);
    return *this;
  }

  IniSection operator+(const std::string &s) { return IniSection(*this) += s; }

private:
  void setEntry(const std::unordered_map<std::string, std::string> &_entry) {
    for (auto &it : _entry) {
      this->entry[it.first] = it.second;
    }
  }

  /**
   * @brief setEntry as "Type = neuralnetwork | decayrate = 0.96 | -epochs = 1"
   * will delete epochs, and overwrite type and decayrate
   */
  void setEntry(const std::string &entry_str);

  std::string section_name;
  std::unordered_map<std::string, std::string> entry;

  friend std::ostream &operator<<(std::ostream &os, const IniSection &section) {
    return os << section.section_name;
  }
};

namespace initest {
typedef enum {
  LOAD = 1 << 0, /**< should fail at load */
  INIT = 1 << 1, /**< should fail at init */
} IniFailAt;
};

class nntrainerIniTest
  : public ::testing::TestWithParam<
      std::tuple<const char *, const std::vector<IniSection>, int>> {
protected:
  virtual void SetUp() {
    name = std::string(std::get<0>(GetParam()));
    std::cout << "starting test case : " << name << std::endl << std::endl;

    sections = std::get<1>(GetParam());
    failAt = std::get<2>(GetParam());
    save_ini();
  }

  virtual void TearDown() { erase_ini(); }

  bool failAtLoad() { return failAt & initest::IniFailAt::LOAD; }

  bool failAtInit() { return failAt & initest::IniFailAt::INIT; }

  std::string getIniName() { return name + ".ini"; }

  std::ofstream getIni() {
    std::ofstream out(getIniName().c_str());
    if (!out.good()) {
      throw std::runtime_error("cannot open ini");
    }
    return out;
  }

  virtual void save_ini() {
    std::ofstream out = getIni();
    for (auto &it : sections) {
      it.print(std::cout);
      std::cout << std::endl;
      it.print(out);
      out << std::endl;
    }

    out.close();
  }

  nntrainer::NeuralNetwork NN;

private:
  void erase_ini() { name.clear(); }
  int failAt;
  std::string name;
  std::vector<IniSection> sections;
};

/**
 * @brief make ini test case from given parameter
 */
std::tuple<const char *, const std::vector<IniSection>, int>
mkIniTc(const char *name, const std::vector<IniSection> vec, int flag);

/// @todo: migrate this to datafile unittest
const std::string config_str = "[Model]"
                               "\n"
                               "Type = NeuralNetwork"
                               "\n"
                               "Learning_rate = 0.0001"
                               "\n"
                               "Decay_rate = 0.96"
                               "\n"
                               "Decay_steps = 1000"
                               "\n"
                               "Epochs = 1"
                               "\n"
                               "Optimizer = adam"
                               "\n"
                               "Loss = cross"
                               "\n"
                               "Weight_Regularizer = l2norm"
                               "\n"
                               "weight_regularizer_constant = 0.005"
                               "\n"
                               "Save_Path = 'model.bin'"
                               "\n"
                               "batch_size = 32"
                               "\n"
                               "beta1 = 0.9"
                               "\n"
                               "beta2 = 0.9999"
                               "\n"
                               "epsilon = 1e-7"
                               "\n"
                               "[DataSet]"
                               "\n"
                               "BufferSize=100"
                               "\n"
                               "TrainData = trainingSet.dat"
                               "\n"
                               "ValidData = valSet.dat"
                               "\n"
                               "LabelData = label.dat"
                               "\n"
                               "[inputlayer]"
                               "\n"
                               "Type = input"
                               "\n"
                               "Input_Shape = 1:1:62720"
                               "\n"
                               "bias_initializer = zeros"
                               "\n"
                               "Normalization = true"
                               "\n"
                               "Activation = sigmoid"
                               "\n"
                               "[outputlayer]"
                               "\n"
                               "Type = fully_connected"
                               "\n"
                               "Unit = 10"
                               "\n"
                               "bias_initializer = zeros"
                               "\n"
                               "Activation = softmax"
                               "\n";

const std::string config_str2 = "[Model]"
                                "\n"
                                "Type = NeuralNetwork"
                                "\n"
                                "Learning_rate = 0.0001"
                                "\n"
                                "Decay_rate = 0.96"
                                "\n"
                                "Decay_steps = 1000"
                                "\n"
                                "Epochs = 1"
                                "\n"
                                "Optimizer = adam"
                                "\n"
                                "Loss = cross"
                                "\n"
                                "Weight_Regularizer = l2norm"
                                "\n"
                                "weight_regularizer_constant = 0.005"
                                "\n"
                                "Model = 'model.bin'"
                                "\n"
                                "batch_size = 32"
                                "\n"
                                "beta1 = 0.9"
                                "\n"
                                "beta2 = 0.9999"
                                "\n"
                                "epsilon = 1e-7"
                                "\n"
                                "[DataSet]"
                                "\n"
                                "BufferSize=100"
                                "\n"
                                "TrainData = trainingSet.dat"
                                "\n"
                                "ValidData = valSet.dat"
                                "\n"
                                "LabelData = label.dat"
                                "\n"
                                "[conv2dlayer]"
                                "\n"
                                "Type = conv2d"
                                "\n"
                                "Input_Shape = 3:28:28"
                                "\n"
                                "bias_initializer = zeros"
                                "\n"
                                "Activation = sigmoid"
                                "\n"
                                "weight_regularizer=l2norm"
                                "\n"
                                "weight_regularizer_constant=0.005"
                                "\n"
                                "filters=6"
                                "\n"
                                "kernel_size=5,5"
                                "\n"
                                "stride=1,1"
                                "\n"
                                "padding=0,0"
                                "\n"
                                "weight_initializer=xavier_uniform"
                                "\n"
                                "flatten = false"
                                "\n"
                                "[outputlayer]"
                                "\n"
                                "Type = fully_connected"
                                "\n"
                                "Unit = 10"
                                "\n"
                                "bias_initializer = zeros"
                                "\n"
                                "Activation = softmax"
                                "\n";

#define GEN_TEST_INPUT(input, eqation_i_j_k_l) \
  do {                                         \
    for (int i = 0; i < batch; ++i) {          \
      for (int j = 0; j < channel; ++j) {      \
        for (int k = 0; k < height; ++k) {     \
          for (int l = 0; l < width; ++l) {    \
            float val = eqation_i_j_k_l;       \
            input.setValue(i, j, k, l, val);   \
          }                                    \
        }                                      \
      }                                        \
    }                                          \
  } while (0)

#define ASSERT_EXCEPTION(TRY_BLOCK, EXCEPTION_TYPE, MESSAGE)                  \
  try {                                                                       \
    TRY_BLOCK                                                                 \
    FAIL() << "exception '" << MESSAGE << "' not thrown at all!";             \
  } catch (const EXCEPTION_TYPE &e) {                                         \
    EXPECT_EQ(std::string(MESSAGE), e.what())                                 \
      << " exception message is incorrect. Expected the following "           \
         "message:\n\n"                                                       \
      << MESSAGE << "\n";                                                     \
  } catch (...) {                                                             \
    FAIL() << "exception '" << MESSAGE << "' not thrown with expected type '" \
           << #EXCEPTION_TYPE << "'!";                                        \
  }

#define RESET_CONFIG(conf_name)                              \
  do {                                                       \
    std::ifstream file_stream(conf_name, std::ifstream::in); \
    if (file_stream.good()) {                                \
      if (std::remove(conf_name) != 0)                       \
        ml_loge("Error: Cannot delete file: %s", conf_name); \
      else                                                   \
        ml_logi("Info: deleteing file: %s", conf_name);      \
    }                                                        \
  } while (0)

/**
 * @brief return a tensor filled with contant value with dimension
 */
nntrainer::Tensor constant(float value, unsigned int batch, unsigned channel,
                           unsigned height, unsigned width);

/**
 * @brief replace string and save in file
 * @param[in] from string to be replaced
 * @param[in] to string to repalce with
 * @param[in] n file name to save
 * @retval void
 */
void replaceString(const std::string &from, const std::string &to,
                   const std::string n, std::string str);

/**
 * @brief      get data which size is batch for train
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_train(float **outVec, float **outLabel, bool *last,
                   void *user_data);

/**
 * @brief      get data which size is batch for val
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getBatch_val(float **outVec, float **outLabel, bool *last, void *user_data);

#endif /* __cplusplus */
#endif /* __NNTRAINER_TEST_UTIL_H__ */
