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

#include <errno.h>
#include <fstream>
#include <string.h>
#include <unordered_map>
#include <utility>

#include <compiler_fwd.h>
#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <realizer.h>
#include <tensor.h>

/** tolerance is reduced for packaging, but CI runs at full tolerance */
#ifdef REDUCE_TOLERANCE
#define tolerance 1.0e-4
#else
#define tolerance 1.0e-5
#endif

/** Enum values to get model accuracy and loss. Sync with internal CAPI header
 */
#define ML_TRAIN_SUMMARY_MODEL_TRAIN_LOSS 101
#define ML_TRAIN_SUMMARY_MODEL_VALID_LOSS 102
#define ML_TRAIN_SUMMARY_MODEL_VALID_ACCURACY 103

/** Gtest compatibility for parameterize google test API  */
#ifdef GTEST_BACKPORT
#define GTEST_PARAMETER_TEST INSTANTIATE_TEST_CASE_P
#else
#define GTEST_PARAMETER_TEST INSTANTIATE_TEST_SUITE_P
#endif

/**
 * @brief This class wraps IniWrapper. This generates real ini file when
 * construct, and remove real ini file when destroy
 *
 */
class ScopedIni {
public:
  /**
   * @brief Construct a new Scoped Ini object
   *
   * @param ini_ ini wrapper
   */
  ScopedIni(const nntrainer::IniWrapper &ini_) : ini(ini_) { ini.save_ini(); }

  /**
   * @brief Construct a new Scoped Ini object
   *
   * @param name_ name
   * @param sections_ sequenes of sections to save
   */
  ScopedIni(const std::string &name_,
            const nntrainer::IniWrapper::Sections &sections_) :
    ini(name_, sections_) {
    ini.save_ini();
  }

  /**
   * @brief Get the Ini Name object
   *
   * @return std::string ini name
   */
  std::string getIniName() { return ini.getIniName(); }

  /**
   * @brief Destroy the Scoped Ini object
   *
   */
  ~ScopedIni() { ini.erase_ini(); }

private:
  nntrainer::IniWrapper ini;
};

#define GEN_TEST_INPUT_NHWC(input, eqation_i_j_k_l) \
  do {                                              \
    for (int i = 0; i < batch; ++i) {               \
      for (int j = 0; j < height; ++j) {            \
        for (int k = 0; k < width; ++k) {           \
          for (int l = 0; l < channel; ++l) {       \
            float val = eqation_i_j_k_l;            \
            input.setValue(i, l, j, k, val);        \
          }                                         \
        }                                           \
      }                                             \
    }                                               \
  } while (0)

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

/**
 * @brief return a tensor filled with contant value with dimension
 */
nntrainer::Tensor
constant(float value, unsigned int d0, unsigned d1, unsigned d2, unsigned d3,
         nntrainer::Tformat fm = nntrainer::Tformat::NCHW,
         nntrainer::Tdatatype d_type = nntrainer::Tdatatype::FP32);

/**
 * @brief return a tensor filled with ranged value with given dimension
 */
nntrainer::Tensor
ranged(unsigned int batch, unsigned channel, unsigned height, unsigned width,
       nntrainer::Tformat fm = nntrainer::Tformat::NCHW,
       nntrainer::Tdatatype d_type = nntrainer::Tdatatype::FP32);

/**
 * @brief return a tensor filled with random value with given dimension
 */
nntrainer::Tensor
randUniform(unsigned int batch, unsigned channel, unsigned height,
            unsigned width, float min = -1, float max = 1,
            nntrainer::Tformat fm = nntrainer::Tformat::NCHW,
            nntrainer::Tdatatype d_type = nntrainer::Tdatatype::FP32);

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

/**
 * @brief Create a user data for training
 *
 * @return DataInformation
 */
DataInformation createTrainData();

/**
 * @brief Create a user data for validataion
 *
 * @return DataInformation
 */
DataInformation createValidData();

/**
 * @brief      get data which size is batch
 * @param[out] outVec
 * @param[out] outLabel
 * @param[out] last if the data is finished
 * @param[in] user_data private data for the callback
 * @retval status for handling error
 */
int getSample(float **outVec, float **outLabel, bool *last, void *user_data);

/**
 * @brief Get the Res Path object
 * @note if NNTRAINER_RESOURCE_PATH environment variable is given, @a
 * fallback_base is ignored and NNTRINAER_RESOURCE_PATH is directly used as a
 * base
 *
 * @param filename filename if omitted, ${prefix}/${base} will be returned
 * @param fallback_base list of base to attach when NNTRAINER_RESOURCE_PATH is
 * not given
 * @return const std::string path,
 */
const std::string
getResPath(const std::string &filename,
           const std::initializer_list<const char *> fallback_base = {});

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;

/**
 * @brief make graph of a representation
 *
 * @param layer_reps layer representation (pair of type, properties)
 * @return nntrainer::GraphRepresentation synthesized graph representation
 */
nntrainer::GraphRepresentation
makeGraph(const std::vector<LayerRepresentation> &layer_reps);

/**
 * @brief make graph of a representation after compile
 *
 * @param layer_reps layer representation (pair of type, properties)
 * @param realizers GraphRealizers to modify graph before compile
 * @param loss_layer loss layer to compile with
 * @return nntrainer::GraphRepresentation synthesized graph representation
 */
nntrainer::GraphRepresentation makeCompiledGraph(
  const std::vector<LayerRepresentation> &layer_reps,
  std::vector<std::unique_ptr<nntrainer::GraphRealizer>> &realizers,
  const std::string &loss_layer = "");

/**
 * @brief read tensor after reading tensor size
 *
 * @param t tensor to fill
 * @param file file name
 * @param error_msg error msg
 */
void sizeCheckedReadTensor(nntrainer::Tensor &t, std::ifstream &file,
                           const std::string &error_msg = "");

template <typename Ta = float, typename Tb = float>
double cosine_similarity(Ta *A /*Predict*/, Tb *B /*Reference */,
                         uint32_t size) {
  double dot = 0.0, denom_a = 0.0, denom_b = 0.0;
  for (uint32_t i = 0u; i < size; ++i) {
    double pred = A[i];
    double ref = B[i];
    dot += pred * ref;
    denom_a += pred * pred;
    denom_b += ref * ref;
  }

  if (sqrt(denom_a) == 0 && sqrt(denom_b) == 0)
    return 1;

  double cosine_sim = dot / (sqrt(denom_a) * sqrt(denom_b));
  return cosine_sim;
}

template <typename Ta = float, typename Tb = float>
float mse(Ta *A /* Predicted */, Tb *B /* Reference */, uint32_t size) {
  float pred;
  float ref;
  float mse_error = 0;
  for (uint32_t i = 0; i < size; i++) {
    pred = A[i];
    ref = B[i];
    float diff = pred - ref;
    mse_error += pow(diff, 2);
  }
  float mse = mse_error / size;
  return mse;
}

#endif /* __cplusplus */
#endif /* __NNTRAINER_TEST_UTIL_H__ */
