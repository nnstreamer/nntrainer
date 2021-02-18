/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer Tensor_Filter, nntrainer Module
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file	tensor_filter_nntrainer.hh
 * @date	09 Sept 2020
 * @brief	nntrainer inference module for tensor_filter gstreamer plugin header
 * @note  The clas has been exposed from tensor_filter_nntrainer.cc to unittest
 * @see		http://github.com/nnstreamer/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (e.g) tensorflow-lite) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */
#include <map>
#include <memory>
#include <vector>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

/**
 * @brief	Internal data structure for nntrainer
 */
typedef struct {
  int rank;
  std::vector<std::int64_t> dims;
} nntrainer_tensor_info_s;

/**
 * @brief NNTrainerInference wrapper for nnstreamer filter subplugin
 *
 */
class NNTrainerInference {
public:
  /**
   * @brief Construct a new NNTrainerInference object
   *
   * @param model_config_ config address
   * @param prop tensor filter property
   */
  NNTrainerInference(const char *model_config_,
                     const GstTensorFilterProperties *prop);

  /**
   * @brief Destroy the NNTrainerInference object
   *
   */
  ~NNTrainerInference();

  /**
   * @brief Get the Model Config object
   *
   * @return const char* config, do not free
   */
  const char *getModelConfig();

  /**
   * @brief Get the Input Tensor Dim object
   *
   * @param[out] info copied tensor info, free after use
   * @return int 0 if success
   */
  int getInputTensorDim(GstTensorsInfo *info);

  /**
   * @brief Get the Output Tensor Dim object
   *
   * @param info copied tensor info, free after use
   * @return int 0 if success
   */
  int getOutputTensorDim(GstTensorsInfo *info);

  /**
   * @brief run inference
   *
   * @param input input tensor memory
   * @param output output tensor memory
   * @return int 0 if success
   */
  int run(const GstTensorMemory *input, GstTensorMemory *output);

  /**
   * @brief free output tensor
   *
   * @param data reference to the output data to free
   */
  void freeOutputTensor(void *data);

private:
  void loadModel();
  void validateTensor(const GstTensorsInfo *tensorInfo, bool is_input);

  char *model_config;
  nntrainer::NeuralNetwork *model;

  GstTensorsInfo inputTensorMeta;
  GstTensorsInfo outputTensorMeta;

  std::vector<nntrainer_tensor_info_s> input_tensor_info;
  std::map<void *, std::shared_ptr<nntrainer::Tensor>> outputTensorMap;
};
