/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer Tensor_Filter, nntrainer Module
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file   tensor_filter_nntrainer.hh
 * @date   09 Sept 2020
 * @brief  nntrainer inference module for tensor_filter gstreamer plugin header
 * @note   The clas has been exposed from tensor_filter_nntrainer.cc to unittest
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
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

#include <model.h>
#include <tensor_dim.h>

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
   */
  NNTrainerInference(const std::string &model_config);

  /**
   * @brief Destroy the NNTrainerInference object
   *
   */
  ~NNTrainerInference() = default;

  /**
   * @brief Get the Model Config object
   *
   * @return const char* config, do not free
   */
  const char *getModelConfig();

  /**
   * @brief Set the Batch Size
   *
   * @param batch batch size
   */
  void setBatchSize(unsigned int batch) {
    std::stringstream ss;
    ss << "batch_size=" << batch;
    model->setProperty({ss.str()});
  }

  /**
   * @brief Get the Input Dimension object
   *
   * @return const std::vector<nntrainer::TensorDim> input dimensions
   */
  const std::vector<ml::train::TensorDim> getInputDimension() {
    return model->getInputDimension();
  }

  /**
   * @brief Get the Output Dimension object
   *
   * @return const std::vector<nntrainer::TensorDim> output dimensions
   */
  const std::vector<ml::train::TensorDim> getOutputDimension() {
    return model->getOutputDimension();
  }

  /**
   * @brief run inference, output
   *
   * @param input input tensor memory
   * @param output output tensor memory
   * @return int 0 if success
   */
  int run(const GstTensorMemory *input, GstTensorMemory *output);

private:
  void loadModel();

  std::string model_config;
  std::unique_ptr<ml::train::Model> model;
};
