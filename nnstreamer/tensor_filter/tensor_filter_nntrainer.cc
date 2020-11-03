/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * GStreamer Tensor_Filter, nntrainer Module
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file	tensor_filter_nntrainer.cc
 * @date	09 Sept 2020
 * @brief	nntrainer inference module for tensor_filter gstreamer plugin
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (tensorflow-lite) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include <algorithm>
#include <limits>
#include <map>
#include <sstream>
#include <unistd.h>

#include <nnstreamer_plugin_api.h>
#include <nnstreamer_plugin_api_filter.h>

#include <neuralnet.h>

#define ml_loge g_critical

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

#define NUM_DIM 4

static const gchar *nntrainer_accl_support[] = {NULL};

/**
 * @brief	Internal data structure for nntrainer
 */
typedef struct {
  int rank;
  std::vector<std::int64_t> dims;
} nntrainer_tensor_info_s;

class NNTrainer {
public:
  /**
   * member functions.
   */
  NNTrainer(const char *model_config_, const GstTensorFilterProperties *prop);
  ~NNTrainer();

  const char *getModelConfig();

  int getInputTensorDim(GstTensorsInfo *info);
  int getOutputTensorDim(GstTensorsInfo *info);
  int run(const GstTensorMemory *input, GstTensorMemory *output);
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

void init_filter_nntrainer(void) __attribute__((constructor));
void fini_filter_nntrainer(void) __attribute__((destructor));

NNTrainer::NNTrainer(const char *model_config_,
                     const GstTensorFilterProperties *prop) {
  gst_tensors_info_init(&inputTensorMeta);
  gst_tensors_info_init(&outputTensorMeta);

  model_config = g_strdup(model_config_);
  loadModel();

  validateTensor(&prop->input_meta, true);
  validateTensor(&prop->output_meta, false);

  model->init();
  model->readModel();

  gst_tensors_info_copy(&inputTensorMeta, &prop->input_meta);
  gst_tensors_info_copy(&outputTensorMeta, &prop->output_meta);
}

NNTrainer::~NNTrainer() {
  if (model != nullptr) {
    delete model;
  }

  gst_tensors_info_free(&inputTensorMeta);
  gst_tensors_info_free(&outputTensorMeta);
  g_free(model_config);
}

const char *NNTrainer::getModelConfig() { return model_config; }

void NNTrainer::validateTensor(const GstTensorsInfo *tensorInfo,
                               bool is_input) {

  nntrainer::TensorDim dim;
  nntrainer_tensor_info_s info_s;

  if (is_input)
    dim = model->getInputDimension()[0];
  else
    dim = model->getOutputDimension()[0];

  if (tensorInfo->info[0].type != _NNS_FLOAT32)
    throw std::invalid_argument(
      "only float32 is supported for input and output");

  info_s.rank = NUM_DIM;

  for (unsigned int i = 0; i < NUM_DIM; ++i) {
    if (tensorInfo->info[0].dimension[i] != dim.getDim()[NUM_DIM - i - 1])
      throw std::invalid_argument("Tensor dimension doesn't match");

    info_s.dims.push_back(dim.getDim()[i]);
  }

  if (is_input)
    input_tensor_info.push_back(info_s);
}

void NNTrainer::loadModel() {
#if (DBG)
  gint64 start_time = g_get_real_time();
#endif
  if (model_config == nullptr)
    throw std::invalid_argument("model config is null!");

  model = new nntrainer::NeuralNetwork();
  model->loadFromConfig(model_config);
#if (DBG)
  gint64 stop_time = g_get_real_time();
  g_message("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
}

int NNTrainer::getInputTensorDim(GstTensorsInfo *info) {
  gst_tensors_info_copy(info, &inputTensorMeta);
  return 0;
}

int NNTrainer::getOutputTensorDim(GstTensorsInfo *info) {
  gst_tensors_info_copy(info, &outputTensorMeta);
  return 0;
}

int NNTrainer::run(const GstTensorMemory *input, GstTensorMemory *output) {
#if (DBG)
  gint64 start_time = g_get_real_time();
#endif
  std::shared_ptr<nntrainer::Tensor> out;

  std::vector<std::int64_t> d = input_tensor_info[0].dims;
  nntrainer::Tensor X =
    nntrainer::Tensor(nntrainer::TensorDim(d[0], d[1], d[2], d[3]),
                      static_cast<float *>(input[0].data));

  std::shared_ptr<const nntrainer::Tensor> o;

  try {
    o = model->inference({MAKE_SHARED_TENSOR(X)})[0];
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return -2;
  } catch (...) {
    ml_loge("unknown error type thrown");
    return -3;
  }

  if (o == nullptr) {
    return -1;
  }

  out = std::const_pointer_cast<nntrainer::Tensor>(o);
  output[0].data = out->getData();

  outputTensorMap.insert(std::make_pair(output[0].data, out));

#if (DBG)
  gint64 stop_time = g_get_real_time();
  g_message("Run() is finished: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

void NNTrainer::freeOutputTensor(void *data) {
  if (data != nullptr) {
    std::map<void *, std::shared_ptr<nntrainer::Tensor>>::iterator it =
      outputTensorMap.find(data);
    if (it != outputTensorMap.end()) {
      outputTensorMap.erase(data);
    }
  }
}

static void nntrainer_close(const GstTensorFilterProperties *prop,
                            void **private_data) {
  NNTrainer *nntrainer = static_cast<NNTrainer *>(*private_data);

  if (!nntrainer)
    return;
  delete nntrainer;
  *private_data = NULL;
}

static int nntrainer_loadModelFile(const GstTensorFilterProperties *prop,
                                   void **private_data) {
  if (prop->num_models != 1)
    return -1;

  NNTrainer *nntrainer = static_cast<NNTrainer *>(*private_data);
  const gchar *model_file = prop->model_files[0];
  if (nntrainer != NULL) {
    if (g_strcmp0(model_file, nntrainer->getModelConfig()) == 0)
      return 1; /* skipped */

    nntrainer_close(prop, private_data);
  }

  try {
    nntrainer = new NNTrainer(model_file, prop);
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return -1;
  } catch (...) {
    ml_loge("unknown error type thrown");
    return -3;
  }
  *private_data = nntrainer;

  return 0;
}

static int nntrainer_open(const GstTensorFilterProperties *prop,
                          void **private_data) {
  int status = nntrainer_loadModelFile(prop, private_data);
  return status;
}

static int nntrainer_run(const GstTensorFilterProperties *prop,
                         void **private_data, const GstTensorMemory *input,
                         GstTensorMemory *output) {
  NNTrainer *nntrainer = static_cast<NNTrainer *>(*private_data);
  g_return_val_if_fail(nntrainer && input && output, -EINVAL);

  return nntrainer->run(input, output);
}

static int nntrainer_getInputDim(const GstTensorFilterProperties *prop,
                                 void **private_data, GstTensorsInfo *info) {
  NNTrainer *nntrainer = static_cast<NNTrainer *>(*private_data);
  g_return_val_if_fail(nntrainer && info, -EINVAL);
  return nntrainer->getInputTensorDim(info);
}

static int nntrainer_getOutputDim(const GstTensorFilterProperties *prop,
                                  void **private_data, GstTensorsInfo *info) {
  NNTrainer *nntrainer = static_cast<NNTrainer *>(*private_data);
  g_return_val_if_fail(nntrainer && info, -EINVAL);
  return nntrainer->getOutputTensorDim(info);
}

static void nntrainer_destroyNotify(void **private_data, void *data) {
  NNTrainer *nntrainer = static_cast<NNTrainer *>(*private_data);
  if (nntrainer) {
    nntrainer->freeOutputTensor(data);
  }
}

static int nntrainer_checkAvailability(accl_hw hw) {
  if (g_strv_contains(nntrainer_accl_support, get_accl_hw_str(hw)))
    return 0;

  return -ENOENT;
}

static gchar filter_subplugin_nntrainer[] = "nntrainer";

static GstTensorFilterFramework NNS_support_nntrainer = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = nntrainer_open,
  .close = nntrainer_close,
};

void init_filter_nntrainer(void) {
  NNS_support_nntrainer.name = filter_subplugin_nntrainer;
  NNS_support_nntrainer.allow_in_place = FALSE;
  NNS_support_nntrainer.allocate_in_invoke = TRUE;
  NNS_support_nntrainer.run_without_model = FALSE;
  NNS_support_nntrainer.verify_model_path = FALSE;
  NNS_support_nntrainer.invoke_NN = nntrainer_run;
  NNS_support_nntrainer.getInputDimension = nntrainer_getInputDim;
  NNS_support_nntrainer.getOutputDimension = nntrainer_getOutputDim;
  NNS_support_nntrainer.destroyNotify = nntrainer_destroyNotify;
  NNS_support_nntrainer.checkAvailability = nntrainer_checkAvailability;

  nnstreamer_filter_probe(&NNS_support_nntrainer);
}

void fini_filter_nntrainer(void) {
  nnstreamer_filter_exit(NNS_support_nntrainer.name);
}
