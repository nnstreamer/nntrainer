/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer Tensor_Filter, nntrainer Module
 * Copyright (C) 2020 Jijoong Moon <jijoong.moon@samsung.com>
 */
/**
 * @file   tensor_filter_nntrainer.cc
 * @date   09 Sept 2020
 * @brief  nntrainer inference module for tensor_filter gstreamer plugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 * This is the per-NN-framework plugin (e.g) tensorflow-lite) for tensor_filter.
 * Fill in "GstTensorFilterFramework" for tensor_filter.h/c
 *
 */

#include <algorithm>
#include <limits>
#include <sstream>
#include <unistd.h>

#include <nntrainer_error.h>

#include "tensor_filter_nntrainer.hh"

#ifdef ml_loge
#undef ml_loge
#endif

#define ml_loge g_critical

/**
 * @brief Macro for debug mode.
 */
#ifndef DBG
#define DBG FALSE
#endif

static const gchar *nntrainer_accl_support[] = {NULL};

/**
 * @brief   startup constructor
 *
 */
void init_filter_nntrainer(void) __attribute__((constructor));

/**
 * @brief   startdown destructor
 *
 */
void fini_filter_nntrainer(void) __attribute__((destructor));

/**
 * @brief   Wrapper for glib deletor
 */
struct gNewDeletor {
  template <typename T> void operator()(T *ptr) const { g_free(ptr); }
};

static std::unique_ptr<GstTensorInfo, gNewDeletor>
to_nnst_tensor_dim(const ml::train::TensorDim &dim) {
  auto info =
    std::unique_ptr<GstTensorInfo, gNewDeletor>(g_new(GstTensorInfo, 1));
  gst_tensor_info_init(info.get());

  info->type = _NNS_FLOAT32;
  for (unsigned int i = 0; i < ml::train::TensorDim::MAXDIM; ++i) {
    info->dimension[i] = dim.getTensorDim(ml::train::TensorDim::MAXDIM - i - 1);
  }

  return info;
}

static ml::train::TensorDim to_nntr_tensor_dim(const GstTensorInfo *info) {
  const tensor_dim &d = info->dimension;
  return {d[3], d[2], d[1], d[0]};
}

NNTrainerInference::NNTrainerInference(const std::string &model_config_) :
  batch_size(1),
  model_config(model_config_) {
  loadModel();
  model->compile();
  model->initialize();
}

const char *NNTrainerInference::getModelConfig() {
  return model_config.c_str();
}

void NNTrainerInference::loadModel() {
#if (DBG)
  gint64 start_time = g_get_real_time();
#endif
  model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
  model->load(model_config, ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN);
#if (DBG)
  gint64 stop_time = g_get_real_time();
  g_message("Model is loaded: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
}

int NNTrainerInference::run(const GstTensorMemory *input,
                            GstTensorMemory *output) {
#if (DBG)
  gint64 start_time = g_get_real_time();
#endif

  auto input_dims = getInputDimension();
  auto output_dims = getOutputDimension();

  std::vector<float *> inputs;
  inputs.reserve(input_dims.size());

  for (size_t idx = 0; idx < input_dims.size(); idx++)
    // do not allocate new, but instead use tensor::Map
    inputs.emplace_back(static_cast<float *>(input[idx].data));

  std::vector<float *> outputs;
  std::vector<float *> labels;

  try {
    outputs = model->inference(batch_size, inputs, labels);
  } catch (std::exception &e) {
    ml_loge("%s %s", typeid(e).name(), e.what());
    return -2;
  } catch (...) {
    ml_loge("unknown error type thrown");
    return -3;
  }

  for (size_t idx = 0; idx < output_dims.size(); idx++) {
    if (outputs[idx] == nullptr) {
      return -1;
    }

    output[idx].data = static_cast<void *>(outputs[idx]);
  }

#if (DBG)
  gint64 stop_time = g_get_real_time();
  g_message("Run() is finished: %" G_GINT64_FORMAT, (stop_time - start_time));
#endif
  return 0;
}

static void nntrainer_close(const GstTensorFilterProperties *prop,
                            void **private_data) {
  NNTrainerInference *nntrainer =
    static_cast<NNTrainerInference *>(*private_data);

  if (!nntrainer)
    return;
  delete nntrainer;
  *private_data = NULL;
}

static int nntrainer_loadModelFile(const GstTensorFilterProperties *prop,
                                   void **private_data) {
  if (prop->num_models != 1)
    return -1;

  NNTrainerInference *nntrainer =
    static_cast<NNTrainerInference *>(*private_data);
  const gchar *model_file = prop->model_files[0];
  if (nntrainer != NULL) {
    if (g_strcmp0(model_file, nntrainer->getModelConfig()) == 0)
      return 1; /* skipped */

    nntrainer_close(prop, private_data);
  }

  try {
    nntrainer = new NNTrainerInference(model_file);
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
  NNTrainerInference *nntrainer =
    static_cast<NNTrainerInference *>(*private_data);
  g_return_val_if_fail(nntrainer && input && output, -EINVAL);

  return nntrainer->run(input, output);
}

static int nntrainer_setInputDim(const GstTensorFilterProperties *prop,
                                 void **private_data,
                                 const GstTensorsInfo *in_info,
                                 GstTensorsInfo *out_info) {
  NNTrainerInference *nntrainer =
    static_cast<NNTrainerInference *>(*private_data);
  g_return_val_if_fail(prop && nntrainer && in_info && out_info, -EINVAL);

  auto num_input = in_info->num_tensors;
  g_return_val_if_fail(num_input != 0, -EINVAL);

  auto batch_size = in_info->info[0].dimension[3];

  /// this does not allocate the memory for the inference, so setting batch here
  /// does not have a large effect on the first inference call as of now.
  /// we can make a call to nntrainer->allocate();
  /// which would wrap around NeuralNetwork::allocate(false);
  /// However, it might not be a good choice in therms of migrating to api.
  nntrainer->setBatchSize(batch_size);

  auto model_inputs = nntrainer->getInputDimension();
  /// check number of in
  g_return_val_if_fail(num_input == model_inputs.size(), -EINVAL);

  /// check each in dimension matches
  for (unsigned int i = 0; i < num_input; ++i) {
    model_inputs[i].batch(batch_size);
    g_return_val_if_fail(in_info->info[i].type == _NNS_FLOAT32, -EINVAL);
    g_return_val_if_fail(
      model_inputs[i] == to_nntr_tensor_dim(in_info->info + i), -EINVAL);
  }

  auto model_outputs = nntrainer->getOutputDimension();
  g_return_val_if_fail(!model_outputs.empty(), -EINVAL);
  /// set gstTensorInfo
  out_info->num_tensors = model_outputs.size();
  for (unsigned int i = 0; i < out_info->num_tensors; ++i) {
    model_outputs[i].batch(batch_size);
    gst_tensor_info_copy(out_info->info + i,
                         to_nnst_tensor_dim(model_outputs[i]).get());
  }

  return 0;
}

static void nntrainer_destroyNotify(void **private_data, void *data) {}

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
  NNS_support_nntrainer.setInputDimension = nntrainer_setInputDim;
  NNS_support_nntrainer.destroyNotify = nntrainer_destroyNotify;
  NNS_support_nntrainer.checkAvailability = nntrainer_checkAvailability;
  NNS_support_nntrainer.getInputDimension = NULL;
  NNS_support_nntrainer.getOutputDimension = NULL;

  nnstreamer_filter_probe(&NNS_support_nntrainer);
}

void fini_filter_nntrainer(void) {
  nnstreamer_filter_exit(NNS_support_nntrainer.name);
}
