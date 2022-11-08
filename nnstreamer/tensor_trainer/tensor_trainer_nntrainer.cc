/* SPDX-License-Identifier: Apache-2.0 */
/**
 * NNStreamer tensor_trainer subplugin for nntrainer
 * Copyright (C) 2022 Hyunil Park <hyunil46.park@samsung.com>
 */
/**
 * @file   tensor_trainer_nntrainer.cc
 * @date   07 November 2022
 * @brief  NNStreamer tensor_trainer subplugin
 * @see    http://github.com/nnstreamer/nnstreamer
 * @author Hyunil Park <hyunil46.park@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include <algorithm>
#include <limits>
#include <sstream>
#include <unistd.h>

#include <nntrainer_error.h>

#include "ml-api-common.h"
#include "nnstreamer.h"
#include "nnstreamer_plugin_api.h"
#include "tensor_trainer_nntrainer.hh"

#include <iostream>
#define cout_log(x) std::cout<<__FILE__<<":"<<__LINE__<<":"<<__func__ << ":" <<#x<<":"<< x <<std::endl;

/**
 * @brief   startup constructor
 */
void init_subplugin_nntrainer(void) __attribute__((constructor));

/**
 * @brief   startdown destructor
 */
void fini_subplugin_nntrainer(void) __attribute__((destructor));

static gchar subplugin_name[] = "nntrainer";

static GstTensorFilterFramework NNS_Trainer_support_nntrainer = {
  .version =  GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = NULL,
  .close = NULL,
};

void init_subplugin_nntrainer(void) {
  NNS_Trainer_support_nntrainer.name = subplugin_name;
  NNS_Trainer_support_nntrainer.allow_in_place = FALSE;
  NNS_Trainer_support_nntrainer.allocate_in_invoke = TRUE;
  NNS_Trainer_support_nntrainer.run_without_model = FALSE;
  NNS_Trainer_support_nntrainer.verify_model_path = FALSE;
  NNS_Trainer_support_nntrainer.invoke_NN = NULL;
  NNS_Trainer_support_nntrainer.destroyNotify = NULL;
  NNS_Trainer_support_nntrainer.checkAvailability = NULL;
  NNS_Trainer_support_nntrainer.getInputDimension = NULL;
  NNS_Trainer_support_nntrainer.getOutputDimension = NULL;
  NNS_Trainer_support_nntrainer.setInputDimension = NULL;

  nnstreamer_filter_probe(&NNS_Trainer_support_nntrainer);
}

void fini_subplugin_nntrainer(void) {
  nnstreamer_filter_exit(NNS_Trainer_support_nntrainer.name);
}
