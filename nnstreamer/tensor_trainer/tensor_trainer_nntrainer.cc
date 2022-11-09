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
#include "nnstreamer_plugin_api_filter.h"
#include "tensor_trainer_nntrainer.hh"

#include <iostream>

#define UNUSED(expr) \
  do {               \
    (void)(expr);    \
  } while (0)

/**
 * @brief startup constructor
 */
void init_subplugin_nntrainer(void) __attribute__((constructor));

/**
 * @brief startdown destructor
 */
void fini_subplugin_nntrainer(void) __attribute__((destructor));

static void nntrainer_model_destructor(const GstTensorFilterProperties *prop,
                                       void **private_data) {
  NNTrainer::NNTrainerTrain *nntrainer =
    static_cast<NNTrainer::NNTrainerTrain *>(*private_data);

  if (!nntrainer)
    return;
  delete nntrainer;
  *private_data = NULL;
}

#if 0 /** To avoid build errors : defined but not used */
static void nntrainer_model_train(const GstTensorFilterProperties *prop,
                                  void **private_data) {
  NNTrainer::NNTrainerTrain *nntrainer =
    static_cast<NNTrainer::NNTrainerTrain *>(*private_data);
  if (!nntrainer) {
    std::cerr << "Failed get model" << std::endl;
  }

  try {
    nntrainer->trainModel();
  } catch (std::exception &e) {
    std::cerr << "Error" << typeid(e).name() << "," << e.what() << std::endl;
  }
}
#endif

void NNTrainer::NNTrainerTrain::trainModel() {
  try {
    model->train();
    training_loss = model->getTrainingLoss();
    validation_loss = model->getValidationLoss();
  } catch (std::exception &e) {
    std::cerr << "Error" << typeid(e).name() << "," << e.what() << std::endl;
  }
}

void NNTrainer::NNTrainerTrain::createModel() {

  try {
    model = ml::train::createModel(ml::train::ModelType::NEURAL_NET);
    model->load(model_config,
                ml::train::ModelFormat::MODEL_FORMAT_INI_WITH_BIN);
  } catch (std::exception &e) {
    std::cerr << "Failed loading model configuration file" << typeid(e).name()
              << e.what() << std::endl;
  }

  try {
    model->compile();
    model->initialize();
    model->setDataset(ml::train::DatasetModeType::MODE_TRAIN, dataset_train);
    model->setDataset(ml::train::DatasetModeType::MODE_VALID, dataset_valid);
  } catch (std::exception &e) {
    std::cerr << "Error" << typeid(e).name() << "," << e.what() << std::endl;
  }
}

NNTrainer::NNTrainerTrain::NNTrainerTrain(const std::string &_model_config) :
  model_config(_model_config) {
  createModel();
}

static int
nntrainer_model_construct_with_conf(const GstTensorFilterProperties *prop,
                                    void **private_data) {

  NNTrainer::NNTrainerTrain *nntrainer =
    static_cast<NNTrainer::NNTrainerTrain *>(*private_data);
  if (!nntrainer)
    nntrainer_model_destructor(prop, private_data);

  try {
    nntrainer =
      new NNTrainer::NNTrainerTrain("/home/hyunil/nnstreamer/mnist.ini");
  } catch (std::exception &e) {
    std::cerr << "Error" << typeid(e).name() << "," << e.what() << std::endl;
  }

  return 0;
}

static int nntrainer_model_construct(const GstTensorFilterProperties *prop,
                                     void **private_data) {
  int status = nntrainer_model_construct_with_conf(prop, private_data);

  return status;
}

static gchar subplugin_name[] = "nntrainer";

/** GstTensorFilterFramework will be changed */
static GstTensorFilterFramework NNS_Trainer_support_nntrainer = {
  .version = GST_TENSOR_FILTER_FRAMEWORK_V0,
  .open = nntrainer_model_construct, //.construct = nntrainer_model_construct,
  .close =
    nntrainer_model_destructor, //.destructor = nntrainer_model_destructor,
  //.train = nntrainer_model_train
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
  /* nnstreamer_filter_probe function will be changed */
  nnstreamer_filter_probe(&NNS_Trainer_support_nntrainer);
}

void fini_subplugin_nntrainer(void) {
  /* nnstreamer_filter_exit function will be changed */
  nnstreamer_filter_exit(NNS_Trainer_support_nntrainer.name);
}
