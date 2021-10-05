// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_models_recurrent.cpp
 * @date 05 Oct 2021
 * @brief unittest models for recurrent ones
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <databuffer.h>
#include <dataset.h>
#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

using namespace nntrainer;

static nntrainer::IniSection nn_base("model", "type = NeuralNetwork");
static std::string fc_base = "type = Fully_connected";
static nntrainer::IniSection sgd_base("optimizer", "Type = sgd");
static nntrainer::IniSection constant_loss("loss",
                                           "type = constant_derivative");

int getSample(float **outVec, float **outLabel, bool *last, void *user_data) {
  **outVec = 1;
  **outLabel = 1;
  *last = true;
  return 0;
};

TEST(FcOnly, fcHandUnrolled) {
  ScopedIni fc_only_hand_unrolled(
    "fc_only_hand_unrolled", {nn_base, sgd_base,
                              IniSection("fc_1") + fc_base +
                                "unit=1 | weight_initializer=ones | "
                                "bias_initializer=ones | input_shape=1:1:1",
                              IniSection("fc_2") + fc_base +
                                "unit=1 | weight_initializer=ones | "
                                "bias_initializer=ones | shared_from = fc_1",
                              IniSection("fc_3") + fc_base +
                                "unit=1 | weight_initializer=ones | "
                                "bias_initializer=ones | shared_from = fc_1",
                              constant_loss});

  NeuralNetwork nn;
  nn.load(fc_only_hand_unrolled.getIniName(),
          ml::train::ModelFormat::MODEL_FORMAT_INI);

  EXPECT_EQ(nn.compile(), 0);
  EXPECT_EQ(nn.initialize(), 0);

  auto db = ml::train::createDataset(ml::train::DatasetType::GENERATOR,
                                     getSample, nullptr);

  nn.setDataset(DatasetModeType::MODE_TRAIN, std::move(db));
  EXPECT_EQ(nn.train(), 0);
}
