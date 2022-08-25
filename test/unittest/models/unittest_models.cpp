// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file unittest_models_v2.cpp
 * @date 25 Nov 2021
 * @brief unittest models for v2 version
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <memory>

#include <ini_wrapper.h>
#include <neuralnet.h>
#include <nntrainer_test_util.h>

#include <models_golden_test.h>

using namespace nntrainer;

static inline constexpr const int NOT_USED_ = 1;

static IniSection nn_base("model", "type = NeuralNetwork");
static std::string fc_base = "type = Fully_connected";
static std::string red_mean_base = "type = reduce_mean";
static IniSection sgd_base("optimizer", "Type = sgd");
static IniSection constant_loss("loss", "type = constant_derivative");
static IniSection act_base("activation", "Type = Activation");

IniWrapper reduce_mean_last("reduce_mean_last",
                            {
                              nn_base + "batch_size=3",
                              sgd_base + "learning_rate=0.1",
                              IniSection("fc_1") + fc_base +
                                "unit=7 | input_shape=1:1:2",
                              IniSection("red_mean") + red_mean_base + "axis=3",
                              constant_loss,
                            });

IniWrapper fc_relu_decay(
  "fc_relu_decay",
  {nn_base + "Loss=mse | batch_size = 3", sgd_base + "learning_rate = 0.1",
   IniSection("input") + "type=input" + "input_shape = 1:1:3",
   IniSection("dense") + fc_base + "unit = 10" + "weight_decay=0.9",
   IniSection("act") + act_base + "Activation = relu",
   IniSection("dense_1") + fc_base + "unit = 2" + "bias_decay=0.9",
   IniSection("act_1") + act_base + "Activation = sigmoid"});

static std::unique_ptr<NeuralNetwork> makeMolAttention() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=in3", "input_shape=1:1:5"}},
    {"input", {"name=in2", "input_shape=1:4:6"}},
    {"input", {"name=in1", "input_shape=1:1:6"}},
    {"mol_attention",
     {"name=mol", "input_layers=in1,in2,in3", "unit=8", "mol_k=5"}},
    {"constant_derivative", {"name=loss1", "input_layers=mol(0)"}},
    {"constant_derivative", {"name=loss2", "input_layers=mol(1)"}},
  });

  nn->setProperty({"label_layers=loss1,loss2"});
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMolAttentionMasked() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=in4", "input_shape=1:1:1"}},
    {"input", {"name=in3", "input_shape=1:1:5"}},
    {"input", {"name=in2", "input_shape=1:4:6"}},
    {"input", {"name=in1", "input_shape=1:1:6"}},
    {"mol_attention",
     {"name=mol", "input_layers=in1,in2,in3,in4", "unit=8", "mol_k=5"}},
    {"constant_derivative", {"name=loss1", "input_layers=mol(0)"}},
    {"constant_derivative", {"name=loss2", "input_layers=mol(1)"}},
  });

  nn->setProperty({"label_layers=loss1,loss2"});
  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

static std::unique_ptr<NeuralNetwork>
makeMultiHeadAttention_disable_need_weights() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:6"}},
    {"input", {"name=input_2", "input_shape=1:2:6"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_1, input_2",
      "disable_bias=true", "num_heads=2"}},
    {"mse", {"name=loss", "input_layers=multi_head_attention"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0, input_1, input_2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:6"}},
    {"input", {"name=input_2", "input_shape=1:2:6"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_1, input_2",
      "num_heads=2", "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty(
    {"input_layers=input_0, input_1, input_2", "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention_kdim_vdim() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:4"}},
    {"input", {"name=input_2", "input_shape=1:2:5"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_1, input_2",
      "num_heads=2", "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty(
    {"input_layers=input_0, input_1, input_2", "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention_float_attn_mask() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"input", {"name=input_1", "input_shape=1:2:6"}},
    {"input", {"name=input_2", "input_shape=1:2:6"}},
    {"input", {"name=input_3", "input_shape=2:3:2"}},
    {"multi_head_attention",
     {"name=multi_head_attention",
      "input_layers=input_0, input_1, input_2, input_3", "num_heads=2",
      "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0, input_1, input_2, input_3",
                   "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makeMultiHeadAttention_self_attention() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input_0", "input_shape=1:3:6"}},
    {"multi_head_attention",
     {"name=multi_head_attention", "input_layers=input_0, input_0, input_0",
      "num_heads=2", "return_attention_weight=after"}},
    {"mse", {"name=loss1", "input_layers=multi_head_attention(0)"}},
    {"mse", {"name=loss2", "input_layers=multi_head_attention(1)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  nn->setProperty({"input_layers=input_0", "label_layers=loss1, loss2"});

  return nn;
}

static std::unique_ptr<NeuralNetwork> makePositionalEncoding() {
  std::unique_ptr<NeuralNetwork> nn(new NeuralNetwork());
  nn->setProperty({"batch_size=3"});

  auto outer_graph = makeGraph({
    {"input", {"name=input", "input_shape=5:1:6"}},
    {"reshape", {"name=reshape", "target_shape=1:5:6"}},
    {"positional_encoding", {"name=positional_encoding", "max_timestep=7"}},
    {"multi_head_attention",
     {"name=multi_head_attention",
      "input_layers=positional_encoding, positional_encoding, "
      "positional_encoding",
      "num_heads=2"}},
    {"mse", {"name=loss", "input_layers=multi_head_attention(0)"}},
  });

  for (auto &node : outer_graph) {
    nn->addLayer(node);
  }

  nn->setOptimizer(ml::train::createOptimizer("sgd", {"learning_rate = 0.1"}));
  return nn;
}

GTEST_PARAMETER_TEST(
  model, nntrainerModelTest,
  ::testing::ValuesIn({
    mkModelIniTc(reduce_mean_last, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeMolAttention, "mol_attention",
                 ModelTestOption::COMPARE_V2),
    mkModelTc_V2(makeMolAttentionMasked, "mol_attention_masked",
                 ModelTestOption::COMPARE_RUN_V2),
    mkModelTc_V2(makeMultiHeadAttention_disable_need_weights,
                 "multi_head_attention_disable_need_weights",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention, "multi_head_attention",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention_kdim_vdim,
                 "multi_head_attention_kdim_vdim", ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention_float_attn_mask,
                 "multi_head_attention_float_attn_mask",
                 ModelTestOption::ALL_V2),
    /** @todo:change model if bool type tensor is supported */
    mkModelTc_V2(makeMultiHeadAttention_float_attn_mask,
                 "multi_head_attention_pseudo_bool_attn_mask",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makeMultiHeadAttention_self_attention,
                 "multi_head_attention_self_attention",
                 ModelTestOption::ALL_V2),
    mkModelTc_V2(makePositionalEncoding, "positional_encoding",
                 ModelTestOption::ALL_V2),
    mkModelIniTc(fc_relu_decay, DIM_UNUSED, NOT_USED_,
                 ModelTestOption::COMPARE_V2),
  }),
  [](const testing::TestParamInfo<nntrainerModelTest::ParamType> &info) {
    return std::get<1>(info.param);
  });
