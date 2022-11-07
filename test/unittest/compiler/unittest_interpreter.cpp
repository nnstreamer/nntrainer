// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_interpreter.cpp
 * @date 02 April 2021
 * @brief interpreter test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Donghak Park <donghak.park@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <functional>
#include <gtest/gtest.h>
#include <memory>

#include <app_context.h>
#include <execution_mode.h>
#include <flatten_realizer.h>
#include <ini_interpreter.h>
#include <interpreter.h>
#include <layer.h>
#include <layer_node.h>
#include <network_graph.h>
#include <node_exporter.h>
#include <realizer.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tflite_interpreter.h>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#endif

#include <app_context.h>
#include <compiler_test_util.h>
#include <nntrainer_test_util.h>

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;

auto ini_interpreter = std::make_shared<nntrainer::IniGraphInterpreter>(
  nntrainer::AppContext::Global(), compilerPathResolver);

/**
 * @brief nntrainer Interpreter Test setup
 *
 * @note Proposing an evolutional path of current test
 * 1. A reference graph vs given parameter
 * 2. A reference graph vs list of models
 * 3. A reference graph vs (pick two models) a -> b -> a graph, b -> a -> b
 * graph
 */
class nntrainerInterpreterTest
  : public ::testing::TestWithParam<
      std::tuple<nntrainer::GraphRepresentation, const char *,
                 std::shared_ptr<nntrainer::GraphInterpreter>>> {

protected:
  virtual void SetUp() {
    auto params = GetParam();

    reference = std::get<0>(params);
    file_path = compilerPathResolver(std::get<1>(params));
    interpreter = std::move(std::get<2>(params));
  }

  nntrainer::GraphRepresentation reference;
  std::shared_ptr<nntrainer::GraphInterpreter> interpreter;
  std::string file_path;
};

/**
 * @brief Check two compiled graph is equal
 * @note later this will be more complicated (getting N graph and compare each
 * other)
 *
 */
TEST_P(nntrainerInterpreterTest, graphEqual) {
  // int status = reference->compile("");
  // EXPECT_EQ(status, ML_ERROR_NONE);
  auto g = interpreter->deserialize(file_path);

  /// @todo: change this to something like graph::finalize
  // status = g->compile("");
  // EXPECT_EQ(status, ML_ERROR_NONE);

  /// @todo: make a proper graph equal
  /// 1. having same number of nodes
  /// 2. layer name is identical (this is too strict though)
  /// 3. attributes of layer is identical
  EXPECT_NO_THROW(graphEqual(g, reference));
}

/**
 * @brief graph serialize after deserialize, compare if they are the same
 *
 */
TEST_P(nntrainerInterpreterTest, graphSerializeAfterDeserialize) {
  auto g = interpreter->deserialize(file_path);

  auto out_file_path = file_path + ".out";

  /// @todo: change this to something like graph::finalize
  // int status = g->compile("");
  // EXPECT_EQ(status, ML_ERROR_NONE);
  interpreter->serialize(g, out_file_path);
  auto new_g = interpreter->deserialize(out_file_path);

  graphEqual(g, new_g);

  EXPECT_EQ(remove(out_file_path.c_str()), 0) << strerror(errno);
}

TEST_P(nntrainerInterpreterTest, deserialize_01_n) {
  EXPECT_THROW(interpreter->deserialize(""), std::invalid_argument);
}

TEST_P(nntrainerInterpreterTest, deserialize_02_n) {
  EXPECT_THROW(interpreter->deserialize("not_existing_file"),
               std::invalid_argument);
}

auto fc0 = LayerRepresentation("fully_connected",
                               {"name=fc0", "unit=2", "input_shape=1:1:100"});
auto fc1 = LayerRepresentation("fully_connected", {"name=fc1", "unit=2"});

auto flatten = LayerRepresentation("flatten", {"name=flat"});

/**
 * TODO: update tflite interpreter after the change of semantics that tensors
 * are different between input and output of a layer but the underlying data
 * is same. Once the interpreter is updated, this test can be enabled.
 */
TEST(nntrainerInterpreterTflite, simple_fc) {

  nntrainer::TfliteInterpreter interpreter;

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});

  auto fc0_zeroed = LayerRepresentation(
    "fully_connected", {"name=fc0", "unit=2", "input_layers=in0",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto fc1_zeroed = LayerRepresentation(
    "fully_connected", {"name=fc1", "unit=2", "bias_initializer=ones",
                        "weight_initializer=ones", "input_layers=fc0"});

  auto g = makeGraph({input0, fc0_zeroed, fc1_zeroed});

  nntrainer::NetworkGraph ng;

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);
  interpreter.serialize(g, "test.tflite");
  ng.deallocateTensors();

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("test.tflite");
  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  nntrainer::Tensor in(nntrainer::TensorDim({1, 1, 1, 1}));
  in.setValue(2.0f);
  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 2, 1}));

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 2, 1}));
  ans.setValue(7.0f);

  EXPECT_EQ(out, ans);

  if (remove("test.tflite")) {
    std::cerr << "remove ini "
              << "test.tflite"
              << "failed, reason: " << strerror(errno);
  }
}

TEST(nntrainerInterpreterTflite, part_of_resnet_0) {

  nntrainer::TfliteInterpreter interpreter;

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=1:1:1"});

  auto averagepool0 = LayerRepresentation(
    "pooling2d", {"name=averagepool0", "input_layers=in0", "pooling=average",
                  "pool_size=1,1", "stride=1,1", "padding=valid"});

  auto reshape0 =
    LayerRepresentation("reshape", {"name=reshape0", "target_shape=1:1:1",
                                    "input_layers=averagepool0"});

  auto fc0 = LayerRepresentation(
    "fully_connected", {"name=fc0", "unit=2", "input_layers=reshape0",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto softmax0 = LayerRepresentation(
    "activation", {"name=softmax0", "activation=softmax", "input_layers=fc0"});

  auto g = makeGraph({input0, averagepool0, reshape0, fc0, softmax0});

  nntrainer::NetworkGraph ng;

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);
  interpreter.serialize(g, "part_of_resnet.tflite");
  ng.deallocateTensors();

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("part_of_resnet.tflite");
  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  nntrainer::Tensor in(nntrainer::TensorDim({1, 1, 2, 2}));
  in.setValue(2.0f);
  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 1, 2}));

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 1, 2}));
  ans.setValue(0.5f);

  EXPECT_EQ(out, ans);

  if (remove("part_of_resnet.tflite")) {
    std::cerr << "remove ini "
              << "part_of_resnet.tflite"
              << "failed, reason: " << strerror(errno);
  }
}

/**
 * @brief make ini test case from given parameter
 */
static std::tuple<nntrainer::GraphRepresentation, const char *,
                  std::shared_ptr<nntrainer::GraphInterpreter>>
mkTc(nntrainer::GraphRepresentation graph, const char *file,
     std::shared_ptr<nntrainer::GraphInterpreter> interpreter) {
  return std::make_tuple(graph, file, interpreter);
}

// clang-format off
GTEST_PARAMETER_TEST(nntrainerAutoInterpreterTest, nntrainerInterpreterTest,
                        ::testing::Values(
  mkTc(makeGraph({fc0, flatten}), "simple_fc.ini", ini_interpreter),
  mkTc(makeGraph({fc0, flatten}), "simple_fc_backbone.ini", ini_interpreter)
));
// clang-format on

/**
 * @brief Fully Connected Layer weights transpose(NCHW -> NHWC) unittest
 *
 */

TEST(nntrainerInterpreterTflite, simple_flatten) {

  nntrainer::TfliteInterpreter interpreter;
  nntrainer::FlattenRealizer fr;

  auto input0 = LayerRepresentation(
    "input", {"name=in0", "input_shape=3:2:4", "flatten=true"});

  auto fc0_zeroed = LayerRepresentation(
    "fully_connected", {"name=fc0", "unit=1", "input_layers=in0",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto g = fr.realize(makeGraph({input0, fc0_zeroed}));

  nntrainer::NetworkGraph ng;

  /// @todo Disable Support Inplace  --> should be support Inplace()
  ng.setMemoryOptimizations(false);

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);

  unsigned int UNIT = 1;
  unsigned int HEIGHT = 2;
  unsigned int WIDTH = 4;
  unsigned int CHANNEL = 3;

  auto weight_data = ng.getLayerNode("fc0")->getWeight(0).getData();
  auto *ptr = const_cast<float *>(weight_data);

  // Set initial Weights
  int count = 0;
  for (unsigned int h = 0; h < HEIGHT; h++) {
    for (unsigned int w = 0; w < WIDTH; w++) {
      for (unsigned int c = 0; c < CHANNEL; c++) {
        for (unsigned int u = 0; u < UNIT; u++) {
          int now_position = (u * (CHANNEL * WIDTH * HEIGHT)) +
                             h * (WIDTH * CHANNEL) + w * CHANNEL + c;

          ptr[count] = now_position;
          count += 1;
        }
      }
    }
  }

  interpreter.serialize(g, "FC_weight_test.tflite");

  auto weight_data2 = ng.getLayerNode("fc0")->getWeight(0).getData();
  auto *ptr2 = const_cast<float *>(weight_data2);

  ng.deallocateTensors();

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("FC_weight_test.tflite");

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 2, 4}));

  count = 0;
  // Set Tensor with (1,3,4,2)(NCHW) Shape
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 2; h++) {
      for (int w = 0; w < 4; w++) {
        in.setValue(0, c, h, w, count);
        count++;
      }
    }
  }

  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 1, 1}));

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 1, 1}));
  ans.setValue(4325);
  EXPECT_EQ(out, ans);

  if (remove("FC_weight_test.tflite")) {
    std::cerr << "remove ini "
              << "FC_weight_test.tflite"
              << "failed, reason: " << strerror(errno);
  }
}

TEST(nntrainerInterpreterTflite, simple_flatten2) {

  nntrainer::TfliteInterpreter interpreter;
  nntrainer::FlattenRealizer fr;

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=3:2:4"});

  auto flatten =
    LayerRepresentation("flatten", {"name=flat", "input_layers=in0"});

  auto fc0_zeroed = LayerRepresentation(
    "fully_connected",
    {"name=fc0", "unit=4", "input_layers=flat", "bias_initializer=ones",
     "weight_initializer=ones", "activation=softmax"});

  auto g = fr.realize(makeGraph({input0, flatten, fc0_zeroed}));

  nntrainer::NetworkGraph ng;

  /// @todo Disable Support Inplace  --> should be support Inplace()
  ng.setMemoryOptimizations(false);

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);

  unsigned int UNIT = 4;
  unsigned int HEIGHT = 2;
  unsigned int WIDTH = 4;
  unsigned int CHANNEL = 3;

  auto weight_data = ng.getLayerNode("fc0")->getWeight(0).getData();
  auto *ptr = const_cast<float *>(weight_data);

  // Set initial Weights
  int count = 0;
  int now_weights_value = 0;

  for (unsigned int h = 0; h < HEIGHT; h++) {
    for (unsigned int w = 0; w < WIDTH; w++) {
      for (unsigned int c = 0; c < CHANNEL; c++) {
        for (unsigned int u = 0; u < UNIT; u++) {

          ptr[count] = now_weights_value;
          count += 1;
        }
        now_weights_value++;
      }
    }
  }

  interpreter.serialize(g, "FC_weight_test2.tflite");

  ng.deallocateTensors();

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("FC_weight_test2.tflite");

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 2, 4}));

  count = 0;
  // Set Tensor with (1,3,4,2)(NCHW) Shape
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 2; h++) {
      for (int w = 0; w < 4; w++) {
        in.setValue(0, c, h, w, count);
        count++;
      }
    }
  }

  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 4, 1}));

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 4, 1}));
  ans.setValue(4325);
  EXPECT_EQ(out, ans);

  if (remove("FC_weight_test2.tflite")) {
    std::cerr << "remove ini "
              << "FC_weight_test2.tflite"
              << "failed, reason: " << strerror(errno);
  }
}

TEST(nntrainerInterpreterTflite, simple_flatten3) {

  nntrainer::TfliteInterpreter interpreter;
  nntrainer::FlattenRealizer fr;

  auto input0 = LayerRepresentation(
    "input", {"name=in0", "input_shape=3:2:4", "flatten=true"});

  auto fc0_zeroed = LayerRepresentation(
    "fully_connected", {"name=fc0", "unit=4", "input_layers=in0",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto fc1_zeroed = LayerRepresentation(
    "fully_connected", {"name=fc1", "unit=1", "input_layers=fc0",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto g = fr.realize(makeGraph({input0, fc0_zeroed, fc1_zeroed}));

  nntrainer::NetworkGraph ng;

  /// @todo Disable Support Inplace  --> should be support Inplace()
  ng.setMemoryOptimizations(false);

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);

  unsigned int UNIT = 4;
  unsigned int HEIGHT = 2;
  unsigned int WIDTH = 4;
  unsigned int CHANNEL = 3;

  auto weight_data = ng.getLayerNode("fc0")->getWeight(0).getData();
  auto *ptr = const_cast<float *>(weight_data);

  // Set initial Weights
  int count = 0;
  int now_weights_value = 0;

  for (unsigned int h = 0; h < HEIGHT; h++) {
    for (unsigned int w = 0; w < WIDTH; w++) {
      for (unsigned int c = 0; c < CHANNEL; c++) {
        for (unsigned int u = 0; u < UNIT; u++) {

          ptr[count] = now_weights_value;
          count += 1;
        }
        now_weights_value++;
      }
    }
  }

  interpreter.serialize(g, "./FC_weight_test3.tflite");

  ng.deallocateTensors();

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("./FC_weight_test3.tflite");

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 2, 4}));

  count = 0;
  // Set Tensor with (1,3,4,2)(NCHW) Shape
  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 2; h++) {
      for (int w = 0; w < 4; w++) {
        in.setValue(0, c, h, w, count);
        count++;
      }
    }
  }

  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 1, 1}));

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 1, 1}));
  ans.setValue(17301);
  EXPECT_EQ(out, ans);

  if (remove("FC_weight_test3.tflite")) {
    std::cerr << "remove ini "
              << "FC_weight_test3.tflite"
              << "failed, reason: " << strerror(errno);
  }
}

/**
 * @brief Construct a new TEST object
 *
 */
TEST(nntrainerInterpreterTflite, flatten_test) {

  nntrainer::TfliteInterpreter interpreter;
  nntrainer::FlattenRealizer fr;

  auto input0 = LayerRepresentation("input", {"name=in0", "input_shape=3:2:4"});

  auto flat = LayerRepresentation("flatten", {"name=flat", "input_layers=in0"});

  auto g = fr.realize(makeGraph({input0, flat}));

  nntrainer::NetworkGraph ng;

  /// @todo Disable Support Inplace  --> should be support Inplace()
  ng.setMemoryOptimizations(false);

  for (auto &node : g) {
    ng.addLayer(node);
  }
  EXPECT_EQ(ng.compile(""), ML_ERROR_NONE);
  EXPECT_EQ(ng.initialize(), ML_ERROR_NONE);

  ng.allocateTensors(nntrainer::ExecutionMode::INFERENCE);
  interpreter.serialize(g, "flatten_test.tflite");
  ng.deallocateTensors();

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("flatten_test.tflite");

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 2, 4}));

  int count = 0;

  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 2; h++) {
      for (int w = 0; w < 4; w++) {
        in.setValue(0, c, h, w, count);
        count++;
      }
    }
  }

  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 1, 24}));

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 1, 24}));
  int ans_array[24] = {0, 8,  16, 1, 9,  17, 2, 10, 18, 3, 11, 19,
                       4, 12, 20, 5, 13, 21, 6, 14, 22, 7, 15, 23};
  for (int i = 0; i < 24; i++) {
    ans.setValue(0, 0, 0, i, ans_array[i]);
  }
  EXPECT_EQ(out, ans);

  if (remove("flatten_test.tflite")) {
    std::cerr << "remove ini "
              << "flatten_test.tflite"
              << "failed, reason: " << strerror(errno);
  }
}

TEST(nntrainerInterpreterTflite, MNIST_OUTPUT_TEST) {

  //  Tensor declare :
  //  { in : NNTrainer & TFLite input,
  //    ans : NNtrainer result,
  //    out : TFlite result }
  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 1, 10}));
  nntrainer::Tensor in(nntrainer::TensorDim({1, 1, 28, 28}));
  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 1, 10}));
  int count = 0;

  for (int h = 0; h < 28; h++) {
    for (int w = 0; w < 28; w++) {
      in.setValue(0, 0, h, w, count);
      count++;
    }
  }

  // 1. Load ini file and Build NN model for export and inference
  nntrainer::NeuralNetwork NN;
  NN.loadFromConfig("./res/test/test_models/models/tf_mnist.ini");
  NN.compile();
  NN.initialize();

  // 2. NNTrainer inference & save result at ans Tensor
  auto output2 = NN.inference({MAKE_SHARED_TENSOR(in)}, false)[0];
  for (int i = 0; i < 10; i++) {
    ans.setValue(0, 0, 0, i, output2->getValue(0, 0, 0, i));
  }

  // 3. export NNtrainer to tensorflow lite
  NN.exports(ml::train::ExportMethods::METHOD_TFLITE, "MNIST_TEST.tflite");

  // 4. load tflite file ans invoke inference
  nntrainer::TfliteInterpreter interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("MNIST_TEST.tflite");

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  // 5. compare results
  EXPECT_EQ(out, ans);

  if (remove("MNIST_TEST.tflite")) {
    std::cerr << "remove "
              << "MNIST_TEST.tflite "
              << "failed, reason: " << strerror(errno);
  }
}

TEST(nntrainerInterpreterTflite, RESNET_OUTPUT_TEST) {

  //  Tensor declare :
  //  { in : NNTrainer & TFLite input,
  //    ans : NNtrainer result,
  //    out : TFlite result }
  nntrainer::Tensor in(nntrainer::TensorDim({1, 3, 32, 32}));
  nntrainer::Tensor out(nntrainer::TensorDim({1, 1, 1, 100}));
  nntrainer::Tensor ans(nntrainer::TensorDim({1, 1, 1, 100}));
  int count = 0;

  for (int c = 0; c < 3; c++) {
    for (int h = 0; h < 32; h++) {
      for (int w = 0; w < 32; w++) {
        in.setValue(0, c, h, w, count);
        count++;
      }
    }
  }

  // 1. Load ini file and Build NN model for export and inference
  nntrainer::NeuralNetwork NN;
  NN.loadFromConfig("./res/test/test_models/models/tf_resnet.ini");
  NN.compile();
  NN.initialize();

  // 2. NNTrainer inference & save result at ans Tensor
  auto output2 = NN.inference({MAKE_SHARED_TENSOR(in)}, false)[0];
  for (int i = 0; i < 100; i++) {
    ans.setValue(0, 0, 0, i, output2->getValue(0, 0, 0, i));
  }

  // 3. export NNtrainer to tensorflow lite
  NN.exports(ml::train::ExportMethods::METHOD_TFLITE, "ResNet_TEST.tflite");

  // 4. load tflite file ans invoke inference
  nntrainer::TfliteInterpreter interpreter;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> tf_interpreter;
  std::unique_ptr<tflite::FlatBufferModel> model =
    tflite::FlatBufferModel::BuildFromFile("ResNet_TEST.tflite");

  EXPECT_NE(model, nullptr);
  tflite::InterpreterBuilder(*model, resolver)(&tf_interpreter);
  EXPECT_NE(tf_interpreter, nullptr);

  EXPECT_EQ(tf_interpreter->AllocateTensors(), kTfLiteOk);

  auto in_indices = tf_interpreter->inputs();
  for (size_t idx = 0; idx < in_indices.size(); idx++) {
    tf_interpreter->tensor(in_indices[idx])->data.raw =
      reinterpret_cast<char *>(in.getData());
  }

  auto out_indices = tf_interpreter->outputs();
  for (size_t idx = 0; idx < out_indices.size(); idx++) {
    tf_interpreter->tensor(out_indices[idx])->data.raw =
      reinterpret_cast<char *>(out.getData());
  }

  int status = tf_interpreter->Invoke();
  EXPECT_EQ(status, TfLiteStatus::kTfLiteOk);

  // 5. compare results
  EXPECT_EQ(out, ans);

  if (remove("ResNet_TEST.tflite")) {
    std::cerr << "remove "
              << "ResNet_TEST.tflite "
              << "failed, reason: " << strerror(errno);
  }
}
