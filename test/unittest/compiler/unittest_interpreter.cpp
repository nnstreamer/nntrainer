// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_interpreter.cpp
 * @date 02 April 2021
 * @brief interpreter test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <functional>
#include <gtest/gtest.h>
#include <memory>

#include <app_context.h>
#include <ini_interpreter.h>
#include <interpreter.h>
#include <layer.h>
#include <layer_node.h>

#ifdef ENABLE_TFLITE_INTERPRETER
#include <tflite_interpreter.h>

#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/model.h>
#endif

#include <nntrainer_test_util.h>

using LayerRepresentation = std::pair<std::string, std::vector<std::string>>;

auto &ac = nntrainer::AppContext::Global();

static std::shared_ptr<nntrainer::GraphRepresentation>
makeGraph(const std::vector<LayerRepresentation> &layer_reps) {
  auto graph = std::make_shared<nntrainer::GraphRepresentation>();

  for (const auto &layer_representation : layer_reps) {
    /// @todo Use unique_ptr here
    std::shared_ptr<nntrainer::LayerNode> layer = createLayerNode(
      ac.createObject<nntrainer::Layer>(layer_representation.first),
      layer_representation.second);
    graph->addLayer(layer);
  }

  return graph;
}

const std::string pathResolver(const std::string &path) {
  return getResPath(path, {"test", "test_models", "models"});
}

auto ini_interpreter =
  std::make_shared<nntrainer::IniGraphInterpreter>(ac, pathResolver);

/**
 * @brief prototypical version of checking graph is equal
 *
 * @param lhs compiled(later, finalized) graph to be compared
 * @param rhs compiled(later, finalized) graph to be compared
 * @retval true graph is equal
 * @retval false graph is not equal
 */
static void graphEqual(const nntrainer::GraphRepresentation &lhs,
                       const nntrainer::GraphRepresentation &rhs) {
  EXPECT_EQ(lhs.size(), rhs.size());

  auto is_node_equal = [](const nntrainer::LayerNode &l,
                          const nntrainer::LayerNode &r) {
    nntrainer::Exporter lhs_export;
    nntrainer::Exporter rhs_export;

    l.exportTo(lhs_export, nntrainer::ExportMethods::METHOD_STRINGVECTOR);
    r.exportTo(rhs_export, nntrainer::ExportMethods::METHOD_STRINGVECTOR);

    /*** fixme, there is one caveat that order matters in this form */
    EXPECT_EQ(
      *lhs_export.getResult<nntrainer::ExportMethods::METHOD_STRINGVECTOR>(),
      *rhs_export.getResult<nntrainer::ExportMethods::METHOD_STRINGVECTOR>());
  };

  if (lhs.size() == rhs.size()) {
    auto lhs_iter = lhs.cbegin();
    auto rhs_iter = rhs.cbegin();
    for (; lhs_iter != lhs.cend(), rhs_iter != rhs.cend();
         lhs_iter++, rhs_iter++) {
      auto lhs = *lhs_iter;
      auto rhs = *rhs_iter;
      is_node_equal(*lhs.get(), *rhs.get());
    }
  }
}

/**
 * @brief nntrainer Interpreter Test setup
 *
 * @note Proposing an evolutional path of current test
 * 1. A reference graph vs given paramter
 * 2. A reference graph vs list of models
 * 3. A reference graph vs (pick two models) a -> b -> a graph, b -> a -> b
 * graph
 */
class nntrainerInterpreterTest
  : public ::testing::TestWithParam<
      std::tuple<std::shared_ptr<nntrainer::GraphRepresentation>, const char *,
                 std::shared_ptr<nntrainer::GraphInterpreter>>> {

protected:
  virtual void SetUp() {
    auto params = GetParam();

    reference = std::get<0>(params);
    file_path = pathResolver(std::get<1>(params));
    interpreter = std::move(std::get<2>(params));
  }

  std::shared_ptr<nntrainer::GraphRepresentation> reference;
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
  std::cout << "testing " << file_path << '\n';

  int status = reference->compile("");
  EXPECT_EQ(status, ML_ERROR_NONE);
  auto g = interpreter->deserialize(file_path);

  /// @todo: change this to something like graph::finalize
  status = g->compile("");
  EXPECT_EQ(status, ML_ERROR_NONE);

  /// @todo: make a proper graph equal
  /// 1. having same number of nodes
  /// 2. layer name is identical (this is too strict though)
  /// 3. attributes of layer is identical
  graphEqual(*g, *reference);
}

/**
 * @brief graph serialize after deserialize, compare if they are the same
 *
 */
TEST_P(nntrainerInterpreterTest, graphSerializeAfterDeserialize) {
  auto g = interpreter->deserialize(file_path);

  auto out_file_path = file_path + ".out";

  /// @todo: change this to something like graph::finalize
  int status = g->compile("");
  EXPECT_EQ(status, ML_ERROR_NONE);
  interpreter->serialize(*g, out_file_path);
  auto new_g = interpreter->deserialize(out_file_path);

  graphEqual(*g, *new_g);

  EXPECT_EQ(remove(out_file_path.c_str()), 0) << strerror(errno);
}

auto fc0 = LayerRepresentation("fully_connected",
                               {"name=fc0", "unit=2", "input_shape=1:1:100"});
auto fc1 = LayerRepresentation("fully_connected", {"name=fc1", "unit=2"});

auto flatten = LayerRepresentation("flatten", {"name=flat"});

#ifdef ENABLE_TFLITE_INTERPRETER
TEST(nntrainerInterpreterTflite, simple_fc) {

  nntrainer::TfliteInterpreter interpreter;

  auto fc0_zeroed = LayerRepresentation(
    "fully_connected", {"name=fc0", "unit=2", "input_shape=1:1:1",
                        "bias_initializer=ones", "weight_initializer=ones"});

  auto fc1_zeroed = LayerRepresentation(
    "fully_connected",
    {"name=fc1", "unit=2", "bias_initializer=ones", "weight_initializer=ones"});

  auto g = makeGraph({fc0_zeroed, fc1_zeroed});
  EXPECT_EQ(g->compile(""), ML_ERROR_NONE);
  EXPECT_EQ(g->initialize(), ML_ERROR_NONE);

  g->initializeWeights();
  g->allocateWeights();
  interpreter.serialize(*g, "test.tflite");
  g->deallocateTensors();

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
  ans.setValue(7.0f);

  EXPECT_EQ(out, ans);

  if (remove("test.tflite")) {
    std::cerr << "remove ini "
              << "test.tflite"
              << "failed, reason: " << strerror(errno);
  }
}
#endif
/**
 * @brief make ini test case from given parameter
 */
static std::tuple<std::shared_ptr<nntrainer::GraphRepresentation>, const char *,
                  std::shared_ptr<nntrainer::GraphInterpreter>>
mkTc(std::shared_ptr<nntrainer::GraphRepresentation> graph, const char *file,
     std::shared_ptr<nntrainer::GraphInterpreter> interpreter) {
  return std::make_tuple(graph, file, interpreter);
}

// clang-format off
INSTANTIATE_TEST_CASE_P(nntrainerAutoInterpreterTest, nntrainerInterpreterTest,
                        ::testing::Values(
  mkTc(makeGraph({fc0, flatten}), "simple_fc.ini", ini_interpreter),
  mkTc(makeGraph({fc0, flatten}), "simple_fc_backbone.ini", ini_interpreter)
));
// clang-format on
