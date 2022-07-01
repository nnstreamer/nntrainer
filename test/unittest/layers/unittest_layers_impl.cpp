// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_layer_impl.cpp
 * @date 16 June 2021
 * @brief Layer Impl test
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <tuple>

#include <gtest/gtest.h>
#include <layers_common_tests.h>

#include <layer_context.h>
#include <layer_impl.h>
#include <string>

namespace {

using namespace nntrainer;
/**
 * @brief Minimal implementation of layer impl to test layer impl itself
 *
 */
class MockLayer final : public LayerImpl {
public:
  ~MockLayer() = default;

  inline static const std::string type = "mock_";
  const std::string getType() const override { return type; }
  void finalize(InitLayerContext &context) override {
    context.setOutputDimensions(context.getInputDimensions());
  }
  void forwarding(RunLayerContext &context, bool training = true) override {
    /** do nothing */
  }
  void calcDerivative(RunLayerContext &context) override { /** do nothing */
  }

  void setProperty(const std::vector<std::string> &values) override {
    LayerImpl::setProperty(values);
  }

  void exportTo(Exporter &exporter,
                const ml::train::ExportMethods &method) const override {
    LayerImpl::exportTo(exporter, method);
  }

  bool supportBackwarding() const override { return true; }
};
} // namespace

auto semantic_tc = LayerSemanticsParamType(nntrainer::createLayer<MockLayer>,
                                           MockLayer::type, {}, 0, false, 1);
GTEST_PARAMETER_TEST(LayerImpl, LayerSemantics, ::testing::Values(semantic_tc));

// GTEST_PARAMETER_TEST(
//   LayerImpl, LayerGoldenTest,
//   ::testing::Values(
//     "test") /**< format of type, properties, num_batch, golden file name */);
