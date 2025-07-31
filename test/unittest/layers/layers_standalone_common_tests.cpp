// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file layer_common_tests.cpp
 * @date 15 June 2021
 * @brief Common test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Debadri Samaddar <s.debadri@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <layers_common_tests.h>

#include <layer_context.h>
#include <layer_devel.h>
#include <nntrainer_error.h>
#include <tensor_dim.h>

constexpr unsigned SAMPLE_TRIES = 10;

TEST_P(LayerSemantics, setProperties_n) {
  /** must not crash */
  EXPECT_THROW(layer->setProperty({"unknown_props=2"}), std::invalid_argument);
}

TEST_P(LayerPropertySemantics, setPropertiesValidInvalidOnly_n) {
  EXPECT_THROW(layer->setProperty(valid_properties), std::invalid_argument);
}

TEST_P(LayerSemantics, gettersValidate_p) {
  std::string type;

  EXPECT_NO_THROW(type = layer->getType());
  EXPECT_GT(type.size(), size_t(0));
  EXPECT_NO_THROW(layer->supportInPlace());
  EXPECT_NO_THROW(layer->requireLabel());
  EXPECT_NO_THROW(layer->supportBackwarding());
}

TEST_P(LayerSemantics, finalizeValidate_p) {
  ml::train::TensorDim in_dim({1, 1, 1, 1});
  std::vector<ml::train::TensorDim> input_dims(num_inputs, in_dim);
  nntrainer::InitLayerContext init_context =
    nntrainer::InitLayerContext(input_dims, {true}, false, "layer");
  EXPECT_EQ(init_context.validate(), true);

  // set necessary properties only
  EXPECT_NO_THROW(layer->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(layer->finalize(init_context));

    for (auto const &spec : init_context.getOutSpecs())
      EXPECT_GT(spec.variable_spec.dim.getDataLen(), size_t(0));
    for (auto const &ws : init_context.getWeightsSpec())
      EXPECT_GT(std::get<0>(ws).getDataLen(), size_t(0));
    for (auto const &ts : init_context.getTensorsSpec())
      EXPECT_GT(std::get<0>(ts).getDataLen(), size_t(0));
  } else {
    EXPECT_THROW(layer->finalize(init_context),
                 nntrainer::exception::not_supported);
  }
}

TEST_P(LayerSemantics, setBatchValidate_p) {
  ml::train::TensorDim in_dim({1, 1, 1, 1});
  std::vector<ml::train::TensorDim> input_dims(num_inputs, in_dim);
  nntrainer::InitLayerContext init_context =
    nntrainer::InitLayerContext(input_dims, {true}, false, "layer");
  EXPECT_EQ(init_context.validate(), true);

  // set necessary properties only
  EXPECT_NO_THROW(layer->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(layer->finalize(init_context));
  } else {
    EXPECT_THROW(layer->finalize(init_context),
                 nntrainer::exception::not_supported);
  }
}

TEST_P(LayerSemantics, exportTo_p) {
  EXPECT_NO_THROW(layer->setProperty(valid_properties));

  nntrainer::Exporter e;
  EXPECT_NO_THROW(
    layer->exportTo(e, ml::train::ExportMethods::METHOD_STRINGVECTOR));
}

#ifdef ENABLE_OPENCL
TEST_P(LayerSemanticsGpu, setProperties_n) {
  /** must not crash */
  EXPECT_THROW(layer->setProperty({"unknown_props=2"}), std::invalid_argument);
}

TEST_P(LayerSemanticsGpu, gettersValidate_p) {
  std::string type;

  EXPECT_NO_THROW(type = layer->getType());
  EXPECT_GT(type.size(), size_t(0));
  EXPECT_NO_THROW(layer->supportInPlace());
  EXPECT_NO_THROW(layer->requireLabel());
  EXPECT_NO_THROW(layer->supportBackwarding());
}

TEST_P(LayerSemanticsGpu, finalizeValidate_p) {
  ml::train::TensorDim in_dim({1, 1, 1, 1});
  std::vector<ml::train::TensorDim> input_dims(num_inputs, in_dim);
  nntrainer::InitLayerContext init_context =
    nntrainer::InitLayerContext(input_dims, {true}, false, "layer");
  EXPECT_EQ(init_context.validate(), true);

  // set necessary properties only
  EXPECT_NO_THROW(layer->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(layer->finalize(init_context));

    for (auto const &spec : init_context.getOutSpecs())
      EXPECT_GT(spec.variable_spec.dim.getDataLen(), size_t(0));
    for (auto const &ws : init_context.getWeightsSpec())
      EXPECT_GT(std::get<0>(ws).getDataLen(), size_t(0));
    for (auto const &ts : init_context.getTensorsSpec())
      EXPECT_GT(std::get<0>(ts).getDataLen(), size_t(0));
  } else {
    EXPECT_THROW(layer->finalize(init_context),
                 nntrainer::exception::not_supported);
  }
}

TEST_P(LayerSemanticsGpu, setBatchValidate_p) {
  ml::train::TensorDim in_dim({1, 1, 1, 1});
  std::vector<ml::train::TensorDim> input_dims(num_inputs, in_dim);
  nntrainer::InitLayerContext init_context =
    nntrainer::InitLayerContext(input_dims, {true}, false, "layer");
  EXPECT_EQ(init_context.validate(), true);

  // set necessary properties only
  EXPECT_NO_THROW(layer->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(layer->finalize(init_context));
  } else {
    EXPECT_THROW(layer->finalize(init_context),
                 nntrainer::exception::not_supported);
  }
}
#endif
