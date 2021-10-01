// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2020 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file layer_common_tests.cpp
 * @date 15 June 2021
 * @brief Common test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <layers_common_tests.h>

#include <layer_devel.h>

constexpr unsigned SAMPLE_TRIES = 10;

TEST_P(LayerSemantics, DISABLED_setProperties_p) {
  /// @todo check if setProperties does not collide with layerNode designated
  /// properties
  EXPECT_EQ(1, 1); /**< no assert tc from TCM, this is disabled test */
}

TEST_P(LayerSemantics, setProperties_n) {
  /** must not crash */
  EXPECT_THROW(layer->setProperty({"unknown_props=2"}), std::invalid_argument);
}

TEST_P(LayerSemantics, DISABLED_setPropertiesValidWithInvalid_n) {
  EXPECT_EQ(1, 1); /**< no assert tc from TCM, this is disabled test */
}

TEST_P(LayerSemantics, DISABLED_setPropertiesValidInvalidOnly_n) {
  EXPECT_EQ(1, 1); /**< no assert tc from TCM, this is disabled test */
}

TEST_P(LayerSemantics, finalizeValidate_p) {
  nntrainer::TensorDim in_dim({1, 1, 1, 1});
  std::vector<nntrainer::TensorDim> input_dims(num_inputs, in_dim);
  nntrainer::InitLayerContext init_context =
    nntrainer::InitLayerContext(input_dims, 1, "layer");
  EXPECT_EQ(init_context.validate(), true);

  // set necessary properties only
  EXPECT_NO_THROW(layer->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(layer->finalize(init_context));

    EXPECT_EQ(init_context.getOutputDimensions().size(),
              init_context.getNumOutputs());

    for (auto const &dim : init_context.getOutputDimensions())
      EXPECT_GT(dim.getDataLen(), size_t(0));
    for (auto const &ws : init_context.getWeightsSpec())
      EXPECT_GT(std::get<0>(ws).getDataLen(), size_t(0));
    for (auto const &ts : init_context.getTensorsSpec())
      EXPECT_GT(std::get<0>(ts).getDataLen(), size_t(0));
  } else {
    EXPECT_THROW(layer->finalize(init_context),
                 nntrainer::exception::not_supported);
  }
}

TEST_P(LayerSemantics, getTypeValidate_p) {
  std::string type;

  EXPECT_NO_THROW(type = layer->getType());
  EXPECT_GT(type.size(), size_t(0));
}

TEST_P(LayerSemantics, gettersValidate_p) {
  EXPECT_NO_THROW(layer->supportInPlace());
  EXPECT_NO_THROW(layer->requireLabel());
  EXPECT_NO_THROW(layer->supportBackwarding());
}

TEST_P(LayerSemantics, setBatchValidate_p) {
  nntrainer::TensorDim in_dim({1, 1, 1, 1});
  std::vector<nntrainer::TensorDim> input_dims(num_inputs, in_dim);
  nntrainer::InitLayerContext init_context =
    nntrainer::InitLayerContext(input_dims, 1, "layer");
  EXPECT_EQ(init_context.validate(), true);

  // set necessary properties only
  EXPECT_NO_THROW(layer->setProperty(valid_properties));

  if (!must_fail) {
    EXPECT_NO_THROW(layer->finalize(init_context));
    EXPECT_NO_THROW(layer->setBatch(
      init_context, init_context.getInputDimensions()[0].batch() + 10));
  } else {
    EXPECT_THROW(layer->finalize(init_context),
                 nntrainer::exception::not_supported);
  }
}
