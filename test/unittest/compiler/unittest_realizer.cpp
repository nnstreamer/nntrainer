// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file unittest_realizer.h
 * @date 09 October 2021
 * @brief NNTrainer graph realizer related tests
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */

#include <gtest/gtest.h>

#include <vector>

#include <flatten_realizer.h>
#include <realizer.h>
#include <recurrent_realizer.h>
#include <remap_realizer.h>

#include <compiler_test_util.h>

using namespace nntrainer;

/**
 * @brief check realize and equal
 *
 * @param realizer realizer to use
 * @param input input
 * @param expected expected output
 */
static void realizeAndEqual(GraphRealizer &realizer,
                            const std::vector<LayerRepresentation> &input,
                            const std::vector<LayerRepresentation> &expected) {
  auto processed = realizer.realize(makeGraph(input));
  auto expected_graph = makeGraph(expected);
  graphEqual(processed, expected_graph);
}

TEST(FlattenRealizer, flatten_p) {
  FlattenRealizer fr;

  LayerRepresentation input1 = {"fully_connected",
                                {"name=layer1", "flatten=true"}};
  LayerRepresentation expected1 = {"fully_connected",
                                   {"name=layer1/flatten_realized"}};
  LayerRepresentation expected2 = {"flatten", {"name=layer1"}};

  realizeAndEqual(fr, {input1}, {expected1, expected2});
}

TEST(RecurrentRealizer, recurrent_p) {

  RecurrentRealizer r({"unroll_for=3", "return_sequences=true",
                       "input_layers=fc1", "output_layers=fc2",
                       "recurrent_input=fc_1", "recurrent_output=fc_2"},
                      {"outter_input"});

  LayerRepresentation input1 = {
    "fully_connected", {"name=fc1", "flatten=true", "input_layers=fc1,fc1"}};

  LayerRepresentation expected1 = {"fully_connected",
                                   {"name=outter_input", "flatten=true",
                                    "input_layers=outter_input,outter_input"}};

  realizeAndEqual(r, {input1}, {expected1});
}

TEST(RemapRealizer, remap_p) {

  RemapRealizer r([](std::string &name) { name = "scoped/" + name; });

  LayerRepresentation input1 = {
    "fully_connected", {"name=layer1", "flatten=true", "input_layers=1,2"}};

  LayerRepresentation expected1 = {
    "fully_connected",
    {"name=scoped/layer1", "flatten=true", "input_layers=scoped/1,scoped/2"}};

  realizeAndEqual(r, {input1}, {expected1});
}
