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

TEST(RecurrentRealizer, recurrent_no_return_sequence_p) {

  RecurrentRealizer r({"unroll_for=3", "return_sequences=false",
                       "input_layers=initial_source", "output_layers=fc_out",
                       "recurrent_input=fc_in", "recurrent_output=fc_out"},
                      {"out_source"});

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc_in", "input_layers=initial_source"}},
    {"fully_connected", {"name=fc_out", "input_layers=fc_in"}}};

  std::vector<LayerRepresentation> expected = {
    {"fully_connected", {"name=fc_in/0", "input_layers=out_source"}},
    {"fully_connected", {"name=fc_out/0", "input_layers=fc_in/0"}},
    {"fully_connected",
     {"name=fc_in/1", "input_layers=fc_out/0", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=fc_in/1", "shared_from=fc_out/0"}},
    {"fully_connected",
     {"name=fc_in", "input_layers=fc_out/1", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out", "input_layers=fc_in", "shared_from=fc_out/0"}},
  };

  realizeAndEqual(r, before, expected);
}

TEST(RecurrentRealizer, recurrent_return_sequence_p) {

  RecurrentRealizer r({"unroll_for=3", "return_sequences=true",
                       "input_layers=initial_source", "output_layers=fc_out",
                       "recurrent_input=fc_in", "recurrent_output=fc_out"},
                      {"out_source"});

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc_in", "input_layers=initial_source"}},
    {"fully_connected", {"name=fc_out", "input_layers=fc_in"}}};

  std::vector<LayerRepresentation> expected = {
    {"fully_connected", {"name=fc_in/0", "input_layers=out_source"}},
    {"fully_connected", {"name=fc_out/0", "input_layers=fc_in/0"}},
    {"fully_connected",
     {"name=fc_in/1", "input_layers=fc_out/0", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=fc_in/1", "shared_from=fc_out/0"}},
    {"fully_connected",
     {"name=fc_in/2", "input_layers=fc_out/1", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=fc_in/2", "shared_from=fc_out/0"}},
    {"concat", {"name=fc_out", "input_layers=fc_out/0,fc_out/1,fc_out/2"}},
  };

  realizeAndEqual(r, before, expected);
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
