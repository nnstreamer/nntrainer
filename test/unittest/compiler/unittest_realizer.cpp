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

#include <activation_realizer.h>
#include <connection.h>
#include <flatten_realizer.h>
#include <input_realizer.h>
#include <multiout_realizer.h>
#include <previous_input_realizer.h>
#include <realizer.h>
#include <recurrent_realizer.h>
#include <remap_realizer.h>
#include <slice_realizer.h>

#include <compiler_test_util.h>
#include <nntrainer_test_util.h>

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
  LayerRepresentation expected2 = {
    "flatten", {"name=layer1", "input_layers=layer1/flatten_realized"}};

  realizeAndEqual(fr, {input1}, {expected1, expected2});
}

TEST(RecurrentRealizer, recurrent_no_return_sequence_p) {

  RecurrentRealizer r({"unroll_for=3", "return_sequences=false",
                       "recurrent_input=fc_in", "recurrent_output=fc_out"},
                      {"source"}, {"fc_out"});

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc_in", "input_layers=source"}},
    {"fully_connected", {"name=fc_out", "input_layers=fc_in"}}};

  std::vector<LayerRepresentation> expected = {
    {"fully_connected", {"name=fc_in/0", "input_layers=source"}},
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
                       "recurrent_input=lstm", "recurrent_output=fc_out"},
                      {"source"}, {"fc_out"});

  std::vector<LayerRepresentation> before = {
    {"lstm", {"name=lstm", "input_layers=source"}},
    {"fully_connected", {"name=fc_out", "input_layers=lstm"}}};

  std::vector<LayerRepresentation> expected = {
    {"lstm",
     {"name=lstm/0", "input_layers=source", "max_timestep=3", "timestep=0"}},
    {"fully_connected", {"name=fc_out/0", "input_layers=lstm/0"}},
    {"lstm",
     {"name=lstm/1", "input_layers=fc_out/0", "shared_from=lstm/0",
      "max_timestep=3", "timestep=1"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=lstm/1", "shared_from=fc_out/0"}},
    {"lstm",
     {"name=lstm/2", "input_layers=fc_out/1", "shared_from=lstm/0",
      "max_timestep=3", "timestep=2"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=lstm/2", "shared_from=fc_out/0"}},
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
  RemapRealizer r2(
    [](std::string &name, unsigned &_) { name = "scoped/" + name; });

  LayerRepresentation expected2 = {
    "fully_connected",
    {"name=layer1", "flatten=true", "input_layers=scoped/1,scoped/2"}};

  realizeAndEqual(r2, {input1}, {expected2});
}

TEST(SliceRealizer, slice_01_p) {
  /**
   * graph architecture
   *
   * a1  a2
   *  |   |
   * b1   b2    b3
   *  \  /  \  /
   *   c1    c2
   *  / \
   * d1   d2
   */
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a1"}},
    {"fully_connected", {"name=a2"}},
    {"fully_connected", {"name=b1", "input_layers=a1"}},
    {"fully_connected", {"name=b2", "input_layers=a2"}},
    {"fully_connected", {"name=b3"}},
    {"fully_connected", {"name=c1", "input_layers=b1,b2"}},
    {"fully_connected", {"name=c2", "input_layers=b2,b3"}},
    {"fully_connected", {"name=d1", "input_layers=c1"}},
    {"fully_connected", {"name=d2", "input_layers=c1"}},
  };

  /**
   * graph architecture
   * start_layer = a1, b1, b2
   * end_layer = a1, d1, d2
   *
   * a1 (was input port)
   *  |
   * b1   b2 (orphaned)
   *  \  /
   *   c1
   *  / \
   * d1   d2
   */
  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=a1"}},
    {"fully_connected", {"name=b1", "input_layers=a1"}},
    {"fully_connected", {"name=b2", "input_layers=a2"}},
    {"fully_connected", {"name=c1", "input_layers=b1,b2"}},
    {"fully_connected", {"name=d1", "input_layers=c1"}},
    {"fully_connected", {"name=d2", "input_layers=c1"}},
  };

  SliceRealizer r({"a1", "b1", "b2"}, {"a1", "d1", "d2"});

  realizeAndEqual(r, before, after);
}

TEST(SliceRealizer, slice_02_p) {
  /**
   * graph architecture
   *
   * a1----
   *  |   |
   * b1   |
   *  \  /
   *   c1
   */
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a1"}},
    {"fully_connected", {"name=b1", "input_layers=a1"}},
    {"concat", {"name=c1", "input_layers=a1, b1"}},
  };

  /**
   * a1----
   *  |   |
   * b1   |
   *  \  /
   *   c1
   */
  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=a1"}},
    {"fully_connected", {"name=b1", "input_layers=a1"}},
    {"concat", {"name=c1", "input_layers=a1, b1"}},
  };

  SliceRealizer r({"a1"}, {"c1"});

  realizeAndEqual(r, before, after);
}

TEST(InputRealizer, remap_p) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}}, // no input_layers specified
    {"fully_connected",
     {"name=fc2", "input_layers=none1,fc1"}}, // single orphaned node
    {"fully_connected",
     {"name=fc3", "input_layers=none2,fc2,none3"}}, // multi orphaned node
  };

  std::vector<LayerRepresentation> after = {
    {"fully_connected",
     {"name=fc1", "input_layers=in1"}}, // no input_layers specified
    {"fully_connected",
     {"name=fc2", "input_layers=in2,fc1"}}, // single orphaned node
    {"fully_connected",
     {"name=fc3", "input_layers=in3,fc2,in4"}}, // multi orphaned node
  };

  InputRealizer r({"fc1", "fc2", "fc3"}, {"in1", "in2", "in3", "in4"});
  realizeAndEqual(r, before, after);
}

TEST(PreviousInputRealizer, previous_p) {
  { /// realization without identifying custom input
    std::vector<LayerRepresentation> before = {
      {"fully_connected", {"name=fc1", "input_shape=1"}}, // model input
      {"fully_connected", {"name=fc2"}}, // auto connected to fc 1
      {"fully_connected", {"name=fc3", "input_layers=fc1"}},
      {"fully_connected", {"name=fc4"}}, // auto connected to fc 3
    };

    std::vector<LayerRepresentation> after = {
      {"fully_connected", {"name=fc1", "input_shape=1"}},
      {"fully_connected", {"name=fc2", "input_layers=fc1"}},
      {"fully_connected", {"name=fc3", "input_layers=fc1"}},
      {"fully_connected", {"name=fc4", "input_layers=fc3"}},
    };
    PreviousInputRealizer r({});
    realizeAndEqual(r, before, after);
  }
  { /// realization identifying fc1, fc4 is input layer
    std::vector<LayerRepresentation> before = {
      {"fully_connected", {"name=fc1"}}, // will be identified as model input
      {"fully_connected", {"name=fc2"}}, // auto connected to fc 1
      {"fully_connected", {"name=fc3", "input_layers=fc1"}},
      {"fully_connected", {"name=fc4"}}, // will be identified as model input
    };
    std::vector<LayerRepresentation> after = {
      {"fully_connected", {"name=fc1"}},
      {"fully_connected", {"name=fc2", "input_layers=fc1"}},
      {"fully_connected", {"name=fc3", "input_layers=fc1"}},
      {"fully_connected", {"name=fc4"}},
    };
    PreviousInputRealizer r({Connection("fc1"), Connection("fc4")});
    realizeAndEqual(r, before, after);
  }
  { /// intermediate node is auto input
    std::vector<LayerRepresentation> before = {
      {"fully_connected",
       {"name=fc1", "input_layers=fc2"}},                 // intermediate node
      {"fully_connected", {"name=fc2", "input_shape=1"}}, // model input
      {"fully_connected", {"name=fc3"}}, // auto connected to fc3
      {"fully_connected", {"name=fc4"}}, // auto connected to fc 3
    };

    std::vector<LayerRepresentation> after = {
      {"fully_connected", {"name=fc1", "input_layers=fc2"}},
      {"fully_connected", {"name=fc2", "input_shape=1"}},
      {"fully_connected", {"name=fc3", "input_layers=fc2"}},
      {"fully_connected", {"name=fc4", "input_layers=fc3"}},
    };
    PreviousInputRealizer r({});
    realizeAndEqual(r, before, after);
  }
}

TEST(PreviousInputRealizer, user_not_identifying_first_input_n) {
  /// realization without identifying custom input
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}}, // this should be model input but
                                       // nothing has been connected
    {"fully_connected", {"name=fc2"}}, // auto connected to fc 1
    {"fully_connected", {"name=fc3", "input_layers=fc1"}},
    {"fully_connected", {"name=fc4"}}, // auto connected to fc 3
  };
  PreviousInputRealizer r({});
  EXPECT_ANY_THROW(realizeAndEqual(r, before, {}));
}
TEST(MultioutRealizer, multiout_p) {
  { // source has single output, referred multiple times
    std::vector<LayerRepresentation> before = {
      {"fully_connected", {"name=a"}},
      {"fully_connected", {"name=b", "input_layers=a"}},
      {"fully_connected", {"name=c", "input_layers=a"}},
      {"fully_connected", {"name=d", "input_layers=a"}},
    };
    std::vector<LayerRepresentation> after = {
      {"fully_connected", {"name=a"}},
      {"multiout", {"name=a/generated_out_0", "input_layers=a"}},
      {"fully_connected", {"name=b", "input_layers=a/generated_out_0(0)"}},
      {"fully_connected", {"name=c", "input_layers=a/generated_out_0(1)"}},
      {"fully_connected", {"name=d", "input_layers=a/generated_out_0(2)"}},
    };

    MultioutRealizer r;
    realizeAndEqual(r, before, after);
  }

  { // source has single output, all are referred multiple times
    std::vector<LayerRepresentation> before = {
      {"split", {"name=a", "input_shape=1:1:1:2", "axis=3"}},
      {"fully_connected", {"name=b", "input_layers=a"}},
      {"fully_connected", {"name=c", "input_layers=a(0)"}},
      {"fully_connected", {"name=d", "input_layers=a(1)"}},
      {"fully_connected", {"name=e", "input_layers=a(1)"}},
    };
    std::vector<LayerRepresentation> after = {
      {"split", {"name=a", "input_shape=1:1:1:2", "axis=3"}},
      {"multiout", {"name=a/generated_out_0", "input_layers=a(0)"}},
      {"multiout", {"name=a/generated_out_1", "input_layers=a(1)"}},
      {"fully_connected", {"name=b", "input_layers=a/generated_out_0(0)"}},
      {"fully_connected", {"name=c", "input_layers=a/generated_out_0(1)"}},
      {"fully_connected", {"name=d", "input_layers=a/generated_out_1(0)"}},
      {"fully_connected", {"name=e", "input_layers=a/generated_out_1(1)"}},
    };

    MultioutRealizer r;
    realizeAndEqual(r, before, after);
  }

  { // source has single output, some are referred multiple times
    std::vector<LayerRepresentation> before = {
      {"split", {"name=a", "input_shape=1:1:1:2", "axis=3"}},
      {"fully_connected", {"name=b", "input_layers=a(0)"}},
      {"fully_connected", {"name=c", "input_layers=a(0)"}},
      {"fully_connected", {"name=d", "input_layers=a(1)"}},
    };
    std::vector<LayerRepresentation> after = {
      {"split", {"name=a", "input_shape=1:1:1:2", "axis=3"}},
      {"multiout", {"name=a/generated_out_0", "input_layers=a(0)"}},
      {"fully_connected", {"name=b", "input_layers=a/generated_out_0(0)"}},
      {"fully_connected", {"name=c", "input_layers=a/generated_out_0(1)"}},
      {"fully_connected", {"name=d", "input_layers=a(1)"}},
    };

    MultioutRealizer r;
    realizeAndEqual(r, before, after);
  }
}

TEST(MultioutRealizer, multiout_clashing_name_n) {
  std::vector<LayerRepresentation> before = {
    {"split", {"name=a", "input_shape=1:1:1:2", "axis=3"}},
    {"fully_connected", {"name=a", "input_layers=a(0)"}},
  };

  MultioutRealizer r;
  EXPECT_ANY_THROW(realizeAndEqual(r, before, {}));
}

TEST(ActivationRealizer, activation_p) {
  ActivationRealizer ar;

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a"}},
    {"activation",
     {"name=b", "activation=relu", "input_layers=a"}}, /// not realized
    {"fully_connected",
     {"name=c", "input_layers=b", "activation=softmax"}}, // realized
  };

  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=a"}},
    {"activation", {"name=b", "activation=relu", "input_layers=a"}},
    {"fully_connected",
     {"name=c/activation_realized", "input_layers=b", "activation=none"}},
    {"activation",
     {"name=c", "input_layers=c/activation_realized", "activation=softmax"}},
  };

  realizeAndEqual(ar, before, after);
}

TEST(ActivationRealizer, activation_unknown_n) {
  ActivationRealizer ar;

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a"}},
    {"activation",
     {"name=b", "activation=relu", "input_layers=a"}}, /// not realized
    {"fully_connected",
     {"name=c", "input_layers=b", "activation=unknown"}}, // unknown
  };

  EXPECT_ANY_THROW(realizeAndEqual(ar, before, {}));
}