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
#include <bn_realizer.h>
#include <connection.h>
#include <flatten_realizer.h>
#include <input_realizer.h>
#include <loss_realizer.h>
#include <multiout_realizer.h>
#include <nntrainer_error.h>
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

static void compileAndRealizeAndEqual(
  GraphRealizer &realizer,
  std::vector<std::unique_ptr<GraphRealizer>> &realizers,
  const std::vector<LayerRepresentation> &input,
  const std::vector<LayerRepresentation> &expected) {
  auto processed = realizer.realize(makeCompiledGraph(input, realizers));
  std::vector<std::unique_ptr<nntrainer::GraphRealizer>> defalute_realizers;
  auto expected_graph = makeCompiledGraph(expected, realizers);

  graphEqual(processed, expected_graph);
}

TEST(FlattenRealizer, flatten_p) {
  FlattenRealizer fr;

  LayerRepresentation input1 = {
    "fully_connected",
    {"name=layer1", "flatten=true"},
  };
  LayerRepresentation expected1 = {"fully_connected", {"name=layer1"}};
  LayerRepresentation expected2 = {
    "flatten",
    {"name=layer1/flatten_realized", "input_layers=layer1"},
  };

  EXPECT_NO_THROW(realizeAndEqual(fr, {input1}, {expected1, expected2}));
}

TEST(RecurrentRealizer, recurrent_no_return_sequence_p) {
  using C = Connection;

  RecurrentRealizer r(
    {"unroll_for=3", "recurrent_input=fc_in", "recurrent_output=fc_out"},
    {C("source")}, {C("fc_out")});

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc_in", "input_layers=source"}},
    {"fully_connected", {"name=fc_out", "input_layers=fc_in"}}};

  std::vector<LayerRepresentation> expected = {
    /// t - 0
    {"fully_connected",
     {"name=fc_in/0", "input_layers=source", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/0", "input_layers=fc_in/0", "shared_from=fc_out/0"}},

    /// t - 1
    {"fully_connected",
     {"name=fc_in/1", "input_layers=fc_out/0", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=fc_in/1", "shared_from=fc_out/0"}},

    /// t - 2
    {"fully_connected",
     {"name=fc_in/2", "input_layers=fc_out/1", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=fc_in/2", "shared_from=fc_out/0"}},

    /// mapping
    {"identity", {"name=fc_out", "input_layers=fc_out/2"}},
  };

  EXPECT_NO_THROW(realizeAndEqual(r, before, expected));
}

TEST(RecurrentRealizer, recurrent_input_is_sequence_p) {
  using C = Connection;

  RecurrentRealizer r({"unroll_for=3", "input_is_sequence=source",
                       "recurrent_input=fc_out", "recurrent_output=fc_out"},
                      {C("source")}, {C("fc_out")});

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc_in", "input_layers=source"}},
    {"fully_connected", {"name=fc_out", "input_layers=fc_in"}}};

  std::vector<LayerRepresentation> expected = {
    /// t - 0
    {"fully_connected",
     {"name=fc_in/0", "input_layers=source/0", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/0", "input_layers=fc_in/0", "shared_from=fc_out/0"}},

    /// t - 1
    {"fully_connected",
     {"name=fc_in/1", "input_layers=source/1", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=fc_out/0", "shared_from=fc_out/0"}},

    /// t - 2
    {"fully_connected",
     {"name=fc_in/2", "input_layers=source/2", "shared_from=fc_in/0"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=fc_out/1", "shared_from=fc_out/0"}},

    /// mapping
    {"identity", {"name=fc_out", "input_layers=fc_out/2"}},
  };

  EXPECT_NO_THROW(realizeAndEqual(r, before, expected));
}

TEST(RecurrentRealizer, recurrent_return_sequence_single_p) {
  using C = Connection;
  RecurrentRealizer r({"unroll_for=3", "as_sequence=fc_out",
                       "recurrent_input=lstmcell", "recurrent_output=fc_out"},
                      {C("source")}, {C("fc_out")});

  std::vector<LayerRepresentation> before = {
    {"lstmcell", {"name=lstmcell", "input_layers=source"}},
    {"fully_connected", {"name=fc_out", "input_layers=lstmcell"}}};

  std::vector<LayerRepresentation> expected = {
    /// t - 0
    {"lstmcell",
     {"name=lstmcell/0", "input_layers=source", "shared_from=lstmcell/0"}},
    {"fully_connected",
     {"name=fc_out/0", "input_layers=lstmcell/0", "shared_from=fc_out/0"}},

    /// t - 1
    {"lstmcell",
     {"name=lstmcell/1", "input_layers=fc_out/0", "shared_from=lstmcell/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=lstmcell/1", "shared_from=fc_out/0"}},

    /// t - 2
    {"lstmcell",
     {"name=lstmcell/2", "input_layers=fc_out/1", "shared_from=lstmcell/0"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=lstmcell/2", "shared_from=fc_out/0"}},

    /// mapping
    {"concat",
     {"name=fc_out/concat_0", "input_layers=fc_out/0,fc_out/1,fc_out/2"}},
    {"identity", {"name=fc_out", "input_layers=fc_out/concat_0"}},
  };

  EXPECT_NO_THROW(realizeAndEqual(r, before, expected));
}

TEST(RecurrentRealizer, recurrent_multi_inout_return_seq_p) {
  using C = Connection;
  RecurrentRealizer r(
    {
      "unroll_for=3",
      "as_sequence=fc_out",
      "recurrent_input=lstmcell,add(2)",
      "recurrent_output=fc_out,split(1)",
    },
    {C("source"), C("source2"), C("source3")}, {C("fc_out")});

  /// @note for below graph,
  /// 1. fc_out feds back to lstmcell
  /// 2. ouput_dummy feds back to source2_dummy
  /// ========================================================
  /// lstmcell        -------- addition - split ---- fc_out (to_lstmcell)
  /// source2_dummy   --/                  \----- (to addition 3)
  std::vector<LayerRepresentation> before = {
    {"lstmcell", {"name=lstmcell", "input_layers=source"}},
    {"addition", {"name=add", "input_layers=lstmcell,source2,source3"}},
    {"split", {"name=split", "input_layers=add"}},
    {"fully_connected", {"name=fc_out", "input_layers=split(0)"}},
  };

  std::vector<LayerRepresentation> expected = {
    /// timestep 0
    {"lstmcell",
     {"name=lstmcell/0", "input_layers=source", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/0", "input_layers=lstmcell/0,source2,source3",
      "shared_from=add/0"}},
    {"split", {"name=split/0", "input_layers=add/0", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/0", "input_layers=split/0(0)", "shared_from=fc_out/0"}},

    /// timestep 1
    {"lstmcell",
     {"name=lstmcell/1", "input_layers=fc_out/0", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/1", "input_layers=lstmcell/1,source2,split/0(1)",
      "shared_from=add/0"}},
    {"split", {"name=split/1", "input_layers=add/1", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=split/1(0)", "shared_from=fc_out/0"}},

    /// timestep 2
    {"lstmcell",
     {"name=lstmcell/2", "input_layers=fc_out/1", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/2", "input_layers=lstmcell/2,source2,split/1(1)",
      "shared_from=add/0"}},
    {"split", {"name=split/2", "input_layers=add/2", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=split/2(0)", "shared_from=fc_out/0"}},

    /// mapping
    {"concat",
     {"name=fc_out/concat_0", "input_layers=fc_out/0,fc_out/1,fc_out/2"}},
    {"identity", {"name=fc_out", "input_layers=fc_out/concat_0"}},
  };

  EXPECT_NO_THROW(realizeAndEqual(r, before, expected));
}

TEST(RecurrentRealizer, recurrent_multi_inout_using_connection_p) {
  using C = Connection;
  RecurrentRealizer r(
    {
      "unroll_for=3",
      "recurrent_input=lstmcell,add(2)",
      "recurrent_output=fc_out,split(1)",
    },
    {C("source"), C("source2"), C("source3")}, {C("fc_out")});

  /// @note for below graph,
  /// 1. fc_out feds back to lstmcell
  /// 2. ouput_dummy feds back to source2_dummy
  /// ========================================================
  /// lstmcell        -------- addition - split ---- fc_out (to_lstmcell)
  /// source2_dummy   --/                  \----- (to addition 3)
  std::vector<LayerRepresentation> before = {
    {"lstmcell", {"name=lstmcell", "input_layers=source"}},
    {"addition", {"name=add", "input_layers=lstmcell,source2,source3"}},
    {"split", {"name=split", "input_layers=add"}},
    {"fully_connected", {"name=fc_out", "input_layers=split(0)"}},
  };

  std::vector<LayerRepresentation> expected = {
    /// timestep 0
    {"lstmcell",
     {"name=lstmcell/0", "input_layers=source", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/0", "input_layers=lstmcell/0,source2,source3",
      "shared_from=add/0"}},
    {"split", {"name=split/0", "input_layers=add/0", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/0", "input_layers=split/0(0)", "shared_from=fc_out/0"}},

    /// timestep 1
    {"lstmcell",
     {"name=lstmcell/1", "input_layers=fc_out/0", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/1", "input_layers=lstmcell/1,source2,split/0(1)",
      "shared_from=add/0"}},
    {"split", {"name=split/1", "input_layers=add/1", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=split/1(0)", "shared_from=fc_out/0"}},

    /// timestep 2
    {"lstmcell",
     {"name=lstmcell/2", "input_layers=fc_out/1", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/2", "input_layers=lstmcell/2,source2,split/1(1)",
      "shared_from=add/0"}},
    {"split", {"name=split/2", "input_layers=add/2", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=split/2(0)", "shared_from=fc_out/0"}},

    /// mapping
    {"identity", {"name=fc_out", "input_layers=fc_out/2"}},
  };

  EXPECT_NO_THROW(realizeAndEqual(r, before, expected));
}

TEST(RecurrentRealizer, recurrent_multi_inout_multi_connection_end_p) {
  using C = Connection;
  RecurrentRealizer r(
    {
      "unroll_for=3",
      "recurrent_input=lstmcell,add(2)",
      "recurrent_output=fc_out,split(1)",
      "as_sequence=split(1)",
    },
    {
      C("source"),
      C("source2"),
      C("source3"),
    },
    {
      C("split(0)"),
      C("split(1)"),
    });

  /// @note for below graph,
  /// 1. fc_out feds back to lstmcell
  /// 2. ouput_dummy feds back to source2_dummy
  /// ========================================================
  /// lstmcell        -------- addition - split ---- fc_out (to_lstmcell)
  /// source2_dummy   --/                  \----- (to addition 3)
  std::vector<LayerRepresentation> before = {
    {"lstmcell", {"name=lstmcell", "input_layers=source"}},
    {"addition", {"name=add", "input_layers=lstmcell,source2,source3"}},
    {"split", {"name=split", "input_layers=add"}},
    {"fully_connected", {"name=fc_out", "input_layers=split(0)"}},
  };

  std::vector<LayerRepresentation> expected = {
    /// timestep 0
    {"lstmcell",
     {"name=lstmcell/0", "input_layers=source", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/0", "input_layers=lstmcell/0,source2,source3",
      "shared_from=add/0"}},
    {"split", {"name=split/0", "input_layers=add/0", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/0", "input_layers=split/0(0)", "shared_from=fc_out/0"}},

    /// timestep 1
    {"lstmcell",
     {"name=lstmcell/1", "input_layers=fc_out/0", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/1", "input_layers=lstmcell/1,source2,split/0(1)",
      "shared_from=add/0"}},
    {"split", {"name=split/1", "input_layers=add/1", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/1", "input_layers=split/1(0)", "shared_from=fc_out/0"}},

    /// timestep 2
    {"lstmcell",
     {"name=lstmcell/2", "input_layers=fc_out/1", "shared_from=lstmcell/0"}},
    {"addition",
     {"name=add/2", "input_layers=lstmcell/2,source2,split/1(1)",
      "shared_from=add/0"}},
    {"split", {"name=split/2", "input_layers=add/2", "shared_from=split/0"}},
    {"fully_connected",
     {"name=fc_out/2", "input_layers=split/2(0)", "shared_from=fc_out/0"}},

    /// mapping
    {"concat",
     {"name=split/concat_1", "input_layers=split/0(1),split/1(1),split/2(1)"}},
    {"identity", {"name=split", "input_layers=split/2(0),split/concat_1"}},
  };

  EXPECT_NO_THROW(realizeAndEqual(r, before, expected));
}

TEST(RemapRealizer, remap_01_n) {
  std::function<void(std::string &, unsigned &)> remap_connection_function =
    nullptr;

  EXPECT_THROW(RemapRealizer r(remap_connection_function),
               std::invalid_argument);
}

TEST(RemapRealizer, remap_02_n) {
  std::function<void(std::string &)> remap_function = nullptr;

  EXPECT_THROW(RemapRealizer r(remap_function), std::invalid_argument);
}

TEST(RemapRealizer, remap_03_n) {
  auto model_graph = NetworkGraph();
  LayerRepresentation input = {"input", {"name=layer1", "input_shape=1:1:1"}};

  auto graph = makeGraph({input});
  EXPECT_NO_THROW(model_graph.addLayer(graph[0]));
  EXPECT_EQ(model_graph.compile("mse"), ML_ERROR_NONE);
  EXPECT_NO_THROW(model_graph.finalizeContext(graph[0], {}));

  RemapRealizer r([](std::string &name) { name = "scoped/" + name; });
  EXPECT_THROW(r.realize(graph), std::invalid_argument);
}

TEST(RemapRealizer, remap_04_n) {
  auto model_graph = NetworkGraph();
  LayerRepresentation input = {"input", {"name=layer1", "input_shape=1:1:1"}};

  auto graph = makeGraph({input});
  EXPECT_NO_THROW(model_graph.addLayer(graph[0]));
  EXPECT_EQ(model_graph.compile("mse"), ML_ERROR_NONE);
  EXPECT_NO_THROW(model_graph.finalizeContext(graph[0], {}));

  RemapRealizer r(
    [](std::string &name, unsigned &_) { name = "scoped/" + name; });
  EXPECT_THROW(r.realize(graph), std::invalid_argument);
}

TEST(RemapRealizer, remap_05_p) {
  LayerRepresentation input1 = {
    "fully_connected", {"name=layer1", "flatten=true", "input_layers=1,2"}};

  {
    RemapRealizer r([](std::string &name) { name = "scoped/" + name; });

    LayerRepresentation expected1 = {
      "fully_connected",
      {"name=scoped/layer1", "flatten=true", "input_layers=scoped/1,scoped/2"}};

    EXPECT_NO_THROW(realizeAndEqual(r, {input1}, {expected1}));
  }
  {
    RemapRealizer r2(
      [](std::string &name, unsigned &_) { name = "scoped/" + name; });

    LayerRepresentation expected2 = {
      "fully_connected",
      {"name=layer1", "flatten=true", "input_layers=scoped/1,scoped/2"}};

    EXPECT_NO_THROW(realizeAndEqual(r2, {input1}, {expected2}));
  }
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

  using C = Connection;
  SliceRealizer r(
    {
      C("a1"),
      C("b1"),
      C("b2"),
    },
    {
      C("a1"),
      C("d1"),
      C("d2"),
    });

  EXPECT_NO_THROW(realizeAndEqual(r, before, after));
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

  SliceRealizer r({Connection("a1")}, {Connection("c1")});

  EXPECT_NO_THROW(realizeAndEqual(r, before, after));
}

TEST(SliceRealizer, slice_03_n) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a1", "input_layers=a2"}},
    {"fully_connected", {"name=a2", "input_layers=a1"}},
  };

  std::vector<LayerRepresentation> after = {};

  using C = Connection;
  SliceRealizer r({}, {C("a2")});

  EXPECT_THROW(realizeAndEqual(r, before, after), std::runtime_error);
}

TEST(SliceRealizer, slice_04_n) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a1", "input_layers=a2"}},
    {"fully_connected", {"name=a2", "input_layers=a1"}},
  };

  std::vector<LayerRepresentation> after = {};

  using C = Connection;
  SliceRealizer r({C("a1")}, {});

  EXPECT_THROW(realizeAndEqual(r, before, after), std::runtime_error);
}

TEST(SliceRealizer, slice_05_n) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a1"}},
    {"fully_connected", {"name=a2", "input_layers=a1"}},
  };

  std::vector<LayerRepresentation> after = {};

  using C = Connection;
  SliceRealizer r({C("a2")}, {C("a1")});

  EXPECT_THROW(realizeAndEqual(r, before, after), std::invalid_argument);
}

TEST(InputRealizer, input_p) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=none1,fc1"}},
    {"fully_connected", {"name=fc3", "input_layers=none2,fc2,none3"}},
  };

  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=fc1", "input_layers=in1(0)"}},
    {"fully_connected", {"name=fc2", "input_layers=in2,fc1"}},
    {"fully_connected", {"name=fc3", "input_layers=in3(3),fc2,in4"}},
  };

  using C = Connection;
  InputRealizer r(
    {
      C("fc1"),
      C("fc2(0)"),
      C("fc3(0)"),
      C("fc3(2)"),
    },
    {
      C("in1(0)"),
      C("in2"),
      C("in3(3)"),
      C("in4"),
    });
  EXPECT_NO_THROW(realizeAndEqual(r, before, after));
}

TEST(InputRealizer, input_start_num_not_match_n) {
  using C = Connection;
  EXPECT_ANY_THROW(InputRealizer r(
    {
      C("fc1"),
    },
    {
      C("in1(0)"),
      C("in2"),
      C("in3(3)"),
      C("in4"),
    }));
}

TEST(InputRealizer, start_empty_conn_not_defined_n) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=none1,fc1"}},
    {"fully_connected", {"name=fc3", "input_layers=none2,fc2,none3"}},
  };

  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=fc1", "input_layers=in1(0)"}},
    {"fully_connected", {"name=fc2", "input_layers=in2,fc1"}},
    {"fully_connected", {"name=fc3", "input_layers=in3(3),fc2,in4"}},
  };

  using C = Connection;
  InputRealizer r(
    {
      C("fc1(2)"), /**< connection not defined, although fc1(0) is allowed */
      C("fc2(0)"),
      C("fc3(0)"),
      C("fc3(2)"),
    },
    {
      C("in1(0)"),
      C("in2"),
      C("in3(3)"),
      C("in4"),
    });
  EXPECT_ANY_THROW(realizeAndEqual(r, before, after));
}

TEST(InputRealizer, intermediate_conn_not_defined_n) {
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=none1,fc1"}},
    {"fully_connected", {"name=fc3", "input_layers=none2,fc2,none3"}},
  };

  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=fc1", "input_layers=in1(0)"}},
    {"fully_connected", {"name=fc2", "input_layers=in2,fc1"}},
    {"fully_connected", {"name=fc3", "input_layers=in3(3),fc2,in4"}},
  };

  using C = Connection;
  InputRealizer r(
    {
      C("fc1"),
      C("fc2(4)"), /**< connection not defined */
      C("fc3(0)"),
      C("fc3(2)"),
    },
    {
      C("in1(0)"),
      C("in2"),
      C("in3(3)"),
      C("in4"),
    });
  EXPECT_ANY_THROW(realizeAndEqual(r, before, after));
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
    EXPECT_NO_THROW(realizeAndEqual(r, before, after));
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
    EXPECT_NO_THROW(realizeAndEqual(r, before, after));
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
    EXPECT_NO_THROW(realizeAndEqual(r, before, after));
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
    EXPECT_NO_THROW(realizeAndEqual(r, before, after));
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
    EXPECT_NO_THROW(realizeAndEqual(r, before, after));
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
    EXPECT_NO_THROW(realizeAndEqual(r, before, after));
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
  ActivationRealizer r;

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
    {"fully_connected", {"name=c", "input_layers=b", "activation=none"}},
    {"activation",
     {"name=c/activation_realized", "input_layers=c", "activation=softmax"}},
  };

  EXPECT_NO_THROW(realizeAndEqual(r, before, after));
}

TEST(ActivationRealizer, activation_unknown_n) {
  ActivationRealizer r;

  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=a"}},
    {"activation",
     {"name=b", "activation=relu", "input_layers=a"}}, /// not realized
    {"fully_connected",
     {"name=c", "input_layers=b", "activation=unknown"}}, // unknown
  };

  EXPECT_ANY_THROW(realizeAndEqual(r, before, {}));
}

TEST(BnRealizer, bn_realizer_p) {
  /// realization without identifying custom input
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}},
    {"batch_normalization", {"name=bn1", "input_layers=fc1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=bn1"}},
    {"fully_connected", {"name=fc2", "input_layers=ac1"}},
    {"batch_normalization", {"name=bn2", "input_layers=fc2"}},
    {"activation", {"name=ac2", "activation=relu", "input_layers=bn2"}},
    {"fully_connected", {"name=fc3", "input_layers=ac2"}},
  };
  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=fc1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=ac1"}},
    {"activation", {"name=ac2", "activation=relu", "input_layers=fc2"}},
    {"fully_connected", {"name=fc3", "input_layers=ac2"}},
  };
  BnRealizer r;
  std::vector<std::unique_ptr<nntrainer::GraphRealizer>> realizers;
  EXPECT_NO_THROW(compileAndRealizeAndEqual(r, realizers, before, after));
}

TEST(BnRealizer, bn_realizer_resblock_p) {
  std::vector<LayerRepresentation> before = {
    {"input", {"name=input0"}},
    {"conv2d", {"name=conv0", "kernel_size=3,3", "input_layers=input0"}},
    {"batch_normalization", {"name=first_bn", "input_layers=conv0"}},
    {"activation", {"name=ac0", "activation=relu", "input_layers=first_bn"}},
    {"conv2d", {"name=a1", "kernel_size=3,3", "input_layers=ac0"}},
    {"batch_normalization", {"name=bn1", "input_layers=a1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=bn1"}},
    {"conv2d", {"name=a2", "kernel_size=3,3", "input_layers=ac1"}},
    {"conv2d", {"name=b1", "kernel_size=3,3", "input_layers=ac0"}},
    {"addition", {"name=c1", "input_layers=a2,b1"}},
    {"batch_normalization", {"name=bn2", "input_layers=c1"}},
    {"activation", {"name=ac2", "activation=relu", "input_layers=bn2"}},
    {"fully_connected", {"name=fc3", "input_layers=ac2"}},
  };
  std::vector<LayerRepresentation> after = {
    {"input", {"name=input0"}},
    {"conv2d", {"name=conv0", "kernel_size=3,3", "input_layers=input0"}},
    {"activation", {"name=ac0", "activation=relu", "input_layers=conv0"}},
    {"conv2d", {"name=a1", "kernel_size=3,3", "input_layers=ac0"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=a1"}},
    {"conv2d", {"name=a2", "kernel_size=3,3", "input_layers=ac1"}},
    {"conv2d", {"name=b1", "kernel_size=3,3", "input_layers=ac0"}},
    {"addition", {"name=c1", "input_layers=a2,b1"}},
    {"activation", {"name=ac2", "activation=relu", "input_layers=c1"}},
    {"fully_connected", {"name=fc3", "input_layers=ac2"}},
  };
  std::vector<std::unique_ptr<nntrainer::GraphRealizer>> realizers;
  realizers.emplace_back(new nntrainer::MultioutRealizer());
  BnRealizer r;
  EXPECT_NO_THROW(compileAndRealizeAndEqual(r, realizers, before, after));
}

TEST(LossRealizer, loss_realizer_p) {
  /// realization without identifying custom input
  std::vector<LayerRepresentation> before = {
    {"fully_connected", {"name=fc1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=ac1"}},
    {"activation", {"name=ac2", "activation=relu", "input_layers=fc2"}},
    {"fully_connected", {"name=fc3", "input_layers=ac2"}},
    {"mse", {"name=loss", "input_layers=fc3"}},
  };
  std::vector<LayerRepresentation> after = {
    {"fully_connected", {"name=fc1"}},
    {"activation", {"name=ac1", "activation=relu", "input_layers=fc1"}},
    {"fully_connected", {"name=fc2", "input_layers=ac1"}},
    {"activation", {"name=ac2", "activation=relu", "input_layers=fc2"}},
    {"fully_connected", {"name=fc3", "input_layers=ac2"}},
  };
  LossRealizer r;
  std::vector<std::unique_ptr<nntrainer::GraphRealizer>> realizers;
  EXPECT_NO_THROW(compileAndRealizeAndEqual(r, realizers, before, after));
}
