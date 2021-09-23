// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file layer_golden_tests.cpp
 * @date 09 Sept 2021
 * @brief Common golden test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <layers_common_tests.h>

#include <fstream>
#include <type_traits>

#include <base_properties.h>
#include <layer_context.h>
#include <layer_devel.h>
#include <nntrainer_test_util.h>
#include <util_func.h>
#include <var_grad.h>
#include <weight.h>

#pragma GCC diagnostic ignored "-Wunused-local-typedefs"

using namespace nntrainer;

using TensorPacks = std::tuple<
  std::vector<Weight> /**< weight */, std::vector<Var_Grad> /**< in */,
  std::vector<Var_Grad> /**< out */, std::vector<Var_Grad> /**< tensors */>;

/**
 * @brief read Tensor with size check
 *
 * @param t tensor to read
 * @param file file stream
 */
static void sizeCheckedReadTensor(Tensor &t, std::ifstream &file,
                                  const std::string &error_msg = "") {
  unsigned int sz = 0;
  checkedRead(file, (char *)&sz, sizeof(unsigned));
  NNTR_THROW_IF(t.getDim().getDataLen() != sz, std::invalid_argument)
    << "[ReadFail] dimension does not match at " << error_msg << " sz: " << sz
    << " dimsize: " << t.getDim().getDataLen() << '\n';
  t.read(file);
}

/**
 * @brief Get the layer Path object
 *
 * @param file_name file name
 * @return const std::string model path
 */
static const std::string getGoldenPath(const std::string &file_name) {
  return getResPath(file_name, {"test", "unittest_layers"});
}

static InitLayerContext createInitContext(Layer *layer,
                                          const std::string &input_shape_str) {
  struct shape_parser_ : Property<TensorDim> {
    using prop_tag = dimension_prop_tag;
  };

  std::vector<shape_parser_> parsed;
  from_string(input_shape_str, parsed);

  InitLayerContext context({parsed.begin(), parsed.end()}, 1, "golden_test");
  layer->finalize(context);

  return context;
}

static TensorPacks prepareTensors(const InitLayerContext &context,
                                  std::ifstream &file) {
  auto allocate_inouts = [&file](const auto &dims) {
    std::vector<Var_Grad> vg;
    vg.reserve(dims.size());

    for (auto &dim : dims) {
      vg.emplace_back(dim, Tensor::Initializer::NONE, true, true);
      sizeCheckedReadTensor(vg.back().getVariableRef(), file,
                            vg.back().getName());
    }
    return vg;
  };

  auto allocate_tensors = [](const auto &specs) {
    std::vector<Var_Grad> vg;
    vg.reserve(specs.size());

    for (auto &spec : specs) {
      vg.emplace_back(spec, true);
    }
    return vg;
  };

  auto allocate_weights = [&file](const auto &specs) {
    std::vector<Weight> weights;
    weights.reserve(specs.size());

    for (auto &spec : specs) {
      weights.emplace_back(spec, true);
      sizeCheckedReadTensor(weights.back().getVariableRef(), file,
                            weights.back().getName());
    }
    return weights;
  };

  return {
    allocate_weights(context.getWeightsSpec()),
    allocate_inouts(context.getInputDimensions()),
    allocate_inouts(context.getOutputDimensions()),
    allocate_tensors(context.getTensorsSpec()),
  };
}

static RunLayerContext prepareRunContext(const TensorPacks &packs) {
  auto &[weights, ins, outs, tensors] = packs;
  auto create_view = [](const auto &var_grads) {
    using ptr_type_ = std::add_pointer_t<
      typename std::decay_t<decltype(var_grads)>::value_type>;
    std::vector<std::remove_cv_t<ptr_type_>> ret;
    ret.reserve(var_grads.size());

    for (auto &vg : var_grads) {
      ret.push_back(const_cast<ptr_type_>(&vg));
    }

    return ret;
  };

  auto rc =
    RunLayerContext("golden", 0.0f, create_view(weights), create_view(ins),
                    create_view(outs), create_view(tensors));

  auto num_outputs = rc.getNumOutputs();

  for (unsigned i = 0; i < num_outputs; ++i) {
    rc.getOutput(i).setRandUniform(); /// randomize output
    rc.getIncomingDerivative(i).setValue(
      2.0); /// incoming derivative is fixed to 2
  }

  return rc;
}

static void compareRunContext(RunLayerContext &rc, std::ifstream &file) {
  file.seekg(0, std::ios::beg);
  auto compare_tensors = [&file](unsigned length, auto tensor_getter, auto pred,
                                 const std::string &name) {
    for (unsigned i = 0; i < length; ++i) {
      if (!pred(i)) {
        continue;
      }
      const auto &tensor = tensor_getter(i);
      auto answer = tensor.clone();
      sizeCheckedReadTensor(answer, file, name + " at " + std::to_string(i));

      if (name == "initial_weights") {
        continue;
      }
      EXPECT_EQ(tensor, answer) << name << " at " << std::to_string(i);
    }
  };

  auto always = [](unsigned idx) { return true; };
  auto only_trainable = [&rc](unsigned idx) {
    return rc.weightHasGradient(idx);
  };

  compare_tensors(rc.getNumWeights(),
                  [&rc](unsigned idx) { return rc.getWeight(idx); }, always,
                  "initial_weights");
  compare_tensors(rc.getNumInputs(),
                  [&rc](unsigned idx) { return rc.getInput(idx); }, always,
                  "inputs");
  compare_tensors(rc.getNumOutputs(),
                  [&rc](unsigned idx) { return rc.getOutput(idx); }, always,
                  "outputs");
  compare_tensors(rc.getNumWeights(),
                  [&rc](unsigned idx) { return rc.getWeightGrad(idx); },
                  only_trainable, "gradients");
  compare_tensors(rc.getNumWeights(),
                  [&rc](unsigned idx) { return rc.getWeight(idx); }, always,
                  "weights");
  compare_tensors(rc.getNumInputs(),
                  [&rc](unsigned idx) { return rc.getOutgoingDerivative(idx); },
                  always, "derivatives");
}

LayerGoldenTest::~LayerGoldenTest() {}

void LayerGoldenTest::SetUp() {}

void LayerGoldenTest::TearDown() {}

TEST_P(LayerGoldenTest, run) {
  auto f = std::get<0>(GetParam());
  auto layer = f(std::get<1>(GetParam()));
  auto golden_file = checkedOpenStream<std::ifstream>(
    getGoldenPath(std::get<3>(GetParam())), std::ios::in | std::ios::binary);
  auto &input_dims = std::get<2>(GetParam());

  auto ic = createInitContext(layer.get(), input_dims);
  auto tensors = prepareTensors(ic, golden_file);
  auto rc = prepareRunContext(tensors);

  layer->forwarding(rc, true);
  layer->calcGradient(rc);
  layer->calcDerivative(rc);

  compareRunContext(rc, golden_file);

  EXPECT_TRUE(true); // stub test for tcm
}
