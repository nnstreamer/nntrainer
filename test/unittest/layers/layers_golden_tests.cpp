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

  InitLayerContext context({parsed.begin(), parsed.end()}, {true}, false,
                           "golden_test");
  layer->finalize(context);

  return context;
}

static TensorPacks prepareTensors(const InitLayerContext &context,
                                  std::ifstream &file) {
  auto allocate_inouts = [&file](const auto &dims) {
    std::vector<Var_Grad> vg;
    vg.reserve(dims.size());

    for (auto &dim : dims) {
      vg.emplace_back(dim, Tensor::Initializer::NONE, true, true, "golden");
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
      weights.back().getGradientRef().setZero();
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
    RunLayerContext("golden", true, 0.0f, false, create_view(weights),
                    create_view(ins), create_view(outs), create_view(tensors));

  auto num_outputs = rc.getNumOutputs();

  for (unsigned i = 0; i < num_outputs; ++i) {
    rc.getOutput(i).setRandUniform(); /// randomize output
    rc.getOutputGradUnsafe(i).setValue(
      2.0); /// incoming derivative is fixed to 2
  }

  return rc;
}

static void compareRunContext(RunLayerContext &rc, std::ifstream &file,
                              bool skip_grad, bool skip_deriv,
                              bool dropout_match) {
  file.seekg(0, std::ios::beg);
  auto compare_percentage_tensors = [](const Tensor &t1, const Tensor &t2,
                                       unsigned int match_percentage) -> bool {
    if (match_percentage == 100) {
      EXPECT_EQ(t1, t2);
      return t1 == t2;
    }

    if (t1.getDim() != t2.getDim())
      return false;

    unsigned int total = t1.size();
    unsigned int weak_match = 0;
    unsigned int strong_match = 0;

    for (unsigned int idx = 0; idx < total; idx++) {
      auto d1 = t1.getValue(idx);
      auto d2 = t2.getValue(idx);
      /** either both the values must be equal or 1 must be zero */
      weak_match +=
        std::min((d1 == d2) + (d1 == 0 && d2 != 0) + (d1 != 0 && d2 == 0), 1);
      strong_match += (d1 == d2);
    }

    return (weak_match == total) &
           (strong_match >= (total * match_percentage) / 100);
  };

  auto compare_tensors = [&file, compare_percentage_tensors](
                           unsigned length, auto tensor_getter, auto pred,
                           bool skip_compare, const std::string &name,
                           unsigned int match_percentage = 100) {
    for (unsigned i = 0; i < length; ++i) {
      if (!pred(i)) {
        continue;
      }
      const auto &tensor = tensor_getter(i);
      auto answer = tensor.clone();
      sizeCheckedReadTensor(answer, file, name + " at " + std::to_string(i));

      if (skip_compare) {
        continue;
      }
      EXPECT_TRUE(compare_percentage_tensors(tensor, answer, match_percentage))
        << name << " at " << std::to_string(i);
    }
  };

  auto always_read = [](unsigned idx) { return true; };
  auto only_read_trainable = [&rc](unsigned idx) {
    return rc.weightHasGradient(idx);
  };

  int match_percentage = 100;
  if (dropout_match)
    match_percentage = 60;

  constexpr bool skip_compare = true;

  compare_tensors(rc.getNumWeights(),
                  [&rc](unsigned idx) { return rc.getWeight(idx); },
                  always_read, skip_compare, "initial_weights");
  compare_tensors(rc.getNumInputs(),
                  [&rc](unsigned idx) { return rc.getInput(idx); }, always_read,
                  !skip_compare, "inputs");
  compare_tensors(rc.getNumOutputs(),
                  [&rc](unsigned idx) { return rc.getOutput(idx); },
                  always_read, !skip_compare, "outputs", match_percentage);
  compare_tensors(rc.getNumWeights(),
                  [&rc](unsigned idx) { return rc.getWeightGrad(idx); },
                  only_read_trainable, skip_grad, "gradients");
  compare_tensors(rc.getNumWeights(),
                  [&rc](unsigned idx) { return rc.getWeight(idx); },
                  always_read, !skip_compare, "weights");
  compare_tensors(rc.getNumInputs(),
                  [&rc](unsigned idx) { return rc.getOutgoingDerivative(idx); },
                  always_read, skip_deriv, "derivatives", match_percentage);
}

LayerGoldenTest::~LayerGoldenTest() {}

void LayerGoldenTest::SetUp() {}

void LayerGoldenTest::TearDown() {}

bool LayerGoldenTest::shouldMatchDropout60Percent() {
  return std::get<int>(GetParam()) &
         LayerGoldenTestParamOptions::DROPOUT_MATCH_60_PERCENT;
}

bool LayerGoldenTest::shouldForwardWithInferenceMode() {
  return std::get<int>(GetParam()) &
         LayerGoldenTestParamOptions::FORWARD_MODE_INFERENCE;
}

bool LayerGoldenTest::shouldSkipCalcDeriv() {
  return std::get<int>(GetParam()) &
         LayerGoldenTestParamOptions::SKIP_CALC_DERIV;
}

bool LayerGoldenTest::shouldSkipCalcGrad() {
  return std::get<int>(GetParam()) &
         LayerGoldenTestParamOptions::SKIP_CALC_GRAD;
}

TEST_P(LayerGoldenTest, run) {
  auto f = std::get<0>(GetParam());
  auto layer = f(std::get<1>(GetParam()));
  auto golden_file = checkedOpenStream<std::ifstream>(
    getGoldenPath(std::get<3>(GetParam())), std::ios::in | std::ios::binary);
  auto &input_dims = std::get<2>(GetParam());

  auto ic = createInitContext(layer.get(), input_dims);
  auto tensors = prepareTensors(ic, golden_file);
  auto rc = prepareRunContext(tensors);

  bool skip_calc_grad = shouldSkipCalcGrad();
  bool skip_calc_deriv = shouldSkipCalcDeriv();
  bool dropout_compare_60_percent = shouldMatchDropout60Percent();

  for (int i = 0; i < 4; ++i) {
    /// warm layer multiple times
    layer->forwarding(rc, !shouldForwardWithInferenceMode());
  }

  layer->forwarding(rc, !shouldForwardWithInferenceMode());
  if (!skip_calc_grad) {
    layer->calcGradient(rc);
  }
  if (!skip_calc_deriv) {
    layer->calcDerivative(rc);
  }

  compareRunContext(rc, golden_file, skip_calc_grad, skip_calc_deriv,
                    dropout_compare_60_percent);

  EXPECT_TRUE(true); // stub test for tcm
}
