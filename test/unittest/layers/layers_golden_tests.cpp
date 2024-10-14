// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
 *
 * @file layer_golden_tests.cpp
 * @date 09 Sept 2021
 * @brief Common golden test for nntrainer layers (Param Tests)
 * @see	https://github.com/nnstreamer/nntrainer
 * @author Jihoon Lee <jhoon.it.lee@samsung.com>
 * @author Sungsik Kong <ss.kong@samsung.com>
 * @bug No known bugs except for NYI items
 */
#include <layers_common_tests.h>
#include <tensor_wrap_specs.h>

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

#define EXPECT_IN_RANGE(VAL, MIN, MAX) \
  EXPECT_GE((VAL), (MIN));             \
  EXPECT_LE((VAL), (MAX))

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

static InitLayerContext
createInitContext(Layer *layer, const std::string &input_shape_str,
                  std::array<std::string, 3> tensor_type) {
  struct shape_parser_ : Property<TensorDim> {
    using prop_tag = dimension_prop_tag;
  };

  std::vector<shape_parser_> parsed;
  from_string(input_shape_str, parsed);

  /// @todo tensor_type should not affect input layer data type since
  /// technically a layer should not have information about its previous layer
  for (auto &par : parsed) {
    par.get().setFormat(
      str_converter<enum_class_prop_tag,
                    nntrainer::TensorFormatInfo>::from_string(tensor_type[0]));
    if (tensor_type[2] == "fp16" && layer->getType() != "embedding") {
      par.get().setDataType(ml::train::TensorDim::DataType::FP16);
    }
  }

  InitLayerContext context({parsed.begin(), parsed.end()}, {true}, false,
                           "golden_test", "", 0.0, tensor_type);
  layer->finalize(context);

  for (auto &dim : context.getMutableInputDimensions()) {
    if (tensor_type[2] == "fp16" && layer->getType() != "embedding") {
      dim.setDataType(ml::train::TensorDim::DataType::FP16);
    }
    dim.setFormat(
      str_converter<enum_class_prop_tag,
                    nntrainer::TensorFormatInfo>::from_string(tensor_type[0]));
  }

  return context;
}

static TensorPacks prepareTensors(const InitLayerContext &context,
                                  std::ifstream &file) {
  auto allocate_inouts = [&file](const auto &dims) {
    std::vector<Var_Grad> vg;
    vg.reserve(dims.size());

    for (auto &dim : dims) {
      vg.emplace_back(dim, Initializer::NONE, true, true, "golden");
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

  auto allocate_tensors_v2 = [](const std::vector<VarGradSpecV2> &specs) {
    std::vector<Var_Grad> vg;
    vg.reserve(specs.size());

    for (auto &spec : specs) {
      /// @todo initializer should be depending is as well
      vg.emplace_back(spec.variable_spec.dim, Initializer::NONE, true, true,
                      "golden");
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
    allocate_tensors_v2(context.getOutSpecs()),
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
                              bool dropout_match, bool skip_cos_sim) {
  file.seekg(0, std::ios::beg);

  auto compare_percentage_tensors = [](const Tensor &t1, const Tensor &t2,
                                       unsigned int match_percentage,
                                       bool skip_cos_sim) -> bool {
    if (t1.getDim() != t2.getDim())
      return false;

    unsigned int total = t1.size();
    unsigned int weak_match = 0;

    if (t1.getDim().getDataType() == ml::train::TensorDim::DataType::FP32 &&
        t2.getDim().getDataType() == ml::train::TensorDim::DataType::FP32) {

      if (match_percentage == 100) {

        if (!skip_cos_sim) {
          auto tensor = t1.clone();
          auto answer = t2.clone();
          const float epsilon = 1e-6;

          auto cos_sim = cosine_similarity<float>(
            answer.getData<float>(), tensor.getData<float>(), tensor.size());
          EXPECT_IN_RANGE(cos_sim, 1 - epsilon, 1 + epsilon);
        }

        EXPECT_EQ(t1, t2);
        return t1 == t2;
      }

      for (unsigned int idx = 0; idx < total; idx++) {
        auto d1 = t1.getValue(idx);
        auto d2 = t2.getValue(idx);
        auto float_eq = [](float a, float b) {
          constexpr auto eps = 1e-6;
          return std::abs(a - b) < eps;
        };
        /** either both the values must be equal or 1 must be zero */
        weak_match += std::min(float_eq(d1, d2) + (float_eq(d1, 0) && d2 != 0) +
                                 (d1 != 0 && float_eq(d2, 0)),
                               1);
      }

      return (weak_match == total);
    } else if (t1.getDim().getDataType() ==
                 ml::train::TensorDim::DataType::FP16 &&
               t2.getDim().getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      if (match_percentage == 100) {
        auto tensor = t1.clone();
        auto answer = t2.clone();

        const float cos_sim_range = 1e-5;
        float mse_range = 1e-3;
        mse_range *= tensor.size();

        if (!skip_cos_sim) {
          auto cos_sim = cosine_similarity<_FP16>(
            answer.getData<_FP16>(), tensor.getData<_FP16>(), tensor.size());
          EXPECT_IN_RANGE(cos_sim, 1 - cos_sim_range, 1 + cos_sim_range);
        }

        auto mean_squared_error = mse<_FP16>(
          answer.getData<_FP16>(), tensor.getData<_FP16>(), tensor.size());
        EXPECT_IN_RANGE(mean_squared_error, 0, mse_range);
      }

      for (unsigned int idx = 0; idx < total; idx++) {
        auto d1 = t1.getValue<_FP16>(idx);
        auto d2 = t2.getValue<_FP16>(idx);
        auto float_eq = [&](_FP16 a, _FP16 b) {
          constexpr auto eps = 1e-2;
          constexpr auto min_fp16 = 6.104e-5;
          if (a < b)
            std::swap(a, b);
          // out-of-fp16-range near-zero values
          if ((b > 0 && b < min_fp16) || (b < 0 && b > -min_fp16))
            b = 0;
          if ((a > 0 && a < min_fp16) || (a < 0 && a > -min_fp16))
            a = 0;
          if (a - b < eps)
            return a - b < eps;
          if (b != 0) {
            double relative_error = (a - b) / b;
            return (relative_error > 0) ? relative_error < eps * total
                                        : -relative_error < eps * total;
          } else
            return (a - b) < eps * total;
        };
        /** either both the values must be equal or 1 must be zero */
        weak_match += std::min(float_eq(d1, d2) + (float_eq(d1, 0) && d2 != 0) +
                                 (d1 != 0 && float_eq(d2, 0)),
                               1);
      }

      return (weak_match == total);
#else
      throw std::invalid_argument("Error: enable-fp16 is not enabled");
#endif
    } else
      return false;
  };

  auto compare_tensors = [&file, compare_percentage_tensors](
                           unsigned length, auto tensor_getter, auto pred,
                           bool skip_compare, bool skip_cos_sim,
                           const std::string &name,
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
      EXPECT_TRUE(compare_percentage_tensors(tensor, answer, match_percentage,
                                             skip_cos_sim))
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

  compare_tensors(
    rc.getNumWeights(),
    [&rc](unsigned idx) -> const auto & { return rc.getWeight(idx); },
    always_read, skip_compare, skip_cos_sim, "initial_weights");
  compare_tensors(
    rc.getNumInputs(),
    [&rc](unsigned idx) -> const auto & { return rc.getInput(idx); },
    always_read, !skip_compare, skip_cos_sim, "inputs");
  compare_tensors(
    rc.getNumOutputs(),
    [&rc](unsigned idx) -> const auto & { return rc.getOutput(idx); },
    always_read, !skip_compare, skip_cos_sim, "outputs", match_percentage);
  compare_tensors(
    rc.getNumWeights(),
    [&rc](unsigned idx) -> const auto & { return rc.getWeightGrad(idx); },
    only_read_trainable, skip_grad, skip_cos_sim, "gradients");
  compare_tensors(
    rc.getNumWeights(),
    [&rc](unsigned idx) -> const auto & { return rc.getWeight(idx); },
    always_read, !skip_compare, skip_cos_sim, "weights");
  compare_tensors(
    rc.getNumInputs(),
    [&rc](unsigned idx) -> const auto & {
      return rc.getOutgoingDerivative(idx);
    },
    always_read, skip_deriv, skip_cos_sim, "derivatives", match_percentage);
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

bool LayerGoldenTest::shouldUseIncForward() {
  return std::get<int>(GetParam()) &
         LayerGoldenTestParamOptions::USE_INC_FORWARD;
}

bool LayerGoldenTest::shouldSkipCosineSimilarity() {
  return std::get<int>(GetParam()) &
         LayerGoldenTestParamOptions::SKIP_COSINE_SIMILARITY;
}

TEST_P(LayerGoldenTest, run) {
  const auto &f = std::get<0>(GetParam());
  auto layer = f(std::get<1>(GetParam()));
  std::string format = std::get<5>(GetParam());
  std::string type_w = std::get<6>(GetParam());
  std::string type_a = std::get<7>(GetParam());

  auto golden_file = checkedOpenStream<std::ifstream>(
    getGoldenPath(std::get<3>(GetParam())), std::ios::in | std::ios::binary);
  auto &input_dims = std::get<2>(GetParam());

  auto ic =
    createInitContext(layer.get(), input_dims, {format, type_w, type_a});
  auto tensors = prepareTensors(ic, golden_file);
  auto rc = prepareRunContext(tensors);

  bool skip_calc_grad = shouldSkipCalcGrad();
  bool skip_calc_deriv = shouldSkipCalcDeriv();
  bool use_inc_forward = shouldUseIncForward();
  bool dropout_compare_60_percent = shouldMatchDropout60Percent();
  bool skip_cos_sim = shouldSkipCosineSimilarity();

  Tensor &input = rc.getInput(0);
  TensorDim input_dim = input.getDim();
  size_t inputHeight = input_dim.height();

  for (int i = 0; i < 4; ++i) {
    /// warm layer multiple times
    if (use_inc_forward) {
      layer->incremental_forwarding(rc, 0, inputHeight,
                                    !shouldForwardWithInferenceMode());
    } else {
      layer->forwarding(rc, !shouldForwardWithInferenceMode());
    }
  }

  if (use_inc_forward) {
    layer->incremental_forwarding(rc, 0, inputHeight,
                                  !shouldForwardWithInferenceMode());
  } else {
    layer->forwarding(rc, !shouldForwardWithInferenceMode());
  }

  if (!skip_calc_grad) {
    layer->calcGradient(rc);
  }
  if (!skip_calc_deriv) {
    layer->calcDerivative(rc);
  }

  compareRunContext(rc, golden_file, skip_calc_grad, skip_calc_deriv,
                    dropout_compare_60_percent, skip_cos_sim);
}
