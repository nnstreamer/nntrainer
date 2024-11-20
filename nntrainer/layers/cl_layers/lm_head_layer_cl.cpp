// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Yash Singh <yash.singh@samsung.com>
 *
 * @file   lm_head_layer_cl.cpp
 * @date   1 Oct 2024
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Yash Singh <yash.singh@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Implementation of custom lm head layer
 */

#include <algorithm>
#include <blas_kernel_interface.h>
#include <iostream>
#include <map>

#include <lm_head_layer_cl.h>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum LMHeadParams { weight, bias, candidate_weight, candidate_hidden_step };

CustomLMHeadLayerCl::CustomLMHeadLayerCl() :
  LayerImpl(),
  custom_lm_head_props(nntrainer::props::Unit(), props::UseVocabSelection(),
                       props::LshChoices()) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
}

void CustomLMHeadLayerCl::finalize(nntrainer::InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<nntrainer::props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<nntrainer::props::WeightRegularizerConstant>(*layer_impl_props);
  auto weight_initializer = nntrainer::props::InitializerInfo::Enum::ZEROS;
  auto &weight_decay =
    std::get<nntrainer::props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<nntrainer::props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer =
    std::get<nntrainer::props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias =
    std::get<nntrainer::props::DisableBias>(*layer_impl_props);

  auto unit = std::get<nntrainer::props::Unit>(custom_lm_head_props).get();

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "lm head layer takes only one input";

  std::vector<ml::train::TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == nntrainer::Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];
  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration
  ml::train::TensorDim bias_dim(
    1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0001 : 0b0100);

  ml::train::TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    ml::train::TensorDim::TensorType(context.getFormat(),
                                     context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[LMHeadParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", true);

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[LMHeadParams::bias] = context.requestWeight(
      bias_dim, bias_initializer, nntrainer::WeightRegularizer::NONE, 1.0f,
      bias_decay, "bias", true);
  }

  auto use_vocab_selection =
    std::get<props::UseVocabSelection>(custom_lm_head_props).get();

  if (use_vocab_selection) {
    auto lsh_choices = std::get<props::LshChoices>(custom_lm_head_props).get();

    ml::train::TensorDim candidate_weight_dim(
      1, is_nchw ? 1 : lsh_choices, is_nchw ? lsh_choices : in_dim.channel(),
      is_nchw ? in_dim.width() : 1,
      ml::train::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));

    weight_idx[LMHeadParams::candidate_weight] = context.requestTensor(
      candidate_weight_dim, "candidate_weight", Initializer::NONE, false,
      nntrainer::TensorLifespan::ITERATION_LIFESPAN);

    ml::train::TensorDim candidate_hidden_step_dim(
      1, 1, 1, lsh_choices,
      ml::train::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));

    weight_idx[LMHeadParams::candidate_hidden_step] = context.requestTensor(
      candidate_hidden_step_dim, "candidate_hidden_step", Initializer::NONE,
      false, nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  }
}

void CustomLMHeadLayerCl::forwarding(nntrainer::RunLayerContext &context,
                                     bool training) {
  // NYI
}

void CustomLMHeadLayerCl::initVocabSelection(
  LshType lshType, int lshChoices, nntrainer::RunLayerContext &context) {
  nntrainer::Tensor w;
  nntrainer::Tensor &weight = w;
  context.getWeight(weight, weight_idx[LMHeadParams::weight]);
  this->vocabSelection =
    std::shared_ptr<VocabSelection>(new VocabSelectionNNTrainer(
      lshType, lshChoices, weight.height(), weight.width(), weight));
  weight_T = std::make_unique<nntrainer::Tensor>(weight.transpose("0:2:1"));

  weight_T->reshape({weight_T->width(), weight_T->height()});
}

void CustomLMHeadLayerCl::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::Tensor w;
  nntrainer::Tensor &weight = w;
  context.getWeight(weight, weight_idx[LMHeadParams::weight]);

  nntrainer::Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);

  ml::train::TensorDim input_dim = input_.getDim();
  ml::train::TensorDim hidden_dim = hidden_.getDim();

  ml::train::TensorDim input_step_dim = input_dim;
  ml::train::TensorDim hidden_step_dim = hidden_dim;

  unsigned int _from = from;

  if (from) {
    NNTR_THROW_IF(to - from != 1, std::invalid_argument)
      << "incremental step size is not 1";
    from = 0;
    to = 1;
  }

  input_step_dim.batch(1);
  input_step_dim.height(1);
  hidden_step_dim.batch(1);
  hidden_step_dim.height(1);

  // bool smart_reply = std::get<props::SmartReply>(custom_lm_head_props).get();

  unsigned int b_size = input_dim.batch();
  unsigned omp_num = 4;
  // if (smart_reply && !_from) {
  //   b_size = 1;
  //   omp_num = 1;
  // }

  // #pragma omp parallel for num_threads(omp_num)
  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor input_step = input_.getSharedDataTensor(
      input_step_dim,
      b * input_dim.getFeatureLen() +
        (to - from == 1 ? 0 : (to - 1) * input_.width()),
      true);
    nntrainer::Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim,
      b * hidden_dim.getFeatureLen() +
        (to - from == 1 ? 0 : (to - 1) * hidden_.width()),
      true);

    auto unit = std::get<nntrainer::props::Unit>(custom_lm_head_props).get();
    auto use_vocab_selection =
      std::get<props::UseVocabSelection>(custom_lm_head_props).get();

    if (use_vocab_selection) {
      auto lsh_choices =
        std::get<props::LshChoices>(custom_lm_head_props).get();
      auto vocab = vocabSelection->getVocabs(input_step);

      hidden_step.setValue(0);

      ml::train::TensorDim weight_T_ith_choice_dim = weight_T->getDim();
      weight_T_ith_choice_dim.width(1);
      ml::train::TensorDim hidden_step_ith_choice_dim = hidden_step_dim;
      hidden_step_ith_choice_dim.width(1);
      nntrainer::Tensor weight_T_ith_choice;

      for (unsigned int i = 0; i < lsh_choices; ++i) {
        weight_T_ith_choice = weight_T->getSharedDataTensor(
          weight_T_ith_choice_dim, vocab[0][i] * input_step.width(), true);
        nntrainer::Tensor hidden_step_ith_choice =
          hidden_step.getSharedDataTensor(hidden_step_ith_choice_dim,
                                          vocab[0][i], true);

        dotCl(input_step, weight_T_ith_choice, hidden_step_ith_choice);
      }
    } else {
      dotCl(input_step, weight, hidden_step);
    }

    if (auto &disable_bias =
          std::get<nntrainer::props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      nntrainer::Tensor &bias =
        context.getWeight(weight_idx[LMHeadParams::bias]);

      add_i_cl(hidden_step, bias);
    }
  }
}

void CustomLMHeadLayerCl::calcDerivative(nntrainer::RunLayerContext &context) {}

void CustomLMHeadLayerCl::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, custom_lm_head_props);
  LayerImpl::setProperty(remain_props);
}
} // namespace nntrainer
