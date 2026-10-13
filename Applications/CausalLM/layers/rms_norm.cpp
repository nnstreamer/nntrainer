// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <cpu_backend.h>
#include <iostream>

#include "generation_prefill_tail.h"
#include "htp_decode_hook.h"
#include "htp_strict.h"
#include "moe_decode_lookahead.h"
#include "moe_lookahead_trace.h"
#include "rms_norm.h"

namespace causallm {

// Installed at model load, before any forwarding. The experimental virtual
// expert cache, like its existing prefill path, supports one model per process.
static MoeDecodeLookaheadCallback lookahead_callback = nullptr;

void setMoeDecodeLookaheadCallback(MoeDecodeLookaheadCallback callback) {
  lookahead_callback = callback;
}

void moeDecodeLookaheadNorm(const std::string &name, unsigned from, unsigned to,
                            const float *input, const float *addend,
                            const float *gamma, unsigned width, float epsilon) {
  if (lookahead_callback)
    lookahead_callback(name, from, to, input, addend, gamma, width, epsilon);
}

static constexpr size_t SINGLE_INOUT_IDX = 0;

void RMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  NNTR_THROW_IF(dim.size() != 1 && dim.size() != 2, std::invalid_argument)
    << "RMSNorm expects one input or a residual pair";
  NNTR_THROW_IF(dim.size() == 2 && (!htpStrict() || dim[0] != dim[1] ||
                                    dim[0].getDataType() !=
                                      nntrainer::TensorDim::DataType::FP32),
                std::invalid_argument)
    << "Residual RMSNorm requires matching FP32 inputs and HTP strict mode";
  deferred_consumer_ = std::get<props::DeferredNormConsumer>(rms_props).get();
  NNTR_THROW_IF(deferred_consumer_ != "none" &&
                  (deferred_consumer_ != "conv" &&
                   deferred_consumer_ != "router" &&
                   deferred_consumer_ != "qkv"),
                std::invalid_argument)
    << "Unknown deferred norm consumer";
  NNTR_THROW_IF(deferred_consumer_ != "none" && !htpStrict(),
                std::invalid_argument)
    << "Deferred norm requires a strict HTP consumer";
  context.setOutputDimensions(dim);

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  // gamma is unquantized and stored as FP32 in the bin. Request it as FP32
  // regardless of the activation dtype; declaring it FP16 reinterprets the
  // on-disk FP32 bytes as FP16 and corrupts gamma (≈FP16-max garbage). The
  // FP16 forward path casts gamma down to FP16 at the multiply site.
  nntrainer::TensorDim gamma_dim(
    1, 1, 1, dim[0].width(),
    nntrainer::TensorDim::TensorType(context.getFormat(),
                                     nntrainer::TensorDim::DataType::FP32));
  wt_idx[RMSParams::gamma] = context.requestWeight(
    gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
    nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", true);
}

void RMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                              bool training) {}

void RMSNormLayer::incremental_forwarding(nntrainer::RunLayerContext &context,
                                          unsigned int from, unsigned int to,
                                          bool training) {
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  const unsigned tail_offset = generationPrefillTailOffset(
    context.getName(), from, to, in_dim.batch(), training);
  const unsigned rows = to - from - tail_offset;
  const size_t tail_elements =
    static_cast<size_t>(tail_offset) * in_dim.width();
  in_step_dim.batch(1);
  in_step_dim.height(rows);
  out_step_dim.batch(1);
  out_step_dim.height(rows);

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step = in.getSharedDataTensor(
      in_step_dim, b * in_dim.getFeatureLen() + tail_elements, true);
    nntrainer::Tensor out_step = out.getSharedDataTensor(
      out_step_dim, b * out_dim.getFeatureLen() + tail_elements, true);

    if (b_size == 1 &&
        in_step.getDataType() == ml::train::TensorDim::DataType::FP32)
      moeDecodeLookaheadNorm(
        context.getName(), from, to, in_step.getData<float>(),
        context.getNumInputs() == 2 ? context.getInput(1).getData<float>()
                                    : nullptr,
        gamma.getData<float>(), in_step.width(), epsilon);

    // Snapshot the residual operands before a deferred consumer can reuse them.
    // Offline analysis reconstructs RMSNorm and validates it against routing.
    if (moeLookaheadDirectory() && from && to - from == 1 && b_size == 1 &&
        in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const std::string name = context.getName();
      const std::string step = name + "_" + std::to_string(from);
      moeLookaheadDump(name + "_gamma", gamma.getData<float>(), in_step.width(),
                       true);
      moeLookaheadDump(step + "_input", in_step.getData<float>(),
                       in_step.width());
      if (context.getNumInputs() == 2)
        moeLookaheadDump(step + "_addend", context.getInput(1).getData<float>(),
                         in_step.width());
    }

    if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
      const auto &dim = in_step.getDim();
      if (context.getNumInputs() == 2) {
        auto residual = context.getInput(1).getSharedDataTensor(
          in_step_dim, b * in_dim.getFeatureLen() + tail_elements, true);
        auto sum = context.getOutput(1).getSharedDataTensor(
          out_step_dim, b * out_dim.getFeatureLen() + tail_elements, true);
        if (b_size == 1 && dim.height() == 1 && deferred_consumer_ != "none" &&
            htpStrictOps()->defer_decode_norm(
              deferred_consumer_, dim.width(), epsilon,
              in_step.getData<float>(), residual.getData<float>(),
              gamma.getData<float>(), sum.getData<float>(),
              out_step.getData<float>()))
          continue;
        if (!htpStrictOps()->add_rmsnorm_fp32(
              dim.height(), dim.width(), epsilon, in_step.getData<float>(),
              residual.getData<float>(), gamma.getData<float>(),
              sum.getData<float>(), out_step.getData<float>()))
          throw std::runtime_error("HTP residual RMSNorm is unsupported");
        continue;
      }
      if (htpStrict()) {
        if (b_size == 1 && dim.height() == 1 && deferred_consumer_ != "none" &&
            htpStrictOps()->defer_decode_norm(
              deferred_consumer_, dim.width(), epsilon,
              in_step.getData<float>(), nullptr, gamma.getData<float>(),
              nullptr, out_step.getData<float>()))
          continue;
        htpStrictMath(nntrainer::ComputeOps::InferenceMath::RmsNorm,
                      dim.height(), dim.width(), dim.width(), epsilon, 1.0f,
                      in_step.getData<float>(), nullptr, gamma.getData<float>(),
                      out_step.getData<float>());
        continue;
      }
      // [#130] one decode row: the HTP runs it when RMSNORM is resident
      if (to - from == 1 && b_size == 1 &&
          gamma.getDataType() == ml::train::TensorDim::DataType::FP32 &&
          htpDecodeRmsNorm(from, in_step.getData<float>(),
                           out_step.getData<float>(), gamma.getData<float>(),
                           dim.width(), epsilon))
        continue;
#ifdef ENABLE_FP16
      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);

      // DO NOT USE rms_norm_wrt_width_fp16_intrinsic. It causes overflow!

      // nntrainer::rms_norm_wrt_width_fp16_intrinsic(
      //   in_step.getData<float>(), out_step.getData<float>(), dim.height(),
      //   dim.width(), epsilon);
#else

      nntrainer::rms_norm_wrt_width_fp32_intrinsic(
        in_step.getData<float>(), out_step.getData<float>(), dim.height(),
        dim.width(), epsilon);
#endif
#ifdef ENABLE_FP16
    } else if (in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      const auto &dim = in_step.getDim();
      // FP16 activation: this kernel accumulates the sum-of-squares in FP32
      // (so a wide residual row cannot overflow FP16) and reads/writes FP16.
      nntrainer::rms_norm_wrt_width_fp16_intrinsic(
        in_step.getData<_FP16>(), out_step.getData<_FP16>(), dim.height(),
        dim.width(), epsilon);
#endif
    } else {
      throw std::invalid_argument(
        "Error: not yet implemented for this data type");
    }
    // gamma (unquantized) may be stored at a different dtype than the FP16
    // activation; cast it to match before the elementwise multiply.
    if (gamma.getDataType() != out_step.getDataType()) {
      nntrainer::Tensor gamma_cast = gamma.clone(out_step.getDataType());
      out_step.multiply_i(gamma_cast);
    } else {
      out_step.multiply_i(gamma);
    }
#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << "gamma:" << gamma << std::endl;
#endif
  }
}

void RMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  for (unsigned i = 0; i < input_dimensions.size(); ++i) {
    context.updateInput(i, input_dimensions[i]);
    context.updateOutput(i, input_dimensions[i]);
  }
}

void RMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  std::throw_with_nested(std::runtime_error("Training is not supported yet."));
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new RMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
