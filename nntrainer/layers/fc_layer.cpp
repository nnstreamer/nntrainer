/**
 * Copyright (C) 2020 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *   http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * @file	fc_layer.cpp
 * @date	14 May 2020
 * @brief	This is Fully Connected Layer Class for Neural Network
 * @see		https://github.com/nntrainer/nntrainer
 * @author	Jijoong Moon <jijoong.moon@samsung.com>
 * @author	Anirudh Bocha <b.saianirud@samsung.com>
 * @author	Pranjal Thapliyal <p.thapliyal@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 */

#include <common_properties.h>
#include <compute_ops.h>
#include <fc_layer.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <quantizer.h>
#include <util_func.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <iostream>
#include <numeric>

namespace nntrainer {

static constexpr size_t SINGLE_INOUT_IDX = 0;

enum FCParams { weight, bias };
enum LORAParams { loraA, loraB, loraTmp, loraOut };

std::mutex FullyConnectedLayer::s_registry_mutex;
std::unordered_map<std::string, FullyConnectedLayer::LoRAQATStats>
  FullyConnectedLayer::s_qat_registry;
std::unordered_map<std::string,
                   std::pair<std::vector<float>, std::vector<float>>>
  FullyConnectedLayer::s_block_d_registry;

FullyConnectedLayer::FullyConnectedLayer() :
  LayerImpl(),
  lora_scaling(1.0f),
  fc_props(props::Unit(), props::LoraRank(), props::LoraAlpha(),
           props::LoraQAT(), props::LoraWeightQ4(), props::LoraQatEmaMomentum(),
           props::LoraGPU(), props::FusedActivation(),
           props::PlanLastRowOnly()),
  quantizer(nullptr) {
  weight_idx.fill(std::numeric_limits<unsigned>::max());
  lora_idx.fill(std::numeric_limits<unsigned>::max());
}

// Per-block Q4_0 fake-quantization with EMA block scales tracked in N x K
// layout.
//
// Blocks are defined in N x K layout (the transposed layout the Q4_0 GEMM
// kernel and build_q4_0_forced_blocks both use at save time), so the EMA
// scales computed here line up exactly with the blocks force-fed at save
// time.
//
// Training:   compute a fresh per-block scale, bootstrap the EMA on first
//             use, then update block_d[b] = (1-m)*block_d[b] + m*d_fresh,
//             and quantize with the updated EMA scale.
// Validation: quantize with the current EMA scale without updating it.
// STE:        the caller (calcDerivative/calcGradient) treats the round in
//             this function as the identity for gradient purposes; this
//             function itself only computes the forward value.
Tensor FullyConnectedLayer::fakeQuantizeQ4_0(const Tensor &x,
                                             std::vector<float> &block_d,
                                             bool training) {
  // x is stored K x N in nntrainer (height=K, width=N).
  const size_t K = x.getDim().height();
  const size_t N = x.getDim().width();
  const size_t num_blocks_NK = (K * N) / 32;

  if (block_d.empty())
    block_d.resize(num_blocks_NK, 0.0f);

  // Compute fresh per-block scales in N x K layout.
  // For N x K linear index nk = b*32+j: n = nk/K, k = nk%K -> K x N index =
  // k*N+n.
  for (size_t b = 0; b < num_blocks_NK; ++b) {
    float max_abs = 0.0f;
    for (size_t j = 0; j < 32; ++j) {
      const size_t nk = b * 32 + j;
      const size_t n = nk / K;
      const size_t k = nk % K;
      max_abs = std::max(max_abs, std::abs(x.getValue<float>(k * N + n)));
    }
    const float d_fresh = (max_abs > 1e-8f) ? max_abs / 8.0f : 0.0f;
    if (training) {
      if (block_d[b] < 1e-10f)
        block_d[b] = d_fresh;
      else
        block_d[b] = (1.0f - momentum) * block_d[b] + momentum * d_fresh;
    }
  }

  // Apply fake-quant iterating K x N order; map each element to its N x K
  // block. apply() iterates K x N linearly: index i = k*N + n -> k = i/N,
  // n = i%N -> N x K index nk = n*K + k -> block = nk/32.
  Tensor x_fq = x.clone();
  size_t i = 0;
  std::function<float(float)> fn = [&i, K, N, &block_d](float v) -> float {
    const size_t k = i / N;
    const size_t n = i % N;
    const size_t b = (n * K + k) / 32;
    ++i;
    const float d = block_d[b];
    if (d < 1e-10f)
      return v;
    float q = std::round(v / d);
    q = std::max(-8.0f, std::min(7.0f, q));
    return q * d;
  };
  x_fq.apply<float>(fn, x_fq);
  return x_fq;
}

void FullyConnectedLayer::gemm(const Tensor &A, const Tensor &B, Tensor &C,
                               bool trans_a, bool trans_b, float beta) {
  if (gemm_ops_ != nullptr && A.getDataType() == Tdatatype::FP32 &&
      B.getDataType() == Tdatatype::FP32 &&
      C.getDataType() == Tdatatype::FP32 && A.getFormat() == Tformat::NCHW &&
      B.getFormat() == Tformat::NCHW && C.getFormat() == Tformat::NCHW) {
    const TensorDim &ad = A.getDim();
    const TensorDim &bd = B.getDim();
    const unsigned int a_rows = ad.batch() * ad.channel() * ad.height();
    const unsigned int a_cols = ad.width();
    const unsigned int b_rows = bd.batch() * bd.channel() * bd.height();
    const unsigned int b_cols = bd.width();
    const unsigned int M = trans_a ? a_cols : a_rows;
    const unsigned int K = trans_a ? a_rows : a_cols;
    const unsigned int Kb = trans_b ? b_cols : b_rows;
    const unsigned int N = trans_b ? b_rows : b_cols;
    if (K == Kb && C.size() == (size_t)M * N) {
      if (beta == 0.0f) {
        static const char *check_env = std::getenv("NNTR_TRAIN_GPU_CHECK");
        static const bool check = check_env != nullptr;
        static const bool use_ref = check && check_env[0] == '2';
        static int logged = 0, logged_big = 0, calls = 0, bad = 0;
        static double worst = 0;
        const auto t0 = std::chrono::steady_clock::now();
        if (gemm_ops_->try_sgemm_fp32_host(
              trans_a, trans_b, M, N, K, A.getData<float>(), B.getData<float>(),
              C.getData<float>())) {
          if (check) {
            const double gpu_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t0)
                                    .count();
            Tensor ref(C.getDim());
            const auto t1 = std::chrono::steady_clock::now();
            A.dot(B, ref, trans_a, trans_b, 0.0f);
            const double cpu_ms = std::chrono::duration<double, std::milli>(
                                    std::chrono::steady_clock::now() - t1)
                                    .count();
            const float *g = C.getData<float>();
            const float *r = ref.getData<float>();
            double max_abs = 0, max_ref = 0;
            for (size_t i = 0; i < (size_t)M * N; ++i) {
              max_abs = std::max(max_abs, (double)std::abs(g[i] - r[i]));
              max_ref = std::max(max_ref, (double)std::abs(r[i]));
            }
            const double rel = max_ref > 0 ? max_abs / max_ref : max_abs;
            ++calls;
            worst = std::max(worst, rel);
            const bool is_bad = rel > 1e-4;
            if (is_bad)
              ++bad;
            const bool big = (size_t)M * N * K >= (size_t(1) << 24);
            if (is_bad || (big ? logged_big++ < 12 : logged++ < 4) ||
                calls % 500 == 0)
              std::cout << "[GEMM-CHECK] #" << calls << " tA=" << trans_a
                        << " tB=" << trans_b << " M=" << M << " N=" << N
                        << " K=" << K << " gpu=" << gpu_ms
                        << "ms cpu=" << cpu_ms << "ms rel_err=" << rel
                        << " worst=" << worst << " bad=" << bad << "\n";
            if (use_ref)
              std::memcpy(C.getData<float>(), r, sizeof(float) * (size_t)M * N);
          }
          return;
        }
      } else {
        Tensor prod(C.getDim());
        if (gemm_ops_->try_sgemm_fp32_host(
              trans_a, trans_b, M, N, K, A.getData<float>(), B.getData<float>(),
              prod.getData<float>())) {
          if (beta != 1.0f)
            C.multiply_i(beta);
          C.add_i(prod);
          return;
        }
      }
    }
  }
  A.dot(B, C, trans_a, trans_b, beta);
}

FullyConnectedLayer::LoRAQATStats
FullyConnectedLayer::getRegisteredStats(const std::string &layer_name) {
  std::lock_guard<std::mutex> lock(s_registry_mutex);
  auto it = s_qat_registry.find(layer_name);
  if (it != s_qat_registry.end())
    return it->second;
  return {};
}

std::pair<std::vector<float>, std::vector<float>>
FullyConnectedLayer::getRegisteredBlockScales(const std::string &layer_name) {
  std::lock_guard<std::mutex> lock(s_registry_mutex);
  auto it = s_block_d_registry.find(layer_name);
  if (it != s_block_d_registry.end())
    return it->second;
  return {};
}

void FullyConnectedLayer::finalize(InitLayerContext &context) {
  auto &weight_regularizer =
    std::get<props::WeightRegularizer>(*layer_impl_props);
  auto &weight_regularizer_constant =
    std::get<props::WeightRegularizerConstant>(*layer_impl_props);
  auto &weight_initializer =
    std::get<props::WeightInitializer>(*layer_impl_props);
  auto &weight_decay = std::get<props::WeightDecay>(*layer_impl_props);
  auto &bias_decay = std::get<props::BiasDecay>(*layer_impl_props);
  auto &bias_initializer = std::get<props::BiasInitializer>(*layer_impl_props);
  auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);

  const auto &unit = std::get<props::Unit>(fc_props).get();
  const auto &lora_rank = (std::get<props::LoraRank>(fc_props).empty())
                            ? 0
                            : std::get<props::LoraRank>(fc_props).get();
  lora_scaling = (lora_rank && !std::get<props::LoraAlpha>(fc_props).empty())
                   ? (float)std::get<props::LoraAlpha>(fc_props) / lora_rank
                   : 1;
  if (!std::get<props::SkipPrefill>(*layer_impl_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(*layer_impl_props).get();

  gemm_ops_ = nullptr;
#ifdef ENABLE_OPENCL
  if (lora_rank && !std::get<props::LoraGPU>(fc_props).empty() &&
      std::get<props::LoraGPU>(fc_props).get())
    gemm_ops_ = get_cl_ops();
#endif

  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "Fully connected layer takes only one input";

  std::vector<TensorDim> output_dims(1);

  /// @todo fc actaully supports multidimensions. EffDimFlag shouldn't be fixed
  /// like this.
  context.setEffDimFlagInputDimension(0, 0b1001);
  context.setDynDimFlagInputDimension(0, 0b1000);

  bool is_nchw = (context.getFormat() == Tformat::NCHW);
  /** set output dimensions */
  auto const &in_dim = context.getInputDimensions()[0];

  /// The CPU kernels take the input in the activation dtype. With an FP32
  /// input in an FP16-activation model the dot is dispatched on the FP32
  /// input and writes FP32 results into the FP16 output, so the layer returns
  /// NaN or garbage without an error. The input layer keeps the dtype it is
  /// declared with, so an FP16 model fed straight from an input layer has to
  /// declare input_dtype=FP16 there.
  const auto act_dtype = context.getActivationDataType();
  const auto in_dtype = in_dim.getDataType();
  const auto float_name = [](TensorDim::DataType t) -> const char * {
    return t == TensorDim::DataType::FP32   ? "FP32"
           : t == TensorDim::DataType::FP16 ? "FP16"
                                            : nullptr;
  };
  NNTR_THROW_IF(float_name(act_dtype) && float_name(in_dtype) &&
                  in_dtype != act_dtype,
                std::invalid_argument)
    << "fully_connected layer '" << context.getName() << "': input dtype is "
    << float_name(in_dtype) << " but the model activation dtype is "
    << float_name(act_dtype) << "; feed it in " << float_name(act_dtype)
    << " (e.g. input_dtype=" << float_name(act_dtype) << " on the input layer)";

  output_dims[0] = in_dim;
  is_nchw ? output_dims[0].width(unit) : output_dims[0].channel(unit);

  // Same planning contract the neutral FC states, so the property reads the
  // same whichever engine resolves "fully_connected": a head the model has
  // declared to produce only its last row is planned at height 1 instead of
  // the graph-build height, whose rows above the first are allocated and never
  // written. Read with skip_prefill, which is what makes the rest of the plane
  // dead in the first place.
  auto &plan_last_row_only = std::get<props::PlanLastRowOnly>(fc_props);
  if (skip_prefill && !plan_last_row_only.empty() && plan_last_row_only.get())
    output_dims[0].height(1);

  output_dims[0].setTensorType(
    {context.getFormat(), context.getActivationDataType()});

  context.setOutputDimensions(output_dims);

  /** set weight specifications */
  // @todo : This NCHW format setting is just temporal, it needs to be set by
  // global configuration

  /** Bias Dimension : (1, 1, 1, unit) */
  /// @note Bias is un-quantized and added directly to the activation. Its
  /// storage dtype must match how it is laid out on disk:
  ///  - float weight (FP16/FP32): bias is stored in the activation dtype, so
  ///    request it as such (no cast needed at the add site).
  ///  - quantized weight (Q4_0/Q6_K/QINT*/...): bias is stored FP32 on disk;
  ///    requesting it as the (possibly FP16) activation dtype would reinterpret
  ///    the FP32 bytes and corrupt it. Request FP32 and cast to the activation
  ///    dtype at the add site below.
  const auto weight_dtype = context.getWeightDataType();
  const bool weight_is_float = (weight_dtype == TensorDim::DataType::FP32 ||
                                weight_dtype == TensorDim::DataType::FP16);
  const auto bias_dtype = weight_is_float ? context.getActivationDataType()
                                          : TensorDim::DataType::FP32;
  TensorDim bias_dim(1, is_nchw ? 1 : unit, 1, is_nchw ? unit : 1,
                     TensorDim::TensorType(context.getFormat(), bias_dtype),
                     is_nchw ? 0b0001 : 0b0100);

  /** Weight Dimension : (1, 1, in_dim.width(), unit)*/
  TensorDim weight_dim(
    1, is_nchw ? 1 : unit, is_nchw ? in_dim.width() : 1,
    is_nchw ? unit : in_dim.channel(),
    TensorDim::TensorType(context.getFormat(), context.getWeightDataType()),
    is_nchw ? 0b0011 : 0b0101);

  weight_idx[FCParams::weight] = context.requestWeight(
    weight_dim, weight_initializer, weight_regularizer,
    weight_regularizer_constant, weight_decay, "weight", (lora_rank == 0));

  if (disable_bias.empty() || disable_bias.get() == false) {
    weight_idx[FCParams::bias] =
      context.requestWeight(bias_dim, bias_initializer, WeightRegularizer::NONE,
                            1.0f, bias_decay, "bias", (lora_rank == 0));
  }

  /** create weights for LoRA */
  if (lora_rank) {

    const bool lora_qat_mode = !std::get<props::LoraQAT>(fc_props).empty() &&
                               std::get<props::LoraQAT>(fc_props).get();
    const bool lora_q4 = !std::get<props::LoraWeightQ4>(fc_props).empty() &&
                         std::get<props::LoraWeightQ4>(fc_props).get();
    // Inference-only Q4_0 adapter (no QAT): register loraA/loraB as real
    // Q4_0 tensors so the W4A8 GEMM kernel fires at runtime. Otherwise
    // (plain FP32 training, or QAT training, which needs FP32 gradients and
    // fake-quantizes in forwarding/calcDerivative/calcGradient instead),
    // keep them in the base weight's dtype.
    // LoRA adapters are always FP32 for training (gradients need full
    // precision). Only inference-only Q4_0 adapters (lora_q4 && !QAT)
    // use Q4_0 dtype so the W4A8 GEMM kernel fires at runtime.
    const bool use_q4_tensors = lora_q4 && !lora_qat_mode;
    const auto lora_dtype =
      use_q4_tensors ? TensorDim::DataType::Q4_0 : TensorDim::DataType::FP32;

    /** loraA Dimension : (1, 1, in_dim.width, lora_rank) */
    TensorDim loraA_dim(1, is_nchw ? 1 : lora_rank,
                        is_nchw ? in_dim.width() : 1,
                        is_nchw ? lora_rank : in_dim.channel(),
                        TensorDim::TensorType(context.getFormat(), lora_dtype),
                        is_nchw ? 0b0011 : 0b0101);

    /** loraB Dimension : (1, 1, lora_rank, unit) */
    TensorDim loraB_dim(1, is_nchw ? 1 : unit, is_nchw ? lora_rank : 1,
                        is_nchw ? unit : lora_rank,
                        TensorDim::TensorType(context.getFormat(), lora_dtype),
                        is_nchw ? 0b0011 : 0b0101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), lora_rank) */
    TensorDim loraTmp_dim(
      in_dim.batch(), is_nchw ? 1 : lora_rank, is_nchw ? in_dim.height() : 1,
      is_nchw ? lora_rank : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /** loraTmp Dimension : (B, 1, in_dim.height(), unit) */
    TensorDim loraOut_dim(
      in_dim.batch(), is_nchw ? 1 : unit, is_nchw ? in_dim.height() : 1,
      is_nchw ? unit : in_dim.width(),
      TensorDim::TensorType(context.getFormat(),
                            context.getActivationDataType()),
      is_nchw ? 0b1011 : 0b1101);

    /**
     * @note Standard LoRA initialization (Hu et al. 2021): A is drawn from a
     * zero-mean distribution and B is zero, so the adapter contributes
     * B*A == 0 at step 0 and the pretrained model is reproduced exactly.
     *
     * The reverse assignment (A zero, B random) also gives B*A == 0, but is
     * badly conditioned: with A == 0 the only non-zero gradient at step 0 is
     * dL/dA = x^T (dy B^T) * scaling, i.e. the first update to A is steered
     * entirely by the *random* B, so the initial step is a large step in an
     * arbitrary direction. Empirically that made training diverge above
     * lr ~1e-6 on Qwen3-0.6B, whereas this ordering is stable at 1e-4.
     *
     * When loraA/loraB are registered as real Q4_0 tensors (inference-only,
     * see use_q4_tensors above), there is nothing to initialize randomly:
     * the values come from the saved adapter file, so use NONE and mark
     * them non-trainable.
     */
    const Initializer loraA_init =
      use_q4_tensors ? Initializer::NONE : Initializer::LECUN_NORMAL;
    const Initializer loraB_init =
      use_q4_tensors ? Initializer::NONE : Initializer::ZEROS;

    lora_idx[LORAParams::loraA] = context.requestWeight(
      loraA_dim, loraA_init, weight_regularizer, weight_regularizer_constant,
      weight_decay, "loraA", !use_q4_tensors);

    lora_idx[LORAParams::loraB] = context.requestWeight(
      loraB_dim, loraB_init, weight_regularizer, weight_regularizer_constant,
      weight_decay, "loraB", !use_q4_tensors);

    lora_idx[LORAParams::loraTmp] =
      context.requestTensor(loraTmp_dim, "hidden_tmp_lora", Initializer::NONE,
                            true, TensorLifespan::FORWARD_GRAD_LIFESPAN);

    lora_idx[LORAParams::loraOut] =
      context.requestTensor(loraOut_dim, "hidden_lora", Initializer::NONE, true,
                            TensorLifespan::FORWARD_FUNC_LIFESPAN);

    if (lora_qat_mode)
      layer_name_ = context.getName();
  }

  ///@todo this quantizaer should be moved to tensor, not layer!
  switch (context.getWeightDataType()) {
  case ml::train::TensorDim::DataType::QINT4:
  case ml::train::TensorDim::DataType::QINT8:
  case ml::train::TensorDim::DataType::QINT16:
    quantizer =
      Quantization::createQuantizer(nntrainer::QScheme::PER_TENSOR_AFFINE);
    break;
  default:
    quantizer = nullptr;
    break;
  }

  // inline fused activation: if the FusionRealizer moved an activation onto
  // this FC (fused_activation set), build the SAME ActiFunc the standalone
  // ActivationLayer would use, so the fused forward is value-identical.
  auto &fused_act = std::get<props::FusedActivation>(fc_props);
  rejectFusedActivationOnTrainingGraph(fused_act, context.getExecutionMode(),
                                       "FC");
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE) {
    if (context.getActivationDataType() == TensorDim::DataType::FP16) {
#ifdef ENABLE_FP16
      acti_func.setActiFunc<_FP16>(fused_act.get());
#else
      throw std::invalid_argument(
        "[FC] fused fp16 activation needs enable-fp16");
#endif
    } else {
      acti_func.setActiFunc<float>(fused_act.get());
    }
  }
}

void FullyConnectedLayer::exportTo(
  Exporter &exporter, const ml::train::ExportMethods &method) const {
  LayerImpl::exportTo(exporter, method);
  exporter.saveResult(fc_props, method, this);
}

void FullyConnectedLayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, fc_props);
  auto &ema_momentum = std::get<props::LoraQatEmaMomentum>(fc_props);
  if (!ema_momentum.empty())
    momentum = ema_momentum.get();
  LayerImpl::setProperty(remain_props);
}

void FullyConnectedLayer::setBatch(nntrainer::RunLayerContext &context,
                                   unsigned int batch) {
  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // update Lora Tensor's batch info.
    context.updateTensor(lora_idx[LORAParams::loraTmp], batch);
    context.updateTensor(lora_idx[LORAParams::loraOut], batch);
  }
}

void FullyConnectedLayer::pack(RunLayerContext &context) {
  for (auto &w : context.getWeights()) {
    Tensor &var = w->getVariableRef();
    if (var.getDataType() == TensorDim::DataType::QS4CX)
      var.pack();
  }
}

namespace {

/**
 * [PERF] Dequantize @a weight into a persistent arena-backed FP32 scratch and
 * hand back a view of it via @a view_out. Materializing a fresh FP32 copy of
 * every quantized weight on each forwarding/calcDerivative call costs a large
 * malloc/munmap cycle (page-fault churn) on top of the dequantize itself.
 * Layer execution is sequential, so one file-scope arena is safe; it grows
 * once to the largest layer weight and is then reused warm.
 */
void dequantizeToArena(const Tensor &weight, Quantizer &quantizer,
                       Tensor &view_out) {
  static Tensor arena;
  const unsigned int elems = weight.getDim().getDataLen();
  if (arena.size() < elems) {
    arena = Tensor(TensorDim(1, 1, 1, elems, Tformat::NCHW, Tdatatype::FP32));
  }
  const TensorDim &d = weight.getDim();
  view_out = arena.getSharedDataTensor(
    TensorDim(d.batch(), d.channel(), d.height(), d.width(), Tformat::NCHW,
              Tdatatype::FP32),
    0);
  /// [PERF-DEQ] diagnostic timing (parallel dequantize; see quantizer.cpp)
  const auto deq_t0 = std::chrono::steady_clock::now();
  quantizer.dequantize(weight, view_out);
  std::cout << "[PERF-DEQ] " << d.height() << "x" << d.width() << "="
            << std::chrono::duration<double, std::milli>(
                 std::chrono::steady_clock::now() - deq_t0)
                 .count()
            << "ms\n";
}

} // namespace

void FullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  ///@todo This dequantization action should be moved to tensor.dot()
  if (quantizer != nullptr) {
    Tensor weight_arena_view;
    dequantizeToArena(weight, *quantizer, weight_arena_view);
    input_.dot(weight_arena_view, hidden_, false, false);
  } else {
    input_.dot(weight, hidden_, false, false);
  }

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &hidden_tmp_lora = context.getTensor(lora_idx[LORAParams::loraTmp]);
    Tensor &hidden_out_lora = context.getTensor(lora_idx[LORAParams::loraOut]);

    const bool lora_qat = !std::get<props::LoraQAT>(fc_props).empty() &&
                          std::get<props::LoraQAT>(fc_props).get();
    if (lora_qat) {
      // Per-block EMA fake-quant: training updates the EMA, validation
      // reads it without updating.
      a_fq = fakeQuantizeQ4_0(loraA, lora_a_block_d, training);
      b_fq = fakeQuantizeQ4_0(loraB, lora_b_block_d, training);

      // Publish the current calibration to the registries so the
      // Application layer can force-feed these scales at save time.
      if (!lora_a_block_d.empty() && !lora_b_block_d.empty()) {
        LoRAQATStats s;
        s.a_min = loraA.minValue();
        s.a_max = loraA.maxValue();
        s.a_scale =
          std::accumulate(lora_a_block_d.begin(), lora_a_block_d.end(), 0.0f) /
          static_cast<float>(lora_a_block_d.size());
        s.b_min = loraB.minValue();
        s.b_max = loraB.maxValue();
        s.b_scale =
          std::accumulate(lora_b_block_d.begin(), lora_b_block_d.end(), 0.0f) /
          static_cast<float>(lora_b_block_d.size());
        s.valid = true;
        std::lock_guard<std::mutex> lk(s_registry_mutex);
        s_qat_registry[layer_name_] = s;
        s_block_d_registry[layer_name_] = {lora_a_block_d, lora_b_block_d};
      }
      gemm(input_, a_fq, hidden_tmp_lora, false, false);
      gemm(hidden_tmp_lora, b_fq, hidden_out_lora, false, false);
    } else {
      gemm(input_, loraA, hidden_tmp_lora, false, false);
      gemm(hidden_tmp_lora, loraB, hidden_out_lora, false, false);
    }
    hidden_out_lora.multiply_i(lora_scaling);
    hidden_.add_i(hidden_out_lora);
  }

  if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
      disable_bias.empty() || disable_bias.get() == false) {
    Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
    if (bias.getDataType() != hidden_.getDataType()) {
      Tensor bias_cast = bias.clone(hidden_.getDataType());
      hidden_.add_i(bias_cast);
    } else {
      hidden_.add_i(bias);
    }
  }

  // fused activation epilogue: apply the activation inline on the GEMM
  // output, eliminating the separate ActivationLayer node. In-place when the
  // activation supports it (relu/sigmoid/tanh); otherwise via a temp input
  // copy, exactly mirroring ActivationLayer's run_fn(input, output) ->
  // value-identical.
  auto &fused_act = std::get<props::FusedActivation>(fc_props);
  if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE) {
    if (acti_func.supportInPlace()) {
      acti_func.run_fn(hidden_, hidden_);
    } else {
      Tensor in_copy = hidden_.clone();
      acti_func.run_fn(in_copy, hidden_);
    }
  }
}

void FullyConnectedLayer::incremental_forwarding(RunLayerContext &context,
                                                 unsigned int from,
                                                 unsigned int to,
                                                 bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor loraA, loraB;
  TensorDim hidden_tmp_lora_dim, hidden_out_lora_dim;

  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill)
    return;

  const bool has_lora = !std::get<props::LoraRank>(fc_props).empty();
  if (has_lora) {
    loraA = context.getWeight(lora_idx[LORAParams::loraA]);
    loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    /**
     * @note loraTmp/loraOut are requested with FORWARD_GRAD_LIFESPAN /
     * FORWARD_FUNC_LIFESPAN, which are not allocated when the graph is
     * compiled without a backward pass (pure inference). This function is
     * the inference/decode path, so only shape metadata is read from them
     * here (always valid, independent of data allocation); the actual
     * scratch computation below uses freshly-allocated local tensors rather
     * than a shared-data view into this possibly-unallocated storage.
     */
    hidden_tmp_lora_dim =
      context.getTensor(lora_idx[LORAParams::loraTmp]).getDim();
    hidden_out_lora_dim =
      context.getTensor(lora_idx[LORAParams::loraOut]).getDim();
  }

  TensorDim input_dim = input_.getDim();
  TensorDim hidden_dim = hidden_.getDim();

  TensorDim input_step_dim = input_dim;
  TensorDim hidden_step_dim = hidden_dim;

  input_step_dim.batch(1);
  if (input_dim.height() > 1)
    input_step_dim.height(to - from);
  hidden_step_dim.batch(1);
  if (hidden_dim.height() > 1)
    hidden_step_dim.height(to - from);

  auto &fused_act = std::get<props::FusedActivation>(fc_props);

  // @todo make it parallelized with batch axis
  for (unsigned int b = 0; b < hidden_.batch(); ++b) {
    Tensor input_step = input_.getSharedDataTensor(
      input_step_dim, b * hidden_dim.getFeatureLen(), true);
    Tensor hidden_step = hidden_.getSharedDataTensor(
      hidden_step_dim, b * hidden_dim.getFeatureLen(), true);

    input_step.dot(weight, hidden_step, false, false);

    if (has_lora) {
      nntrainer::TensorDim hidden_tmp_lora_step_dim = hidden_tmp_lora_dim;
      hidden_tmp_lora_step_dim.batch(1);
      if (hidden_tmp_lora_step_dim.height() > 1)
        hidden_tmp_lora_step_dim.height(to - from);

      nntrainer::TensorDim hidden_out_lora_step_dim = hidden_out_lora_dim;
      hidden_out_lora_step_dim.batch(1);
      if (hidden_out_lora_step_dim.height() > 1)
        hidden_out_lora_step_dim.height(to - from);

      nntrainer::Tensor hidden_tmp_lora_step(hidden_tmp_lora_step_dim, true);
      nntrainer::Tensor hidden_out_lora_step(hidden_out_lora_step_dim, true);

      input_step.dot(loraA, hidden_tmp_lora_step, false, false);
      hidden_tmp_lora_step.dot(loraB, hidden_out_lora_step, false, false);
      hidden_out_lora_step.multiply_i(lora_scaling);
      hidden_step.add_i(hidden_out_lora_step);
    }

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &bias = context.getWeight(weight_idx[FCParams::bias]);
      if (bias.getDataType() != hidden_step.getDataType()) {
        Tensor bias_cast = bias.clone(hidden_step.getDataType());
        hidden_step.add_i(bias_cast);
      } else {
        hidden_step.add_i(bias);
      }
    }

    // The same epilogue forwarding() applies. Without it one layer with one
    // fused_activation value would produce two different results depending on
    // which entry point ran, and the incremental path -- the one an LLM graph
    // takes -- would return pre-activation values.
    if (!fused_act.empty() && fused_act.get() != ActivationType::ACT_NONE) {
      if (acti_func.supportInPlace()) {
        acti_func.run_fn(hidden_step, hidden_step);
      } else {
        Tensor in_copy = hidden_step.clone();
        acti_func.run_fn(in_copy, hidden_step);
      }
    }
  }
}

void FullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // Dequantize the base weight to FP32 if it is stored in a quantized
    // format (Q4_0 / Q6_K). The backward pass needs FP32 to compute
    // gradients correctly; the base weight is frozen during LoRA training,
    // so this dequantization is exact and has no gradient through it.
    using DT = TensorDim::DataType;
    // Reuse a persistent arena-backed scratch for the dequantized weight
    // across layers and iterations (see dequantizeToArena): materializing a
    // fresh FP32 copy of every quantized weight each pass costs a large
    // malloc/munmap cycle (page-fault churn) on top of the dequantize itself.
    Tensor w_arena_view;
    const Tensor *w = &weight;
    if (quantizer != nullptr) {
      dequantizeToArena(weight, *quantizer, w_arena_view);
      w = &w_arena_view;
    } else if (weight.getDataType() == DT::Q4_0) {
      auto dq = Quantization::createQuantizer(nntrainer::QScheme::Q4_0);
      dequantizeToArena(weight, *dq, w_arena_view);
      w = &w_arena_view;
    } else if (weight.getDataType() == DT::Q6_K) {
      auto dq = Quantization::createQuantizer(nntrainer::QScheme::Q6_K);
      dequantizeToArena(weight, *dq, w_arena_view);
      w = &w_arena_view;
    }

    const bool lora_qat_deriv = !std::get<props::LoraQAT>(fc_props).empty() &&
                                std::get<props::LoraQAT>(fc_props).get();
    // dX = dH @ (W + sAB)^T = dH @ W^T + s * (dH @ B^T) @ A^T.
    // The LoRA part is a rank-limited chain — never materialize the
    // full out*in LoRA matrix (sAB) nor a full-size (W + sAB) temp.
    gemm(derivative_, *w, ret_, false, true);
    Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &lora_B = context.getWeight(lora_idx[LORAParams::loraB]);
    // Straight-through estimator: dL/dx uses the same fake-quantized
    // a_fq/b_fq the forward pass used, not the raw (unquantized) weights.
    const Tensor &dB = lora_qat_deriv ? b_fq : lora_B;
    const Tensor &dA = lora_qat_deriv ? a_fq : lora_A;
    // (dH @ B^T) is [rows, rank], then @ A^T is [rows, in].
    const TensorDim &dd = derivative_.getDim();
    Tensor dtmp_r(TensorDim(dd.batch(), dd.channel(), dd.height(),
                            dB.getDim().height(), dd.getTensorType()));
    gemm(derivative_, dB, dtmp_r, false, true);
    Tensor dtmp(TensorDim(dd.batch(), dd.channel(), dd.height(),
                          dA.getDim().height(), dd.getTensorType()));
    gemm(dtmp_r, dA, dtmp, false, true);
    dtmp.multiply_i(lora_scaling);
    ret_.add_i(dtmp);
  } else {
    ret_.dot_deriv_wrt_1(weight, derivative_, false, false);
  }
}

void FullyConnectedLayer::calcGradient(RunLayerContext &context) {
  /** (default) calcGradient - compute gradient of weight and bias */
  if (std::get<props::LoraRank>(fc_props).empty()) {
    Tensor &djdw = context.getWeightGrad(weight_idx[FCParams::weight]);
    djdw.setZero();

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

    if (auto &disable_bias = std::get<props::DisableBias>(*layer_impl_props);
        disable_bias.empty() || disable_bias.get() == false) {
      Tensor &djdb = context.getWeightGrad(weight_idx[FCParams::bias]);
      djdb.setZero();

      if (context.isGradientFirstAccess(weight_idx[FCParams::bias])) {
        derivative_.sum({0, 1, 2}, djdb);
      } else {
        /// @todo optimize below by adding beta to Tensor::sum
        Tensor t = derivative_.sum({0, 1, 2});
        djdb.add_i(t);
      }
    }

    input_.dot_deriv_wrt_2(
      djdw, derivative_, false, false,
      !context.isGradientFirstAccess(weight_idx[FCParams::weight]));
  } else {
    /** (lora) calcGradient - compute gradients of LoRA params only */
    Tensor &djdla = context.getWeightGrad(lora_idx[LORAParams::loraA]);
    Tensor &djdlb = context.getWeightGrad(lora_idx[LORAParams::loraB]);
    Tensor &djdtmp = context.getTensorGrad(lora_idx[LORAParams::loraTmp]);

    const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
    Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);
    Tensor &loraB = context.getWeight(lora_idx[LORAParams::loraB]);
    Tensor &loraTmp = context.getTensor(lora_idx[LORAParams::loraTmp]);
    const auto &lora_derivative_ = derivative_.multiply(lora_scaling);

    gemm(loraTmp, lora_derivative_, djdlb, true, false,
         !context.isGradientFirstAccess(lora_idx[LORAParams::loraB]) ? 1.0f
                                                                     : 0.0f);

    const bool lora_qat_grad = !std::get<props::LoraQAT>(fc_props).empty() &&
                               std::get<props::LoraQAT>(fc_props).get();
    if (lora_qat_grad) {
      // Straight-through estimator: dL/d(loraTmp) uses b_fq (the same
      // fake-quantized loraB the forward pass used), not the raw loraB.
      gemm(lora_derivative_, b_fq, djdtmp, false, true,
           !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp])
             ? 1.0f
             : 0.0f);
    } else {
      gemm(lora_derivative_, loraB, djdtmp, false, true,
           !context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp])
             ? 1.0f
             : 0.0f);
    }
    gemm(input_, djdtmp, djdla, true, false,
         !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]) ? 1.0f
                                                                     : 0.0f);
  }
}

} /* namespace nntrainer */
