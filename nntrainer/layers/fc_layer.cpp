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
#include <fp16.h>
#include <ggml_interface.h>
#include <layer_context.h>
#include <lazy_tensor.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <node_exporter.h>
#include <qs4cx_tensor.h>
#include <quantizer.h>
#include <thread_manager.h>
#include <train_prof.h>
#include <util_func.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <map>
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

namespace {
/** @brief mean magnitude of the (signed) per-block QAT scales */
float meanAbs(const std::vector<float> &v) {
  float sum = 0.0f;
  for (float d : v)
    sum += std::abs(d);
  return v.empty() ? 0.0f : sum / static_cast<float>(v.size());
}
} // namespace

// Per-block Q4_0 fake-quantization with EMA block scales tracked in N x K
// layout.
//
// Blocks are defined in N x K layout (the transposed layout
// build_q4_0_forced_blocks uses at save time), so the EMA scales computed
// here line up exactly with the blocks force-fed at save time.
//
// Scale:      ggml's Q4_0 convention: d = m / -8 for the block's
//             largest-magnitude element m, so that element lands exactly on
//             code -8 (the grid is -8..7: a plain max|w|/8 clips a positive
//             maximum to 7). block_d holds the signed d; the EMA runs on its
//             magnitude and the sign follows the current maximum.
// Training:   update the EMA, bootstrapping it on first use:
//             |d| = (1-m)*|d| + m*|d_fresh|; validation reads it unchanged.
// FP16:       every stored scale is rounded to FP16, the precision the
//             saved adapter keeps, so the values trained against are exactly
//             the values deployed.
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

  // Blocks run over the N x K linear index nk = b*32+j: n = nk/K, k = nk%K,
  // which is K x N element k*N+n. Raw-pointer loops (the per-element virtual
  // getValue and std::function dispatch dominated this op).
  const float *src = x.getData<float>();
  Tensor x_fq = x.clone();
  float *dst = x_fq.getData<float>();

  if (training) {
    for (size_t b = 0; b < num_blocks_NK; ++b) {
      float amax = 0.0f, m = 0.0f;
      for (size_t j = 0; j < 32; ++j) {
        const size_t nk = b * 32 + j;
        const float v = src[(nk % K) * N + nk / K];
        if (std::abs(v) > amax) {
          amax = std::abs(v);
          m = v;
        }
      }
      if (amax <= 1e-8f)
        continue; // all-zero block: keep its scale (0 until it has one)
      const float fresh = amax / 8.0f;
      const float prev = std::abs(block_d[b]);
      const float mag =
        prev < 1e-10f ? fresh : (1.0f - momentum) * prev + momentum * fresh;
      block_d[b] =
        compute_fp16_to_fp32(compute_fp32_to_fp16(m > 0.0f ? -mag : mag));
    }
  }

  for (size_t i = 0, total = K * N; i < total; ++i) {
    const size_t k = i / N;
    const size_t n = i % N;
    const float d = block_d[(n * K + k) / 32];
    if (std::abs(d) < 1e-10f)
      continue; // x_fq already holds v
    float q = std::round(dst[i] / d);
    q = std::max(-8.0f, std::min(7.0f, q));
    dst[i] = q * d;
  }
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
  if (train_prof::enabled())
    std::cout << "[PERF-DEQ] " << d.height() << "x" << d.width() << "="
              << std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - deq_t0)
                   .count()
              << "ms\n";
}

/**
 * @brief Dequantize a QS4CX weight (logical K x N; stored as N rows of
 * ceil(K/2) nibble bytes, even k in the low nibble, value + 8, then N fp32
 * per-output-channel scales at getScale()) into a K x N FP32 arena view.
 */
void dequantizeQs4cxToArena(const Tensor &weight, Tensor &view_out) {
  static Tensor arena;
  const TensorDim &d = weight.getDim();
  const unsigned int K = d.height(), N = d.width();
  const size_t elems = (size_t)K * N;
  if (arena.size() < elems)
    arena = Tensor(TensorDim(1, 1, 1, elems, Tformat::NCHW, Tdatatype::FP32));
  view_out = arena.getSharedDataTensor(
    TensorDim(d.batch(), d.channel(), K, N, Tformat::NCHW, Tdatatype::FP32), 0);
  const uint8_t *q = weight.getData<uint8_t>();
  const float *scale = static_cast<const float *>(weight.getScale());
  float *out = view_out.getData<float>();
  const size_t row_bytes = (K + 1) / 2;
  ThreadManager::Global().parallel_for(0, N, [=](size_t n) {
    const uint8_t *row = q + n * row_bytes;
    const float s = scale[n];
    for (unsigned int k = 0; k < K; ++k) {
      const uint8_t b = row[k / 2];
      const int v = ((k & 1) ? (b >> 4) : (b & 0x0F)) - 8;
      out[(size_t)k * N + n] = (float)v * s;
    }
  });
}

/**
 * @brief Build the transposed unit-scale QS4CX copy of a QS4CX weight W
 * (logical K x N, N rows of ceil(K/2) nibble bytes + N scales): returns a
 * QS4CX tensor of logical N x K whose K output channels hold column k of the
 * codes with scale 1.
 */
[[maybe_unused]] Tensor buildQs4cxTransposed(const Tensor &weight) {
  const TensorDim &d = weight.getDim();
  const unsigned int K = d.height(), N = d.width();
  Tensor wt(TensorDim(1, 1, N, K, Tformat::NCHW, Tdatatype::QS4CX), true);
  const uint8_t *src = weight.getData<uint8_t>();
  uint8_t *dst = wt.getData<uint8_t>();
  const size_t src_row = (K + 1) / 2, dst_row = (N + 1) / 2;
  ThreadManager::Global().parallel_for(0, K, [=](size_t k) {
    uint8_t *drow = dst + k * dst_row;
    std::memset(drow, 0, dst_row);
    for (unsigned int n = 0; n < N; ++n) {
      const uint8_t b = src[n * src_row + k / 2];
      const uint8_t nib = (k & 1) ? (b >> 4) : (b & 0x0F);
      drow[n / 2] |= (n & 1) ? (uint8_t)(nib << 4) : nib;
    }
  });
  float *sc = static_cast<float *>(wt.getScale());
  std::fill(sc, sc + K, 1.0f);
  return wt;
}

/** KleidiAI dX for a QS4CX base: NNTR_TRAIN_DX_KAI=0 falls back to Q8_0. */
[[maybe_unused]] bool useKaiDx() {
  static const bool on = [] {
    const char *e = std::getenv("NNTR_TRAIN_DX_KAI");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

/**
 * Rank-r (LoRA) skinny GEMMs, FP32 row-major, split over the compute threads.
 * These shapes are memory bound (one operand is M x K or M x N, the other is
 * rank-wide), so a single-threaded BLAS call left most of the bandwidth idle.
 */
inline float skinny_dot(const float *a, const float *b, unsigned int n) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  unsigned int d = 0;
  for (; d + 8 <= n; d += 8)
    for (unsigned int t = 0; t < 8; ++t)
      acc[t] += a[d + t] * b[d + t];
  float r = ((acc[0] + acc[4]) + (acc[1] + acc[5])) +
            ((acc[2] + acc[6]) + (acc[3] + acc[7]));
  for (; d < n; ++d)
    r += a[d] * b[d];
  return r;
}

inline void skinny_axpy(float *y, float a, const float *x, unsigned int n) {
  for (unsigned int i = 0; i < n; ++i)
    y[i] += a * x[i];
}

/** C (M x r) = X (M x K) . W (K x r) */
template <unsigned int R>
void skinny_xw_r(const float *X, const float *W, float *C, unsigned int M,
                 unsigned int K) {
  ThreadManager::Global().parallel_for(0, (M + 7) / 8, [=](size_t blk) {
    const unsigned int m_end = std::min<unsigned int>(M, (blk + 1) * 8);
    for (unsigned int m = blk * 8; m < m_end; ++m) {
      float acc[R] = {};
      const float *x = X + (size_t)m * K;
      for (unsigned int k = 0; k < K; ++k) {
        const float xv = x[k];
        const float *w = W + (size_t)k * R;
        for (unsigned int j = 0; j < R; ++j)
          acc[j] += xv * w[j];
      }
      std::memcpy(C + (size_t)m * R, acc, sizeof(acc));
    }
  });
}

void skinny_xw(const float *X, const float *W, float *C, unsigned int M,
               unsigned int K, unsigned int r) {
  if (r == 8)
    return skinny_xw_r<8>(X, W, C, M, K);
  if (r == 16)
    return skinny_xw_r<16>(X, W, C, M, K);
  ThreadManager::Global().parallel_for(0, M, [=](size_t m) {
    for (unsigned int j = 0; j < r; ++j) {
      float a = 0.0f;
      for (unsigned int k = 0; k < K; ++k)
        a += X[m * K + k] * W[(size_t)k * r + j];
      C[m * r + j] = a;
    }
  });
}

/** C (M x r) = scale * Y (M x N) . B^T, B is (r x N) */
void skinny_ybt(const float *Y, const float *B, float *C, unsigned int M,
                unsigned int N, unsigned int r, float scale) {
  ThreadManager::Global().parallel_for(0, (M + 7) / 8, [=](size_t blk) {
    const unsigned int m_end = std::min<unsigned int>(M, (blk + 1) * 8);
    for (unsigned int m = blk * 8; m < m_end; ++m)
      for (unsigned int j = 0; j < r; ++j)
        C[(size_t)m * r + j] =
          scale * skinny_dot(Y + (size_t)m * N, B + (size_t)j * N, N);
  });
}

/** C (M x Nd) += scale * U (M x r) . V (r x Nd) */
void skinny_uv_acc(const float *U, const float *V, float *C, unsigned int M,
                   unsigned int r, unsigned int Nd, float scale) {
  ThreadManager::Global().parallel_for(0, (M + 7) / 8, [=](size_t blk) {
    const unsigned int m_end = std::min<unsigned int>(M, (blk + 1) * 8);
    for (unsigned int m = blk * 8; m < m_end; ++m) {
      float *c = C + (size_t)m * Nd;
      for (unsigned int j = 0; j < r; ++j)
        skinny_axpy(c, scale * U[(size_t)m * r + j], V + (size_t)j * Nd, Nd);
    }
  });
}

/** D (r x N) = scale * T^T (r x M) . Y (M x N), column blocks per thread */
void skinny_tty(const float *T, const float *Y, float *D, unsigned int M,
                unsigned int r, unsigned int N, float scale) {
  constexpr unsigned int NB = 256;
  ThreadManager::Global().parallel_for(0, (N + NB - 1) / NB, [=](size_t blk) {
    const unsigned int n0 = blk * NB;
    const unsigned int len = std::min<unsigned int>(NB, N - n0);
    for (unsigned int j = 0; j < r; ++j)
      std::fill(D + (size_t)j * N + n0, D + (size_t)j * N + n0 + len, 0.0f);
    for (unsigned int m = 0; m < M; ++m) {
      const float *y = Y + (size_t)m * N + n0;
      for (unsigned int j = 0; j < r; ++j)
        skinny_axpy(D + (size_t)j * N + n0, T[(size_t)m * r + j], y, len);
    }
    if (scale != 1.0f)
      for (unsigned int j = 0; j < r; ++j)
        for (unsigned int n = 0; n < len; ++n)
          D[(size_t)j * N + n0 + n] *= scale;
  });
}

/** D (K x r) (+)= X^T (K x M) . T (M x r), row blocks of K per thread */
template <unsigned int R>
void skinny_xtt_r(const float *X, const float *T, float *D, unsigned int M,
                  unsigned int K, bool accumulate) {
  constexpr unsigned int KB = 64;
  ThreadManager::Global().parallel_for(0, (K + KB - 1) / KB, [=](size_t blk) {
    const unsigned int k0 = blk * KB;
    const unsigned int k1 = std::min<unsigned int>(K, k0 + KB);
    if (!accumulate)
      std::fill(D + (size_t)k0 * R, D + (size_t)k1 * R, 0.0f);
    for (unsigned int m = 0; m < M; ++m) {
      const float *x = X + (size_t)m * K;
      const float *t = T + (size_t)m * R;
      for (unsigned int k = k0; k < k1; ++k) {
        const float xv = x[k];
        float *d = D + (size_t)k * R;
        for (unsigned int j = 0; j < R; ++j)
          d[j] += xv * t[j];
      }
    }
  });
}

void skinny_xtt(const float *X, const float *T, float *D, unsigned int M,
                unsigned int K, unsigned int r, bool accumulate) {
  if (r == 8)
    return skinny_xtt_r<8>(X, T, D, M, K, accumulate);
  if (r == 16)
    return skinny_xtt_r<16>(X, T, D, M, K, accumulate);
  ThreadManager::Global().parallel_for(0, K, [=](size_t k) {
    for (unsigned int j = 0; j < r; ++j) {
      float a = accumulate ? D[k * r + j] : 0.0f;
      for (unsigned int m = 0; m < M; ++m)
        a += X[(size_t)m * K + k] * T[(size_t)m * r + j];
      D[k * r + j] = a;
    }
  });
}

/** [DIAG] NNTR_LORA_SKINNY_CHECK=1: max |a-b| / max |b| of two buffers */
[[maybe_unused]] void skinnyCheck(const char *what, const float *a,
                                  const float *b, size_t n) {
  static const bool on = std::getenv("NNTR_LORA_SKINNY_CHECK") != nullptr;
  static std::map<std::string, int> printed;
  if (!on || printed[what] > 40)
    return;
  double md = 0, mr = 0;
  for (size_t i = 0; i < n; ++i) {
    md = std::max(md, (double)std::abs(a[i] - b[i]));
    mr = std::max(mr, (double)std::abs(b[i]));
  }
  if (mr == 0.0 && md == 0.0)
    return; // trivially equal (e.g. LoRA B still zero): not informative
  ++printed[what];
  std::cout << "[SKINNY-CHECK] " << what << " rel=" << (mr > 0 ? md / mr : md)
            << " (max|ref|=" << mr << ")\n";
}

inline bool skinnyCheckOn() {
  static const bool on = std::getenv("NNTR_LORA_SKINNY_CHECK") != nullptr;
  return on;
}

/**
 * [DIAG] NNTR_TRAIN_GRAD_DUMP=<dir>: for the first NNTR_TRAIN_GRAD_DUMP_ITERS
 * (default 3) calcGradient calls of every LoRA FC, write the pre-update
 * loraA / loraB and their gradients as raw FP32 to <dir>/<layer>_it<k>_*.bin.
 * Consumed by Applications/CausalLM/tools/lora_gradcheck (reference gradients
 * and optimizer-update checks against PyTorch).
 */
[[maybe_unused]] void loraGradDump(const std::string &layer, const Tensor &A,
                                   const Tensor &B, const Tensor &dA,
                                   const Tensor &dB) {
  static const char *dir = std::getenv("NNTR_TRAIN_GRAD_DUMP");
  if (dir == nullptr)
    return;
  static const int max_it = [] {
    const char *e = std::getenv("NNTR_TRAIN_GRAD_DUMP_ITERS");
    return e ? std::atoi(e) : 3;
  }();
  static std::map<std::string, int> count;
  const int it = ++count[layer];
  if (it > max_it)
    return;
  auto w = [&](const char *tag, const Tensor &t) {
    const std::string path = std::string(dir) + "/" + layer + "_it" +
                             std::to_string(it) + "_" + tag + ".bin";
    if (FILE *f = std::fopen(path.c_str(), "wb")) {
      std::fwrite(t.getData<float>(), sizeof(float), t.size(), f);
      std::fclose(f);
    }
  };
  w("A", A);
  w("B", B);
  w("dA", dA);
  w("dB", dB);
}

/** whether the rank-r LoRA GEMMs may take the skinny kernels */
bool skinnyOk(const Tensor &t) {
  return t.getDataType() == Tdatatype::FP32 && t.getContiguous() &&
         t.getFormat() == Tformat::NCHW;
}

/**
 * @brief Whether the frozen-weight dX of a LoRA FC runs as an int8 (Q8_0)
 * GEMM. NNTR_TRAIN_DX_Q8=0/1 overrides; default on for aarch64 (i8mm) only.
 */
bool useQ8Dx() {
  static const bool on = [] {
    if (const char *e = std::getenv("NNTR_TRAIN_DX_Q8"))
      return e[0] != '0';
#if defined(__aarch64__)
    return true;
#else
    return false;
#endif
  }();
  return on;
}

} // namespace

void FullyConnectedLayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);
  Tensor &hidden_ = context.getOutput(SINGLE_INOUT_IDX);
  Tensor &input_ = context.getInput(SINGLE_INOUT_IDX);

  ///@todo This dequantization action should be moved to tensor.dot()
  {
    train_prof::Scope tp(std::get<props::LoraRank>(fc_props).empty()
                           ? "fc.base_fwd.nolora"
                           : "fc.base_fwd.lora",
                         train_prof::FWD);
    /// [DIAG] NNTR_TRAIN_FWD_FP32=1: a QS4CX base weight's training forward
    /// as FP32 activation x dequantized W instead of the KleidiAI kernel
    /// (whose per-row int8 activation quantization the FP32 backward does
    /// not see), so gradients can be compared exactly against a reference.
    static const bool fwd_fp32 = std::getenv("NNTR_TRAIN_FWD_FP32") != nullptr;
    if (fwd_fp32 && training &&
        weight.getDataType() == TensorDim::DataType::QS4CX) {
      Tensor weight_arena_view;
      dequantizeQs4cxToArena(weight, weight_arena_view);
      input_.dot(weight_arena_view, hidden_, false, false);
    } else if (quantizer != nullptr) {
      Tensor weight_arena_view;
      dequantizeToArena(weight, *quantizer, weight_arena_view);
      input_.dot(weight_arena_view, hidden_, false, false);
    } else {
      input_.dot(weight, hidden_, false, false);
    }
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
      train_prof::Scope tp_fq("fc.lora_fakequant+stats", train_prof::FWD);
      a_fq = fakeQuantizeQ4_0(loraA, lora_a_block_d, training);
      b_fq = fakeQuantizeQ4_0(loraB, lora_b_block_d, training);

      // Publish the current calibration to the registries so the
      // Application layer can force-feed these scales at save time. Only a
      // training forward moves the scales.
      if (training && !lora_a_block_d.empty() && !lora_b_block_d.empty()) {
        LoRAQATStats s;
        s.a_min = loraA.minValue();
        s.a_max = loraA.maxValue();
        s.a_scale = meanAbs(lora_a_block_d);
        s.b_min = loraB.minValue();
        s.b_max = loraB.maxValue();
        s.b_scale = meanAbs(lora_b_block_d);
        s.valid = true;
        std::lock_guard<std::mutex> lk(s_registry_mutex);
        s_qat_registry[layer_name_] = s;
        s_block_d_registry[layer_name_] = {lora_a_block_d, lora_b_block_d};
      }
      tp_fq.stop();
    }
    // hidden_ += s * (x A) B. loraTmp keeps the unscaled x A (calcGradient
    // reads it); the scale goes on the rank-sized copy, and the second GEMM
    // accumulates into hidden_ (beta = 1) -- no full-size LoRA output plane,
    // no separate scale and add passes over it.
    train_prof::Scope tp_g("fc.lora_fwd_gemm", train_prof::FWD);
    const Tensor &fa = lora_qat ? a_fq : loraA;
    const Tensor &fb = lora_qat ? b_fq : loraB;
    const unsigned int l_rank = fa.getDim().width();
    const unsigned int l_in = fa.getDim().height();
    const unsigned int l_out = fb.getDim().width();
    const unsigned int l_rows = input_.getDim().batch() *
                                input_.getDim().channel() *
                                input_.getDim().height();
    if (gemm_ops_ == nullptr && skinnyOk(input_) && skinnyOk(fa) &&
        skinnyOk(fb) && skinnyOk(hidden_) && skinnyOk(hidden_tmp_lora) &&
        input_.getDim().width() == l_in &&
        hidden_.size() == (size_t)l_rows * l_out) {
      Tensor h_before;
      if (skinnyCheckOn())
        h_before = hidden_.clone();
      skinny_xw(input_.getData<float>(), fa.getData<float>(),
                hidden_tmp_lora.getData<float>(), l_rows, l_in, l_rank);
      skinny_uv_acc(hidden_tmp_lora.getData<float>(), fb.getData<float>(),
                    hidden_.getData<float>(), l_rows, l_rank, l_out,
                    lora_scaling);
      if (skinnyCheckOn()) {
        Tensor t_ref(hidden_tmp_lora.getDim());
        input_.dot(fa, t_ref, false, false);
        skinnyCheck("fwd T", hidden_tmp_lora.getData<float>(),
                    t_ref.getData<float>(), t_ref.size());
        Tensor o_ref(hidden_.getDim());
        t_ref.multiply(lora_scaling).dot(fb, o_ref, false, false);
        o_ref.add_i(h_before);
        skinnyCheck("fwd H", hidden_.getData<float>(), o_ref.getData<float>(),
                    o_ref.size());
      }
    } else {
      gemm(input_, fa, hidden_tmp_lora, false, false);
      Tensor tmp_scaled = hidden_tmp_lora.multiply(lora_scaling);
      if (hidden_.getDataType() == hidden_tmp_lora.getDataType()) {
        gemm(tmp_scaled, fb, hidden_, false, false, 1.0f);
      } else {
        gemm(tmp_scaled, fb, hidden_out_lora, false, false);
        hidden_.add_i(hidden_out_lora);
      }
    }
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

void FullyConnectedLayer::releaseDxPlainPayloads(Tensor &weight) {
  if (dx_plain_released_)
    return;
  static const bool keep = [] {
    const char *e = std::getenv("NNTR_TRAIN_DROP_PLAIN");
    return e != nullptr && e[0] == '0';
  }();
  // NNTR_TRAIN_FWD_FP32 dequantizes W from its plain payload every forward.
  static const bool fwd_fp32 = std::getenv("NNTR_TRAIN_FWD_FP32") != nullptr;
  if (keep || fwd_fp32 || !weight.isPacked() || !dx_wt_qs4t_.isPacked())
    return;
  dx_plain_released_ = true;
  // dx_wt_qs4t_ is a standalone tensor and always owns its payload. W owns
  // its payload only under the QS4CX heap bypass; otherwise it is a slice of
  // the weight pool and stays.
  auto release = [](Tensor &t) {
    const void *base = t.getData<uint8_t>();
    const size_t bytes = t.getMemoryBytes();
    if (releaseHostPages(base, bytes))
      markQs4cxPayloadDropped(base, bytes);
  };
  release(dx_wt_qs4t_);
  if (qs4cxHeapBypassOn())
    release(weight);
}

void FullyConnectedLayer::calcDerivative(RunLayerContext &context) {
  Tensor &weight = context.getWeight(weight_idx[FCParams::weight]);

  const Tensor &derivative_ = context.getIncomingDerivative(SINGLE_INOUT_IDX);
  Tensor &ret_ = context.getOutgoingDerivative(SINGLE_INOUT_IDX);

  if (!std::get<props::LoraRank>(fc_props).empty()) {
    // dX = dY . W^T for the frozen base weight W (no gradient flows into W).
    // A quantized W reaches it one of three ways (forward never dequantizes:
    // Q4_0 / QS4CX dispatch straight to their integer GEMMs in Tensor::dot):
    //  - QS4CX: a unit-scale transposed copy of the int4 codes with the
    //    per-channel scale folded into dY, on the KleidiAI GEMM (kai_dx);
    //  - Q4_0 / Q6_K / QINT*: W re-quantized once to Q8_0 along N and the
    //    ggml int8 GEMM (q8_dx);
    //  - otherwise (or NNTR_TRAIN_DX_Q8=0, non-aarch64): W dequantized to FP32
    //    in a persistent arena (see dequantizeToArena) and an FP32 GEMM.
    using DT = TensorDim::DataType;
    Tensor w_arena_view;
    const Tensor *w = &weight;
    auto dequantize_base = [&]() {
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
      } else if (weight.getDataType() == DT::QS4CX) {
        // dotQs4cx ignores trans flags, so a QS4CX base must never reach the
        // FP32 gemm(dY, W, false, true) below undequantized.
        dequantizeQs4cxToArena(weight, w_arena_view);
        w = &w_arena_view;
      }
    };

    // int8 dX: W (K x N, frozen) is re-quantized once to Q8_0 blocks along N
    // -- the reduction axis of dX = dY * W^T, so the per-block scales factor
    // out of the int8 dot -- and kept repacked for the i8mm GEMM. The Q4_0
    // forward layout cannot be reused: its blocks run along K.
    const TensorDim &wd = weight.getDim();
    const TensorDim &dd0 = derivative_.getDim();
    const unsigned int w_k = wd.height(), w_n = wd.width();
    const unsigned int d_rows = dd0.batch() * dd0.channel() * dd0.height();
    const bool q8_dx =
      useQ8Dx() &&
      (quantizer != nullptr || weight.getDataType() == DT::Q4_0 ||
       weight.getDataType() == DT::Q6_K || weight.getDataType() == DT::QS4CX) &&
      w_k % 4 == 0 && w_n % 32 == 0 && dd0.width() == w_n &&
      derivative_.getDataType() == DT::FP32 && ret_.getDataType() == DT::FP32 &&
      derivative_.getContiguous() && ret_.getContiguous() &&
      ret_.size() == (size_t)d_rows * w_k;
#if defined(__aarch64__)
    const bool kai_dx =
      useQ8Dx() && useKaiDx() && weight.getDataType() == DT::QS4CX &&
      dd0.width() == w_n && w_n % 32 == 0 &&
      derivative_.getDataType() == DT::FP32 && ret_.getDataType() == DT::FP32 &&
      derivative_.getContiguous() && ret_.getContiguous();
#else
    const bool kai_dx = false;
#endif
    if (kai_dx) {
      train_prof::Scope tp("fc.deriv_base_dx_kai", train_prof::DERIV);
      if (dx_wt_qs4t_.empty())
        dx_wt_qs4t_ = buildQs4cxTransposed(weight);
      if (dx_scale_.empty()) {
        const float *ws = static_cast<const float *>(weight.getScale());
        dx_scale_.assign(ws, ws + w_n);
      }
      if (dx_scaled_dy_.empty() || dx_scaled_dy_.getDim() != dd0)
        dx_scaled_dy_ = Tensor(dd0, true);
      // G = dY * s (per output channel): s is W's scale, folded out of the
      // unit-scale transposed codes.
      const float *s = dx_scale_.data();
      const float *g_in = derivative_.getData<float>();
      float *g = dx_scaled_dy_.getData<float>();
      ThreadManager::Global().parallel_for(0, d_rows, [=](size_t r) {
        const float *a = g_in + r * w_n;
        float *o = g + r * w_n;
        for (unsigned int n = 0; n < w_n; ++n)
          o[n] = a[n] * s[n];
      });
      dx_scaled_dy_.dot(dx_wt_qs4t_, ret_, false, false);
      static const bool kai_check =
        std::getenv("NNTR_TRAIN_DX_CHECK") != nullptr;
      if (!kai_check)
        releaseDxPlainPayloads(weight);
      static int kai_checked = 0;
      if (kai_check && kai_checked < 64) {
        ++kai_checked;
        dequantize_base();
        Tensor ref(ret_.getDim());
        derivative_.dot(*w, ref, false, true);
        const float *r = ref.getData<float>();
        const float *q = ret_.getData<float>();
        double num = 0, den = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
          num += ((double)q[i] - r[i]) * ((double)q[i] - r[i]);
          den += (double)r[i] * r[i];
        }
        std::cout << "[PERF-DXCHECK] kai " << layer_name_
                  << " rel_l2=" << (den > 0 ? std::sqrt(num / den) : 0) << "\n";
      }
    }
    train_prof::Scope tp_dq("fc.deriv_dequant", train_prof::DERIV);
    if (kai_dx) {
      // dX already computed above
    } else if (q8_dx && dx_wt_q8x4_.empty()) {
      dequantize_base();
      const size_t bytes = (size_t)w_k * (w_n / 32) * 34; // block_q8_0
      std::vector<uint8_t> plain(bytes);
      __ggml_quantize_q8_0(w->getData<float>(), plain.data(), w_k, w_n,
                           nullptr);
      dx_wt_q8x4_.resize(bytes);
      __ggml_repack_q8_0_to_q8_0_4x8(dx_wt_q8x4_.data(), plain.data(), bytes,
                                     w_k, w_n);
    } else if (!q8_dx) {
      dequantize_base();
    }

    const bool lora_qat_deriv = !std::get<props::LoraQAT>(fc_props).empty() &&
                                std::get<props::LoraQAT>(fc_props).get();
    // dX = dH @ (W + sAB)^T = dH @ W^T + s * (dH @ B^T) @ A^T.
    // The LoRA part is a rank-limited chain — never materialize the
    // full out*in LoRA matrix (sAB) nor a full-size (W + sAB) temp.
    tp_dq.stop();
    if (kai_dx) {
      // dX already computed above
    } else if (q8_dx) {
      train_prof::Scope tp("fc.deriv_base_dx_q8gemm", train_prof::DERIV);
      __ggml_q8_0_4x8_q8_0_GEMM(d_rows, w_k, w_n, derivative_.getData<float>(),
                                w_n, dx_wt_q8x4_.data(), w_n,
                                ret_.getData<float>(), w_k);
      /// [PERF-DXCHECK] NNTR_TRAIN_DX_CHECK=1: compare against the FP32 path
      static const bool dx_check =
        std::getenv("NNTR_TRAIN_DX_CHECK") != nullptr;
      static int dx_checked = 0;
      if (dx_check && dx_checked < 64) {
        ++dx_checked;
        dequantize_base();
        Tensor ref(ret_.getDim());
        derivative_.dot(*w, ref, false, true);
        const float *r = ref.getData<float>();
        const float *q = ret_.getData<float>();
        double num = 0, den = 0, max_abs = 0, max_ref = 0;
        for (size_t i = 0; i < ref.size(); ++i) {
          const double e = (double)q[i] - r[i];
          num += e * e;
          den += (double)r[i] * r[i];
          max_abs = std::max(max_abs, std::abs(e));
          max_ref = std::max(max_ref, (double)std::abs(r[i]));
        }
        std::cout << "[PERF-DXCHECK] " << layer_name_ << " M=" << d_rows
                  << " K=" << w_k << " N=" << w_n
                  << " rel_l2=" << (den > 0 ? std::sqrt(num / den) : 0)
                  << " max_abs/max_ref="
                  << (max_ref > 0 ? max_abs / max_ref : 0) << "\n";
      }
    } else {
      train_prof::Scope tp("fc.deriv_base_dx_fp32gemm", train_prof::DERIV);
      gemm(derivative_, *w, ret_, false, true);
    }
    train_prof::Scope tp_l("fc.deriv_lora_dx", train_prof::DERIV);
    Tensor &lora_A = context.getWeight(lora_idx[LORAParams::loraA]);
    Tensor &lora_B = context.getWeight(lora_idx[LORAParams::loraB]);
    // Straight-through estimator: dL/dx uses the same fake-quantized
    // a_fq/b_fq the forward pass used, not the raw (unquantized) weights.
    const Tensor &dB = lora_qat_deriv ? b_fq : lora_B;
    const Tensor &dA = lora_qat_deriv ? a_fq : lora_A;
    // (dH @ B^T) is [rows, rank], then @ A^T is [rows, in].
    const TensorDim &dd = derivative_.getDim();
    const unsigned int l_rank = dA.getDim().width();
    const unsigned int l_in = dA.getDim().height();
    const unsigned int l_out = dB.getDim().width();
    const unsigned int l_rows = dd.batch() * dd.channel() * dd.height();
    if (gemm_ops_ == nullptr && skinnyOk(derivative_) && skinnyOk(dA) &&
        skinnyOk(dB) && skinnyOk(ret_) &&
        ret_.size() == (size_t)l_rows * l_in) {
      // s * dY . B^T: reuse the loraTmp gradient calcGradient just formed
      // (same B), else compute it.
      Tensor dtmp_r;
      const float *u = nullptr;
      if (lora_dtmp_ready_ &&
          lora_dtmp_copy_.size() == (size_t)l_rows * l_rank) {
        u = lora_dtmp_copy_.data();
      } else {
        dtmp_r = Tensor(TensorDim(1, 1, l_rows, l_rank), true);
        skinny_ybt(derivative_.getData<float>(), dB.getData<float>(),
                   dtmp_r.getData<float>(), l_rows, l_out, l_rank,
                   lora_scaling);
        u = dtmp_r.getData<float>();
      }
      lora_dtmp_ready_ = false;
      // dX += U . A^T: A is (in x rank); transpose it once (rank x in).
      std::vector<float> at((size_t)l_rank * l_in);
      const float *a = dA.getData<float>();
      for (unsigned int k = 0; k < l_in; ++k)
        for (unsigned int j = 0; j < l_rank; ++j)
          at[(size_t)j * l_in + k] = a[(size_t)k * l_rank + j];
      Tensor dx_before;
      if (skinnyCheckOn())
        dx_before = ret_.clone();
      skinny_uv_acc(u, at.data(), ret_.getData<float>(), l_rows, l_rank, l_in,
                    1.0f);
      if (skinnyCheckOn()) {
        Tensor u_ref(TensorDim(dd.batch(), dd.channel(), dd.height(), l_rank,
                               dd.getTensorType()));
        derivative_.dot(dB, u_ref, false, true);
        u_ref.multiply_i(lora_scaling);
        skinnyCheck("deriv U", u, u_ref.getData<float>(), u_ref.size());
        Tensor dx_ref(ret_.getDim());
        u_ref.dot(dA, dx_ref, false, true);
        dx_ref.add_i(dx_before);
        skinnyCheck("deriv dX", ret_.getData<float>(), dx_ref.getData<float>(),
                    dx_ref.size());
      }
    } else {
      Tensor dtmp_r(TensorDim(dd.batch(), dd.channel(), dd.height(),
                              dB.getDim().height(), dd.getTensorType()));
      gemm(derivative_, dB, dtmp_r, false, true);
      // Scale the rank-sized intermediate and accumulate into ret_ (beta = 1)
      // instead of materializing, scaling and adding a full-size dX plane.
      dtmp_r.multiply_i(lora_scaling);
      gemm(dtmp_r, dA, ret_, false, true, 1.0f);
    }
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

    const bool lora_qat_grad0 = !std::get<props::LoraQAT>(fc_props).empty() &&
                                std::get<props::LoraQAT>(fc_props).get();
    const bool first_b =
      context.isGradientFirstAccess(lora_idx[LORAParams::loraB]);
    const bool first_tmp =
      context.isGradientFirstAccess(lora_idx[LORAParams::loraTmp]);
    const Tensor &gB = lora_qat_grad0 ? b_fq : loraB;
    const unsigned int g_rank = gB.getDim().height();
    const unsigned int g_out = gB.getDim().width();
    const unsigned int g_in = input_.getDim().width();
    const unsigned int g_rows = input_.getDim().batch() *
                                input_.getDim().channel() *
                                input_.getDim().height();
    if (first_b && first_tmp && gemm_ops_ == nullptr && skinnyOk(loraTmp) &&
        skinnyOk(derivative_) && skinnyOk(gB) && skinnyOk(input_) &&
        skinnyOk(djdlb) && skinnyOk(djdtmp) && skinnyOk(djdla) &&
        djdlb.size() == (size_t)g_rank * g_out &&
        djdtmp.size() == (size_t)g_rows * g_rank &&
        djdla.size() == (size_t)g_in * g_rank) {
      skinny_tty(loraTmp.getData<float>(), derivative_.getData<float>(),
                 djdlb.getData<float>(), g_rows, g_rank, g_out, lora_scaling);
      skinny_ybt(derivative_.getData<float>(), gB.getData<float>(),
                 djdtmp.getData<float>(), g_rows, g_out, g_rank, lora_scaling);
      skinny_xtt(input_.getData<float>(), djdtmp.getData<float>(),
                 djdla.getData<float>(), g_rows, g_in, g_rank,
                 !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]));
      lora_dtmp_copy_.assign(djdtmp.getData<float>(),
                             djdtmp.getData<float>() + djdtmp.size());
      lora_dtmp_ready_ = true;
      loraGradDump(context.getName(),
                   context.getWeight(lora_idx[LORAParams::loraA]), loraB, djdla,
                   djdlb);
      if (skinnyCheckOn()) {
        Tensor db_ref(djdlb.getDim());
        loraTmp.dot(derivative_, db_ref, true, false);
        db_ref.multiply_i(lora_scaling);
        skinnyCheck("grad dB", djdlb.getData<float>(), db_ref.getData<float>(),
                    db_ref.size());
        Tensor dt_ref(djdtmp.getDim());
        derivative_.dot(gB, dt_ref, false, true);
        dt_ref.multiply_i(lora_scaling);
        skinnyCheck("grad dT", djdtmp.getData<float>(), dt_ref.getData<float>(),
                    dt_ref.size());
        Tensor da_ref(djdla.getDim());
        input_.dot(dt_ref, da_ref, true, false);
        skinnyCheck("grad dA", djdla.getData<float>(), da_ref.getData<float>(),
                    da_ref.size());
      }
      return;
    }
    if (first_b && first_tmp) {
      // dB = s * T^T dY and dT = s * dY B^T: apply s to the rank-sized
      // results rather than to a full-size copy of dY.
      gemm(loraTmp, derivative_, djdlb, true, false, 0.0f);
      djdlb.multiply_i(lora_scaling);
      gemm(derivative_, lora_qat_grad0 ? b_fq : loraB, djdtmp, false, true,
           0.0f);
      djdtmp.multiply_i(lora_scaling);
      gemm(input_, djdtmp, djdla, true, false,
           !context.isGradientFirstAccess(lora_idx[LORAParams::loraA]) ? 1.0f
                                                                       : 0.0f);
      return;
    }
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
