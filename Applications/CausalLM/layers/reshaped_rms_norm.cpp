// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Seungbaek Hong <sb92.hong@samsung.com>
 *
 * @file   reshaped_rms_norm.cpp
 * @date   19 July 2023
 * @brief  Implementation of custom RMS normalization function
 * @see    https://github.com/nntrainer/nntrainer
 * @author Seungbaek Hong <sb92.hong@samsung.com>
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Anirudh Bocha <b.saianirud@samsung.com>
 * @bug    No known bugs except for NYI items
 *
 */

#include <cmath>
#include <vector>
#include <cpu_backend.h>
#include <env_compat.h>
#include <layer_prof.h>
#include <reshaped_rms_norm.h>

#if defined(ENABLE_OPENCL)
// OpenCL GPU rmsnorm kernels (rmsnorm_cl / rmsnorm_cl_fp16). Guarded so the
// enable-opencl=false build compiles host-only.
#include <blas_kernels.h>
#endif
#include <memory_data.h>

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
#include <cuda_context_manager.h>
#include <cuda_rmsnorm.h>
#include <cuda_runtime.h>
#include <cuda_stream_manager.h>
#endif

namespace causallm {

static constexpr size_t SINGLE_INOUT_IDX = 0;

void ReshapedRMSNormLayer::finalize(nntrainer::InitLayerContext &context) {
  std::vector<nntrainer::TensorDim> dim = context.getInputDimensions();
  context.setOutputDimensions(dim);
  feature_size = std::get<props::FeatureSize>(rms_props);
  use_gamma = std::get<props::UseGamma>(rms_props).get();

  if (!std::get<nntrainer::props::SkipPrefill>(rms_props).empty())
    skip_prefill = std::get<nntrainer::props::SkipPrefill>(rms_props).get();

  NNTR_THROW_IF(dim[0].width() % feature_size != 0, std::invalid_argument)
    << "feature size must be a divisor of width";

  if (use_gamma) {
    // gamma is an unquantized weight read straight out of the model .bin, and
    // the .bin has no per-tensor dtype: every weight's file offset is derived
    // from the dtype the graph *requests* (NeuralNetwork::load() walks the
    // graph accumulating getMemoryBytes()). So the request must reproduce the
    // dtype the exporting graph used, which for a packed=false norm is the
    // activation dtype -- that is exactly what getWeightDataType() returns
    // here, and what nntr_quantize records in model_tensor_type. Hard-coding
    // FP32 both misreads an FP16-stored gamma and shifts the offset of every
    // weight after it. It would also disagree with the sibling
    // implementations of this same on-disk weight (RMSNormLayerGPU,
    // nntrainer::CudaRMSNormLayer, RMSNormLayerCl), which all use
    // getWeightDataType(). A package that genuinely stores FP32 gamma
    // declares an FP32 activation dtype and lands here as FP32 anyway; the
    // forward path below still casts gamma at the multiply site and keeps the
    // dtype-matched device kernels gated, so a mixed case stays correct.
    nntrainer::TensorDim gamma_dim(
      1, 1, 1, feature_size,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       context.getWeightDataType()));
    wt_idx[RMSParams::gamma] = context.requestWeight(
      gamma_dim, nntrainer::props::InitializerInfo::Enum::NONE,
      nntrainer::WeightRegularizer::NONE, 1.0f, 0.0f, "gamma", false);
  }

  /**
   * @note inv_rms cache: one value per feature_size-wide chunk, populated by
   * computeRMSNorm() during forwarding() and read back verbatim by
   * calcDerivative()/calcGradient(), instead of being recomputed from the
   * input tensor at backward time -- see RMSNormLayer's identical cache in
   * rms_norm.cpp for the full rationale (this layer has the same
   * cross-layer buffer-reuse hazard; it is used for Qwen3's per-head
   * Q-norm/K-norm, which sit just as deep in the 28-layer stack).
   */
  cache_inv_rms =
    (context.getExecutionMode() == ml::train::ExecutionMode::TRAIN);
  if (cache_inv_rms) {
    const unsigned int chunks_per_batch = dim[0].getFeatureLen() / feature_size;
    nntrainer::TensorDim inv_rms_dim(
      dim[0].batch(), 1, chunks_per_batch, 1,
      nntrainer::TensorDim::TensorType(context.getFormat(),
                                       nntrainer::TensorDim::DataType::FP32));
    wt_idx[RMSParams::inv_rms] = context.requestTensor(
      inv_rms_dim, "inv_rms", nntrainer::Initializer::NONE, false,
      nntrainer::TensorLifespan::ITERATION_LIFESPAN);
  }
}

void ReshapedRMSNormLayer::forwarding(nntrainer::RunLayerContext &context,
                                      bool training) {
  // incremental_forwarding() is being phased out as the inference entry point,
  // so run the full-sequence reshaped RMS norm through forwarding(). The shared
  // worker (incremental_forwarding) handles both the host and the GPU-resident
  // (isSVM) paths over the whole input [0, height).
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  incremental_forwarding(context, 0, in.height(), training);
}

void ReshapedRMSNormLayer::incremental_forwarding(
  nntrainer::RunLayerContext &context, unsigned int from, unsigned int to,
  bool training) {
  nntrainer::LayerProfScope _prof("rms_norm", (to - from) == 1);
  auto &epsilon = std::get<nntrainer::props::Epsilon>(rms_props).get();

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  // gamma weight is only requested in finalize() when use_gamma is true
  // (e.g. Gemma4 v_norm sets use_gamma=false). When false, wt_idx[gamma] is
  // left at UINT_MAX, so reading it would index out of bounds -- the norm is
  // gamma-free (identity scale) in that case.
  nntrainer::Tensor *gamma =
    use_gamma ? &context.getWeight(wt_idx[RMSParams::gamma]) : nullptr;

  ml::train::TensorDim in_dim = in.getDim();
  ml::train::TensorDim out_dim = out.getDim();

  ml::train::TensorDim in_step_dim = in_dim;
  ml::train::TensorDim out_step_dim = out_dim;

  // A multi-token step is a prefill even when it does not start at 0 (resumed
  // / chunked prefill), so recognize (to - from) > 1 as prefill too.
  bool is_prefill = !from || (to - from) > 1;
  if (skip_prefill && is_prefill && !training)
    return;

  in_step_dim.batch(1);
  in_step_dim.height(to - from);
  out_step_dim.batch(1);
  out_step_dim.height(to - from);

  // set reshaped dim to (1, 1, -1, feature_size)
  ml::train::TensorDim step_reshaped_dim = in_step_dim;

  step_reshaped_dim.width(feature_size);
  step_reshaped_dim.height(in_step_dim.height() *
                           (in_dim.width() / feature_size));

  unsigned int b_size = in_dim.batch();

  for (unsigned int b = 0; b < b_size; ++b) {
    nntrainer::Tensor in_step =
      in.getSharedDataTensor(in_step_dim, b * in_dim.getFeatureLen(), true);
    nntrainer::Tensor out_step =
      out.getSharedDataTensor(out_step_dim, b * out_dim.getFeatureLen(), true);

    // reshape in_step
    // reshape out_step
    in_step.reshape(step_reshaped_dim);
    out_step.reshape(step_reshaped_dim);

    // GPU-resident path: when the activation, output and gamma are all SVM
    // (the graph runs on the SVM pool), normalize each feature_size chunk on
    // the GPU SVM-direct -- no host round-trip. The GPU rmsnorm kernel folds
    // gamma in, so the separate multiply_i(gamma) is skipped on this path.
    const auto in_md = in_step.getMemoryData();
    const auto out_md = out_step.getMemoryData();
    const auto gamma_md = gamma ? gamma->getMemoryData() : nullptr;
    // The GPU rmsnorm kernel folds gamma in, so the SVM-direct path requires a
    // gamma weight; fall back to the host path when use_gamma is false.
    // The SVM kernels read gamma at the activation dtype, so they may only run
    // when gamma's dtype actually matches. finalize() requests gamma at the
    // graph's weight dtype, which for these packed=false norms is the
    // activation dtype, so this holds for every package whose norm gamma was
    // exported at the activation dtype. A package that pins gamma to another
    // dtype falls through to the host path, which casts at the multiply site.
    const bool gamma_dtype_ok =
      gamma && gamma->getDataType() == in_step.getDataType();
    const bool use_svm = gamma && gamma_dtype_ok && in_md && in_md->isSVM() &&
                         out_md && out_md->isSVM() && gamma_md &&
                         gamma_md->isSVM();
    // Gamma-free norm on the GPU. A norm with use_gamma=false has no gamma
    // weight, so the path above falls through to the host, which costs a full
    // queue drain plus two blocking SVM maps per call. Route it instead to the
    // gamma-free cooperative kernel whenever the operands are SVM FP16: that
    // kernel accumulates the sum of squares in FP32, exactly like the host
    // intrinsic, and skips the gamma fold. This is on by default because a
    // gamma-free norm registered on the GPU context holds device-allocated
    // tensors, on which the host fallback's Tensor ops would fault; the
    // NNTR_VNORM_HOST escape hatch is there for bring-up on a new device.
    static const bool vnorm_host = std::getenv("NNTR_VNORM_HOST") != nullptr;
    const bool use_svm_ng =
      !gamma && !vnorm_host && in_md && in_md->isSVM() && out_md &&
      out_md->isSVM() &&
      in_step.getDataType() == ml::train::TensorDim::DataType::FP16;
    const unsigned int n_rows = in_step.getDim().height();
    bool gpu_done = false;
#if defined(ENABLE_OPENCL)
    // GPU-resident SVM rmsnorm dispatch. Without OpenCL gpu_done stays false
    // and the host RMSNorm loop below runs.
    if (use_svm) {
      if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        nntrainer::rmsnorm_cl(in_step.getData<float>(), gamma->getData<float>(),
                              out_step.getData<float>(), epsilon, n_rows,
                              feature_size, /*use_svm=*/true);
        gpu_done = true;
#ifdef ENABLE_FP16
      } else if (in_step.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
        // Bind the device sub-buffer when an operand lives on the device
        // residency plane (a per-head norm consumes a quantized FC output that
        // is written straight to a device buffer). Without this the SVM path
        // reads a stale host shadow of that plane and produces garbage. gamma
        // stays SVM, matching RMSNormLayerGPU.
        void *in_cl = in_step.isClMem() ? in_step.getClMem() : nullptr;
        void *out_cl = out_step.isClMem() ? out_step.getClMem() : nullptr;
        nntrainer::rmsnorm_cl_fp16(
          in_step.getData<_FP16>(), gamma->getData<_FP16>(),
          out_step.getData<_FP16>(), epsilon, n_rows, feature_size,
          /*use_svm=*/true, out_cl, in_cl, /*feeds_fc=*/false);
        gpu_done = true;
#endif
      }
#ifdef ENABLE_FP16
    } else if (use_svm_ng) {
      // Gamma-free GPU v_norm: same cl_mem/SVM operand binding as above, gamma
      // is null so the wrapper dispatches the _ng kernel (no gamma fold).
      void *in_cl = in_step.isClMem() ? in_step.getClMem() : nullptr;
      void *out_cl = out_step.isClMem() ? out_step.getClMem() : nullptr;
      nntrainer::rmsnorm_cl_fp16(in_step.getData<_FP16>(), /*gamma=*/nullptr,
                                 out_step.getData<_FP16>(), epsilon, n_rows,
                                 feature_size,
                                 /*use_svm=*/true, out_cl, in_cl);
      gpu_done = true;
#endif
    }
#endif // ENABLE_OPENCL

#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1 && defined(ENABLE_FP16)
    // CUDA per-head norm: each feature_size chunk is one rmsnorm row, so this
    // reuses cuda_rmsnorm_fp16 with rows = n_rows and width = feature_size
    // (gamma is null for a gamma-free norm). Keeps the q/k/v norms on the
    // device. Opt-in through NNTR_CUDA_QKNORM.
    if (!gpu_done &&
        in_step.getDataType() == ml::train::TensorDim::DataType::FP16) {
      static const bool gpu = nntr_env_on("NNTR_CUDA_QKNORM");
      if (gpu) {
        auto *ip =
          reinterpret_cast<const unsigned short *>(in_step.getData<_FP16>());
        auto *op =
          reinterpret_cast<unsigned short *>(out_step.getData<_FP16>());
        // Only bind gamma when it is genuinely FP16 storage: reading a
        // non-FP16 gamma's bytes as _FP16 would corrupt the scale. gamma
        // normally matches the activation dtype (finalize), so this binds; a
        // gamma pinned to another dtype falls to the host path below.
        const unsigned short *gp =
          gamma_dtype_ok
            ? reinterpret_cast<const unsigned short *>(gamma->getData<_FP16>())
            : nullptr;
        const bool gamma_bindable = !gamma || gamma_dtype_ok;
        auto dev_ok = [](const void *p) {
          if (!p)
            return true;
          return nntrainer::cuda::dev_accessible(p);
        };
        if (gamma_bindable && dev_ok(ip) && dev_ok(op) && dev_ok(gp) &&
            nntrainer::cuda::cuda_rmsnorm_fp16(ip, gp, op, epsilon, n_rows,
                                               feature_size))
          gpu_done = true;
      }
    }
#endif

    if (!gpu_done) {
#if defined(ENABLE_CUDA) && ENABLE_CUDA == 1
      // The host normalization below reads the GPU-produced input (and gamma)
      // with the CPU, so in async mode the stream has to be drained first.
      // It belongs HERE, not above the device dispatch: when the device norm
      // ran (gpu_done), nothing on this path touches those bytes from the host,
      // and draining there stalls the whole prefill pipeline once per q/k/v
      // norm -- the drain the async mode exists to avoid.
      nntrainer::cuda::drain_if_async();
#endif
      if (in_step.getDataType() == ml::train::TensorDim::DataType::FP32) {
        ///@todo rms_norm_wrt_width_something() should be refactored to
        /// nntrainer::Tensor operation.
        // fp32_intrinsic (not fp16_intrinsic) even under ENABLE_FP16 -- the
        // fp16 variant overflows (upstream RMSNorm FP16 overflow fix).
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_step.getData<float>(), out_step.getData<float>(),
          in_step.getDim().height(), in_step.getDim().width(), epsilon);
      } else if (in_step.getDataType() ==
                 ml::train::TensorDim::DataType::FP16) {
        // FP16 path: the sum-of-squares MUST be computed in FP32. Doing
        // in_step.multiply(in_step) in FP16 squares each element in FP16, so
        // any per-row element |x| > ~256 (sqrt(65504)) overflows to +Inf ->
        // average -> Inf -> inv_sqrt = 0 -> the whole row is zeroed, which
        // zeroes the attention that consumes it and cascades to NaN.
        // Normalize via the same FP32 intrinsic the FP32 branch uses by
        // converting in to FP32, then cast the normalized result back to FP16.
        // This stays inside the residual stream (out is still FP16) and reuses
        // the proven-correct FP32 RMSNorm (no overflow).
        nntrainer::Tensor in_f32 =
          in_step.clone(ml::train::TensorDim::DataType::FP32);
        nntrainer::Tensor out_f32(in_f32.getDim());
        nntrainer::rms_norm_wrt_width_fp32_intrinsic(
          in_f32.getData<float>(), out_f32.getData<float>(),
          in_step.getDim().height(), in_step.getDim().width(), epsilon);
        out_step.copyData(out_f32);
      } else {
        throw std::invalid_argument(
          "reshaped_rms_norm NYI dtype=" +
          std::to_string(static_cast<int>(in_step.getDataType())) +
          " layer=" + context.getName());
      }
      if (gamma) {
        // gamma normally matches the activation dtype; cast when a package
        // pins it to a different one (e.g. FP32 gamma under FP16 activation)
        // before the elementwise multiply.
        if (gamma->getDataType() != out_step.getDataType()) {
          nntrainer::Tensor gamma_cast = gamma->clone(out_step.getDataType());
          out_step.multiply_i(gamma_cast);
        } else {
          out_step.multiply_i(*gamma);
        }
      }

      // Cache inv_rms per chunk for calcDerivative()/calcGradient() to read
      // back verbatim -- see the note in finalize() for why this must not
      // be recomputed from the input tensor at backward time instead. Host
      // path only: training graphs are host-resident FP32.
      if (cache_inv_rms && in_step.getDataType() ==
                             ml::train::TensorDim::DataType::FP32) {
        nntrainer::Tensor &inv_rms_cache =
          context.getTensor(wt_idx[RMSParams::inv_rms]);
        const unsigned int chunks_from =
          from * (in_dim.width() / feature_size);
        float *inv_rms_row = inv_rms_cache.getData<float>() +
                             b * inv_rms_cache.getDim().getFeatureLen() +
                             chunks_from;
        const float *x_chunk = in_step.getData<float>();
        const unsigned int chunks = in_step.getDim().height();
        for (unsigned int c = 0; c < chunks; ++c) {
          float ms = 0.0f;
          for (unsigned int w = 0; w < feature_size; ++w)
            ms += x_chunk[c * feature_size + w] * x_chunk[c * feature_size + w];
          ms /= feature_size;
          inv_rms_row[c] = 1.0f / std::sqrt(ms + epsilon);
        }
      }
    }

    // reshape again out_step
    out_step.reshape(out_step_dim);

#ifdef DEBUG
    std::cout << context.getName() << " \n input:" << in_step
              << "output:" << out_step << std::endl;
#endif
  }
}

void ReshapedRMSNormLayer::updateTensorsByInputDimensions(
  nntrainer::RunLayerContext &context,
  std::vector<nntrainer::TensorDim> input_dimensions) {
  context.updateInput(SINGLE_INOUT_IDX, input_dimensions[0]);
  context.updateOutput(SINGLE_INOUT_IDX, input_dimensions[0]);

  if (cache_inv_rms) {
    const unsigned int chunks_per_batch =
      input_dimensions[0].getFeatureLen() / feature_size;
    nntrainer::TensorDim inv_rms_dim(input_dimensions[0].batch(), 1,
                                     chunks_per_batch, 1);
    context.updateTensor(wt_idx[RMSParams::inv_rms], inv_rms_dim);
  }
}

void ReshapedRMSNormLayer::setBatch(nntrainer::RunLayerContext &context,
                                    unsigned int batch) {
  if (cache_inv_rms)
    context.updateTensor(wt_idx[RMSParams::inv_rms], batch);
}

/**
 * @brief calcDerivative for ReshapedRMSNorm.
 * @details Identical math to RMSNormLayer::calcDerivative, but each row of
 *          width `feature_size` (rather than the full tensor width) is
 *          normalized independently, matching the reshape done in
 *          computeRMSNorm(). dgamma is computed in calcGradient(), which
 *          the framework only calls when the layer is trainable.
 */
void ReshapedRMSNormLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &inv_rms_cache =
    context.getTensor(wt_idx[RMSParams::inv_rms]);

  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[reshaped_rms_norm] calcDerivative only supports FP32 for now";

  nntrainer::Tensor gamma_fp32;
  const float *g = nullptr;
  if (use_gamma) {
    nntrainer::Tensor &gamma = context.getWeight(wt_idx[RMSParams::gamma]);
    gamma_fp32 = (gamma.getDataType() == ml::train::TensorDim::DataType::FP32)
                   ? gamma
                   : gamma.clone(ml::train::TensorDim::DataType::FP32);
    g = gamma_fp32.getData<float>();
  }

  const ml::train::TensorDim &in_dim = in.getDim();
  const unsigned int chunks_per_batch = in_dim.getFeatureLen() / feature_size;
  const unsigned int batch = in_dim.batch();
  const unsigned int inv_rms_stride = inv_rms_cache.getDim().getFeatureLen();

  const float *x = in.getData<float>();
  const float *dy_ = dy.getData<float>();
  const float *inv_rms_ = inv_rms_cache.getData<float>();
  float *dx_ = dx.getData<float>();

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x_b = x + b * in_dim.getFeatureLen();
    const float *dy_b = dy_ + b * in_dim.getFeatureLen();
    float *dx_b = dx_ + b * in_dim.getFeatureLen();
    const float *inv_rms_b = inv_rms_ + b * inv_rms_stride;

    for (unsigned int r = 0; r < chunks_per_batch; ++r) {
      const float *x_row = x_b + r * feature_size;
      const float *dy_row = dy_b + r * feature_size;
      float *dx_row = dx_b + r * feature_size;

      // Read back the inv_rms this chunk's forward pass computed, rather
      // than recomputing it from x_row here -- see the note in finalize()
      // for why re-deriving it from the (possibly since-reused) input
      // buffer at this point is unsafe.
      const float inv_rms = inv_rms_b[r];
      const float inv_rms3 = inv_rms * inv_rms * inv_rms;

      float sum_gdyx = 0.0f;
      for (unsigned int w = 0; w < feature_size; ++w) {
        float gdy = use_gamma ? g[w] * dy_row[w] : dy_row[w];
        sum_gdyx += gdy * x_row[w];
      }
      float mean_gdyx = sum_gdyx / feature_size;

      for (unsigned int w = 0; w < feature_size; ++w) {
        float gdy = use_gamma ? g[w] * dy_row[w] : dy_row[w];
        dx_row[w] = inv_rms * gdy - inv_rms3 * x_row[w] * mean_gdyx;
      }
    }
  }
}

/**
 * @brief calcGradient for ReshapedRMSNorm.
 * @details Same accumulation as RMSNormLayer::calcGradient, but gamma spans
 *          one `feature_size`-wide chunk and every chunk in the tensor
 *          contributes to it:
 *          dL/dgamma[w] = sum over chunks of dy[c][w]*x[c][w]*inv_rms[c].
 *          No-op when use_gamma is false, since then there is no weight.
 */
void ReshapedRMSNormLayer::calcGradient(nntrainer::RunLayerContext &context) {
  if (!use_gamma)
    return;

  nntrainer::Tensor &in = context.getInput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &dy =
    context.getIncomingDerivative(SINGLE_INOUT_IDX);
  nntrainer::Tensor &dgamma = context.getWeightGrad(wt_idx[RMSParams::gamma]);
  const nntrainer::Tensor &inv_rms_cache =
    context.getTensor(wt_idx[RMSParams::inv_rms]);

  NNTR_THROW_IF(in.getDataType() != ml::train::TensorDim::DataType::FP32 ||
                  dgamma.getDataType() !=
                    ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[reshaped_rms_norm] calcGradient only supports FP32 for now";

  const ml::train::TensorDim &in_dim = in.getDim();
  const unsigned int chunks_per_batch = in_dim.getFeatureLen() / feature_size;
  const unsigned int batch = in_dim.batch();
  const unsigned int inv_rms_stride = inv_rms_cache.getDim().getFeatureLen();

  const float *x = in.getData<float>();
  const float *dy_ = dy.getData<float>();
  const float *inv_rms_ = inv_rms_cache.getData<float>();

  std::vector<double> acc(feature_size, 0.0);

  for (unsigned int b = 0; b < batch; ++b) {
    const float *x_b = x + b * in_dim.getFeatureLen();
    const float *dy_b = dy_ + b * in_dim.getFeatureLen();
    const float *inv_rms_b = inv_rms_ + b * inv_rms_stride;

    for (unsigned int r = 0; r < chunks_per_batch; ++r) {
      const float *x_row = x_b + r * feature_size;
      const float *dy_row = dy_b + r * feature_size;
      const float inv_rms = inv_rms_b[r];

      for (unsigned int w = 0; w < feature_size; ++w)
        acc[w] += static_cast<double>(dy_row[w]) * x_row[w] * inv_rms;
    }
  }

  float *dg = dgamma.getData<float>();
  if (context.isGradientFirstAccess(wt_idx[RMSParams::gamma])) {
    for (unsigned int w = 0; w < feature_size; ++w)
      dg[w] = static_cast<float>(acc[w]);
  } else {
    for (unsigned int w = 0; w < feature_size; ++w)
      dg[w] += static_cast<float>(acc[w]);
  }
}

#ifdef PLUGGABLE

nntrainer::Layer *create_rms_norm_layer() {
  auto layer = new ReshapedRMSNormLayer();
  return layer;
}

void destroy_rms_norm_layer(nntrainer::Layer *layer) { delete layer; }

extern "C" {
nntrainer::LayerPluggable ml_train_layer_pluggable{create_rms_norm_layer,
                                                   destroy_rms_norm_layer};
}

#endif

} // namespace causallm
