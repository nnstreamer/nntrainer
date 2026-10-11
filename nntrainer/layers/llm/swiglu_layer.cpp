// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2024 Niket Agarwal <niket.a@samsung.com>
 *
 * @file   swiglu_layer.cpp
 * @date   6 June 2024
 * @see    https://github.com/nntrainer/nntrainer
 * @author Niket Agarwal <niket.a@samsung.com>
 * @author Eunju Yang <ej.yang@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  Backend-neutral SwiGLU activation: out = silu(gate) * up.
 */

#include "swiglu_layer.h"

#include <algorithm>
#include <cmath>

#include <compute_ops.h>
#include <nntrainer_error.h>
#include <node_exporter.h>
#include <tensor.h>
#include <thread_manager.h>

namespace nntrainer {

static constexpr size_t OUT_IDX = 0;
static constexpr size_t INPUT_IDX_1 = 0; /**< gate */
static constexpr size_t INPUT_IDX_2 = 1; /**< up */

void SwiGLULayer::finalize(InitLayerContext &context) {
  context.setOutputDimensions({context.getInputDimensions()[0]});

  if (!std::get<props::SkipPrefill>(swiglu_props).empty())
    skip_prefill = std::get<props::SkipPrefill>(swiglu_props).get();
}

void SwiGLULayer::setProperty(const std::vector<std::string> &values) {
  auto remain_props = loadProperties(values, swiglu_props);
  if (!remain_props.empty()) {
    std::string msg = "[SwiGLULayer] Unknown Layer Properties count " +
                      std::to_string(values.size());
    throw exception::not_supported(msg);
  }
}

namespace {
/**
 * @brief out = silu(in1) * in2 over the first @a rows rows. The host table's
 * swiglu is the scalar reference every backend is checked against; a
 * multi-row (prefill / training) step is split into row blocks across the
 * compute threads -- same per-element math, so bit-identical.
 */
void runSwiglu(Tensor &in1, Tensor &in2, Tensor &out, unsigned int rows) {
  ComputeOps *ops = in1.getOps();
  constexpr unsigned int ROWS_PER_TASK = 16;
  if (ops == getComputeOps() && rows >= 2 * ROWS_PER_TASK) {
    const unsigned int n_tasks = (rows + ROWS_PER_TASK - 1) / ROWS_PER_TASK;
    ThreadManager::Global().parallel_for(0, n_tasks, [&](size_t t) {
      const unsigned int r0 = static_cast<unsigned int>(t) * ROWS_PER_TASK;
      ops->swiglu(in1, in2, out, std::min(ROWS_PER_TASK, rows - r0), r0);
    });
    return;
  }
  ops->swiglu(in1, in2, out, rows, /*row_offset=*/0);
}
} // namespace

void SwiGLULayer::forwarding(RunLayerContext &context, bool training) {
  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // Rows are the (batch, channel, height) axes flattened; width() is the
  // per-row element count.
  runSwiglu(in1, in2, out, in1.batch() * in1.channel() * in1.height());
}

void SwiGLULayer::incremental_forwarding(RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) {
  // A multi-token step is a prefill step, so a KV-shared layer skips it here
  // exactly as it skips the from == 0 one: the branch produces nothing any
  // later layer reads.
  if (skip_prefill && (from == 0 || (to - from) > 1))
    return;

  Tensor &in1 = context.getInput(INPUT_IDX_1);
  Tensor &in2 = context.getInput(INPUT_IDX_2);
  Tensor &out = context.getOutput(OUT_IDX);

  // Every producer writes the live rows starting at the buffer base on every
  // backend, so the offset is always 0 and the count is the live-row window.
  // Rows are (batch, channel, height) flattened, so the window has to be
  // scaled by batch*channel or a batch>1 shape only computes its first
  // (to - from) rows.
  const unsigned int bc = in1.batch() * in1.channel();
  runSwiglu(in1, in2, out, (to - from) * bc);
}

void SwiGLULayer::updateTensorsByInputDimensions(
  RunLayerContext &context, std::vector<TensorDim> input_dimensions) {
  TensorDim input_dim1 = context.getInput(INPUT_IDX_1).getDim();
  TensorDim input_dim2 = context.getInput(INPUT_IDX_2).getDim();
  TensorDim output_dim = context.getOutput(OUT_IDX).getDim();

  input_dim1.height(input_dimensions[0].height());
  input_dim2.height(input_dimensions[0].height());
  output_dim.height(input_dimensions[0].height());

  context.updateInput(INPUT_IDX_1, input_dim1);
  context.updateInput(INPUT_IDX_2, input_dim2);
  context.updateOutput(OUT_IDX, output_dim);
}

/**
 * @brief calcDerivative for SwiGLU.
 * @details y = silu(g) * u, so
 *            dL/du_i = dy_i * silu(g_i)
 *            dL/dg_i = dy_i * u_i * silu'(g_i),
 *          with silu'(g) = sig(g) * (1 + g * (1 - sig(g))).
 *          Inputs are snapshotted per element before either derivative is
 *          written: d_gate aliases gate's buffer (nntrainer reuses the
 *          input's storage for the gradient), so reading g[i] after storing
 *          dg[i] would read the gradient, not the activation. FP32 only --
 *          the QAT training graph is host FP32.
 */
void SwiGLULayer::calcDerivative(RunLayerContext &context) {
  Tensor &gate = context.getInput(INPUT_IDX_1);
  Tensor &up = context.getInput(INPUT_IDX_2);
  const Tensor &dy = context.getIncomingDerivative(OUT_IDX);
  Tensor &d_gate = context.getOutgoingDerivative(INPUT_IDX_1);
  Tensor &d_up = context.getOutgoingDerivative(INPUT_IDX_2);

  NNTR_THROW_IF(gate.getDataType() != ml::train::TensorDim::DataType::FP32,
                std::invalid_argument)
    << "[swiglu] calcDerivative only supports FP32 for now";

  const size_t len = gate.size();
  const float *g = gate.getData<float>();
  const float *u = up.getData<float>();
  const float *dy_ = dy.getData<float>();
  float *dg = d_gate.getData<float>();
  float *du = d_up.getData<float>();

  // Element-independent: split into chunks across the compute threads.
  constexpr size_t CHUNK = 16384;
  const size_t n_chunks = (len + CHUNK - 1) / CHUNK;
  ThreadManager::Global().parallel_for(0, n_chunks, [=](size_t c) {
    const size_t i_end = std::min(len, (c + 1) * CHUNK);
    for (size_t i = c * CHUNK; i < i_end; ++i) {
      // Snapshot all inputs at this index first; see the aliasing note above.
      const float gi = g[i];
      const float ui = u[i];
      const float dyi = dy_[i];

      const float sig = 1.0f / (1.0f + std::exp(-gi));
      const float silu = gi * sig;
      const float dsilu = sig * (1.0f + gi * (1.0f - sig));

      du[i] = dyi * silu;
      dg[i] = dyi * ui * dsilu;
    }
  });
}

} // namespace nntrainer
