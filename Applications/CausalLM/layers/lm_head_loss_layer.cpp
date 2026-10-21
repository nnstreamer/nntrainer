// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Samsung Electronics Co., Ltd. All Rights Reserved.
 *
 * @file   lm_head_loss_layer.cpp
 * @date   12 October 2026
 * @brief  LM head fused with a masked next-token cross-entropy
 * @see    https://github.com/nntrainer/nntrainer
 * @bug    No known bugs except for NYI items
 */

#include <lm_head_loss_layer.h>

#include <compute_ops.h>
#include <cpu_backend.h>
#include <fp16.h>
#include <ggml_interface.h>
#include <nntrainer_error.h>
#include <thread_manager.h>
#include <tie_word_embedding.h>
#include <util_func.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <string>

namespace causallm {

namespace {
constexpr unsigned int SINGLE_INOUT_IDX = 0;
/** scored rows per pass through the head: bounds the logits scratch */
constexpr unsigned int ROWS_PER_CHUNK = 32;
/** vocabulary columns per logits GEMM task */
constexpr unsigned int VOCAB_PER_TASK = 4096;
/** hidden columns per dL/dx GEMM task */
constexpr unsigned int HIDDEN_PER_TASK = 64;
/** hidden rows of W^T gathered per task when building its Q8_0 copy */
constexpr unsigned int WT_ROWS_PER_TASK = 16;
/** bytes of one Q8_0 block (fp16 scale + 32 int8) */
constexpr size_t Q8_0_BLOCK_BYTES = 34;

/**
 * @brief Whether the head's two GEMMs run as int8 (Q8_0 x Q8_0) instead of
 * FP32. NNTR_LMHEAD_Q8=0/1 overrides; default on for aarch64 (i8mm) only,
 * like the frozen-FC dX (NNTR_TRAIN_DX_Q8).
 */
bool useQ8Head() {
  // read per forward (one getenv), not latched, so a test can compare both
  if (const char *e = std::getenv("NNTR_LMHEAD_Q8"))
    return e[0] != '0';
#if defined(__aarch64__)
  return true;
#else
  return false;
#endif
}

/** bytes of one Q4_0 block (fp16 scale + 16 nibble bytes) */
constexpr size_t Q4_0_BLOCK_BYTES = 18;

/**
 * @brief [EXP] NNTR_LMHEAD_Q4=1: the head's two copies are Q4_0 instead of
 * Q8_0, and its GEMMs run as int4 x int8 (Q4_0 x Q8_0). Latched per process.
 */
bool useQ4Head() {
  static const bool on = [] {
    const char *e = std::getenv("NNTR_LMHEAD_Q4");
    return e != nullptr && e[0] != '0';
  }();
  return on;
}

/** quantize @a nrow rows of @a n_per_row floats and repack them for the GEMM */
void quantizeRepackRows(bool q4, const float *src, uint8_t *dst,
                        unsigned int nrow, unsigned int n_per_row) {
  const size_t bytes = (size_t)nrow * (n_per_row / 32) *
                       (q4 ? Q4_0_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
  std::vector<uint8_t> plain(bytes);
  if (q4) {
    nntrainer::__ggml_quantize_q4_0(src, plain.data(), nrow, n_per_row,
                                    nullptr);
    nntrainer::__ggml_repack_q4_0_to_q4_0_4(dst, plain.data(), bytes, nrow,
                                            n_per_row);
  } else {
    nntrainer::__ggml_quantize_q8_0(src, plain.data(), nrow, n_per_row,
                                    nullptr);
    nntrainer::__ggml_repack_q8_0_to_q8_0_4x8(dst, plain.data(), bytes, nrow,
                                              n_per_row);
  }
}

/** C (M x N) = A (M x K) . B^T on the head's quantized copy B */
void headGemm(bool q4, unsigned int M, unsigned int N, unsigned int K,
              const float *A, const uint8_t *B, float *C) {
  if (q4)
    nntrainer::__ggml_q4_0_4x8_q8_0_GEMM<float>(M, N, K, A, K, B, N, C, N);
  else
    nntrainer::__ggml_q8_0_4x8_q8_0_GEMM(M, N, K, A, K, B, K, C, N);
}
} // namespace

LmHeadLossLayer::LmHeadLossLayer() :
  Layer(), props(nntrainer::props::Unit(), props::ReleaseHostWeight()) {}

void LmHeadLossLayer::setProperty(const std::vector<std::string> &values) {
  auto remain = nntrainer::loadProperties(values, props);
  NNTR_THROW_IF(!remain.empty(), std::invalid_argument)
    << "[lm_head_loss] unknown property: " << remain.front();
}

void LmHeadLossLayer::finalize(nntrainer::InitLayerContext &context) {
  NNTR_THROW_IF(context.getNumInputs() != 1, std::invalid_argument)
    << "[lm_head_loss] takes exactly one input";
  NNTR_THROW_IF(std::get<nntrainer::props::Unit>(props).empty(),
                std::invalid_argument)
    << "[lm_head_loss] unit (vocabulary size) is required";
  const unsigned int vocab = std::get<nntrainer::props::Unit>(props).get();

  const ml::train::TensorDim &in_dim = context.getInputDimensions()[0];
  NNTR_THROW_IF(in_dim.channel() != 1, std::invalid_argument)
    << "[lm_head_loss] input must be (B, 1, L, hidden)";

  // one loss / one label (the next token id) per position
  ml::train::TensorDim out_dim(
    in_dim.batch(), 1, 1, in_dim.height(),
    {context.getFormat(), context.getActivationDataType()});
  context.setOutputDimensions({out_dim});

  // Same name and shape as the tied lm_head's weight, so shared_from the
  // embedding (and the name-based checkpoint copy) resolve the same way.
  ml::train::TensorDim w_dim(
    1, 1, vocab, in_dim.width(),
    {context.getFormat(), context.getWeightDataType()});
  layer_name_ = context.getName();
  weight_idx = context.requestWeight(w_dim, nntrainer::Initializer::NONE,
                                     nntrainer::WeightRegularizer::NONE, 1.0f,
                                     0.0f, "Embedding", /*trainable=*/false);
}

namespace {
/**
 * @brief Build the int8 head's two copies of the frozen W (vocab x hidden):
 * W as q8_0x4 with blocks along hidden (logits GEMM) and W^T as q8_0x4 with
 * blocks along vocab (dL/dx GEMM).
 */
void buildQ8Copies(const float *w, unsigned int vocab, unsigned int hidden,
                   std::vector<uint8_t> &w_q8, std::vector<uint8_t> &wt_q8) {
  auto &tm = nntrainer::ThreadManager::Global();
  const bool q4 = useQ4Head();

  // The q8_0x4 repack interleaves groups of 4 rows, so each task quantizes
  // and repacks its own rows straight into place through a task-sized plain
  // staging buffer: no full-size plain copy (165 MB per matrix for a 151k
  // vocabulary) next to the FP32 W and the packed result.

  // W (vocab x hidden) for the logits GEMM: Q8_0 blocks along hidden.
  {
    const size_t blk = q4 ? Q4_0_BLOCK_BYTES : Q8_0_BLOCK_BYTES;
    const size_t row_bytes = (size_t)(hidden / 32) * blk;
    w_q8.resize((size_t)vocab * row_bytes);
    constexpr unsigned int rows_per_task = 256;
    tm.parallel_for(
      0, (vocab + rows_per_task - 1) / rows_per_task, [&](size_t t) {
        const size_t v0 = t * rows_per_task;
        const unsigned int nv =
          std::min<unsigned int>(rows_per_task, vocab - (unsigned int)v0);
        quantizeRepackRows(q4, w + v0 * hidden, w_q8.data() + v0 * row_bytes,
                           nv, hidden);
      });
  }

  // W^T (hidden x vocab) for the dL/dx GEMM: Q8_0 blocks along vocab. W^T
  // is never materialized in FP32 (another 600 MB); each task gathers a few
  // of its rows and quantizes them.
  {
    const size_t blk = q4 ? Q4_0_BLOCK_BYTES : Q8_0_BLOCK_BYTES;
    const size_t row_bytes = (size_t)(vocab / 32) * blk;
    wt_q8.resize((size_t)hidden * row_bytes);
    tm.parallel_for(0, hidden / WT_ROWS_PER_TASK, [&](size_t t) {
      const unsigned int h0 = static_cast<unsigned int>(t) * WT_ROWS_PER_TASK;
      std::vector<float> rows((size_t)WT_ROWS_PER_TASK * vocab);
      for (unsigned int v = 0; v < vocab; ++v) {
        const float *src = w + (size_t)v * hidden + h0;
        for (unsigned int j = 0; j < WT_ROWS_PER_TASK; ++j)
          rows[(size_t)j * vocab + v] = src[j];
      }
      quantizeRepackRows(q4, rows.data(), wt_q8.data() + (size_t)h0 * row_bytes,
                         WT_ROWS_PER_TASK, vocab);
    });
  }
}

/**
 * @brief Build the head's GEMM operands from a plain Q4_0 W (vocab x hidden,
 * the tied embedding's own table): W^T is gathered block by block from the
 * Q4_0 codes and quantized along vocab -- Q4_0 with NNTR_LMHEAD_Q4=1, Q8_0
 * otherwise -- and W itself is repacked to q4_0x4 in place, shared with the
 * embedding lookup (TieWordEmbedding::markQ4_0x4Table). No FP32 copy of W
 * and no second copy of its codes is ever built.
 */
void buildFromQ4_0(uint8_t *w, unsigned int vocab, unsigned int hidden,
                   std::vector<uint8_t> &wt_q, bool wt_q4) {
  auto &tm = nntrainer::ThreadManager::Global();
  const size_t w_row_bytes = (size_t)(hidden / 32) * Q4_0_BLOCK_BYTES;

  // W^T: rows h0 .. h0 + 15 sit in one Q4_0 block of every W row (16 | 32),
  // as its low (h0 % 32 == 0) or high nibbles. Read from the plain codes,
  // so this runs before the in-place repack below.
  static_assert(WT_ROWS_PER_TASK == 16, "one half Q4_0 block per W row");
  const size_t wt_row_bytes =
    (size_t)(vocab / 32) * (wt_q4 ? Q4_0_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
  wt_q.resize((size_t)hidden * wt_row_bytes);
  tm.parallel_for(0, hidden / WT_ROWS_PER_TASK, [&](size_t t) {
    const unsigned int h0 = static_cast<unsigned int>(t) * WT_ROWS_PER_TASK;
    const size_t blk_off = (size_t)(h0 / 32) * Q4_0_BLOCK_BYTES;
    const bool high = (h0 % 32) != 0;
    std::vector<float> rows((size_t)WT_ROWS_PER_TASK * vocab);
    for (unsigned int v = 0; v < vocab; ++v) {
      const uint8_t *blk = w + (size_t)v * w_row_bytes + blk_off;
      uint16_t d16;
      std::memcpy(&d16, blk, sizeof(d16));
      const float d = nntrainer::compute_fp16_to_fp32(d16);
      const uint8_t *qs = blk + sizeof(uint16_t);
      for (unsigned int j = 0; j < WT_ROWS_PER_TASK; ++j) {
        const int q = high ? (qs[j] >> 4) : (qs[j] & 0x0F);
        rows[(size_t)j * vocab + v] = (float)(q - 8) * d;
      }
    }
    quantizeRepackRows(wt_q4, rows.data(),
                       wt_q.data() + (size_t)h0 * wt_row_bytes,
                       WT_ROWS_PER_TASK, vocab);
  });

  // W: plain Q4_0 rows -> q4_0x4 in place (same size, lossless: the same
  // codes interleaved), so the logits GEMM reads the embedding's own table.
  // A repack only interleaves inside groups of 4 rows, so disjoint row
  // chunks repack independently through a chunk-sized staging copy.
  constexpr unsigned int rows_per_task = 256;
  tm.parallel_for(
    0, (vocab + rows_per_task - 1) / rows_per_task, [&](size_t t) {
      const size_t v0 = t * rows_per_task;
      const unsigned int nv =
        std::min<unsigned int>(rows_per_task, vocab - (unsigned int)v0);
      std::vector<uint8_t> plain(w + v0 * w_row_bytes,
                                 w + (v0 + nv) * w_row_bytes);
      nntrainer::__ggml_repack_q4_0_to_q4_0_4(
        w + v0 * w_row_bytes, plain.data(), plain.size(), nv, hidden);
    });
  nntrainer::TieWordEmbedding::markQ4_0x4Table(w);
}

/** int8 copies staged by LmHeadLossLayer::stageInt8Weight, by layer name */
struct StagedInt8 {
  std::vector<uint8_t> w, wt;
};
std::mutex &stagedMutex() {
  static std::mutex m;
  return m;
}
std::map<std::string, StagedInt8> &stagedInt8() {
  static std::map<std::string, StagedInt8> m;
  return m;
}

/** whether a head of this shape runs its GEMMs as int8 (see useQ8Head) */
bool int8Head(unsigned int vocab, unsigned int hidden) {
  return useQ8Head() && hidden % 32 == 0 && vocab % 32 == 0 &&
         hidden % WT_ROWS_PER_TASK == 0 && vocab % 4 == 0;
}
} // namespace

void LmHeadLossLayer::buildQ8(const float *w, unsigned int vocab,
                              unsigned int hidden) {
  buildQ8Copies(w, vocab, hidden, w_q8_, wt_q8_);
  w_is_q4_ = wt_is_q4_ = useQ4Head();
  w_q8_src_ = w;
}

bool LmHeadLossLayer::stageInt8Weight(const std::string &layer_name,
                                      const float *src, unsigned int vocab,
                                      unsigned int hidden) {
  if (!int8Head(vocab, hidden) ||
      std::getenv("NNTR_LMHEAD_KEEP_FP32") != nullptr)
    return false;
  StagedInt8 staged;
  buildQ8Copies(src, vocab, hidden, staged.w, staged.wt);
  std::lock_guard<std::mutex> lock(stagedMutex());
  stagedInt8()[layer_name] = std::move(staged);
  return true;
}

LmHeadLossLayer::~LmHeadLossLayer() {
  // a staged copy that no forward adopted (e.g. the run stopped first)
  std::lock_guard<std::mutex> lock(stagedMutex());
  stagedInt8().erase(layer_name_);
}

void LmHeadLossLayer::forwarding(nntrainer::RunLayerContext &context,
                                 bool training) {
  using DT = ml::train::TensorDim::DataType;
  const nntrainer::Tensor &x = context.getInput(SINGLE_INOUT_IDX);
  nntrainer::Tensor &out = context.getOutput(SINGLE_INOUT_IDX);
  const nntrainer::Tensor &w = context.getWeight(weight_idx);
  const bool w_q4_0 = w.getDataType() == DT::Q4_0;
  NNTR_THROW_IF(x.getDataType() != DT::FP32 ||
                  (w.getDataType() != DT::FP32 && !w_q4_0),
                std::invalid_argument)
    << "[lm_head_loss] FP32 input and an FP32 or Q4_0 weight only";

  const ml::train::TensorDim x_dim = x.getDim();
  const unsigned int batch = x_dim.batch();
  const unsigned int len = x_dim.height();
  const unsigned int hidden = x_dim.width();
  const unsigned int vocab = w.getDim().height();
  const float *x_data = x.getData<float>();
  // a Q4_0 W is read only through the copies built from its codes below
  const float *w_data = w_q4_0 ? nullptr : w.getData<float>();
  float *out_data = out.getData<float>();
  std::fill(out_data, out_data + out.size(), 0.0f);

  if (!context.isLabelAvailable(SINGLE_INOUT_IDX)) {
    context.setLoss(0.0f);
    return;
  }
  const float *label = context.getLabel(SINGLE_INOUT_IDX).getData<float>();

  // the scored positions of the whole batch, as flat (b * len + i) rows
  std::vector<unsigned int> rows;
  std::vector<unsigned int> targets;
  for (unsigned int r = 0; r < batch * len; ++r) {
    const float t = label[r];
    if (t >= 0.0f && t < static_cast<float>(vocab)) {
      rows.push_back(r);
      targets.push_back(static_cast<unsigned int>(t));
    }
  }

  if (training) {
    dx_.assign((size_t)batch * len * hidden, 0.0f);
  }
  if (rows.empty()) {
    context.setLoss(0.0f);
    return;
  }

  // dL/dlogits of the batch-mean loss over the scored positions
  const float inv_n = 1.0f / static_cast<float>(rows.size());
  const unsigned int chunk =
    std::min<unsigned int>(ROWS_PER_CHUNK, rows.size());
  logits_.resize((size_t)chunk * vocab);
  x_rows_.resize((size_t)chunk * hidden);
  dx_rows_.resize((size_t)chunk * hidden);

  auto *ops = nntrainer::getComputeOps();
  auto &tm = nntrainer::ThreadManager::Global();
  // The weight is frozen (tied embedding, never trained here), so its int8
  // copies are built once and reused; rebuilt only if the buffer moves.
  // Copies staged at load (stageInt8Weight) were built from the checkpoint
  // directly: W itself was never filled, so it counts as released.
  if (w_q4_0 && w_q4_table_ == nullptr) {
    NNTR_THROW_IF(hidden % 32 != 0 || vocab % 32 != 0 ||
                    hidden % WT_ROWS_PER_TASK != 0,
                  std::invalid_argument)
      << "[lm_head_loss] a Q4_0 head needs hidden and vocab divisible by 32";
    w_is_q4_ = true;
    wt_is_q4_ = useQ4Head();
    NNTR_THROW_IF(
      nntrainer::TieWordEmbedding::isQ4_0x4Table(w.getData<uint8_t>()),
      std::runtime_error)
      << "[lm_head_loss] the Q4_0 table is already repacked (a second head "
         "on the same weight)";
    buildFromQ4_0(const_cast<uint8_t *>(w.getData<uint8_t>()), vocab, hidden,
                  wt_q8_, wt_is_q4_);
    w_q4_table_ = w.getData<uint8_t>();
  }
  if (!w_q4_0 && w_q8_.empty() && !w_released_) {
    std::lock_guard<std::mutex> lock(stagedMutex());
    auto it = stagedInt8().find(layer_name_);
    if (it != stagedInt8().end()) {
      w_q8_ = std::move(it->second.w);
      wt_q8_ = std::move(it->second.wt);
      w_is_q4_ = wt_is_q4_ = useQ4Head();
      stagedInt8().erase(it);
      w_q8_src_ = w_data;
      w_released_ = true;
      nntrainer::releaseHostPages(w_data, w.getMemoryBytes());
    }
  }
  const bool q8 = w_q4_0 || int8Head(vocab, hidden);
  NNTR_THROW_IF(!q8 && w_released_, std::runtime_error)
    << "[lm_head_loss] the FP32 weight was released after the int8 head was "
       "built; the FP32 head cannot run (NNTR_LMHEAD_Q8=0 needs "
       "NNTR_LMHEAD_KEEP_FP32=1)";
  // Once W is released its address no longer names the weight's contents;
  // the copies built from it stay valid (W is frozen).
  if (!w_q4_0 && q8 && !w_released_ && w_q8_src_ != w_data)
    buildQ8(w_data, vocab, hidden);
  // The int8 copies are all the head reads from now on. When this layer owns
  // W (release_host_weight: nothing else, e.g. a tied embedding, reads it),
  // its FP32 pages are a dead third copy of the vocabulary-sized weight.
  if (!w_q4_0 && q8 && !w_released_ &&
      std::get<props::ReleaseHostWeight>(props).get() &&
      std::getenv("NNTR_LMHEAD_KEEP_FP32") == nullptr) {
    nntrainer::releaseHostPages(w_data, w.getMemoryBytes());
    w_released_ = true;
  }
  /// [PERF-LMHEAD] NNTR_TRAIN_PROF: where the head's time goes
  static const bool prof = std::getenv("NNTR_TRAIN_PROF") != nullptr;
  using clk = std::chrono::steady_clock;
  auto ms_since = [](clk::time_point t0) {
    return std::chrono::duration<double, std::milli>(clk::now() - t0).count();
  };
  clk::time_point tp0;
  const unsigned int row_major = 0;
  double loss_sum = 0.0;
  std::vector<float> row_loss(chunk);

  for (size_t c0 = 0; c0 < rows.size(); c0 += chunk) {
    const unsigned int m =
      static_cast<unsigned int>(std::min<size_t>(chunk, rows.size() - c0));
    for (unsigned int j = 0; j < m; ++j)
      std::memcpy(x_rows_.data() + (size_t)j * hidden,
                  x_data + (size_t)rows[c0 + j] * hidden,
                  sizeof(float) * hidden);
    // The int8 GEMM runs rows in groups of 4 and hands a remainder of 1-3
    // rows to a GEMV each, which streams the whole (vocabulary-sized) weight
    // once per row. Pad the chunk to a multiple of 4 with zero rows instead;
    // their results are never read.
    const unsigned int mg = q8 ? std::min(chunk, (m + 3) / 4 * 4) : m;
    if (mg > m)
      std::fill(x_rows_.data() + (size_t)m * hidden,
                x_rows_.data() + (size_t)mg * hidden, 0.0f);

    if (prof)
      tp0 = clk::now();
    // logits (m x vocab) = x_rows (m x hidden) . W^T, split over vocab
    float *lg = logits_.data();
    const float *xr = x_rows_.data();
    const unsigned int n_vt = (vocab + VOCAB_PER_TASK - 1) / VOCAB_PER_TASK;
    if (q8) {
      headGemm(w_is_q4_, mg, vocab, hidden, xr,
               w_q4_table_ != nullptr ? w_q4_table_ : w_q8_.data(), lg);
    } else {
      tm.parallel_for(0, n_vt, [=](size_t t) {
        const unsigned int v0 = static_cast<unsigned int>(t) * VOCAB_PER_TASK;
        const unsigned int nv = std::min(VOCAB_PER_TASK, vocab - v0);
        ops->sgemm_fp32(row_major, false, true, m, nv, hidden, 1.0f, xr, hidden,
                        w_data + (size_t)v0 * hidden, hidden, 0.0f, lg + v0,
                        vocab);
      });
    }

    if (w_q4_0 && c0 == 0 && std::getenv("NNTR_LMHEAD_CHECK") != nullptr) {
      static int lg_checked = 0;
      if (lg_checked++ < 3) {
        const uint8_t *wq = w.getData<uint8_t>();
        std::vector<float> row(hidden);
        double num = 0, den = 0;
        for (unsigned int v = 0; v < vocab; ++v) {
          nntrainer::TieWordEmbedding::dequantizeRowQ4_0x4(wq, v, hidden,
                                                           row.data());
          double r = 0;
          for (unsigned int h = 0; h < hidden; ++h)
            r += (double)xr[h] * row[h];
          num += (lg[v] - r) * (lg[v] - r);
          den += r * r;
        }
        std::cout << "[LMHEAD-CHECK] logits rel_l2="
                  << (den > 0 ? std::sqrt(num / den) : 0.0) << std::endl;
      }
    }
    if (prof) {
      prof_ms_[0] += ms_since(tp0);
      tp0 = clk::now();
    }
    // log-softmax per row; when training, overwrite with dL/dlogits
    const unsigned int *tg = targets.data() + c0;
    float *rl = row_loss.data();
    tm.parallel_for(0, m, [=](size_t j) {
      float *row = lg + j * vocab;
      // softmax in place on the backend's vectorized kernel (the scalar
      // std::exp over the vocabulary was the cost here); the logits are not
      // needed after this. loss = -log p(target), floored at the smallest
      // denormal so a target that underflowed reads as a large finite loss.
      ops->softmax_fp32(vocab, row, row);
      const float p_t = row[tg[j]];
      rl[j] =
        -std::log(std::max(p_t, std::numeric_limits<float>::denorm_min()));
      if (training) {
        // dL/dlogits = (softmax - onehot) / n
        const float k = inv_n;
        for (unsigned int v = 0; v < vocab; ++v)
          row[v] *= k;
        row[tg[j]] -= inv_n;
      }
    });
    for (unsigned int j = 0; j < m; ++j) {
      out_data[rows[c0 + j]] = row_loss[j];
      loss_sum += row_loss[j];
    }
    if (prof) {
      prof_ms_[1] += ms_since(tp0);
      tp0 = clk::now();
    }

    if (!training)
      continue;
    // dL/dx_rows (m x hidden) = dL/dlogits (m x vocab) . W, split over hidden
    float *dxr = dx_rows_.data();
    const unsigned int n_ht = (hidden + HIDDEN_PER_TASK - 1) / HIDDEN_PER_TASK;
    // The one-hot part of dL/dlogits = (softmax - onehot) / n is a single
    // W row per position: subtract it exactly (from the table itself) and
    // run the int GEMM on softmax / n only. Quantizing the one-hot along with
    // the softmax costs most of the int head's dL/dx error: dL/dx is a small
    // difference E_p[W] - W_target, so the GEMM's error on either term shows
    // up amplified.
    if (q8) {
      headGemm(wt_is_q4_, mg, hidden, vocab, lg, wt_q8_.data(), dxr);
    } else {
      tm.parallel_for(0, n_ht, [=](size_t t) {
        const unsigned int h0 = static_cast<unsigned int>(t) * HIDDEN_PER_TASK;
        const unsigned int nh = std::min(HIDDEN_PER_TASK, hidden - h0);
        ops->sgemm_fp32(row_major, false, false, m, nh, vocab, 1.0f, lg, vocab,
                        w_data + h0, hidden, 0.0f, dxr + h0, hidden);
      });
    }
    /// [DIAG] NNTR_LMHEAD_CHECK=1: check the first chunk's dL/dx (all m rows)
    /// against a direct double-precision sum over the W rows
    static int q4_checked = 0;
    if ((w_q4_0 || !w_released_) && c0 == 0 && q4_checked < 3 &&
        std::getenv("NNTR_LMHEAD_CHECK") != nullptr) {
      ++q4_checked;
      std::vector<double> ref((size_t)m * hidden, 0.0);
      std::vector<float> row(hidden);
      for (unsigned int v = 0; v < vocab; ++v) {
        if (w_q4_0)
          nntrainer::TieWordEmbedding::dequantizeRowQ4_0x4(
            w.getData<uint8_t>(), v, hidden, row.data());
        else
          std::memcpy(row.data(), w_data + (size_t)v * hidden,
                      sizeof(float) * hidden);
        for (unsigned int j = 0; j < m; ++j) {
          const double g = lg[(size_t)j * vocab + v];
          double *r = ref.data() + (size_t)j * hidden;
          for (unsigned int h = 0; h < hidden; ++h)
            r[h] += g * row[h];
        }
      }
      double num = 0, den = 0, worst = 0;
      for (unsigned int j = 0; j < m; ++j) {
        double rn = 0, rd = 0;
        for (unsigned int h = 0; h < hidden; ++h) {
          const double e =
            dxr[(size_t)j * hidden + h] - ref[(size_t)j * hidden + h];
          rn += e * e;
          rd += ref[(size_t)j * hidden + h] * ref[(size_t)j * hidden + h];
        }
        num += rn;
        den += rd;
        worst = std::max(worst, rd > 0 ? std::sqrt(rn / rd) : 0.0);
      }
      std::cout << "[LMHEAD-CHECK] dx rows=" << m
                << " rel_l2=" << (den > 0 ? std::sqrt(num / den) : 0.0)
                << " worst_row=" << worst << " (W^T "
                << (wt_is_q4_ ? "Q4_0" : "Q8_0") << ")" << std::endl;
    }
    for (unsigned int j = 0; j < m; ++j)
      std::memcpy(dx_.data() + (size_t)rows[c0 + j] * hidden,
                  dx_rows_.data() + (size_t)j * hidden, sizeof(float) * hidden);
    if (prof)
      prof_ms_[2] += ms_since(tp0);
  }

  if (prof && training) {
    prof_rows_ += rows.size();
    if (++prof_calls_ % 10 == 0) {
      std::cout << "[PERF-LMHEAD] avg over 10: rows " << prof_rows_ / 10.0
                << " logits_gemm " << prof_ms_[0] / 10 << " ms, softmax "
                << prof_ms_[1] / 10 << " ms, dx_gemm " << prof_ms_[2] / 10
                << " ms" << std::endl;
      prof_ms_[0] = prof_ms_[1] = prof_ms_[2] = 0.0;
      prof_rows_ = 0;
    }
  }

  context.setLoss(static_cast<float>(loss_sum * inv_n));
}

void LmHeadLossLayer::calcDerivative(nntrainer::RunLayerContext &context) {
  nntrainer::Tensor &dx = context.getOutgoingDerivative(SINGLE_INOUT_IDX);
  NNTR_THROW_IF(dx_.size() != dx.size(), std::runtime_error)
    << "[lm_head_loss] calcDerivative without a training forward";
  std::memcpy(dx.getData<float>(), dx_.data(), sizeof(float) * dx_.size());
  const float loss_scale = context.getLossScale();
  if (loss_scale != 1.0f)
    dx.multiply_i(loss_scale);
}

} // namespace causallm
