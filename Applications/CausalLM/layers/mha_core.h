// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2025 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   mha_core.h
 * @date   02 September 2024
 * @see    https://github.com/nntrainer/nntrainer
 *         https://arxiv.org/abs/1706.03762
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is custom_mha_core layer supports
 *         the work of multi_head_attention.
 * @note   Unlike custom_multi_head_attention_layer,
 *         which works all of the attention operations
 *         in a layer, this layer is attached after Q / K / V
 *         fully connected layer to post-process them
 *         including KV-Cache.
 *         For inference, forwarding processes the current input rows
 *         at the KV-cache write position, and incremental_forwarding
 *         takes the step rows [`from`, `to`) (one or more rows).
 *         Training is not supported (forwarding is a no-op when
 *         training is set in internal-cache mode).
 */

#ifndef __MHA_CORE_H__
#define __MHA_CORE_H__

#pragma once
#ifdef _WIN32
#define WIN_EXPORT __declspec(dllexport)
#else
#define WIN_EXPORT
#endif

#include <complex>

#include <acti_func.h>
#include <common_properties.h>
#include <cpu_backend.h>
#include <layer_impl.h>
#include <limits.h>
#include <util_simd.h>

#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

namespace causallm {

namespace props {

/**
 * @brief NumHeads property, NumHeads is number of head in multi head attention
 * of Q
 */
class NumHeads_KV : public nntrainer::PositiveIntegerProperty {
public:
  /**
   * @brief Construct a new NumHeads object with default value 1
   */
  NumHeads_KV(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "num_heads_KV";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief SlidingWindow
 */
class SlidingWindow : public nntrainer::Property<unsigned int> {
public:
  SlidingWindow(unsigned int value = UINT_MAX) { set(value); };
  static constexpr const char *key =
    "sliding_window";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief InitSeqLen -- the model's activation-plane height in query rows.
 * @details The prefill feeds at most this many query rows per forward pass,
 * so it is the ceiling the prefill chunk is clamped to. The layer needs the
 * clamped chunk (not the raw NNTR_PREFILL_CHUNK request) to size its KV window
 * ring exactly as the model side does -- the two must agree to the row or the
 * modulo indexing writes out of bounds. 0 means "not told", and the layer then
 * falls back to its query input height, which is the same plane.
 */
class InitSeqLen : public nntrainer::Property<unsigned int> {
public:
  InitSeqLen(unsigned int value = 0) { set(value); };
  static constexpr const char *key =
    "init_seq_len";                          /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief MaxNewTokens
 */
class MaxNewTokens : public nntrainer::Property<unsigned int> {
public:
  MaxNewTokens(unsigned int value = 1) { set(value); };
  static constexpr const char *key =
    "max_new_tokens";                        /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief MaxNewTokens
 */
class MaxPositionEmbeddings : public nntrainer::Property<unsigned int> {
public:
  MaxPositionEmbeddings(unsigned int value = 40960) { set(value); };
  static constexpr const char *key =
    "max_position_embeddings";               /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

/**
 * @brief RopeTheta
 */
class RopeTheta : public nntrainer::Property<unsigned int> {
public:
  RopeTheta(unsigned int value = 500000) { set(value); };
  static constexpr const char *key = "rope_theta"; /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag;       /**< property type */
};

/**
 * @brief UseRope property (Gemma4/Gemma3n). Accepted but our mha_core derives
 *        rope-enable from theta>0, so it is informational here.
 */
class UseRope : public nntrainer::Property<bool> {
public:
  UseRope(bool value = true) { set(value); };
  static constexpr const char *key = "use_rope"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;     /**< property type */
};

/**
 * @brief UseSink property
 */
class UseSink : public nntrainer::Property<bool> {
public:
  UseSink(bool value = false) { set(value); };
  static constexpr const char *key = "use_sink"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;     /**< property type */
};

/**
 * @brief UseGemmAttention property
 *
 * Opt-in toggle for the GEMM / flash-attention path (built and verified for the
 * ARM FP16 device build; forced off on x86 / non-FP16 builds). Defaults to the
 * reference-attention path so existing behavior is preserved.
 */
class UseGemmAttention : public nntrainer::Property<bool> {
public:
  UseGemmAttention(bool value = false) { set(value); };
  static constexpr const char *key =
    "use_gemm_attention";                    /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag; /**< property type */
};

/**
 * @brief AttnLogitSoftcapping
 */
class AttnLogitSoftcapping : public nntrainer::Property<float> {
public:
  AttnLogitSoftcapping(float value = 0.0f) { set(value); };
  static constexpr const char *key =
    "attn_logit_softcapping";                 /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief IsCausal property
 */
class IsCausal : public nntrainer::Property<bool> {
public:
  IsCausal(bool value = true) { set(value); };
  static constexpr const char *key = "is_causal"; /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag;      /**< property type */
};

/**
 * @brief CrossAttention property
 * @note When true, the layer runs in static read-only cross-attention mode
 *       (Option B): inputs 3/4 (cache_key/cache_value) are treated as a
 *       fully-populated encoder K/V cache of @c enc_len rows; the query
 *       (input 0) attends over ALL @c enc_len cached rows, inputs 1/2 (K/V)
 *       are NOT written into the cache, and cache_index is NOT advanced.
 *       Only meaningful in external-cache (5-input) mode. Default false so
 *       existing self-attention callers see byte-identical behaviour.
 */
class CrossAttention : public nntrainer::Property<bool> {
public:
  CrossAttention(bool value = false) { set(value); };
  static constexpr const char *key =
    "cross_attention";                       /**< unique key to access */
  using prop_tag = nntrainer::bool_prop_tag; /**< property type */
};

/**
 * @brief RopeScalingType
 * - default
 * - yarn
 */
class RopeScalingType : public nntrainer::Property<std::string> {
public:
  RopeScalingType(std::string value = "default") { set(value); };
  static constexpr const char *key =
    "rope_scaling_type";                    /**< unique key to access */
  using prop_tag = nntrainer::str_prop_tag; /**< property type */
};
/**
 * @brief RopeScalingFactor
 */
class RopeScalingFactor : public nntrainer::Property<float> {
public:
  RopeScalingFactor(float value = 1.0) { set(value); };
  static constexpr const char *key =
    "rope_scaling_factor";                    /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief RopePartialRotaryFactor (Gemma4/Gemma3n). Outside yarn, a factor
 *        != 1 routes the RoPE table through _compute_proportional_parameters,
 *        which gives rotary frequencies to only factor * head_dim / 2 of the
 *        head_dim / 2 pairs; the rest pass through unrotated.
 */
class RopePartialRotaryFactor : public nntrainer::Property<float> {
public:
  RopePartialRotaryFactor(float value = 1.0f) { set(value); };
  static constexpr const char *key =
    "rope_partial_rotary_factor";             /**< unique key to access */
  using prop_tag = nntrainer::float_prop_tag; /**< property type */
};

/**
 * @brief RopeScalingMaxPositionEmbeddings
 */
class RopeScalingMaxPositionEmbeddings
  : public nntrainer::Property<unsigned int> {
public:
  RopeScalingMaxPositionEmbeddings(unsigned int value = 4096) { set(value); };
  static constexpr const char *key =
    "rope_scaling_max_position_embeddings";  /**< unique key to access */
  using prop_tag = nntrainer::uint_prop_tag; /**< property type */
};

}; // namespace props

/**
 * @brief Declare that this pack wants the prefill K rotation drained, i.e.
 *        whether nntr_config.json sets `"prefill_kv_drain": true`.
 *
 * The attention core uses it for exactly one decision: whether the prefill K
 * rotation drains before `k_scatter_ohwi_cl` reads the SVM cache slice it
 * wrote (see mha_core.cpp, `_kv_nodrain_env`). Default false, i.e. the
 * per-write no-drain, which is what every pack that has not been observed to
 * need the drain keeps.
 *
 * Read the gate honestly. This is a DECLARATION, recording where the defect
 * has been OBSERVED to decide a token -- it is not a decision the tree makes
 * about a pack, and it is emphatically not a model name. The RISK condition is
 * a consumer reading an SVM write issued in an earlier submission, which every
 * model on this chain does; deciding THAT in code, where the chain is built
 * rather than from a config flag, is the per-dispatch hazard check
 * (Kernel::noteUndrainedSvmPlane, opt-in via NNTR_SVM_HAZARD_DRAIN=1), which
 * is meant to replace this declaration with the thing it stands for. Until it
 * does, a pack that does not set it is unobserved, not proven safe.
 */
void setPrefillKvDrain(bool on);

/** @copydoc setPrefillKvDrain(bool) */
bool prefillKvDrain();

/**
 * @brief [prefill-long] Announce the total key span the next prefill reaches.
 * @details Set by the prefill driver before the chunk loop and cleared after,
 * so the tight-stride V image is laid out once for the whole prompt instead of
 * being re-laid (and every scattered row re-written) at each chunk boundary.
 * 0 means "unknown", which restores the per-chunk growth exactly.
 */
void setPrefillSpanHint(unsigned int span);

/** @copydoc setPrefillSpanHint(unsigned int) */
unsigned int prefillSpanHint();

/**
 * @class MHA Core Layer
 * @brief Part of Multi-Head-Attention Layer.
 *        It should be attached after Q / K / V fc layers and before O fc layer.
 *        custom_mha_core_layer computes attention, while updating KV-cache for
 *        inference mode.
 *
 *    [ Q ]    [ K ]    [ V ]
 *      |        |        |
 *     [      mha_core      ]
 *               |
 *             [ O ]
 *
 */
WIN_EXPORT class MHACoreLayer : public nntrainer::LayerImpl {
public:
  /**
   * @brief Constructor of MhaCore Layer
   */
  WIN_EXPORT MHACoreLayer();

  /**
   * @brief Destructor of MhaCore Layer
   */
  WIN_EXPORT ~MHACoreLayer();

  /**
   *  @brief  Move constructor of MHACoreLayer.
   *  @param[in] MHACoreLayer &&
   */
  WIN_EXPORT
  MHACoreLayer(MHACoreLayer &&rhs) noexcept = default;

  /**
   * @brief  Move assignment operator.
   * @param[in] rhs MHACoreLayer to be moved.
   */
  WIN_EXPORT MHACoreLayer &operator=(MHACoreLayer &&rhs) = default;

  /**
   * @brief Finalize funciton of MhaCore Layer
   */
  WIN_EXPORT void finalize(nntrainer::InitLayerContext &context) override;

  /**
   * @brief forwarding function of MhaCore Layer
   *        This is the inference entry point: it runs attention over the
   *        current input rows and updates the KV cache.
   */
  WIN_EXPORT void forwarding(nntrainer::RunLayerContext &context,
                             bool training) override;

  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim);

  void one_batch_incremental_forwarding(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim, nntrainer::Tensor &sink_step);

  /**
   * @brief GEMM / flash-attention path for one attention step (ARM FP16 device
   *        build; gated by use_gemm_attention). Falls back to the reference
   *        per-row attention kernels elsewhere.
   */
  WIN_EXPORT void gemm_attention(nntrainer::Tensor &query_step,
                                 nntrainer::Tensor &b_cached_key,
                                 nntrainer::Tensor &b_cached_value,
                                 nntrainer::Tensor &attention_output_step,
                                 unsigned int N_kv, unsigned int N_q,
                                 unsigned int cache_from);
  /**
   * @brief Run one batch of an FP32-activation layer on the device attention.
   * @details The device arms (RoPE, image / flash attention) take FP16
   * operands in shared virtual memory. This narrows Q/K/V into FP16 planes on
   * the device, runs one_batch_incremental_forwarding on them as for an
   * FP16-activation model, and widens O back into @a attention_output_step on
   * the device, so no operand is read on the host. The arguments are those of
   * one_batch_incremental_forwarding, with FP32 step operands.
   * @return false, with nothing enqueued that a host arm would read, when this
   *         call cannot take it: no OpenCL device attention in this run, an
   *         int8 KV cache or an attention sink, or operands that are not
   *         plain shared virtual memory. The caller then runs its host path.
   */
  bool attention_fp32_on_device(
    const unsigned int batch, const unsigned int _from, const unsigned int from,
    const unsigned int to, nntrainer::Tensor &query_step,
    nntrainer::Tensor &key_step, nntrainer::Tensor &value_step,
    nntrainer::Tensor &attention_output_step, nntrainer::Tensor &cache_key,
    nntrainer::Tensor &cache_value, ml::train::TensorDim &cache_key_dim,
    ml::train::TensorDim &cache_key_step_dim,
    ml::train::TensorDim &cache_value_dim,
    ml::train::TensorDim &cache_value_step_dim);

  /**
   * @copydoc Layer::incremental_forwarding(RunLayerContext &context, unsigned
   * int from, unsigned int to, bool training)
   */
  WIN_EXPORT void incremental_forwarding(nntrainer::RunLayerContext &context,
                                         unsigned int from, unsigned int to,
                                         bool training) override;

  /**
   * @copydoc Layer::calcDerivative(RunLayerContext &context)
   */
  WIN_EXPORT void calcDerivative(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc Layer::calcGradient(RunLayerContext &context)
   */
  WIN_EXPORT void calcGradient(nntrainer::RunLayerContext &context) override;

  /**
   * @copydoc bool supportBackwarding() const
   * @note Returns true, but backwarding is not implemented yet:
   * calcDerivative() and calcGradient() are no-ops.
   */
  WIN_EXPORT bool supportBackwarding() const override { return true; };

  /**
   * @copydoc Layer::exportTo(Exporter &exporter, ml::train::ExportMethods
   * method)
   */
  WIN_EXPORT void
  exportTo(nntrainer::Exporter &exporter,
           const ml::train::ExportMethods &method) const override;

  /**
   * @copydoc Layer::setProperty(const std::vector<std::string> &values)
   */
  WIN_EXPORT void setProperty(const std::vector<std::string> &values) override;

  /**
   * @copydoc Layer::getType()
   */
  WIN_EXPORT const std::string getType() const override {
    return MHACoreLayer::type;
  };

  /**
   * @copydoc Layer::setBatch(RunLayerContext &context, unsigned int batch)
   */
  WIN_EXPORT void setBatch(nntrainer::RunLayerContext &context,
                           unsigned int batch) override;

  WIN_EXPORT void updateTensorsByInputDimensions(
    nntrainer::RunLayerContext &context,
    std::vector<nntrainer::TensorDim> input_dimensions) override;

  /**
   * @brief Set the cache index for external cache mode.
   *        Must be called before forwarding() when use_external_cache is true.
   * @param[in] idx current write position in the KV cache
   */
  WIN_EXPORT void setCacheIndex(unsigned int idx) { cache_index = idx; }

  /**
   * @brief Get the current cache index
   */
  WIN_EXPORT unsigned int getCacheIndex() const { return cache_index; }

  inline static const std::string type = "mha_core";

private:
  std::tuple<
    nntrainer::props::NumHeads, props::NumHeads_KV,
    nntrainer::props::ProjectedKeyDim, nntrainer::props::ProjectedValueDim,
    nntrainer::props::OutputShape, nntrainer::props::DropOutRate,
    nntrainer::props::ReturnAttentionWeight,
    nntrainer::props::AverageAttentionWeight, nntrainer::props::MaxTimestep,
    props::SlidingWindow, props::MaxNewTokens, props::RopeTheta, props::UseRope,
    props::MaxPositionEmbeddings, props::UseSink, props::RopeScalingType,
    props::RopeScalingFactor, props::RopePartialRotaryFactor,
    props::RopeScalingMaxPositionEmbeddings, props::AttnLogitSoftcapping,
    props::IsCausal, props::UseGemmAttention, props::CrossAttention>
    mha_core_props; /**< mha_core layer properties */

  /** softmax activation operation */
  nntrainer::ActiFunc sm;

  float epsilon;            /** to avoid overflow */
  unsigned int cache_index; /** idx of kv cache */

  /**
   * @brief Honor the LayerImpl `skip_prefill` property. Gemma4 KV-shared
   *        layers (the last num_kv_shared_layers) reuse an earlier layer's
   *        K/V projection as their attention K/V *inputs*, and since
   *        [kv-share] they no longer own a dedicated KV-cache slab either:
   *        their cache_k_l{id}/cache_v_l{id} placeholders are bound by
   *        CausalLM::allocateAndBindKVCache to the SOURCE layer's plane
   *        (same MemoryData, same offset), because the values they used to
   *        write into a private slab were byte-identical to the source's.
   *        Nothing in this layer changes: it still sees one cache tensor per
   *        input and still writes through it. The per-layer ops around mha
   *        (Q/O FC, q_norm, scalar, the post-attention add) all honor
   *        skip_prefill and skip the prefill big-step; mha_core previously
   *        ignored it and STILL ran the (heavy) prefill attention for these
   *        shared layers. When set AND this is the prefill call
   *        (to - from > 1), mha_core still WRITES + scatters K/V into its
   *        cache slab (decode attends to those positions, so they MUST be
   *        populated; under aliasing that write lands on the source's plane
   *        and stores the value already there), then early-returns BEFORE the
   *        attention compute and the (unused, downstream-also-skipped)
   *        attention output write.
   *        cache_index is re-set from the absolute `from` argument on every
   *        forward (CausalLM::setKVCachePosition / incremental_forwarding),
   *        so the skipped tail's internal advance is irrelevant to decode.
   *        Parsed from layer_impl_props::SkipPrefill in finalize().
   */
  bool skip_prefill = false;

  /**
   * @brief Whether to use externally provided cache tensors
   *        (true when num_inputs >= 5, i.e., Q, K, V + cache_key + cache_value)
   *        In external mode mha_core does not allocate its own cache tensors,
   *        and reads cache slots from input[3] (cache_key) and input[4]
   *        (cache_value) which are bound by the host (e.g.
   *        CausalLM::allocateAndBindKVCache).
   */
  bool use_external_cache = false;

  /** intermal info */
  size_t num_heads_Q;
  size_t num_heads_KV;
  size_t head_dim;
  bool cache_shift;
  float theta;
  size_t local_window_size;
  /** [kv-window-ring] Physical ring rows for this layer (0 = full max_seq, no
   * ring). Set in finalize() from causallm::kvRingCap(local_window_size,
   * max_timestep, chunk) -- the same function models/transformer.h sizes the
   * allocation with. When non-zero every cache-row index -- write offset and
   * kernel read alike -- is taken modulo this. */
  unsigned int kv_ring_cap = 0;
  /** Map an absolute cache position to its physical ring row (identity when the
   * ring is off). Used ONLY for cache-storage row offsets: RoPE and the
   * causal/window masks keep the absolute position. */
  inline size_t cacheRow(size_t abs_pos) const {
    return kv_ring_cap ? (abs_pos % (size_t)kv_ring_cap) : abs_pos;
  }
  bool use_sink = false;
  float attn_logit_softcapping = 0.0f;
  bool is_causal;
  bool use_gemm_attention = false;
  bool skip_prefill = false;
  /**
   * @brief static read-only cross-attention mode (Option B). See
   *        props::CrossAttention. Default false; guards every cross-attn change
   *        so self-attention callers are unaffected.
   */
  bool cross_attention = false;

  enum INOUT_INDEX {
    /** input index */
    QUERY = 0,
    KEY = 1,
    VALUE = 2,
    MASK = 3,

    /** output index */
    OUTPUT = 0,
    RETURN_ATTENTION_WEIGHT = 1,
  };

  /**< indices of the weights and tensors */
  enum AttentionParams {
    cache_key,
    cache_value,
    projected_key,
    projected_value,
    /** intended comment for later use of attention_mask */
    // attention_mask,
    attention_weight,
    dropout_mask,
    attention_output,
    /** Per-(token, head) FP16 scales for int8 KV cache (paper section
     * 3.7 int8 KV path; only requested when kv_int8 mode is active). */
    cache_key_scale,
    cache_value_scale,
  };
  std::array<unsigned int, 9> tensor_idx;

  /** True when KV cache is stored as int8 (raw bytes treated as
   * int8) + per-(token, head) FP16 scales. Enabled by setting the
   * NNTR_KV_INT8 env var; default false keeps the FP16 cache layout. */
  bool kv_int8 = false;

  /** Scale tensors for the int8 KV cache; populated by the forwarding
   * entry points when kv_int8 is active. Lifespan is the layer's, so
   * the pointers are valid for as long as the layer node exists. */
  nntrainer::Tensor *kv_int8_key_scale = nullptr;
  nntrainer::Tensor *kv_int8_value_scale = nullptr;
  /** Per-batch raw pointers into the scale tensors above, set by
   * one_batch_incremental_forwarding before the read-path call to
   * compute_kcaches / compute_fp16vcache_transposed / gemm_attention.
   * Single-batch dispatch is sequential, so racing the helpers (which
   * parallelize over heads) against these is safe. */
  const uint16_t *cur_kv_int8_key_scale_batch = nullptr;
  const uint16_t *cur_kv_int8_value_scale_batch = nullptr;
  unsigned int sink_idx;

  /** attention parameters */
  unsigned int max_position_embeddings;

  /** rope_scaling parameters */
  std::string rope_scaling_type;
  float attention_scaling = 1.0f;
  float mscale = 1.0f;
  float scale = 1.0f;
  /** fraction of head_dim that receives rotary embedding (1.0 = full rope).
   * Frequencies past partial_rotary_factor*head_dim/2 are set to 0 (identity
   * passthrough) by _compute_proportional_parameters. */
  float rope_partial_rotary_factor = 1.0f;
  unsigned int original_max_position_embeddings = 4096;

  /** set by incremental_forwarding, used by forwarding */
  unsigned int incremental_step_size = 0;

  /****************** ROTARY EMBEDDING *****************/
  /** static variable - they are all expected to be initialized once */
  /**
   * @brief Rotary frequency cache for FP32 and optional FP16 lookup tables
   */
  struct RopeFreqCache {
    std::vector<std::vector<float>> cos;
    std::vector<std::vector<float>> sin;
#ifdef ENABLE_FP16
    std::vector<std::vector<_FP16>> cos_fp16;
    std::vector<std::vector<_FP16>> sin_fp16;
#endif
  };
  inline static std::unordered_map<std::string, RopeFreqCache> rope_freq_cache;
  inline static std::vector<std::vector<float>> *freqs_cos = {};
  inline static std::vector<std::vector<float>> *freqs_sin = {};
  inline static std::vector<float> thetas;
  std::vector<std::vector<float>> *cached_freqs_cos = {};
  std::vector<std::vector<float>> *cached_freqs_sin = {};
#ifdef ENABLE_FP16
  inline static std::vector<std::vector<_FP16>> *freqs_cos_fp16 = {};
  inline static std::vector<std::vector<_FP16>> *freqs_sin_fp16 = {};
  std::vector<std::vector<_FP16>> *cached_freqs_cos_fp16 = {};
  std::vector<std::vector<_FP16>> *cached_freqs_sin_fp16 = {};
  // Flattened [max_pos * head_dim/2] cos/sin LUT (fp16 bits) for the GPU RoPE
  // path (rope_inplace_f16_cl). PROCESS-WIDE static (inline static) CACHE keyed
  // by (head_dim, theta, max_pos): models that alternate RoPE slots per layer
  // (Gemma4: sliding head_dim=256/theta=1e4, full head_dim=512/theta=1e6) keep
  // a SEPARATE, STABLE table per slot instead of rebuilding + re-uploading the
  // single shared table on every sliding<->full transition. std::map is chosen
  // for NODE STABILITY: the cos/sin vector storage never moves on insert, so
  // the .data() pointer handed to the GPU wrapper stays valid for the whole run
  // -- the rope_inplace_f16_cl device-LUT cache is keyed by that host pointer,
  // so stable pointers keep the upload cached per slot (no per-layer thrash).
  // Single-head_dim models (Qwen3/Gemma2) populate exactly one entry, matching
  // the previous "built once" behaviour.
  using RopeFlatKey = std::tuple<int, float, unsigned int>; // head_dim,theta,mp
  using RopeFlatVal =
    std::pair<std::vector<uint16_t>, std::vector<uint16_t>>; // (cos, sin)
  inline static std::map<RopeFlatKey, RopeFlatVal> rope_flat_cache;
  // Stable pointers into the cache entry for the CURRENT slot (set by
  // ensure_rope_flat_lut). Per-instance: each layer instance points at its own
  // slot's cached table; the underlying storage lives in the shared map so the
  // pointers stay valid (and identical) across calls for a given slot.
  const uint16_t *rope_cos_flat_cur = nullptr;
  const uint16_t *rope_sin_flat_cur = nullptr;

  /**
   * @brief Build (or reuse) the process-wide flat RoPE LUT for THIS instance's
   *        current (head_dim, theta, max_pos) slot. Idempotent: an existing
   *        cache entry is reused (just repoints rope_cos/sin_flat_cur); a new
   *        slot is built (precompute_freqs + the flatten fill) and inserted.
   *        Called from finalize() so the build lands at model load instead of
   *        inside the first timed prefill (KVST "lutcheck" segment), and from
   *        the GPU-RoPE hot path as the fallback.
   */
  void ensure_rope_flat_lut();
  /** @brief positions the RoPE LUT must cover = min(max_position_embeddings,
   *  MaxTimestep). Avoids sizing/uploading the LUT at the theoretical RoPE max
   *  (e.g. 131072) when the live cache is only max_seq_len. */
  unsigned int rope_lut_positions() const;
#endif

  /**
   * @brief pre_compute frequencies for Rotary Embedding.
   * @note it is expected to be called only once at the finalize.
   * @param[in] head_dim dimension of head
   * @param[in] seq_len sequence length
   * @param[in] theta base of theta (default = 10000)
   */
  void precompute_freqs(int head_dim, unsigned int seq_len,
                        float theta = 10000.0, bool is_fp16 = false);

  /**
   * @brief _compute frequency parameters for default ROPE
   */
  void _compute_default_parameters(int head_dim, float theta);

  /**
   * @brief _compute frequency parameters for YaRN ROPE
   */
  void _compute_yarn_parameters(int head_dim, float theta);

  /**
   * @brief _compute frequency parameters for proportional ROPE (also handles
   *        partial rotary via rope_partial_rotary_factor: frequencies beyond
   *        partial_rotary_factor*head_dim/2 are zeroed = identity passthrough)
   */
  void _compute_proportional_parameters(int head_dim, float theta);

  /**
   * @brief     apply rotary embedding
   * @param[in] in input tensor
   * @param[out] out output tensor
   * @param[in] dim hidden dim size
   * @param[in] from sequence order
   * @param[in] apply_rope true to apply rotary embedding, false to only store
   *                       the tensor into the cache dtype
   */
  void apply_rotary_emb_tensor_v2(nntrainer::Tensor &in, nntrainer::Tensor &out,
                                  unsigned int dim, unsigned int from,
                                  bool apply_rope = true);

  template <typename BType>
  void compute(const float *A, const BType *B, float *output, int num_rows,
               int N, int chunk_size, int group_size, int tile_size,
               bool process_all);

  void compute_kcaches(nntrainer::Tensor &in, nntrainer::Tensor &cache,
                       nntrainer::Tensor &out, unsigned int from,
                       size_t sequence_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim);

  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from);

  void softmax_triangle(nntrainer::Tensor &qk_out, size_t row, size_t num_heads,
                        unsigned int from, nntrainer::Tensor &sink_step);

  void compute_vcaches(nntrainer::Tensor &in, nntrainer::Tensor &vcache,
                       nntrainer::Tensor &out, unsigned int from,
                       size_t sequence_len, unsigned int num_heads,
                       unsigned int group_size, unsigned int head_dim);

  void compute_fp16vcache_transposed(nntrainer::Tensor &in,
                                     nntrainer::Tensor &vcache,
                                     nntrainer::Tensor &output, int from,
                                     int num_cache_head, int gqa_size,
                                     int head_dim, int to);

  /************** END OF  ROTARY EMBEDDING *************/

  /**
   * @brief calculate common derivative
   * @param context Context of the layer
   */
  void calcCommonDerivative(nntrainer::RunLayerContext &context);

  size_t calc_attn_index(size_t i);

  /**
   * @brief Windowed cumulative attention score index.
   *
   * Returns the cumulative number of attention scores before absolute query
   * row i, respecting a sliding window of size local_window_size (W):
   *
   *   S(i) = sum_{k=0}^{i-1} min(k+1, W)
   *        = (i <= W) ? i*(i+1)/2 : W*(W+1)/2 + (i - W)*W
   *
   * When W == UINT_MAX (full attention), this reduces exactly to
   * i*(i+1)/2 == calc_attn_index(i), preserving byte-identical behaviour.
   */
  size_t calc_windowed_attn_index(size_t i);

}; // end of class MHACoreLayer
} // namespace causallm

#endif
