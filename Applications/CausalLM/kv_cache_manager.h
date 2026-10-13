// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_cache_manager.h
 * @date   25 April 2026
 * @brief  KV Cache Manager for externalized KV cache management
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#ifndef __KV_CACHE_MANAGER_H__
#define __KV_CACHE_MANAGER_H__

#include <cstddef>
#include <fstream>
#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

#include <tensor.h>
#include <tensor_dim.h>

namespace causallm {

/**
 * @brief KV Cache Manager - manages KV cache externally for all attention
 *        layers in a transformer model.
 *
 * This class owns the KV cache memory and provides tensor views to mha_core
 * layers, replacing the internal cache allocation done by mha_core.
 *
 * Key responsibilities:
 * - Allocate KV cache buffers for all layers
 * - Track the current write position (cache_pos)
 * - Provide write-pointer tensor views for new K/V insertion
 * - Provide full-range tensor views for attention computation
 * - Save/load cache to/from files
 *
 * Future extensions:
 * - Cache eviction policies
 * - Cache compression / quantization
 * - Paged attention support
 */
class KVCacheManager {
public:
  /** @brief Allocator for cache slabs that an accelerator reads in place
   *         (ComputeOps::alloc_shared); nullptr means "heap for this slab". */
  using SharedAlloc = std::function<void *(size_t bytes)>;
  /** @brief Matching release (ComputeOps::free_shared). */
  using SharedFree = std::function<void(void *block)>;

  KVCacheManager() = default;
  ~KVCacheManager();

  // Non-copyable, movable
  KVCacheManager(const KVCacheManager &) = delete;
  KVCacheManager &operator=(const KVCacheManager &) = delete;
  KVCacheManager(KVCacheManager &&) = default;
  KVCacheManager &operator=(KVCacheManager &&other) noexcept;

  /**
   * @brief Route every cache slab through an accelerator-visible allocator.
   *
   * Call before allocate(). Each layer's key and value slab is then one
   * block from @a alloc, wrapped by the cache tensors without a copy, and
   * handed back to @a release when the manager is destroyed or
   * re-allocated. A nullptr from @a alloc puts that slab on the heap
   * instead, so a backend with limited shared memory degrades to the
   * per-call copy rather than failing. Views, save and load are unchanged.
   *
   * @param alloc   block allocator, e.g. ComputeOps::alloc_shared
   * @param release block release, e.g. ComputeOps::free_shared
   */
  void setSharedAllocator(SharedAlloc alloc, SharedFree release);

  /**
   * @brief Whether at least one slab came from the shared allocator.
   */
  bool usesSharedMemory() const { return !shared_blocks_.empty(); }

  /**
   * @brief Whether setSharedAllocator installed an allocator (a backend
   *        may still hand back nullptr and leave the slabs on the heap).
   */
  bool hasSharedAllocator() const { return static_cast<bool>(shared_alloc_); }

  /**
   * @brief Allocate KV cache for all layers
   * @param[in] num_layers number of attention layers
   * @param[in] batch_size batch size
   * @param[in] max_seq_len maximum sequence length (total cache capacity)
   * @param[in] num_heads_kv number of KV heads (for GQA)
   * @param[in] head_dim dimension per head
   * @param[in] dtype data type for cache tensors
   * @param[in] format tensor format
   */
  void allocate(
    unsigned int num_layers, unsigned int batch_size, unsigned int max_seq_len,
    unsigned int num_heads_kv, unsigned int head_dim,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::FP16,
    ml::train::TensorDim::Format format = ml::train::TensorDim::Format::NCHW);

  /**
   * @brief Allocate KV cache with per-layer KV widths.
   * @param[in] num_layers number of attention layers
   * @param[in] batch_size batch size
   * @param[in] max_seq_len maximum sequence length
   * @param[in] kv_widths per-layer width (num_heads_kv * head_dim)
   * @param[in] dtype data type for cache tensors
   * @param[in] format tensor format
   */
  void allocate(
    unsigned int num_layers, unsigned int batch_size, unsigned int max_seq_len,
    const std::vector<unsigned int> &kv_widths,
    ml::train::TensorDim::DataType dtype = ml::train::TensorDim::DataType::FP16,
    ml::train::TensorDim::Format format = ml::train::TensorDim::Format::NCHW,
    const std::vector<unsigned int> &rows = {},
    const std::vector<unsigned int> &keep = {});

  /**
   * @brief A windowed layer -- one allocated with fewer rows than
   *        max_seq_len -- holds absolute position a at row a - getBase():
   *        a sliding-window attention never reads past its window, so only
   *        the last keep rows before the step have to stay.
   * @param[in] layer_idx attention layer index
   * @return absolute position of the layer's row 0 (0 for a full layer)
   */
  unsigned int getBase(unsigned int layer_idx) const {
    return bases_.at(layer_idx);
  }

  /**
   * @brief Rows a layer's cache was allocated with.
   */
  unsigned int getRows(unsigned int layer_idx) const {
    return rows_.at(layer_idx);
  }

  /**
   * @brief Before a step that starts at absolute position pos: every
   *        windowed layer moves its last keep rows to the front and its
   *        base to pos - keep. The caller hands the new bases to the
   *        attention layers (they index and mirror the rows).
   * @param[in] pos absolute position of the next row to be written
   * @return true when a base changed
   */
  bool compact(unsigned int pos);

  /**
   * @brief Check if the manager has been allocated
   */
  bool isAllocated() const { return !layer_caches_.empty(); }

  /**
   * @brief Get current write position in the cache
   */
  unsigned int getPosition() const { return cache_pos_; }

  /**
   * @brief Set current write position (e.g., after loading pre-computed cache)
   * @param[in] pos new position
   */
  void setPosition(unsigned int pos);

  /**
   * @brief Advance the write position by step_size
   * @param[in] step_size number of positions to advance
   */
  void advance(unsigned int step_size);

  /**
   * @brief Reset position to 0 (for new inference session)
   */
  void reset();

  /**
   * @brief Get the full key cache tensor for a layer (for direct access)
   * @param[in] layer_idx attention layer index
   * @return reference to the full key cache tensor
   */
  nntrainer::Tensor &getKeyCache(unsigned int layer_idx);

  /**
   * @brief Get the full value cache tensor for a layer (for direct access)
   * @param[in] layer_idx attention layer index
   * @return reference to the full value cache tensor
   */
  nntrainer::Tensor &getValueCache(unsigned int layer_idx);

  /**
   * @brief Get a write-pointer view into key cache at current position
   *        for a specific batch and step_size.
   *        This is where new K values should be written.
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] step_size number of tokens to write
   * @return Tensor view pointing to the write location
   */
  nntrainer::Tensor getKeyCacheWriteView(unsigned int layer_idx,
                                         unsigned int batch,
                                         unsigned int step_size);

  /**
   * @brief Get a write-pointer view into value cache at current position
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] step_size number of tokens to write
   * @return Tensor view pointing to the write location
   */
  nntrainer::Tensor getValueCacheWriteView(unsigned int layer_idx,
                                           unsigned int batch,
                                           unsigned int step_size);

  /**
   * @brief Get a read view of key cache from position 0 to (cache_pos +
   * step_size) for attention computation (Q @ K^T).
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] read_len total length to read (typically cache_pos + step_size)
   * @return Tensor view covering [0, read_len)
   */
  nntrainer::Tensor getKeyCacheReadView(unsigned int layer_idx,
                                        unsigned int batch,
                                        unsigned int read_len);

  /**
   * @brief Get a read view of value cache from position 0 to read_len
   * @param[in] layer_idx attention layer index
   * @param[in] batch batch index
   * @param[in] read_len total length to read
   * @return Tensor view covering [0, read_len)
   */
  nntrainer::Tensor getValueCacheReadView(unsigned int layer_idx,
                                          unsigned int batch,
                                          unsigned int read_len);

  /**
   * @brief Save KV cache to file (all layers, up to current position)
   * @param[in] path file path
   */
  void save(const std::string &path) const;

  /**
   * @brief Save KV cache to file up to specified length: per layer the K
   *        rows then the V rows, raw. A windowed layer saves its last keep
   *        rows only.
   * @param[in] path file path
   * @param[in] seq_len number of positions to save
   */
  void save(const std::string &path, unsigned int seq_len) const;

  /**
   * @brief Load KV cache from file. The file must be exactly what save
   *        writes for seq_len positions of this allocation, or it throws. A
   *        windowed layer's rows land at row 0 and its base at seq_len -
   *        rows.
   * @param[in] path file path
   * @param[in] seq_len number of positions to load
   */
  void load(const std::string &path, unsigned int seq_len);

  /**
   * @brief Get number of layers
   */
  unsigned int getNumLayers() const {
    return static_cast<unsigned int>(layer_caches_.size());
  }

  /**
   * @brief Get maximum sequence length (cache capacity)
   */
  unsigned int getMaxSeqLen() const { return max_seq_len_; }

  /**
   * @brief Get batch size
   */
  unsigned int getBatchSize() const { return batch_size_; }

  /**
   * @brief Get the KV dimension width (num_heads_kv * head_dim)
   */
  unsigned int getKVWidth() const { return num_heads_kv_ * head_dim_; }

private:
  /**
   * @brief Per-layer cache storage
   */
  struct LayerCache {
    nntrainer::Tensor key_cache;   /**< (batch, 1, max_seq_len, kv_width) */
    nntrainer::Tensor value_cache; /**< (batch, 1, max_seq_len, kv_width) */
  };

  std::vector<LayerCache> layer_caches_; /**< per-layer KV caches */

  /**
   * @brief One cache slab: from the shared allocator when one is set and
   *        it delivers, otherwise an ordinary allocation.
   */
  nntrainer::Tensor makeCacheTensor(const ml::train::TensorDim &dim);

  /**
   * @brief Hands every shared block back to the allocator.
   */
  void releaseShared();

  SharedAlloc shared_alloc_;          /**< see setSharedAllocator */
  SharedFree shared_free_;            /**< see setSharedAllocator */
  std::vector<void *> shared_blocks_; /**< blocks owned via shared_alloc_ */

  unsigned int cache_pos_ = 0;    /**< current write position */
  unsigned int batch_size_ = 0;   /**< batch size */
  unsigned int max_seq_len_ = 0;  /**< max sequence length */
  unsigned int num_heads_kv_ = 0; /**< number of KV heads */
  unsigned int head_dim_ = 0;     /**< head dimension */
  unsigned int kv_width_ = 0;     /**< num_heads_kv * head_dim */
  std::vector<unsigned int> kv_widths_;
  std::vector<unsigned int> rows_;  /**< per-layer allocated rows */
  std::vector<unsigned int> keep_;  /**< per-layer window (windowed layers) */
  std::vector<unsigned int> bases_; /**< per-layer absolute pos of row 0 */

  /** @brief Whether the layer has fewer rows than max_seq_len. */
  bool windowed(unsigned int l) const { return rows_[l] < max_seq_len_; }

  /** @brief First absolute position a dump of rows [.., pos) holds for the
   *         layer: a windowed layer's last keep rows, else all of them. */
  unsigned int liveFrom(unsigned int l, unsigned int pos) const {
    return windowed(l) && pos > keep_[l] ? pos - keep_[l] : 0u;
  }

  ml::train::TensorDim::DataType dtype_ = ml::train::TensorDim::DataType::FP16;
  ml::train::TensorDim::Format format_ = ml::train::TensorDim::Format::NCHW;
};

} // namespace causallm

#endif // __KV_CACHE_MANAGER_H__
