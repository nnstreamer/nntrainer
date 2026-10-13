// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2026 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   kv_cache_manager.cpp
 * @date   25 April 2026
 * @brief  KV Cache Manager implementation
 * @see    https://github.com/nntrainer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 */

#include "kv_cache_manager.h"

#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>

#include <memory_data.h>

namespace causallm {

KVCacheManager::~KVCacheManager() { releaseShared(); }

KVCacheManager &KVCacheManager::operator=(KVCacheManager &&other) noexcept {
  if (this != &other) {
    // The tensors go first so nothing still points into the blocks the
    // allocator gets back.
    layer_caches_.clear();
    releaseShared();
    layer_caches_ = std::move(other.layer_caches_);
    cache_pos_ = other.cache_pos_;
    batch_size_ = other.batch_size_;
    max_seq_len_ = other.max_seq_len_;
    num_heads_kv_ = other.num_heads_kv_;
    head_dim_ = other.head_dim_;
    kv_width_ = other.kv_width_;
    kv_widths_ = std::move(other.kv_widths_);
    rows_ = std::move(other.rows_);
    keep_ = std::move(other.keep_);
    bases_ = std::move(other.bases_);
    dtype_ = other.dtype_;
    format_ = other.format_;
    shared_alloc_ = std::move(other.shared_alloc_);
    shared_free_ = std::move(other.shared_free_);
    shared_blocks_ = std::move(other.shared_blocks_);
    other.shared_blocks_.clear();
  }
  return *this;
}

void KVCacheManager::setSharedAllocator(SharedAlloc alloc, SharedFree release) {
  if (isAllocated()) {
    throw std::logic_error(
      "KVCacheManager::setSharedAllocator: call before allocate()");
  }
  if (static_cast<bool>(alloc) != static_cast<bool>(release)) {
    throw std::invalid_argument(
      "KVCacheManager::setSharedAllocator: need both alloc and release");
  }
  shared_alloc_ = std::move(alloc);
  shared_free_ = std::move(release);
}

nntrainer::Tensor
KVCacheManager::makeCacheTensor(const ml::train::TensorDim &dim) {
  if (shared_alloc_) {
    const size_t bytes = dim.getDataLen() * dim.getDataTypeSize();
    void *block = shared_alloc_(bytes);
    if (block) {
      shared_blocks_.push_back(block);
      // An unallocated tensor of this shape, then bound to the block: the
      // same setData() path the memory pool uses, so every view built from
      // it (getSharedDataTensor) resolves into the block.
      nntrainer::Tensor t(dim, false);
      t.setData(std::make_shared<nntrainer::MemoryData>(block), 0, false);
      return t;
    }
  }
  return nntrainer::Tensor(dim, true);
}

void KVCacheManager::releaseShared() {
  if (shared_free_) {
    for (void *block : shared_blocks_) {
      shared_free_(block);
    }
  }
  shared_blocks_.clear();
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              unsigned int num_heads_kv, unsigned int head_dim,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format) {
  if (num_heads_kv == 0 || head_dim == 0) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: all parameters must be > 0");
  }

  allocate(num_layers, batch_size, max_seq_len,
           std::vector<unsigned int>(num_layers, num_heads_kv * head_dim),
           dtype, format);

  num_heads_kv_ = num_heads_kv;
  head_dim_ = head_dim;
}

void KVCacheManager::allocate(unsigned int num_layers, unsigned int batch_size,
                              unsigned int max_seq_len,
                              const std::vector<unsigned int> &kv_widths,
                              ml::train::TensorDim::DataType dtype,
                              ml::train::TensorDim::Format format,
                              const std::vector<unsigned int> &rows,
                              const std::vector<unsigned int> &keep) {
  if (num_layers == 0 || batch_size == 0 || max_seq_len == 0 ||
      kv_widths.size() != num_layers ||
      (!rows.empty() &&
       (rows.size() != num_layers || keep.size() != num_layers))) {
    throw std::invalid_argument(
      "KVCacheManager::allocate: invalid layer, batch, or KV width count");
  }

  for (auto kv_width : kv_widths) {
    if (kv_width == 0) {
      throw std::invalid_argument(
        "KVCacheManager::allocate: KV widths must be > 0");
    }
  }

  batch_size_ = batch_size;
  max_seq_len_ = max_seq_len;
  num_heads_kv_ = 0;
  head_dim_ = 0;
  kv_width_ = kv_widths[0];
  kv_widths_ = kv_widths;
  dtype_ = dtype;
  format_ = format;
  cache_pos_ = 0;
  rows_.assign(num_layers, max_seq_len);
  keep_.assign(num_layers, 0);
  bases_.assign(num_layers, 0);
  for (unsigned int i = 0; i < rows.size(); ++i) {
    if (rows[i] == 0 || rows[i] >= max_seq_len)
      continue;
    // a step must fit behind the window it attends to
    if (keep[i] == 0 || keep[i] >= rows[i])
      throw std::invalid_argument(
        "KVCacheManager::allocate: a windowed layer needs 0 < keep < rows");
    rows_[i] = rows[i];
    keep_[i] = keep[i];
  }

  // Re-allocation: drop the old tensors before their blocks go back.
  layer_caches_.clear();
  releaseShared();
  layer_caches_.resize(num_layers);
  for (unsigned int i = 0; i < num_layers; ++i) {
    ml::train::TensorDim cache_dim({batch_size, 1, rows_[i], kv_widths_[i]},
                                   {format, dtype});
    layer_caches_[i].key_cache = makeCacheTensor(cache_dim);
    layer_caches_[i].value_cache = makeCacheTensor(cache_dim);
  }
}

void KVCacheManager::setPosition(unsigned int pos) {
  if (pos > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::setPosition: pos exceeds max_seq_len");
  }
  cache_pos_ = pos;
}

void KVCacheManager::advance(unsigned int step_size) {
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::advance: position would exceed max_seq_len");
  }
  cache_pos_ += step_size;
}

void KVCacheManager::reset() {
  cache_pos_ = 0;
  bases_.assign(bases_.size(), 0);
}

bool KVCacheManager::compact(unsigned int pos) {
  bool changed = false;
  for (unsigned int l = 0; l < layer_caches_.size(); ++l) {
    const unsigned int from = liveFrom(l, pos);
    if (from <= bases_[l])
      continue;
    const size_t w = kv_widths_[l];
    const size_t skip = (from - bases_[l]) * w, live = (pos - from) * w;
    for (auto *t : {&layer_caches_[l].key_cache, &layer_caches_[l].value_cache})
      for (unsigned int b = 0; b < batch_size_; ++b) {
        // fp16 bits on every build (FP16 or UINT16 storage)
        uint16_t *d =
          t->getData<uint16_t>() + b * static_cast<size_t>(rows_[l]) * w;
        std::memmove(d, d + skip, live * sizeof(uint16_t));
      }
    bases_[l] = from;
    changed = true;
  }
  return changed;
}

nntrainer::Tensor &KVCacheManager::getKeyCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getKeyCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].key_cache;
}

nntrainer::Tensor &KVCacheManager::getValueCache(unsigned int layer_idx) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range("KVCacheManager::getValueCache: invalid layer_idx");
  }
  return layer_caches_[layer_idx].value_cache;
}

nntrainer::Tensor KVCacheManager::getKeyCacheWriteView(unsigned int layer_idx,
                                                       unsigned int batch,
                                                       unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: invalid layer_idx");
  }
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheWriteView: would exceed max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheWriteView(
  unsigned int layer_idx, unsigned int batch, unsigned int step_size) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: invalid layer_idx");
  }
  if (cache_pos_ + step_size > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheWriteView: would exceed max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim step_dim({1, 1, step_size, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen() + cache_pos_ * kv_width;
  return cache.getSharedDataTensor(step_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getKeyCacheReadView(unsigned int layer_idx,
                                                      unsigned int batch,
                                                      unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getKeyCacheReadView: read_len exceeds max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].key_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

nntrainer::Tensor KVCacheManager::getValueCacheReadView(unsigned int layer_idx,
                                                        unsigned int batch,
                                                        unsigned int read_len) {
  if (layer_idx >= layer_caches_.size()) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: invalid layer_idx");
  }
  if (read_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::getValueCacheReadView: read_len exceeds max_seq_len");
  }

  auto &cache = layer_caches_[layer_idx].value_cache;
  ml::train::TensorDim cache_dim = cache.getDim();
  const unsigned int kv_width = kv_widths_[layer_idx];
  ml::train::TensorDim read_dim({1, 1, read_len, kv_width}, {format_, dtype_});

  size_t offset = batch * cache_dim.getFeatureLen();
  return cache.getSharedDataTensor(read_dim, offset, true);
}

void KVCacheManager::save(const std::string &path) const {
  save(path, cache_pos_);
}

void KVCacheManager::save(const std::string &path, unsigned int seq_len) const {
  if (layer_caches_.empty()) {
    throw std::runtime_error("KVCacheManager::save: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::save: seq_len exceeds max_seq_len");
  }

  std::ofstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::save: cannot open file: " + path);
  }

  for (unsigned int l = 0; l < layer_caches_.size(); ++l) {
    const auto &lc = layer_caches_[l];
    const unsigned int from = liveFrom(l, seq_len);
    if (from < bases_[l] || seq_len - bases_[l] > rows_[l])
      throw std::out_of_range("KVCacheManager::save: layer " +
                              std::to_string(l) + " does not hold rows [" +
                              std::to_string(from) + ", " +
                              std::to_string(seq_len) + ")");
    ml::train::TensorDim save_dim = lc.key_cache.getDim();
    save_dim.height(seq_len - from);
    const size_t off = static_cast<size_t>(from - bases_[l]) * kv_widths_[l];

    nntrainer::Tensor k_slice = const_cast<nntrainer::Tensor &>(lc.key_cache)
                                  .getSharedDataTensor(save_dim, off, true);
    nntrainer::Tensor v_slice = const_cast<nntrainer::Tensor &>(lc.value_cache)
                                  .getSharedDataTensor(save_dim, off, true);

    k_slice.save(f);
    v_slice.save(f);
  }
}

void KVCacheManager::load(const std::string &path, unsigned int seq_len) {
  if (layer_caches_.empty()) {
    throw std::runtime_error("KVCacheManager::load: not allocated");
  }
  if (seq_len > max_seq_len_) {
    throw std::out_of_range(
      "KVCacheManager::load: seq_len exceeds max_seq_len");
  }

  std::ifstream f(path, std::ios::binary);
  if (!f.is_open()) {
    throw std::runtime_error("KVCacheManager::load: cannot open file: " + path);
  }

  // A dump of another model, length or dtype reads without an error and
  // decodes garbage: the file must be exactly the rows asked for.
  auto slice_dim = [&](unsigned int l) {
    ml::train::TensorDim d = layer_caches_[l].key_cache.getDim();
    d.height(seq_len - liveFrom(l, seq_len));
    return d;
  };
  size_t want = 0;
  for (unsigned int l = 0; l < layer_caches_.size(); ++l) {
    const ml::train::TensorDim d = slice_dim(l);
    want += 2 * d.getDataLen() * d.getDataTypeSize();
  }
  f.seekg(0, std::ios::end);
  const size_t have = static_cast<size_t>(f.tellg());
  f.seekg(0, std::ios::beg);
  if (have != want) {
    throw std::runtime_error(
      "KVCacheManager::load: " + path + " is " + std::to_string(have) +
      " bytes, " + std::to_string(want) + " expected for " +
      std::to_string(seq_len) + " rows (another model, length or dtype?)");
  }

  for (unsigned int l = 0; l < layer_caches_.size(); ++l) {
    auto &lc = layer_caches_[l];
    const ml::train::TensorDim load_dim = slice_dim(l);

    nntrainer::Tensor k_slice =
      lc.key_cache.getSharedDataTensor(load_dim, 0, true);
    nntrainer::Tensor v_slice =
      lc.value_cache.getSharedDataTensor(load_dim, 0, true);

    k_slice.read(f);
    v_slice.read(f);
    bases_[l] = liveFrom(l, seq_len);
  }

  cache_pos_ = seq_len;
}

} // namespace causallm
