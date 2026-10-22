// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   memory_pool.cpp
 * @date   11 August 2021
 * @see    https://github.com/nntrainer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Memory Pool Class
 */

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <map>
#include <memory_pool.h>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <numeric>
#include <profiler.h>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <sysinfoapi.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace nntrainer {

namespace {

/**
 * @brief Page size used as the default alignment for pool allocations.
 *
 * Page-aligned buffers are required by RPC / SVM backends and don't
 * hurt the host case (aligned_alloc rounds up the request).
 */
size_t system_page_size() {
#if defined(_WIN32)
  SYSTEM_INFO sys_info;
  GetSystemInfo(&sys_info);
  return sys_info.dwPageSize;
#else
  return static_cast<size_t>(sysconf(_SC_PAGE_SIZE));
#endif
}

} // namespace

unsigned int MemoryPool::requestMemory(size_t bytes, unsigned int start_time,
                                       unsigned int end_time,
                                       std::vector<unsigned int> exec_order,
                                       TensorLifespan lifespan, bool is_wgrad,
                                       bool is_pipeline) {
  if (bytes == 0)
    throw std::invalid_argument("Requesting memory of 0 size");

  if (mem_pool != nullptr || !owned_buffers_.empty())
    throw std::invalid_argument(
      "Deallocate memory pool before requesting more memory");

  if (end_time <= start_time)
    throw std::invalid_argument(
      "Invalid validity range for the requested memory");

  const bool starting_new_request_set = memory_size.empty();
  if (starting_new_request_set)
    n_wgrad = 0;

  if (!memory_offset.empty()) {
    memory_offset.clear();
    planned_memory_size_.clear();
    pool_size = 0;
  }

  memory_size.push_back(bytes);
  memory_validity.push_back({start_time, end_time});
  memory_exec_order.push_back(exec_order);
  memory_is_wgrad.push_back(is_wgrad);
  memory_is_pipeline.push_back(is_pipeline);
  if (is_wgrad)
    n_wgrad++;

  /** invalidate min_pool_size if already there */
  min_pool_size = 0;

  return memory_size.size();
}

double MemoryPool::planLayout(const MemoryPlanner &planner) {
  if (mem_pool != nullptr || !owned_buffers_.empty())
    /** allocated buffers must be deallocated before planLayout */
    throw std::runtime_error("Planning memory layout after allocation");

  if (memory_size.empty())
    throw std::runtime_error("Planning memory layout for empty pool");

  /** calculate min_pool_size if not already calculated */
  if (min_pool_size == 0)
    min_pool_size = calcMinMemoryRequirement();

  const bool use_band =
    pipeline_band_ &&
    std::any_of(memory_is_pipeline.begin(), memory_is_pipeline.end(),
                [](bool p) { return p; });

  if (!use_band) {
    pool_size = planner.planLayout(memory_size, memory_validity, memory_offset,
                                   memory_is_wgrad, n_wgrad);
    band_end_ = 0;
  } else {
    /**
     * Pipeline band partition (two-pass):
     * pass 1 plans the pipeline-band tensors (GPU-region activations and the
     * boundary outputs) in their own validity space starting at offset 0 --
     * their internal reuse is planned exactly as before, just relocated.
     * pass 2 plans the remaining tensors after the band's end offset, so no
     * CPU-region tensor can alias a GPU-region one when sample i's tail runs
     * while sample i+1's prefix executes.
     */
    const size_t n_req = memory_size.size();
    std::vector<size_t> pipe_size;
    std::vector<size_t> rest_size;
    std::vector<std::pair<unsigned int, unsigned int>> pipe_validity;
    std::vector<std::pair<unsigned int, unsigned int>> rest_validity;
    std::vector<bool> pipe_wgrad;
    std::vector<bool> rest_wgrad;
    std::vector<unsigned int> pipe_loc;
    std::vector<unsigned int> rest_loc;
    /** is_wgrad flag must be consistent within the subset for the planner's
     * n_wgrad bookkeeping: count wgrads per subset. */
    size_t pipe_n_wgrad = 0;
    size_t rest_n_wgrad = 0;

    pipe_size.reserve(n_req);
    rest_size.reserve(n_req);
    for (size_t idx = 0; idx < n_req; ++idx) {
      if (memory_is_pipeline[idx]) {
        pipe_size.push_back(memory_size[idx]);
        pipe_validity.push_back(memory_validity[idx]);
        pipe_wgrad.push_back(memory_is_wgrad[idx]);
        pipe_n_wgrad += memory_is_wgrad[idx] ? 1 : 0;
        pipe_loc.push_back(idx);
      } else {
        rest_size.push_back(memory_size[idx]);
        rest_validity.push_back(memory_validity[idx]);
        rest_wgrad.push_back(memory_is_wgrad[idx]);
        rest_n_wgrad += memory_is_wgrad[idx] ? 1 : 0;
        rest_loc.push_back(idx);
      }
    }

    memory_offset.assign(n_req, 0);

    std::vector<size_t> pipe_offset;
    size_t band_size = planner.planLayout(pipe_size, pipe_validity, pipe_offset,
                                          pipe_wgrad, pipe_n_wgrad);

    std::vector<size_t> rest_offset;
    size_t rest_size_total = planner.planLayout(
      rest_size, rest_validity, rest_offset, rest_wgrad, rest_n_wgrad);

    for (size_t i = 0; i < pipe_loc.size(); ++i)
      memory_offset[pipe_loc[i]] = pipe_offset[i];
    for (size_t i = 0; i < rest_loc.size(); ++i)
      memory_offset[rest_loc[i]] = rest_offset[i] + band_size;

    pool_size = band_size + rest_size_total;
    band_end_ = band_size;
  }

  if (pool_size < min_pool_size || !validateLayout())
    throw std::runtime_error("Planned layout is not feasible");

  planned_memory_size_ = memory_size;

  return double(min_pool_size) / double(pool_size);
}

void MemoryPool::allocate() {
  if (pool_size == 0)
    throw std::runtime_error("Allocating memory pool with size 0");

  if (mem_pool != nullptr || !owned_buffers_.empty())
    throw std::runtime_error("Memory pool is already allocated");

  ml_logi("MemoryPool::allocate size: %zu allocator: %s", pool_size,
          allocator_->getName().c_str());

  if (allocator_->getName() == "qnn") {
    const auto &allocation_sizes =
      memory_size.empty() ? planned_memory_size_ : memory_size;
    NNTR_THROW_IF(allocation_sizes.size() != memory_offset.size(),
                  std::runtime_error)
      << "QNN memory layout metadata is stale; call planLayout() before "
         "allocate()";

    std::map<size_t, size_t> offset_size;
    for (size_t i = 0; i < memory_offset.size(); ++i) {
      auto it = offset_size.find(memory_offset[i]);
      if (it == offset_size.end()) {
        offset_size[memory_offset[i]] = allocation_sizes[i];
      } else {
        it->second = std::max(it->second, allocation_sizes[i]);
      }
    }

    const size_t alignment = system_page_size();
    std::map<size_t, void *> offset_ptr;
    for (auto &entry : offset_size) {
      void *ptr = nullptr;
      allocator_->alloc(&ptr, entry.second, alignment);
      owned_buffers_.push_back(ptr);
      offset_ptr[entry.first] = ptr;
#ifdef PROFILE
      static long long seq = 0;
      std::string msg("MemoryPool #");
      msg.append(std::to_string(seq++));
      PROFILE_MEM_ALLOC(ptr, entry.second, msg);
#endif
    }

    for (auto &offset : memory_offset)
      memory_ptrs.push_back(offset_ptr.at(offset));

    return;
  }

  // Single contiguous buffer routed through the per-vendor allocator.
  // For ClSVMAllocator this returns SVM memory directly addressable
  // by both host and device; the base allocator returns page-aligned
  // host memory that is not zero-initialised.
  //
  // [p8] Pipeline-band split allocation. With the band partition active on an
  // SVM allocator, the tail (rest) region belongs to tensors only the HOST
  // touches (the trainable CPU blocks) while the band region is what the GPU
  // prefix walks. On a coarse-grain SVM allocation the producer's blocking
  // map/unmap of a band tensor makes the driver do allocation-wide cache
  // maintenance, and a maintenance sweep that also retires the dirty lines of
  // the tail region the training thread is writing at the same moment reads
  // back as run-to-run drift. Two allocations make that impossible: the
  // driver's maintenance for one cannot reach the other's pages.
  //
  // Default (band off, or non-SVM allocator, or the split declined): the
  // single allocation as before -- byte-identical layout, same numbers.
  const bool split_band = band_split_requested_ && band_end_ > 0 &&
                          band_end_ < pool_size && allocator_->isSVM();
  band_split_active_ = split_band;
  if (split_band) {
    allocator_->alloc(&mem_pool, band_end_, system_page_size());
    rest_pool_ = nullptr;
    MemAllocator base_allocator; // the plain host allocator, not the SVM one
    base_allocator.alloc(&rest_pool_, pool_size - band_end_,
                         system_page_size());
    owned_buffers_.push_back(mem_pool);
    owned_buffers_.push_back(rest_pool_);
    for (size_t i = 0; i < memory_offset.size(); ++i) {
      const size_t off = memory_offset[i];
      memory_ptrs.push_back(
        off < band_end_ ? static_cast<char *>(mem_pool) + off
                        : static_cast<char *>(rest_pool_) + (off - band_end_));
    }
  } else {
    allocator_->alloc(&mem_pool, pool_size, system_page_size());
    rest_pool_ = nullptr;
    owned_buffers_.push_back(mem_pool);

    // Hand out per-token slices off the same buffer at planned offsets.
    for (size_t i = 0; i < memory_offset.size(); ++i) {
      char *ptr = static_cast<char *>(mem_pool) + memory_offset[i];
      memory_ptrs.push_back(ptr);
    }
  }

#ifdef PROFILE
  static long long seq = 0;
  std::string msg("MemoryPool #");
  msg.append(std::to_string(seq++));
  PROFILE_MEM_ALLOC(mem_pool, pool_size, msg);
#endif
}

void MemoryPool::allocateFSU() {
  if (pool_size == 0)
    throw std::runtime_error("Allocating memory pool with size 0");

  if (mem_pool != nullptr || !owned_buffers_.empty())
    throw std::runtime_error("Memory pool is already allocated");

  ml_logi("MemoryPool::allocateFSU size: %zu allocator: %s", pool_size,
          allocator_->getName().c_str());

  // FSU path allocates a separate buffer per distinct offset and
  // shares it across all tokens that map to that offset (typical for
  // double-buffered weight reuse). Earlier code used ALIGNED_ALLOC
  // macros and never freed these — fixed here by tracking unique
  // buffers in owned_buffers_.
  std::map<size_t, void *> offset_ptr;     // offset : ptr
  std::map<size_t, size_t> allocated_size; // offset : memory size
  std::map<size_t, std::vector<int>>
    offset_indices; // offset : indices that share this offset

  const size_t alignment = system_page_size();

  int i = 0;
  for (auto &s : memory_offset) {
    size_t current_size = memory_size.at(i);
    /** An offset a second plane owns outright gets no shared slice at all --
     *  see sharedSliceNeeded(). The token's shared pointer stays null, which
     *  is the truthful value for bytes that exist only on the other plane;
     *  reading it is a bug this makes loud instead of silently serving a
     *  never-written slice. Recorded in offset_ptr so the aliasing tokens at
     *  the same offset take the same answer. */
    if (!sharedSliceNeeded(s)) {
      memory_ptrs.push_back(nullptr);
      offset_ptr[s] = nullptr;
      allocated_size[s] = current_size;
      offset_indices[s].push_back(i);
      i++;
      continue;
    }
    auto it = offset_ptr.find(s);
    if (it == offset_ptr.end()) {
      void *ptr = nullptr;
      allocator_->alloc(&ptr, current_size, alignment);
      memory_ptrs.push_back(ptr);
      owned_buffers_.push_back(ptr);
      offset_ptr[s] = ptr;
      allocated_size[s] = current_size;
      offset_indices[s].push_back(i);
    } else {
      void *existing_ptr = it->second;
      size_t max_size = allocated_size[s];
      if (max_size < current_size) {
        // A larger request lands on the same offset — reallocate the
        // shared buffer and remap every aliasing entry.
        void *new_ptr = nullptr;
        allocator_->alloc(&new_ptr, current_size, alignment);
        for (int idx : offset_indices[s]) {
          memory_ptrs[idx] = new_ptr;
        }
        // Replace in owned_buffers_ so deallocate() releases the new
        // one, not the old one we're about to free.
        auto pos =
          std::find(owned_buffers_.begin(), owned_buffers_.end(), existing_ptr);
        if (pos != owned_buffers_.end())
          *pos = new_ptr;
        allocator_->free(existing_ptr);
        offset_ptr[s] = new_ptr;
        allocated_size[s] = current_size;
      }
      memory_ptrs.push_back(offset_ptr[s]);
      offset_indices[s].push_back(i);
    }
    i++;
  }
}

std::shared_ptr<MemoryData> MemoryPool::getMemory(unsigned int idx) {
  if (mem_pool == nullptr && owned_buffers_.empty())
    throw std::invalid_argument("Getting memory before allocation");

  void *shared = memory_ptrs.at(idx - 1);
  auto mem_data = std::make_shared<MemoryData>(shared);
  // SVM-ness comes from what the allocator produces, not from its name: an
  // OpenCL SVM allocation is both host-addressable and device-visible. See
  // MemAllocator::isSVM() for why this is not a generic unified-memory test.
  //
  // ... EXCEPT for a token that has no shared slice at all. allocateFSU()
  // leaves those null on purpose (sharedSliceNeeded(), above) because the bytes
  // exist only on a second plane, and the comment there promises the null makes
  // a read "loud". It did not: asking the ALLOCATOR meant a null-pointer token
  // still answered isSVM() == true, so every `md->isSVM()` guard downstream
  // passed and handed the null straight to a kernel as an SVM argument --
  // mha_core does exactly that for Q/K/V on the OHWI attention path, and the
  // GPU then reads zeros through its own mapping and the model
  // answers something else, silently. The flags belong to the TOKEN, not to
  // the allocator: no memory, no planes.
  mem_data->setSVM(shared != nullptr && allocator_->isSVM() &&
                   tokenInBand(idx));
  // Host addressability is taken from the allocator alone (a null-slice token
  // is not special-cased here): a device-only plane (cudaMalloc) must be
  // staged, never dereferenced, and the allocator is the only thing that knows.
  mem_data->setHostAddressable(allocator_->isHostAddressable() ||
                               rest_pool_ != nullptr);
  return mem_data;
}

void MemoryPool::deallocate() {
  // Symmetric to allocate/allocateFSU: free every uniquely-owned
  // buffer through the same allocator. Mismatching alloc/free
  // backends would corrupt the heap (e.g. clSVMFree on host memory),
  // so the allocator_ shared_ptr is the single source of truth.
  for (void *buf : owned_buffers_) {
    if (rest_pool_ != nullptr && buf == rest_pool_) {
      // [p8] the split tail buffer came from the BASE host allocator, never
      // from the SVM allocator -- handing it to clSVMFree would be a heap
      // corruption. This is the one pointer whose release path differs.
      MemAllocator base_allocator;
      base_allocator.free(buf);
      continue;
    }
    allocator_->free(buf);
#ifdef PROFILE
    PROFILE_MEM_DEALLOC(buf);
#endif
  }
  owned_buffers_.clear();

  mem_pool = nullptr;
  rest_pool_ = nullptr;
  memory_size.clear();
  memory_validity.clear();
  memory_exec_order.clear();
  memory_is_wgrad.clear();
  memory_is_pipeline.clear();
  memory_ptrs.clear();
}

size_t MemoryPool::size() { return pool_size; }

size_t MemoryPool::minMemoryRequirement() {
  if (memory_size.size() && min_pool_size == 0)
    min_pool_size = calcMinMemoryRequirement();

  return min_pool_size;
}

bool MemoryPool::validateLayout() {
  if (memory_offset.size() != memory_size.size())
    return false;

  if (memory_size.empty())
    return pool_size == 0;

  return validateOverflow() && validateOverlap();
}

bool MemoryPool::validateOverflow() {
  for (unsigned int idx = 0; idx < memory_size.size(); idx++)
    if (memory_offset[idx] + memory_size[idx] > pool_size)
      return false;

  return true;
}

template <typename T> static bool overlap(T s1, T e1, T s2, T e2) {
#if DEBUG
  if (e1 <= s1 || e2 <= s2)
    throw std::invalid_argument("Invalid range for intervals in MemoryPool");
#endif

  return !(e1 <= s2 || e2 <= s1);
}

bool MemoryPool::validateOverlap() {
  std::vector<unsigned int> perm = getSortedPermutation();

  size_t len = perm.size();
  for (unsigned int i = 0; i < len; i++) {
    unsigned int idx = perm[i];
    size_t mem_start = memory_offset[idx], mem_size = memory_size[idx];
    unsigned int valid_start = memory_validity[idx].first,
                 valid_end = memory_validity[idx].second;
    for (unsigned int match = idx + 1; match < len; match++) {
      if (overlap(mem_start, mem_start + mem_size, memory_offset[match],
                  memory_offset[match] + memory_size[match])) {
        if (overlap(valid_start, valid_end, memory_validity[match].first,
                    memory_validity[match].second))
          return false;
      } else {
        break;
      }
    }
  }

  return true;
}

std::vector<unsigned int> MemoryPool::getSortedPermutation() {
  std::vector<unsigned int> perm(memory_size.size());
  std::iota(perm.begin(), perm.end(), 0);
  std::sort(perm.begin(), perm.end(), [&](auto const &idx1, auto const &idx2) {
    if (memory_offset[idx1] == memory_offset[idx2])
      return memory_size[idx1] < memory_size[idx2];

    return memory_offset[idx1] < memory_offset[idx2];
  });

  return perm;
}

size_t MemoryPool::calcMinMemoryRequirement() {
  auto max_interval =
    *std::max_element(memory_validity.begin(), memory_validity.end(),
                      [](auto const &val1, auto const &val2) {
                        return val1.second < val2.second;
                      });
  unsigned int last_interval = max_interval.second;
  if (last_interval == (std::numeric_limits<unsigned int>::max)()) {
    max_interval = *std::max_element(
      memory_validity.begin(), memory_validity.end(),
      [last_interval](auto const &val1, auto const &val2) {
        return ((val2.second != last_interval) && (val1.second < val2.second));
      });
    last_interval = max_interval.second;
    if (last_interval == (std::numeric_limits<unsigned int>::max)())
      last_interval = 1;
  }

  std::vector<size_t> interval_req(last_interval + 1, 0);
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    for (unsigned int interval = memory_validity[idx].first;
         interval < std::min(memory_validity[idx].second, last_interval);
         interval++) {
      interval_req[interval] += memory_size[idx];
    }
  }

  return *std::max_element(interval_req.begin(), interval_req.end());
}

void MemoryPool::clear() {
  if (mem_pool != nullptr || !owned_buffers_.empty())
    throw std::invalid_argument("Cannot clear allocated memory pool");

  memory_size.clear();
  memory_validity.clear();
  memory_offset.clear();
  planned_memory_size_.clear();
  file_offset.clear();
  memory_is_wgrad.clear();
  memory_is_pipeline.clear();

  pool_size = 0;
  min_pool_size = 0;
  n_wgrad = 0;
  band_end_ = 0;
  band_split_active_ = false;
}

bool MemoryPool::isAllocated() const {
  return mem_pool != nullptr || !owned_buffers_.empty();
}

} // namespace nntrainer
