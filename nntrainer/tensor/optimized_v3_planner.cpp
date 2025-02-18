// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2023 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimized_v3_planner.cpp
 * @date   2 January 2023
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Optimized V3 Memory Planner
 *
 */

#include <algorithm>
#include <memory>
#include <nntrainer_error.h>
#include <stdexcept>
#include <vector>

#include <optimized_v3_planner.h>

namespace nntrainer {

/**
 * @brief Memory Request data structure clubbing all the requests
 *
 */
struct MemoryRequest {
  unsigned int start; /**< start of the validity (inclusive) */
  unsigned int end;   /**< end of the validity (exclusive) */
  unsigned int loc;   /**< index/location of the this request */
  size_t size;        /**< size of the request */
  size_t offset;      /**< offset for this request */

  /**
   * @brief Constructor for the Memory Request
   *
   */
  MemoryRequest(size_t s, const std::pair<unsigned int, unsigned int> &valid,
                unsigned int idx) :
    start(valid.first),
    end(valid.second),
    loc(idx),
    size(s),
    offset(0) {}
};

static size_t computeSpace(unsigned int exec_order,
                           std::vector<MemoryRequest *> &sorted_req,
                           std::vector<std::pair<size_t, size_t>> &vacant) {
  size_t bottom = 0;
  size_t max_offset = 0;

  std::sort(sorted_req.begin(), sorted_req.end(),
            [](auto const &v1, auto const &v2) -> int {
              return v1->offset < v2->offset;
              /** TODO: try this */
              //   if (v1.end == v2.end)
              //     return v1.start < v2.start;
              //   return v1.end > v2.end;
            });

  for (unsigned idx = 0; idx < sorted_req.size(); idx++) {
    auto const &sr = sorted_req[idx];
    size_t top = sr->offset + sr->size;

    if (max_offset < top)
      max_offset = top;

    if (sr->offset > bottom) {
      vacant.push_back(std::make_pair(bottom, sr->offset));
    }
    bottom = top;
  }

  return max_offset;
}

/**
 * @brief check if validate interval is overlapping in a very naive way.
 *
 * @param memory_validity validity
 * @param memory_size  size
 * @param memory_offset  offset
 * @param memory_req  request
 */
[[maybe_unused]] static void validateIntervalOverlap(
  const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
  const std::vector<size_t> &memory_size,
  const std::vector<size_t> &memory_offset, size_t memory_req) {
  auto bits = std::make_unique<bool[]>(memory_req);

  for (size_t i = 0; i < memory_req; ++i) {
    bits[i] = 0;
  }

  auto exec_start =
    std::min_element(memory_validity.begin(), memory_validity.end(),
                     [](auto &a, auto &b) { return a.first < b.first; });

  auto exec_end =
    std::max_element(memory_validity.begin(), memory_validity.end(),
                     [](auto &a, auto &b) { return a.second < b.second; });

  auto set = [&](int offset, size_t size, int idx) {
    for (unsigned int i = offset; i < size; ++i) {
      NNTR_THROW_IF(bits[i], std::invalid_argument)
        << " bits taken at i: " << i << " offset: " << offset
        << " size: " << size << " idx: " << idx;
      bits[i] = 1;
    }
  };

  auto unset = [&](int offset, size_t size, int idx) {
    for (unsigned int i = offset; i < size; ++i) {
      NNTR_THROW_IF(!bits[i], std::invalid_argument)
        << "double freeing bits at i: " << i << " offset: " << offset
        << " size: " << size << " idx: " << idx;
      bits[i] = 0;
    }
  };

  for (unsigned int exec = exec_start->first; exec <= exec_end->second;
       ++exec) {

    for (unsigned int idx = 0; idx < memory_validity.size(); ++idx) {
      auto &validity = memory_validity.at(idx);
      auto &sz = memory_size.at(idx);
      auto &offset = memory_offset.at(idx);
      if (validity.first == exec) {
        set(offset, sz, idx);
      }
      if (validity.second == exec) {
        unset(offset, sz, idx);
      }
    }
  }
  // check if there is any dangling memory
  set(0, memory_req, memory_validity.size());
}

/**
 * @copydoc MemoryPlanner::planLayout(
 * const std::vector<size_t> &memory_size,
 * const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
 * std::vector<size_t> &memory_offset,
 * std::vector<bool> &memory_is_wgrad);
 *
 * @details The optimized v1 memory planner assigns memory to the requests whose
 * validity starts first.
 * The requested memories are sorted based on the ascending order of the start
 * timestamps, and descending order using the end timestamps. The
 * sorted memories are given increasing offset based on the memory size.
 * At the end of each timestamp, invalid memories are freed, and offset updated
 * for reuse. This planner allocates overlapping memory for all the required
 * memories.
 *
 */
size_t OptimizedV3Planner::planLayout(
  const std::vector<size_t> &memory_size,
  const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
  std::vector<size_t> &memory_offset, std::vector<bool> &memory_is_wgrad,
  size_t n_wgrad) const {

  /** create memory requests structure array for easier management */
  std::vector<MemoryRequest> requests;
  requests.reserve(memory_size.size());
  for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
    requests.emplace_back(memory_size[idx], memory_validity[idx], idx);
  }

  /**
   * sort the memory requests with ascending order of start time first, and
   * then end time
   */
  std::sort(requests.begin(), requests.end(),
            [](auto const &v1, auto const &v2) -> int {
              if (v1.start == v2.start)
                return v1.end < v2.end;
              return v1.start < v2.start;
              /** TODO: try this */
              //   if (v1.end == v2.end)
              //     return v1.start < v2.start;
              //   return v1.end > v2.end;
            });

  /** all the memories in use sorted by their assigned offset and size */
  std::vector<MemoryRequest *> sorted_req;

  /** iterate over the sorted requests and start allocation of the requests */
  memory_offset.resize(memory_size.size());
  size_t memory_req = 0;
  for (auto &req : requests) {
    sorted_req.erase(
      std::remove_if(sorted_req.begin(), sorted_req.end(),
                     [req](auto elem) { return elem->end <= req.start; }),
      sorted_req.end());

    bool replace_and_fill = false;
    std::vector<std::pair<size_t, size_t>> vacant;

    size_t max_offset = computeSpace(req.start, sorted_req, vacant);

    for (unsigned int idx = 0; idx < vacant.size(); idx++) {
      if (vacant[idx].second - vacant[idx].first >= req.size) {
        req.offset = vacant[idx].first;
        memory_offset[req.loc] = req.offset;
        sorted_req.push_back(&req);
        replace_and_fill = true;
        break;
      }
    }
    vacant.clear();

    if (replace_and_fill) {
      continue;
    }

    req.offset = max_offset;
    memory_offset[req.loc] = max_offset;
    memory_req = std::max(memory_req, req.offset + req.size);
    sorted_req.push_back(&req);
  }

  return memory_req;
}

} // namespace nntrainer
