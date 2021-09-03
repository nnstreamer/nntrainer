// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2021 Parichay Kapoor <pk.kapoor@samsung.com>
 *
 * @file   optimized_v1_planner.cpp
 * @date   3 September 2021
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Optimized V1 Memory Planner
 *
 */

#include <algorithm>
#include <vector>

#include <optimized_v1_planner.h>

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

/**
 * @copydoc MemoryPlanner::planLayout(
 * const std::vector<size_t> &memory_size,
 * const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
 * std::vector<size_t> &memory_offset);
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
size_t OptimizedV1Planner::planLayout(
  const std::vector<size_t> &memory_size,
  const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
  std::vector<size_t> &memory_offset) const {

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
    /** remove expired memories and update offset */
    while (!sorted_req.empty() && sorted_req.back()->end <= req.start)
      sorted_req.pop_back();

    /** if there exists an expired memory with same size (not at the edge),
     * reuse it */
    bool replace_and_fill = false;
    for (int idx = sorted_req.size() - 1; idx >= 0; idx--) {
      auto const &sr = sorted_req[idx];
      /** TODO: reuse if memory size not exactly match */
      if (sr->end <= req.start && sr->size == req.size) {
        req.offset = sr->offset;
        memory_offset[req.loc] = req.offset;
        sorted_req[idx] = &req;
        replace_and_fill = true;
        break;
      }
    }
    if (replace_and_fill) {
      continue;
    }

    size_t offset = 0;
    if (!sorted_req.empty())
      offset = sorted_req.back()->offset + sorted_req.back()->size;

    /** assign offset to the new request and push to queue */
    req.offset = offset;
    memory_offset[req.loc] = offset;
    memory_req = std::max(memory_req, req.offset + req.size);
    sorted_req.push_back(&req);
  }

  return memory_req;
}

} // namespace nntrainer
