// SPDX-License-Identifier: Apache-2.0
/**
 * Copyright (C) 2022 Jijoong Moon <jijoong.moon@samsung.com>
 *
 * @file   optimized_v2_planner.cpp
 * @date   29 December 2022
 * @see    https://github.com/nnstreamer/nntrainer
 * @author Parichay Kapoor <pk.kapoor@samsung.com>
 * @author Jijoong Moon <jijoong.moon@samsung.com>
 * @bug    No known bugs except for NYI items
 * @brief  This is Optimized V2 Memory Planner
 *
 */

#include <algorithm>
#include <memory>
#include <nntrainer_error.h>
#include <nntrainer_log.h>
#include <stdexcept>
#include <vector>

#include <optimized_v2_planner.h>

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
 * @brief Memory Request data structure clubbing for the weight gradient
 * requests
 */
struct WGradMemoryRequest {
  MemoryRequest *mem_req;
  std::vector<std::pair<unsigned int, unsigned int>> start_end;

  WGradMemoryRequest(MemoryRequest *req) : mem_req(req) {}
};

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
size_t OptimizedV2Planner::planLayout(
  const std::vector<size_t> &memory_size,
  const std::vector<std::pair<unsigned int, unsigned int>> &memory_validity,
  std::vector<size_t> &memory_offset, std::vector<bool> &memory_is_wgrad,
  size_t n_wgrad) const {

  std::vector<MemoryRequest> wgrad_requests;
  wgrad_requests.reserve(n_wgrad);

  /** create memory requests structure array for easier management */
  std::vector<MemoryRequest> requests;
  requests.reserve(memory_size.size() - n_wgrad);
  if (n_wgrad) {
    for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
      if (!memory_is_wgrad[idx]) {
        requests.emplace_back(memory_size[idx], memory_validity[idx], idx);
      } else {
        wgrad_requests.emplace_back(memory_size[idx], memory_validity[idx],
                                    idx);
      }
    }
  } else {
    for (unsigned int idx = 0; idx < memory_size.size(); idx++) {
      requests.emplace_back(memory_size[idx], memory_validity[idx], idx);
    }
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

  if (wgrad_requests.size()) {
    /** TODO: We donot need to start from memeory_req. We might find proper
     * offset considering execution order */
    size_t last_offset = memory_req;

    /* sort the memory request with ascending order of size */
    std::sort(
      wgrad_requests.begin(), wgrad_requests.end(),
      [](auto const &v1, auto const &v2) -> int { return v1.size > v2.size; });

    std::vector<WGradMemoryRequest> wgrad_sorted_req;

    bool replace_and_fill = false;
    unsigned int new_grad_cnt = 0;
    unsigned int reused_grad_cnt = 0;
    size_t new_grad_size = 0;
    size_t reused_grad_size = 0;
    for (auto &req : wgrad_requests) {
      for (unsigned int idx = 0; idx < wgrad_sorted_req.size(); idx++) {
        auto const sr = wgrad_sorted_req[idx];
        bool merge = true;
        if (sr.mem_req->size >= req.size) {
          for (auto &interval : sr.start_end) {
            if ((interval.first < req.start && interval.first < req.end &&
                 req.end < interval.second) ||
                (req.start > interval.first && req.start < interval.second &&
                 req.end > interval.second) ||
                (req.start == interval.first && req.end == interval.second)) {
              merge = false;
              break;
            }
          }
        }

        if (merge) {
          req.offset = sr.mem_req->offset;
          memory_offset[req.loc] = req.offset;
          replace_and_fill = true;
          wgrad_sorted_req[idx].start_end.push_back(
            std::make_pair(req.start, req.end));
          reused_grad_size += req.size;
          reused_grad_cnt++;
          break;
        } else {
          replace_and_fill = false;
        }
      }
      if (replace_and_fill) {
        continue;
      }

      size_t offset = last_offset;
      if (!wgrad_sorted_req.empty())
        offset = wgrad_sorted_req.back().mem_req->offset +
                 wgrad_sorted_req.back().mem_req->size;

      req.offset = offset;
      memory_offset[req.loc] = offset;
      memory_req = std::max(memory_req, req.offset + req.size);
      wgrad_sorted_req.push_back(WGradMemoryRequest(&req));
      wgrad_sorted_req.back().start_end.push_back(
        std::make_pair(req.start, req.end));
      new_grad_cnt++;
      new_grad_size += req.size;
    }

    ml_logd("Total Requested Memory(OPTV2): %lf MiB>>>>>>>> \n - new mem for "
            "gradient = %d, "
            "(%lf MiB) & reused mem for gradient = %d (%lf MiB)\n",
            memory_req / 1024, new_grad_cnt, new_grad_size / 1024,
            reused_grad_cnt, reused_grad_size / 1024);
  }

  //   validateIntervalOverlap(memory_validity, memory_size, memory_offset,
  //   memory_req);

  return memory_req;
}

} // namespace nntrainer
